import argparse
import gc
import json
import os
import sys
import time
from typing import Any

import mteb
from mteb.models.model_meta import ModelMeta
import numpy as np
import torch
from torch import nn
from torch.nn.parallel import gather, parallel_apply, replicate, scatter

from transformers import AutoConfig, AutoTokenizer
from peft import PeftModel
from safetensors.torch import safe_open

from llm2vec.models import LlamaBiModel, MistralBiModel, GemmaBiModel, Qwen2BiModel


def get_model_class(config):
    name = config.__class__.__name__
    if name == "LlamaConfig":
        return LlamaBiModel
    elif name == "MistralConfig":
        return MistralBiModel
    elif name == "GemmaConfig":
        return GemmaBiModel
    elif name == "Qwen2Config":
        return Qwen2BiModel
    else:
        raise ValueError(f"Model class {name} not supported.")


class SqrtDNorm(nn.Module):
    def forward(self, hidden_states):
        dim = hidden_states.shape[-1]
        return hidden_states * (dim ** 0.5) / hidden_states.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)


class EncodeModule(nn.Module):
    """Backbone + SAE + pooling as one module, so it can be replicated per GPU."""

    def __init__(self, backbone, sae, sae_norm_scale, bos_token_id, jump_relu_threshold=0.0, sae_top_k=50):
        super().__init__()
        self.backbone = backbone
        self.bos_token_id = bos_token_id
        self.sae = sae
        self.sae_norm_scale = sae_norm_scale
        self.jump_relu_threshold = jump_relu_threshold
        self.sae_top_k = sae_top_k

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0] * self.sae_norm_scale
        sae_pre = self.sae(hidden_states)
        sae_out = torch.log(1 + torch.relu(sae_pre))
        # BOS stays in attention but is excluded from pooling (matches training).
        pool_mask = attention_mask * (input_ids != self.bos_token_id)
        sae_out = sae_out * pool_mask.unsqueeze(-1)
        pooled, _ = sae_out.max(dim=1)
        return pooled


class LayerwiseEncoder:
    def __init__(self, model_name_or_path, peft_model_name_or_path, sae_weights_path,
                 lora_layers, trained_checkpoint_path=None, max_length=1024,
                 torch_dtype=torch.bfloat16, attn_implementation="sdpa"):
        config = AutoConfig.from_pretrained(model_name_or_path)
        model_class = get_model_class(config)
        num_active = lora_layers + 1

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        backbone = model_class.from_pretrained(
            model_name_or_path,
            config=config,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )

        if peft_model_name_or_path is not None:
            backbone = PeftModel.from_pretrained(backbone, peft_model_name_or_path)
            backbone = backbone.merge_and_unload()

        # Truncate to layers we need and free the rest
        total_layers = len(backbone.layers)
        backbone.layers = backbone.layers[:num_active]
        backbone.norm = nn.Identity()
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(f"Truncated backbone: kept {num_active}/{total_layers} layers, "
                  f"GPU memory: {free/1024**3:.1f}GB free / {total/1024**3:.1f}GB total")
        else:
            print(f"Truncated backbone: kept {num_active}/{total_layers} layers")

        if trained_checkpoint_path is not None:
            backbone = PeftModel.from_pretrained(backbone, trained_checkpoint_path)

        with safe_open(sae_weights_path, framework="pt") as f:
            encoder_weight = f.get_tensor("encoder.weight")
            encoder_bias = f.get_tensor("encoder.bias")
        sae = nn.Linear(encoder_weight.shape[1], encoder_weight.shape[0])
        with torch.no_grad():
            sae.weight.copy_(encoder_weight)
            sae.bias.copy_(encoder_bias)
        sae.to(torch_dtype)
        sae.requires_grad_(False)

        # Load SAE hyperparams (JumpReLU threshold, TopK)
        sae_dir = os.path.dirname(os.path.dirname(sae_weights_path))
        sae_hyperparams_path = os.path.join(sae_dir, "hyperparams.json")
        with open(sae_hyperparams_path) as f:
            sae_hyperparams = json.load(f)
        self.jump_relu_threshold = sae_hyperparams["jump_relu_threshold"]
        self.sae_top_k = sae_hyperparams["top_k"]
        activation_norm = sae_hyperparams["dataset_average_activation_norm"]["in"]
        d_model = encoder_weight.shape[1]
        sae_norm_scale = (d_model ** 0.5) / activation_norm
        print(f"SAE hyperparams from {sae_hyperparams_path}:")
        print(f"  jump_relu_threshold: {self.jump_relu_threshold}")
        print(f"  top_k: {self.sae_top_k}")
        print(f"  activation_norm: {activation_norm}")
        print(f"  norm_activation: {sae_hyperparams.get('norm_activation', 'unknown')}")
        print(f"  sae_norm_scale: {sae_norm_scale:.4f}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        encode_module = EncodeModule(
            backbone, sae, sae_norm_scale, self.tokenizer.bos_token_id,
            self.jump_relu_threshold, self.sae_top_k,
        )
        encode_module.to(self.device)
        encode_module.eval()
        self.encode_module = encode_module
        self.device_ids = list(range(self.num_gpus))
        self.replicas = None
        if self.num_gpus > 1:
            # Weights are fixed at eval time, so copy the model to each GPU once
            # (nn.DataParallel re-replicates on every forward).
            with torch.no_grad():
                self.replicas = replicate(encode_module, self.device_ids, detach=True)
            print(f"Replicated model once onto {self.num_gpus} GPUs")
        self.sae_norm_scale = sae_norm_scale
        self.max_length = max_length
        self.d_sae = sae.out_features
        # One persistent module per GPU (the original module when there is one GPU).
        self.gpu_modules = self.replicas if self.replicas is not None else [encode_module]
        free, total = torch.cuda.mem_get_info()
        print(f"Model on GPU: {free/1024**3:.1f}GB free / {total/1024**3:.1f}GB total")
        self.bytes_per_token = self._calibrate_bytes_per_token()

    @torch.no_grad()
    def _calibrate_bytes_per_token(self, batch_size=8):
        """Measure peak activation memory per padded token with one dummy batch.

        Uses the longest allowed sequence length, so the (small) quadratic
        attention cost is covered for every shorter batch too.
        """
        seq_len = self.max_length
        device = torch.device("cuda", self.device_ids[0])
        # Ordinary (non-special) token ids; content does not affect memory.
        input_ids = torch.randint(0, self.tokenizer.bos_token_id, (batch_size, seq_len), device=device)
        attention_mask = torch.ones_like(input_ids)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        base = torch.cuda.memory_allocated(device)
        out = self.gpu_modules[0](input_ids, attention_mask)
        del out
        peak = torch.cuda.max_memory_allocated(device)
        bytes_per_token = (peak - base) / (batch_size * seq_len)
        print(f"Calibrated activation memory: {bytes_per_token / 1024**2:.3f} MB/token "
              f"(batch {batch_size} x {seq_len} tokens, peak +{(peak - base) / 1024**3:.1f}GB)")
        del input_ids, attention_mask
        torch.cuda.empty_cache()
        return bytes_per_token

    def _plan_batches(self, token_lengths, budget_bytes, bytes_per_token):
        """Greedily split texts into (start, end, max_len) batches, each fitting one GPU."""
        batches = []
        start = 0
        while start < len(token_lengths):
            batch_max_len = 0
            batch_end = start
            for i in range(start, len(token_lengths)):
                new_max_len = max(batch_max_len, token_lengths[i])
                new_count = i - start + 1
                if new_count * new_max_len * bytes_per_token > budget_bytes and new_count > 1:
                    break
                batch_max_len = new_max_len
                batch_end = i + 1
            batches.append((start, batch_end, batch_max_len))
            start = batch_end
        return batches

    @torch.no_grad()
    def forward_pooled(self, input_ids, attention_mask):
        """Pooled SAE vectors for a batch, split across the persistent GPU replicas."""
        if self.replicas is None:
            return self.encode_module(input_ids, attention_mask)
        inputs = scatter((input_ids, attention_mask), self.device_ids)
        n = len(inputs)
        outputs = parallel_apply(self.replicas[:n], inputs, devices=self.device_ids[:n])
        return gather(outputs, self.device_ids[0])

    def _pad_batch(self, id_lists, device):
        """Left-pad pre-tokenized sequences into (input_ids, attention_mask) on device."""
        max_len = max(len(ids) for ids in id_lists)
        input_ids = torch.full((len(id_lists), max_len), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(id_lists), max_len), dtype=torch.long)
        for i, ids in enumerate(id_lists):
            input_ids[i, max_len - len(ids):] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, max_len - len(ids):] = 1
        return input_ids.to(device), attention_mask.to(device)

    @torch.no_grad()
    def encode_texts(self, texts, top_k=None, safety_factor=1.25, margin_gb=10.0):
        all_chunks = []

        # Tokenize once; the ids are reused for every batch below.
        all_ids = self.tokenizer(texts, truncation=True, max_length=self.max_length)["input_ids"]
        token_lengths = [len(ids) for ids in all_ids]

        # Plan every batch up front against the tightest GPU's free memory.
        # Each planned batch runs whole on one GPU; consecutive batches go to
        # GPUs 0..n-1 in parallel, so each GPU pads only to its own longest text.
        torch.cuda.empty_cache()
        free_gb = min(torch.cuda.mem_get_info(d)[0] for d in self.device_ids) / 1024**3
        budget_gb = free_gb - margin_gb
        bytes_per_token = self.bytes_per_token * safety_factor
        batches = self._plan_batches(token_lengths, budget_gb * 1024**3, bytes_per_token)
        sizes = [end - start for start, end, _ in batches]
        n_gpus = len(self.gpu_modules)
        print(f"  planned {len(batches)} batches for {len(texts)} texts on {n_gpus} GPU(s) "
              f"(bs min={min(sizes)} max={max(sizes)}), "
              f"min free={free_gb:.1f}GB, per-GPU budget={budget_gb:.1f}GB")

        t0 = time.time()
        done = 0
        rounds = 0
        for r in range(0, len(batches), n_gpus):
            group = batches[r:r + n_gpus]
            k = len(group)
            inputs = [
                self._pad_batch(all_ids[start:end], torch.device("cuda", self.device_ids[i]))
                for i, (start, end, _) in enumerate(group)
            ]
            try:
                outputs = parallel_apply(self.gpu_modules[:k], inputs, devices=self.device_ids[:k])
                pooled = gather(outputs, self.device_ids[0])
                if top_k is not None:
                    vals, idx = pooled.topk(top_k, dim=-1)
                    pooled = torch.zeros_like(pooled)
                    pooled.scatter_(-1, idx, vals)
            except torch.cuda.OutOfMemoryError:
                desc = ", ".join(f"gpu{i}: bs={end - start} len={max_len}"
                                 for i, (start, end, max_len) in enumerate(group))
                print(f"CUDA OOM at text {group[0][0]}/{len(texts)} ({desc})", file=sys.stderr)
                sys.exit(1)

            all_chunks.append(pooled.cpu().float().numpy())
            del inputs, outputs, pooled

            prev, done = done, done + k
            rounds += 1
            if done // 100 > prev // 100 or done == len(batches):
                elapsed = time.time() - t0
                print(f"  batch {done}/{len(batches)}: {group[-1][1]}/{len(texts)} texts, "
                      f"{elapsed:.0f}s elapsed, {elapsed / rounds:.2f}s per round "
                      f"({n_gpus} batches in parallel)", flush=True)

        return np.concatenate(all_chunks, axis=0)


class MTEBWrapper:
    def __init__(self, encoder, query_top_k, doc_top_k, max_length=1024):
        self.encoder = encoder
        self.query_top_k = query_top_k
        self.doc_top_k = doc_top_k
        self._mteb_model_meta = ModelMeta(
            name="custom/layerwise-sparse-encoder",
            revision="0.0.1",
            release_date="2026-07-29",
            languages=["eng-Latn"],
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=max_length,
            embed_dim=32768,
            license=None,
            open_weights=True,
            public_training_code=None,
            public_training_data=None,
            framework=["PyTorch"],
            similarity_fn_name="dot",
            use_instructions=False,
            training_datasets=None,
            loader=None,
        )

    @property
    def mteb_model_meta(self):
        return self._mteb_model_meta

    def encode(self, inputs, *, task_metadata=None, hf_split=None, hf_subset=None,
               prompt_type=None, **kwargs):
        task_name = "unknown"
        if task_metadata is not None:
            if hasattr(task_metadata, "metadata"):
                task_name = task_metadata.metadata.name
            elif hasattr(task_metadata, "name"):
                task_name = task_metadata.name

        enc_type = "unknown"
        if prompt_type is not None:
            enc_type = prompt_type.value if hasattr(prompt_type, "value") else str(prompt_type)

        all_sentences = []
        for batch in inputs:
            sentences = batch["text"] if isinstance(batch, dict) else batch
            if isinstance(sentences, str):
                all_sentences.append(sentences)
            else:
                all_sentences.extend(sentences)

        top_k = self.query_top_k if enc_type in ("query",) else self.doc_top_k
        print(f"[DEBUG encode] task={task_name} enc_type={enc_type} "
              f"num_texts={len(all_sentences)} top_k={top_k}", flush=True)
        embeddings = self.encoder.encode_texts(all_sentences, top_k=top_k)
        nnz = (embeddings != 0).sum(axis=1)
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"[DEBUG encode] shape={embeddings.shape} "
              f"nnz_per_sample: mean={nnz.mean():.0f} min={nnz.min()} max={nnz.max()} "
              f"norm: mean={norms.mean():.2f} min={norms.min():.2f} max={norms.max():.2f}",
              flush=True)
        return embeddings

    def similarity(self, embeddings1, embeddings2):
        if isinstance(embeddings1, np.ndarray):
            embeddings1 = torch.from_numpy(embeddings1)
        if isinstance(embeddings2, np.ndarray):
            embeddings2 = torch.from_numpy(embeddings2)
        scores = torch.mm(embeddings1, embeddings2.T)
        print(f"[DEBUG similarity] shapes={embeddings1.shape}x{embeddings2.shape} "
              f"scores: mean={scores.mean():.2f} min={scores.min():.2f} max={scores.max():.2f}",
              flush=True)
        return scores

    def similarity_pairwise(self, embeddings1, embeddings2):
        if isinstance(embeddings1, np.ndarray):
            embeddings1 = torch.from_numpy(embeddings1)
        if isinstance(embeddings2, np.ndarray):
            embeddings2 = torch.from_numpy(embeddings2)
        return (embeddings1 * embeddings2).sum(dim=-1)


def verify_loss(encoder, hard_negatives_file, num_hard_negatives, temperature,
                lambda_q, lambda_d, max_seq_length, num_samples=128):
    """Compute training loss on a batch to verify the loaded model matches training."""
    with open(hard_negatives_file) as f:
        raw = json.load(f)

    samples = []
    for qid, item in raw.items():
        if not item["positives"]:
            continue
        pos = item["positives"][0]
        neg_items = item["hard_negatives"][:num_hard_negatives]
        if len(neg_items) < num_hard_negatives:
            continue
        query = item["query"]
        positive = pos["text"]
        negs = [n["text"] for n in neg_items]
        reranker_scores = [pos["reranker_score"]] + [n["reranker_score"] for n in neg_items]
        samples.append((query, positive, negs, reranker_scores))
        if len(samples) >= num_samples:
            break

    # Tokenize each text group separately (same as ContrastiveCollator)
    num_texts = 2 + num_hard_negatives  # query + positive + negatives
    text_groups = [[] for _ in range(num_texts)]
    all_reranker_scores = []
    for query, positive, negs, reranker_scores in samples:
        texts = [query, positive] + negs
        for i, t in enumerate(texts):
            text_groups[i].append(t)
        all_reranker_scores.append(reranker_scores)

    reranker_scores_tensor = torch.tensor(all_reranker_scores).to(encoder.device)

    tokenized_groups = []
    for group in text_groups:
        tokenized = encoder.tokenizer(
            group, padding=True, truncation=True, max_length=max_seq_length,
            return_tensors="pt"
        ).to(encoder.device)
        tokenized_groups.append(tokenized)

    # Forward pass (same as LayerwiseModel.forward)
    with torch.no_grad():
        pooled_list = []
        for tg in tokenized_groups:
            pooled = encoder.forward_pooled(tg["input_ids"], tg["attention_mask"])
            pooled_list.append(pooled)

        query_enc = pooled_list[0]
        pos_enc = pooled_list[1]
        neg_encs = pooled_list[2:]

        pos_score = (query_enc * pos_enc).sum(dim=-1, keepdim=True)
        neg_scores = torch.stack([(query_enc * neg).sum(dim=-1) for neg in neg_encs], dim=1)
        scores = torch.cat([pos_score, neg_scores], dim=1)

        log_pred = torch.log_softmax(scores / temperature, dim=1)
        target_probs = torch.softmax(reranker_scores_tensor, dim=1)
        kl_loss = nn.functional.kl_div(log_pred, target_probs, reduction="batchmean")

        query_flops = torch.sum(query_enc.mean(dim=0) ** 2)
        doc_flops = torch.sum(torch.cat(pooled_list[1:], dim=0).mean(dim=0) ** 2)

        loss = kl_loss + lambda_q * query_flops + lambda_d * doc_flops

    print(f"=== Verification on {len(samples)} training samples ===", flush=True)
    print(f"  KL loss:       {kl_loss.item():.6f}", flush=True)
    print(f"  Query FLOPS:   {query_flops.item():.6f}", flush=True)
    print(f"  Doc FLOPS:     {doc_flops.item():.6f}", flush=True)
    print(f"  Total loss:    {loss.item():.6f}", flush=True)
    print(f"  Pos score mean: {pos_score.mean().item():.4f}", flush=True)
    print(f"  Neg score mean: {neg_scores.mean().item():.4f}", flush=True)
    print(f"  Score range:   [{scores.min().item():.4f}, {scores.max().item():.4f}]", flush=True)


MTEB_ENG_V2_RETRIEVAL = [
    "ArguAna",
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
    "ClimateFEVER",
    "DBPedia",
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "MSMARCO",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_checkpoint_path", type=str, required=True,
                        help="Path to trained checkpoint dir (contains train config JSON).")
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--peft_model_name_or_path", type=str)
    parser.add_argument("--sae_weights_path", type=str)
    parser.add_argument("--lora_layers", type=int)
    parser.add_argument("--task_name", type=str, nargs="*")
    parser.add_argument("--task_type", type=str, choices=["retrieval", "all"])
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--query_top_k", type=int, default=40, help="0 disables pruning.")
    parser.add_argument("--doc_top_k", type=int, default=400, help="0 disables pruning.")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--hard_negatives_file", type=str)
    parser.add_argument("--num_hard_negatives", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--lambda_q", type=float)
    parser.add_argument("--lambda_d", type=float)
    parser.add_argument("--max_seq_length", type=int)
    args = parser.parse_args()
    args.query_top_k = args.query_top_k or None
    args.doc_top_k = args.doc_top_k or None

    # Auto-discover train config JSON from parent of checkpoint dir
    parent_dir = os.path.dirname(os.path.normpath(args.trained_checkpoint_path))
    config_candidates = [f for f in os.listdir(parent_dir)
                         if f.endswith(".json")]
    if not config_candidates:
        parser.error(f"No config JSON found in {parent_dir}")
    config_path = os.path.join(parent_dir, config_candidates[0])
    print(f"Loading train config from {config_path}")
    with open(config_path) as f:
        cfg = json.load(f)
    config_keys = [
        "model_name_or_path", "peft_model_name_or_path", "lora_layers",
        "sae_expansion",
        "hard_negatives_file", "num_hard_negatives", "temperature",
        "lambda_q", "lambda_d", "max_seq_length",
    ]
    for key in config_keys:
        if getattr(args, key, None) is None and key in cfg:
            setattr(args, key, cfg[key])

    # Resolve relative model paths against PKQ_DATA_DIR
    data_dir = os.environ.get("PKQ_DATA_DIR")
    if data_dir:
        for key in ("peft_model_name_or_path",):
            val = getattr(args, key, None)
            if val and not os.path.isabs(val):
                resolved = os.path.join(data_dir, val)
                if os.path.exists(resolved):
                    print(f"Resolved {key}: {val} -> {resolved}")
                    setattr(args, key, resolved)

    # Infer sae_weights_path from lora_layers if not provided
    if args.sae_weights_path is None and args.lora_layers is not None:
        from huggingface_hub import hf_hub_download
        sae_expansion = getattr(args, "sae_expansion", None) or 8
        args.sae_weights_path = hf_hub_download(
            f"OpenMOSS-Team/Llama3_1-8B-Base-LXR-{sae_expansion}x",
            f"Llama3_1-8B-Base-L{args.lora_layers}R-{sae_expansion}x/checkpoints/final.safetensors",
            local_files_only=True,
        )
        print(f"Inferred sae_weights_path: {args.sae_weights_path}")

    # Validate required args
    required = ["model_name_or_path", "sae_weights_path", "lora_layers"]
    missing = [k for k in required if getattr(args, k, None) is None]
    if missing:
        parser.error(f"Missing required arguments (set via config or CLI): {missing}")

    import traceback
    try:
        encoder = LayerwiseEncoder(
            model_name_or_path=args.model_name_or_path,
            peft_model_name_or_path=args.peft_model_name_or_path,
            sae_weights_path=args.sae_weights_path,
            lora_layers=args.lora_layers,
            trained_checkpoint_path=args.trained_checkpoint_path,
            max_length=args.max_length,
        )

        if args.hard_negatives_file is not None:
            verify_loss(encoder, args.hard_negatives_file, args.num_hard_negatives,
                        args.temperature, args.lambda_q, args.lambda_d, args.max_seq_length)
            gc.collect()
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                print(f"After verify_loss cleanup: {free/1024**3:.1f}GB free / {total/1024**3:.1f}GB total")
        else:
            print("Skipping verify_loss (no --hard_negatives_file provided)")

        model = MTEBWrapper(encoder, query_top_k=args.query_top_k,
                            doc_top_k=args.doc_top_k, max_length=args.max_length)

        if args.task_name:
            task_names = args.task_name
        else:
            task_names = MTEB_ENG_V2_RETRIEVAL

        tasks = mteb.get_tasks(tasks=task_names)
        evaluation = mteb.MTEB(tasks=tasks)
        print(f"Starting evaluation on {task_names}", flush=True)
        results = evaluation.run(model, output_folder=args.output_dir, overwrite_results=True)
        print(f"Evaluation complete. Results: {results}", flush=True)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

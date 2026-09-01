import argparse
import json
import os
import sys
from typing import Any

import mteb
from mteb.models.model_meta import ModelMeta
import numpy as np
import torch
from torch import nn

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


class LayerwiseEncoder:
    def __init__(self, model_name_or_path, peft_model_name_or_path, sae_weights_path,
                 lora_layers, trained_checkpoint_path=None, max_length=1024,
                 torch_dtype=torch.bfloat16, attn_implementation="sdpa"):
        config = AutoConfig.from_pretrained(model_name_or_path)
        model_class = get_model_class(config)

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

        num_active = lora_layers + 1
        backbone.layers = backbone.layers[:num_active]
        backbone.norm = nn.Identity()

        if trained_checkpoint_path is not None:
            backbone = PeftModel.from_pretrained(backbone, trained_checkpoint_path)

        self.backbone = backbone
        with safe_open(sae_weights_path, framework="pt") as f:
            encoder_weight = f.get_tensor("encoder.weight")
            encoder_bias = f.get_tensor("encoder.bias")
        sae = nn.Linear(encoder_weight.shape[1], encoder_weight.shape[0])
        with torch.no_grad():
            sae.weight.copy_(encoder_weight)
            sae.bias.copy_(encoder_bias)
        sae.to(torch_dtype)
        sae.requires_grad_(False)
        self.sae = sae

        # Load SAE hyperparams (JumpReLU threshold, TopK)
        sae_dir = os.path.dirname(os.path.dirname(sae_weights_path))
        sae_hyperparams_path = os.path.join(sae_dir, "hyperparams.json")
        with open(sae_hyperparams_path) as f:
            sae_hyperparams = json.load(f)
        self.jump_relu_threshold = sae_hyperparams["jump_relu_threshold"]
        self.sae_top_k = sae_hyperparams["top_k"]
        activation_norm = sae_hyperparams["dataset_average_activation_norm"]["in"]
        d_model = encoder_weight.shape[1]
        self.sae_norm_scale = (d_model ** 0.5) / activation_norm
        print(f"SAE hyperparams from {sae_hyperparams_path}:")
        print(f"  jump_relu_threshold: {self.jump_relu_threshold}")
        print(f"  top_k: {self.sae_top_k}")
        print(f"  activation_norm: {activation_norm}")
        print(f"  norm_activation: {sae_hyperparams.get('norm_activation', 'unknown')}")
        print(f"  sae_norm_scale: {self.sae_norm_scale:.4f}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone.to(self.device)
        self.sae.to(self.device)
        self.max_length = max_length
        self.backbone.eval()

    @torch.no_grad()
    def encode_texts(self, texts, batch_size=32, top_k=None):
        all_chunks = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            try:
                inputs = self.tokenizer(
                    batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
                ).to(self.device)

                outputs = self.backbone(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                hidden_states = outputs[0]
                # Dataset-wise normalization for Llama Scope SAE
                hidden_states = hidden_states * self.sae_norm_scale
                sae_out = self.sae(hidden_states)
                # JumpReLU + TopK (SAE activation from hyperparams.json)
                sae_out = torch.where(sae_out > self.jump_relu_threshold, sae_out, torch.zeros_like(sae_out))
                topk_vals, topk_idx = sae_out.topk(self.sae_top_k, dim=-1)
                sae_out = torch.zeros_like(sae_out).scatter_(-1, topk_idx, topk_vals)
                sae_out = torch.log(1 + sae_out)
                pooled, _ = sae_out.max(dim=1)
                if top_k is not None:
                    vals, idx = pooled.topk(top_k, dim=-1)
                    pooled = torch.zeros_like(pooled)
                    pooled.scatter_(-1, idx, vals)
            except torch.cuda.OutOfMemoryError:
                print(f"CUDA OOM at batch {start}-{start+batch_size} / {len(texts)}, batch_size={batch_size}", file=sys.stderr)
                sys.exit(1)

            all_chunks.append(pooled.cpu().float().numpy())
            del inputs, outputs, hidden_states, sae_out, pooled

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
            outputs = encoder.backbone(
                input_ids=tg["input_ids"], attention_mask=tg["attention_mask"]
            )
            hidden_states = outputs[0]
            # Dataset-wise normalization for Llama Scope SAE
            hidden_states = hidden_states * encoder.sae_norm_scale
            sae_out = encoder.sae(hidden_states)
            # JumpReLU + TopK (SAE activation from hyperparams.json)
            sae_out = torch.where(sae_out > encoder.jump_relu_threshold, sae_out, torch.zeros_like(sae_out))
            topk_vals, topk_idx = sae_out.topk(encoder.sae_top_k, dim=-1)
            sae_out = torch.zeros_like(sae_out).scatter_(-1, topk_idx, topk_vals)
            sae_out = torch.log(1 + sae_out)
            pooled, _ = sae_out.max(dim=1)
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
        doc_flops = sum(torch.sum(p.mean(dim=0) ** 2) for p in pooled_list[1:]) / len(pooled_list[1:])

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
    parser.add_argument("--config", type=str, default=None,
                        help="Path to training config JSON. Values are used as defaults; "
                             "CLI args override them.")
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--peft_model_name_or_path", type=str)
    parser.add_argument("--sae_weights_path", type=str)
    parser.add_argument("--trained_checkpoint_path", type=str)
    parser.add_argument("--lora_layers", type=int)
    parser.add_argument("--task_name", type=str, nargs="*")
    parser.add_argument("--task_type", type=str, choices=["retrieval", "all"])
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--query_top_k", type=int)
    parser.add_argument("--doc_top_k", type=int)
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--hard_negatives_file", type=str)
    parser.add_argument("--num_hard_negatives", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--lambda_q", type=float)
    parser.add_argument("--lambda_d", type=float)
    parser.add_argument("--max_seq_length", type=int)
    args = parser.parse_args()

    # Load config JSON and fill in any unset args
    if args.config is not None:
        with open(args.config) as f:
            cfg = json.load(f)
        config_keys = [
            "model_name_or_path", "peft_model_name_or_path", "lora_layers",
            "hard_negatives_file", "num_hard_negatives", "temperature",
            "lambda_q", "lambda_d", "max_seq_length",
        ]
        for key in config_keys:
            if getattr(args, key, None) is None and key in cfg:
                setattr(args, key, cfg[key])

    # Infer sae_weights_path from lora_layers if not provided
    if args.sae_weights_path is None and args.lora_layers is not None:
        args.sae_weights_path = (
            f"../remote_data/llm2vec/"
            f"Llama3_1-8B-Base-L{args.lora_layers}R-8x/checkpoints/final.safetensors"
        )
        print(f"Inferred sae_weights_path: {args.sae_weights_path}")

    # Validate required args
    required = ["model_name_or_path", "sae_weights_path", "lora_layers",
                 "hard_negatives_file", "num_hard_negatives", "temperature",
                 "lambda_q", "lambda_d", "max_seq_length"]
    missing = [k for k in required if getattr(args, k, None) is None]
    if missing:
        parser.error(f"Missing required arguments (set via --config or CLI): {missing}")

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

        verify_loss(encoder, args.hard_negatives_file, args.num_hard_negatives,
                    args.temperature, args.lambda_q, args.lambda_d, args.max_seq_length)

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

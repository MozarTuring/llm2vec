import argparse
import glob
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

        if trained_checkpoint_path is not None:
            backbone = PeftModel.from_pretrained(backbone, trained_checkpoint_path)

        self.backbone = backbone
        self.pre_sae_norm = SqrtDNorm()

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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone.to(self.device)
        self.sae.to(self.device)
        self.max_length = max_length
        self.backbone.eval()

    @torch.no_grad()
    def encode_texts(self, texts, batch_size=32, save_dir=None, top_k=None):
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            meta_path = os.path.join(save_dir, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                if meta["total"] == len(texts):
                    chunks = sorted(glob.glob(os.path.join(save_dir, "chunk_*.npy")))
                    if chunks:
                        return np.concatenate([np.load(p) for p in chunks], axis=0)

        chunk_idx = 0
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
                hidden_states = self.pre_sae_norm(hidden_states)
                sae_out = self.sae(hidden_states)
                sae_out = torch.log(1 + torch.relu(sae_out))
                pooled, _ = sae_out.max(dim=1)
                if top_k is not None:
                    vals, idx = pooled.topk(top_k, dim=-1)
                    pooled = torch.zeros_like(pooled)
                    pooled.scatter_(-1, idx, vals)
            except torch.cuda.OutOfMemoryError:
                print(f"CUDA OOM at batch {start}-{start+batch_size} / {len(texts)}, batch_size={batch_size}", file=sys.stderr)
                sys.exit(1)

            if save_dir is not None:
                np.save(os.path.join(save_dir, f"chunk_{chunk_idx}.npy"),
                        pooled.cpu().float().numpy())
            chunk_idx += 1

            del inputs, outputs, hidden_states, sae_out, pooled
            # torch.cuda.synchronize()
            # torch.cuda.empty_cache()

        if save_dir is not None:
            with open(os.path.join(save_dir, "meta.json"), "w") as f:
                json.dump({"total": len(texts)}, f)
            chunks = sorted(glob.glob(os.path.join(save_dir, "chunk_*.npy")))
            return np.concatenate([np.load(p) for p in chunks], axis=0)

        return np.concatenate([], axis=0)


class MTEBWrapper:
    def __init__(self, encoder, cache_dir, query_top_k, doc_top_k, max_length=1024):
        self.encoder = encoder
        self.cache_dir = cache_dir
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

        save_dir = os.path.join(self.cache_dir, task_name, enc_type)

        all_sentences = []
        for batch in inputs:
            sentences = batch["text"] if isinstance(batch, dict) else batch
            if isinstance(sentences, str):
                all_sentences.append(sentences)
            else:
                all_sentences.extend(sentences)

        top_k = self.query_top_k if enc_type in ("query",) else self.doc_top_k
        return self.encoder.encode_texts(all_sentences, save_dir=save_dir, top_k=top_k)

    def similarity(self, embeddings1, embeddings2):
        if isinstance(embeddings1, np.ndarray):
            embeddings1 = torch.from_numpy(embeddings1)
        if isinstance(embeddings2, np.ndarray):
            embeddings2 = torch.from_numpy(embeddings2)
        return torch.mm(embeddings1, embeddings2.T)

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
            hidden_states = encoder.pre_sae_norm(hidden_states)
            sae_out = encoder.sae(hidden_states)
            sae_out = torch.log(1 + torch.relu(sae_out))
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
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--peft_model_name_or_path", type=str, default=None)
    parser.add_argument("--sae_weights_path", type=str, required=True)
    parser.add_argument("--trained_checkpoint_path", type=str, default=None)
    parser.add_argument("--lora_layers", type=int, required=True)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--task_type", type=str, default="retrieval",
                        choices=["retrieval", "all"])
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--cache_dir", type=str, default="embedding_cache")
    parser.add_argument("--query_top_k", type=int, required=True)
    parser.add_argument("--doc_top_k", type=int, required=True)
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Maximum token length for tokenizer truncation (default: 1024)")
    parser.add_argument("--hard_negatives_file", type=str, required=True)
    parser.add_argument("--num_hard_negatives", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--lambda_q", type=float, required=True)
    parser.add_argument("--lambda_d", type=float, required=True)
    parser.add_argument("--max_seq_length", type=int, required=True)
    args = parser.parse_args()

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

        model = MTEBWrapper(encoder, cache_dir=args.cache_dir,
                            query_top_k=args.query_top_k, doc_top_k=args.doc_top_k,
                            max_length=args.max_length)

        if args.task_name:
            task_names = [args.task_name]
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

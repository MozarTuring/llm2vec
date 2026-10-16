"""MTEB eval of a multi-layer SPLARE checkpoint (trained with run_multilayer_finetune.py).

Each text is encoded once; the residual stream of every layer in sae_layers goes through that
layer's frozen SAE, is max-pooled, and the pooled vectors are concatenated. Pruning to
query_top_k / doc_top_k is over the concatenated vector, so the budget is shared by all layers
and matches a single-layer model at the same top-k.

Per-layer diagnostics:
    encode      share of non-zero features (after pruning) from each layer
    similarity  share of each query's top-10 scores from each layer (per similarity call)

    python experiments/mteb_eval_multilayer.py --trained_checkpoint_path <root>/output/multilayer/<run>/checkpoint-N --task_name SciFact NFCorpus ArguAna SCIDOCS
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback

import mteb
import numpy as np
import torch
from torch import nn
from torch.nn.parallel import replicate

from transformers import AutoConfig, AutoTokenizer
from peft import PeftModel
from safetensors.torch import safe_open

from mteb_eval_layerwise import LayerwiseEncoder, MTEBWrapper, get_model_class, infer_config_path


class MultilayerEncodeModule(nn.Module):
    """Backbone + one SAE per layer + pooling + concat, so it can be replicated per GPU."""

    def __init__(self, backbone, saes, sae_layers, sae_norm_scales, bos_token_id):
        super().__init__()
        self.backbone = backbone
        self.saes = saes
        self.sae_layers = sae_layers
        self.sae_norm_scales = sae_norm_scales
        self.bos_token_id = bos_token_id

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                                output_hidden_states=True)
        # BOS stays in attention but is excluded from pooling (matches training).
        pool_mask = attention_mask * (input_ids != self.bos_token_id)
        pooled = []
        for layer, sae, norm_scale in zip(self.sae_layers, self.saes, self.sae_norm_scales):
            # hidden_states[i + 1] is the output of layer i (matches training)
            sae_out = torch.log(1 + torch.relu(sae(outputs.hidden_states[layer + 1] * norm_scale)))
            sae_out = sae_out * pool_mask.unsqueeze(-1)
            pooled.append(sae_out.max(dim=1)[0])
        return torch.cat(pooled, dim=-1)


class MultilayerEncoder(LayerwiseEncoder):
    """LayerwiseEncoder with a multi-layer encode module; batching/encode_texts are inherited."""

    def __init__(self, model_name_or_path, peft_model_name_or_path, sae_weights_paths,
                 sae_layers, trained_checkpoint_path, max_length=1024,
                 torch_dtype=torch.bfloat16, attn_implementation="sdpa"):
        config = AutoConfig.from_pretrained(model_name_or_path)
        model_class = get_model_class(config)
        num_active = max(sae_layers) + 1

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        backbone = model_class.from_pretrained(
            model_name_or_path, config=config, torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        if peft_model_name_or_path is not None:
            backbone = PeftModel.from_pretrained(backbone, peft_model_name_or_path)
            backbone = backbone.merge_and_unload()

        total_layers = len(backbone.layers)
        backbone.layers = backbone.layers[:num_active]
        backbone.norm = nn.Identity()
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)
        free, total = torch.cuda.mem_get_info()
        print(f"Truncated backbone: kept {num_active}/{total_layers} layers, "
              f"GPU memory: {free/1024**3:.1f}GB free / {total/1024**3:.1f}GB total")

        backbone = PeftModel.from_pretrained(backbone, trained_checkpoint_path)

        saes, sae_norm_scales = nn.ModuleList(), []
        for sae_weights_path in sae_weights_paths:
            with safe_open(sae_weights_path, framework="pt") as f:
                encoder_weight = f.get_tensor("encoder.weight")
                encoder_bias = f.get_tensor("encoder.bias")
            sae = nn.Linear(encoder_weight.shape[1], encoder_weight.shape[0])
            with torch.no_grad():
                sae.weight.copy_(encoder_weight)
                sae.bias.copy_(encoder_bias)
            sae.to(torch_dtype)
            sae.requires_grad_(False)
            saes.append(sae)

            sae_dir = os.path.dirname(os.path.dirname(sae_weights_path))
            with open(os.path.join(sae_dir, "hyperparams.json")) as f:
                activation_norm = json.load(f)["dataset_average_activation_norm"]["in"]
            sae_norm_scales.append((encoder_weight.shape[1] ** 0.5) / activation_norm)
            print(f"SAE {sae_weights_path}: activation_norm={activation_norm} "
                  f"sae_norm_scale={sae_norm_scales[-1]:.4f}")

        self.device = torch.device("cuda")
        self.num_gpus = torch.cuda.device_count()
        encode_module = MultilayerEncodeModule(
            backbone, saes, sae_layers, sae_norm_scales, self.tokenizer.bos_token_id,
        )
        encode_module.to(self.device)
        encode_module.eval()
        self.encode_module = encode_module
        self.device_ids = list(range(self.num_gpus))
        self.replicas = None
        if self.num_gpus > 1:
            with torch.no_grad():
                self.replicas = replicate(encode_module, self.device_ids, detach=True)
            print(f"Replicated model once onto {self.num_gpus} GPUs")
        self.max_length = max_length
        self.d_sae = saes[0].out_features
        self.gpu_modules = self.replicas if self.replicas is not None else [encode_module]
        free, total = torch.cuda.mem_get_info()
        print(f"Model on GPU: {free/1024**3:.1f}GB free / {total/1024**3:.1f}GB total")
        self.bytes_per_token = self._calibrate_bytes_per_token()


class MultilayerMTEBWrapper(MTEBWrapper):
    """MTEBWrapper plus per-layer non-zero and score shares."""

    def __init__(self, encoder, sae_layers, query_top_k, doc_top_k, max_length=1024):
        super().__init__(encoder, query_top_k, doc_top_k, max_length)
        self.sae_layers = sae_layers
        self.d_sae = encoder.d_sae

    def _split(self, emb):
        return [emb[:, i * self.d_sae:(i + 1) * self.d_sae] for i in range(len(self.sae_layers))]

    def encode(self, inputs, **kwargs):
        embeddings = super().encode(inputs, **kwargs)
        nnz = [(part != 0).sum(axis=1) for part in self._split(embeddings)]
        total = np.maximum(sum(nnz), 1)
        shares = "  ".join(f"L{layer}={(n / total).mean():.1%} (nnz mean={n.mean():.0f})"
                           for layer, n in zip(self.sae_layers, nnz))
        print(f"[DEBUG encode] per-layer nnz share: {shares}", flush=True)
        return embeddings

    def similarity(self, embeddings1, embeddings2):
        scores = super().similarity(embeddings1, embeddings2)
        e1, e2 = (torch.from_numpy(e) if isinstance(e, np.ndarray) else e
                  for e in (embeddings1, embeddings2))
        k = min(10, scores.shape[1])
        top = scores.topk(k, dim=1).indices
        total = scores.gather(1, top)
        # Last layer's part is the remainder, saving one matmul
        partial = [torch.mm(q, d.T).gather(1, top)
                   for q, d in zip(self._split(e1)[:-1], self._split(e2)[:-1])]
        partial.append(total - sum(partial))
        total = total.clamp(min=1e-8)
        shares = "  ".join(f"L{layer}={(p / total).mean().item():.1%}"
                           for layer, p in zip(self.sae_layers, partial))
        print(f"[DEBUG similarity] per-layer share of top-{k} scores: {shares}", flush=True)
        return scores


def resolve_multilayer_args(args, parser):
    """Fill model/SAE args from the train config inferred from the checkpoint path (in place)."""
    config_path = infer_config_path(args.trained_checkpoint_path, parser)
    print(f"Loading train config from {config_path}")
    with open(config_path) as f:
        cfg = json.load(f)
    args.model_name_or_path = cfg["model_name_or_path"]
    args.peft_model_name_or_path = cfg.get("peft_model_name_or_path")
    args.sae_layers = sorted(cfg["sae_layers"])

    # Resolve relative model paths against PKQ_DATA_DIR
    data_dir = os.environ.get("PKQ_DATA_DIR")
    val = args.peft_model_name_or_path
    if data_dir and val and not os.path.isabs(val):
        resolved = os.path.join(data_dir, val)
        if os.path.exists(resolved):
            print(f"Resolved peft_model_name_or_path: {val} -> {resolved}")
            args.peft_model_name_or_path = resolved

    from huggingface_hub import hf_hub_download
    sae_expansion = cfg.get("sae_expansion", 8)
    args.sae_weights_paths = [
        hf_hub_download(
            f"OpenMOSS-Team/Llama3_1-8B-Base-LXR-{sae_expansion}x",
            f"Llama3_1-8B-Base-L{layer}R-{sae_expansion}x/checkpoints/final.safetensors",
            local_files_only=True,
        )
        for layer in args.sae_layers
    ]
    for layer, path in zip(args.sae_layers, args.sae_weights_paths):
        print(f"Inferred sae_weights_path for layer {layer}: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_checkpoint_path", type=str, required=True)
    parser.add_argument("--task_name", type=str, nargs="+",
                        default=["SciFact", "NFCorpus", "ArguAna", "SCIDOCS"])
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--query_top_k", type=int, default=40, help="0 disables pruning.")
    parser.add_argument("--doc_top_k", type=int, default=400, help="0 disables pruning.")
    parser.add_argument("--max_length", type=int, default=1024)
    args = parser.parse_args()
    args.query_top_k = args.query_top_k or None
    args.doc_top_k = args.doc_top_k or None
    resolve_multilayer_args(args, parser)

    try:
        encoder = MultilayerEncoder(
            model_name_or_path=args.model_name_or_path,
            peft_model_name_or_path=args.peft_model_name_or_path,
            sae_weights_paths=args.sae_weights_paths,
            sae_layers=args.sae_layers,
            trained_checkpoint_path=args.trained_checkpoint_path,
            max_length=args.max_length,
        )
        model = MultilayerMTEBWrapper(encoder, args.sae_layers, query_top_k=args.query_top_k,
                                      doc_top_k=args.doc_top_k, max_length=args.max_length)
        tasks = mteb.get_tasks(tasks=args.task_name)
        evaluation = mteb.MTEB(tasks=tasks)
        print(f"Starting evaluation on {args.task_name}", flush=True)
        results = evaluation.run(model, output_folder=args.output_dir, overwrite_results=True)
        print(f"Evaluation complete. Results: {results}", flush=True)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

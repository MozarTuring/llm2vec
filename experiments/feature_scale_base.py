"""Effect of the SAE input scale s on the base model: SPLARE's Eq. 1 without s vs the correct reading.

The released Llama Scope checkpoints are raw training weights: the SAE was trained on s * x with
s = sqrt(d_model) / activation_norm, so its pre-activation is pre = W_enc (s x) + b_enc
(feature_drift.SAE.pre). The SPLARE paper writes the encoder as W_enc x + b_enc. On the raw weights
that reads

    pre' = W_enc x + b_enc = SAE.pre(x / s).

Both readings are computed on the same base-model residuals (causal Llama-3.1-8B, layer 16, no
fine-tuning) over the corpus of feature_lexical.py, and compared with each other and with base's
JumpReLU gate on the correct reading (the activations the Neuronpedia labels describe).

Per token (BOS and padding excluded):
    pos_share      share of the 131072 features with a positive value (what log(1+relu) keeps)
    pos_gated      share of the positive values whose feature passes base's gate on that token
Per document, retrieval vector v = max over tokens of log(1 + relu(.)), top doc_top_k kept:
    precision      share of the top-k retrieval mass on features base's gate turns on somewhere in
                   the document (feature_precision.py's own-weighted precision)
    overlap        |top-k(v_scaled) & top-k(v_unscaled)| / k, mean over documents
    cos            cos(v_scaled, v_unscaled) over the full vectors, mean over documents
Per feature, Pearson correlation across documents of v_scaled and v_unscaled (features with v > 0 in
at least min_docs documents in both), mean and mass-weighted with each reading's mass.

One process per GPU: torchrun --nproc_per_node=N shards the documents (strided); statistics are
summed with all_reduce, rank 0 prints and writes results_feature_scale_base/feature_scale_base.json.
Plain python runs on one GPU.

    python experiments/feature_scale_base.py
"""

import argparse
import json
import os
import sys
import traceback

import numpy as np
import torch
import torch.distributed as dist
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

from dump_sparse import load_beir
from feature_drift import SAE, truncate
from feature_lexical import Corr, weighted

READINGS = ["scaled", "unscaled"]


def setup():
    if "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    return rank, world, torch.device("cuda")


def all_sum(t):
    if dist.is_initialized():
        dist.all_reduce(t)
    return t


@torch.no_grad()
def collect(texts, tokenizer, base, sae, args, device):
    W = sae.width
    corr = Corr(W, device)
    mass = {r: torch.zeros(W, dtype=torch.float64, device=device) for r in READINGS}
    hit = {r: torch.zeros(W, dtype=torch.float64, device=device) for r in READINGS}
    c = {k: torch.zeros((), dtype=torch.float64, device=device)
         for k in ["tokens", "docs", "overlap", "cos"]
         + [f"{x}_{r}" for r in READINGS for x in ("pos", "pos_gated")]}
    for i in range(0, len(texts), args.batch_size):
        batch = tokenizer(texts[i:i + args.batch_size], padding=True, truncation=True,
                          max_length=args.max_length, return_tensors="pt").to(device)
        ids, attn = batch["input_ids"], batch["attention_mask"]
        valid = attn.bool() & (ids != tokenizer.bos_token_id)
        x = base(input_ids=ids, attention_mask=attn)[0].float()
        pre = {"scaled": sae.pre(x), "unscaled": sae.pre(x / sae.s)}
        del x
        gate = sae.gate(pre["scaled"]) & valid.unsqueeze(-1)                 # (B, L, W)
        active_doc = gate.any(dim=1)                                          # (B, W)
        c["tokens"] += valid.sum()
        c["docs"] += ids.shape[0]

        v, top = {}, {}
        for r in READINGS:
            pos = (pre[r] > 0) & valid.unsqueeze(-1)
            c[f"pos_{r}"] += pos.sum()
            c[f"pos_gated_{r}"] += (pos & gate).sum()
            del pos
            v[r] = (torch.log1p(torch.relu(pre[r])) * valid.unsqueeze(-1)).max(dim=1)[0]
            vals, idx = v[r].topk(args.doc_top_k, dim=-1)
            vals = vals.double()
            mass[r].scatter_add_(0, idx.flatten(), vals.flatten())
            hit[r].scatter_add_(0, idx.flatten(), (vals * active_doc.gather(1, idx)).flatten())
            top[r] = torch.zeros_like(v[r], dtype=torch.bool).scatter_(1, idx, True)
        del pre, gate
        c["overlap"] += ((top["scaled"] & top["unscaled"]).sum(-1).double() / args.doc_top_k).sum()
        c["cos"] += torch.nn.functional.cosine_similarity(v["scaled"], v["unscaled"], dim=-1).double().sum()
        corr.update(v["scaled"], v["unscaled"])
        if (i // args.batch_size) % 100 == 0 and args.rank == 0:
            print(f"  rank 0: {i + ids.shape[0]}/{len(texts)} docs", flush=True)

    for t in [*mass.values(), *hit.values(), *c.values(),
              corr.sx, corr.sy, corr.sxx, corr.syy, corr.sxy, corr.nx, corr.ny]:
        all_sum(t)
    n = torch.tensor(float(corr.n), dtype=torch.float64, device=device)
    corr.n = int(all_sum(n).item())
    return corr, mass, hit, c


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--sae_expansion", type=int, default=32)
    parser.add_argument("--tasks", type=str, nargs="+", default=["SciFact", "NFCorpus", "SCIDOCS", "ArguAna"])
    parser.add_argument("--docs_per_task", type=int, default=3000)
    parser.add_argument("--doc_top_k", type=int, default=400)
    parser.add_argument("--min_docs", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="results_feature_scale_base")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        args.rank, world, device = setup()
        rng = np.random.default_rng(args.seed)
        texts = []
        for task in args.tasks:
            _, corpus, _ = load_beir(task)
            docs = list(corpus.values())
            texts += [docs[j] for j in rng.permutation(len(docs))[:args.docs_per_task]]
        # balanced shards: strided over length-sorted documents
        order = sorted(range(len(texts)), key=lambda j: len(texts[j]))
        shard = [texts[j] for j in order[args.rank::world]]
        if args.rank == 0:
            print(f"Corpus: {len(texts)} docs from {args.tasks}, {world} process(es)")

        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        tokenizer.padding_side = "right"
        base = truncate(AutoModel.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16,
                                                  attn_implementation="sdpa"), args.layer + 1)
        base = base.to(device).eval()
        sae_path = hf_hub_download(
            f"OpenMOSS-Team/Llama3_1-8B-Base-LXR-{args.sae_expansion}x",
            f"Llama3_1-8B-Base-L{args.layer}R-{args.sae_expansion}x/checkpoints/final.safetensors",
            local_files_only=True)
        sae = SAE(sae_path, device)

        corr, mass, hit, c = collect(shard, tokenizer, base, sae, args, device)

        if args.rank == 0:
            rho, valid = corr.corr(args.min_docs)
            rv = rho[valid]
            res = {"tokens": int(c["tokens"]), "docs": int(c["docs"]),
                   "overlap": float(c["overlap"] / c["docs"]), "cos": float(c["cos"] / c["docs"]),
                   "corr_features": int(valid.sum()), "corr_mean": float(rv.mean()),
                   "corr_mass_weighted_scaled": weighted(rv, mass["scaled"][valid]),
                   "corr_mass_weighted_unscaled": weighted(rv, mass["unscaled"][valid])}
            for r in READINGS:
                res[f"pos_share_{r}"] = float(c[f"pos_{r}"] / (c["tokens"] * sae.width))
                res[f"pos_gated_{r}"] = float(c[f"pos_gated_{r}"] / c[f"pos_{r}"].clamp(min=1))
                res[f"precision_{r}"] = float(hit[r].sum() / mass[r].sum())
                res[f"features_used_{r}"] = int((mass[r] > 0).sum())

            print(f"\n=== base model, layer {args.layer}: SAE input scaled by s={sae.s:.4f} vs unscaled, "
                  f"{res['docs']} docs, {res['tokens']} tokens ===")
            print(f"{'reading':10s}{'pos_share':>11s}{'pos_gated':>11s}{'precision':>11s}{'feat_used':>11s}")
            for r in READINGS:
                print(f"{r:10s}{res[f'pos_share_{r}']:>11.5f}{res[f'pos_gated_{r}']:>11.3f}"
                      f"{res[f'precision_{r}']:>11.3f}{res[f'features_used_{r}']:>11d}")
            print(f"\nscaled vs unscaled: top-{args.doc_top_k} overlap {res['overlap']:.3f}, "
                  f"cos {res['cos']:.3f}")
            print(f"per-feature correlation over {res['corr_features']} features: mean "
                  f"{res['corr_mean']:.3f}, weighted by scaled mass {res['corr_mass_weighted_scaled']:.3f}, "
                  f"by unscaled mass {res['corr_mass_weighted_unscaled']:.3f}")

            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, "feature_scale_base.json"), "w") as f:
                json.dump({"args": {k: v for k, v in vars(args).items()}, "s": sae.s, "results": res},
                          f, indent=2)
            print(f"\nSaved to {args.output_dir}/feature_scale_base.json")
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

"""SAE reconstruction of base-model residuals under the correct reading and under SPLARE's Eq. 1.

Companion of feature_scale_base.py (same corpus, base model only). The SPLARE paper writes the SAE as
z = f(W_enc x + b_enc), x_hat = W_dec z + b_dec. On the released raw Llama Scope weights the correct
form has the input scale s = sqrt(d_model) / activation_norm on both sides (feature_drift.SAE):
pre = W_enc (s x) + b_enc, x_hat = (W_dec a + b_dec) / s. Four variants, on the same residuals x
(BOS and padding excluded):

    scaled_gate     pre,  a = pre * gate(pre),  x_hat = (W_dec a + b_dec) / s     -- correct SAE
    scaled_relu     pre,  a = relu(pre),        x_hat = (W_dec a + b_dec) / s     -- ReLU instead of the gate
    unscaled_gate   pre', a = pre' * gate(pre'), x_hat = W_dec a + b_dec           -- Eq. 1 with JumpReLU
    unscaled_relu   pre', a = relu(pre'),       x_hat = W_dec a + b_dec           -- Eq. 1 with ReLU (SPLARE)

pre' = W_enc x + b_enc = SAE.pre(x / s); gate = feature_drift.SAE.gate (folded threshold rule).
Per variant: FVU = sum_t ||x_t - x_hat_t||^2 / sum_t ||x_t - x_bar||^2 (x_bar = mean residual over all
tokens), the L0 of a, and the mean ||x_hat||. Base's mean ||x|| is printed for scale.

One process per GPU: torchrun --nproc_per_node=N shards the documents (strided over length-sorted
order); sums are combined with all_reduce, rank 0 prints and writes
results_feature_scale_base_fvu/feature_scale_base_fvu.json. Plain python runs on one GPU.

    python experiments/feature_scale_base_fvu.py
"""

import argparse
import json
import os
import sys
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

from dump_sparse import load_beir
from feature_drift import SAE, truncate
from feature_scale_base import all_sum, setup

VARIANTS = ["scaled_gate", "scaled_relu", "unscaled_gate", "unscaled_relu"]


@torch.no_grad()
def collect(texts, tokenizer, base, sae, args, device):
    acc = {k: torch.zeros((), dtype=torch.float64, device=device)
           for k in ["tokens", "x_sq", "x_norm"]
           + [f"{x}_{v}" for v in VARIANTS for x in ("err", "l0", "xhat_norm")]}
    x_sum = torch.zeros(4096, dtype=torch.float64, device=device)
    for i in range(0, len(texts), args.batch_size):
        batch = tokenizer(texts[i:i + args.batch_size], padding=True, truncation=True,
                          max_length=args.max_length, return_tensors="pt").to(device)
        ids, attn = batch["input_ids"], batch["attention_mask"]
        valid = attn.bool() & (ids != tokenizer.bos_token_id)
        x = base(input_ids=ids, attention_mask=attn)[0].float()[valid]          # (n, d)
        acc["tokens"] += x.shape[0]
        acc["x_sq"] += (x.double() ** 2).sum()
        acc["x_norm"] += x.norm(dim=-1).double().sum()
        x_sum += x.double().sum(0)
        for reading in ("scaled", "unscaled"):
            pre = sae.pre(x) if reading == "scaled" else sae.pre(x / sae.s)
            for act in ("gate", "relu"):
                v = f"{reading}_{act}"
                a = pre * sae.gate(pre) if act == "gate" else torch.relu(pre)
                acc[f"l0_{v}"] += (a > 0).sum()
                if reading == "scaled":
                    x_hat = sae.decode(a)
                else:
                    x_hat = F.linear(a.to(torch.bfloat16), sae.W_dec).float() + sae.b_dec
                acc[f"err_{v}"] += ((x - x_hat).double() ** 2).sum()
                acc[f"xhat_norm_{v}"] += x_hat.norm(dim=-1).double().sum()
                del a, x_hat
            del pre
        if (i // args.batch_size) % 100 == 0 and args.rank == 0:
            print(f"  rank 0: {i + ids.shape[0]}/{len(texts)} docs", flush=True)
    for t in [*acc.values(), x_sum]:
        all_sum(t)
    return acc, x_sum


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--sae_expansion", type=int, default=32)
    parser.add_argument("--tasks", type=str, nargs="+", default=["SciFact", "NFCorpus", "SCIDOCS", "ArguAna"])
    parser.add_argument("--docs_per_task", type=int, default=3000)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="results_feature_scale_base_fvu")
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

        acc, x_sum = collect(shard, tokenizer, base, sae, args, device)

        if args.rank == 0:
            n = acc["tokens"]
            total_var = acc["x_sq"] - (x_sum ** 2).sum() / n          # sum_t ||x_t - x_bar||^2
            res = {"tokens": int(n), "s": sae.s, "x_norm": float(acc["x_norm"] / n)}
            print(f"\n=== base model, layer {args.layer}: SAE reconstruction, {int(n)} tokens, "
                  f"s={sae.s:.4f}, mean ||x|| {res['x_norm']:.2f} ===")
            print(f"{'variant':15s}{'FVU':>9s}{'L0':>9s}{'||x_hat||':>11s}")
            for v in VARIANTS:
                res[v] = {"fvu": float(acc[f"err_{v}"] / total_var), "l0": float(acc[f"l0_{v}"] / n),
                          "xhat_norm": float(acc[f"xhat_norm_{v}"] / n)}
                print(f"{v:15s}{res[v]['fvu']:>9.3f}{res[v]['l0']:>9.1f}{res[v]['xhat_norm']:>11.2f}")
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, "feature_scale_base_fvu.json"), "w") as f:
                json.dump({"args": vars(args), "results": res}, f, indent=2)
            print(f"\nSaved to {args.output_dir}/feature_scale_base_fvu.json")
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

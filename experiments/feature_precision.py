"""Firing precision vs base: does a model put retrieval value where base's feature is active?

Pearson correlation with base (feature_lexical.py) counts sparsification as drift: a feature that a
model keeps only on a subset of base's documents (featgate's negative biases) loses correlation
although it fires on nothing new. This measure is one-sided. Per feature i in model m:

    P_{m,i} = sum_d v_{m,d,i} 1[i in top_400(v_{m,d})] 1[i active in base on some token of d]
              / sum_d v_{m,d,i} 1[i in top_400(v_{m,d})]

"active in base" = base's JumpReLU gate (feature_drift.SAE.gate), the activations the Neuronpedia
labels describe. P = 1: all of the feature's retrieval value is on documents where base's feature
fires; firing on fewer documents than base does not lower it. Per-feature version of mass_off_base
(feature_drift.py), over the same 12000-document corpus as feature_lexical.py.

Aggregated over features with retrieval mass w_{m,i} > 0, with three weightings:
    own   w_{m,i}            = share of the model's total retrieval mass on base-active (doc, feature)
    base  w_{b,i}            (same weights for every model)
    mntp  w_{mntp,i}         (same weights for every model)
plus coverage: share of base's mass on features the model uses at all (w_{m,i} > 0).

Models: base (reference: its sub-threshold retrieval value), mntp, splare (the checkpoint), and
featgate (mntp pre-activation * gain + bias) when --featgate_path is given.

    python experiments/feature_precision.py --trained_checkpoint_path <root>/output/<group>/<run>/checkpoint-N [--featgate_path .../featgate.pt]
"""

import argparse
import json
import os
import sys
import traceback

import numpy as np
import torch
from transformers import AutoTokenizer

from dump_sparse import load_beir
from feature_drift import SAE, load_models
from feature_lexical import _no_adapter, pooled_values
from mteb_eval_layerwise import resolve_checkpoint_args

WEIGHTINGS = ["own", "base", "mntp"]


@torch.no_grad()
def collect(texts, tokenizer, base, bi, sae, gain, bias, args, device):
    W = sae.width
    models = ["base", "mntp", "splare"] + (["featgate"] if gain is not None else [])
    mass = {m: torch.zeros(W, dtype=torch.float64, device=device) for m in models}
    hit = {m: torch.zeros(W, dtype=torch.float64, device=device) for m in models}
    for i in range(0, len(texts), args.batch_size):
        chunk = texts[i:i + args.batch_size]
        batch = tokenizer(chunk, padding=True, truncation=True, max_length=args.max_length,
                          return_tensors="pt").to(device)
        ids, attn = batch["input_ids"], batch["attention_mask"]
        valid = attn.bool() & (ids != tokenizer.bos_token_id)

        pooled = {}
        pre, pooled["base"] = pooled_values(base, sae, ids, attn, valid)
        active_b = (sae.gate(pre) & valid.unsqueeze(-1)).any(dim=1)          # (B, W)
        del pre
        pre, pooled["mntp"] = pooled_values(lambda **kw: _no_adapter(bi, **kw), sae, ids, attn, valid)
        if gain is not None:
            gated = torch.log1p(torch.relu(pre * gain + bias)) * valid.unsqueeze(-1)
            pooled["featgate"] = gated.max(dim=1)[0]
            del gated
        del pre
        _, pooled["splare"] = pooled_values(bi, sae, ids, attn, valid)

        for m in models:
            vals, idx = pooled[m].topk(args.doc_top_k, dim=-1)
            vals = vals.double()
            mass[m].scatter_add_(0, idx.flatten(), vals.flatten())
            hit[m].scatter_add_(0, idx.flatten(), (vals * active_b.gather(1, idx)).flatten())
        if (i // args.batch_size) % 100 == 0:
            print(f"  {i + len(chunk)}/{len(texts)} docs", flush=True)
    return models, mass, hit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_checkpoint_path", type=str, required=True)
    parser.add_argument("--featgate_path", type=str, help="featgate.pt; adds the featgate model.")
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--peft_model_name_or_path", type=str)
    parser.add_argument("--sae_weights_path", type=str)
    parser.add_argument("--lora_layers", type=int)
    parser.add_argument("--tasks", type=str, nargs="+", default=["SciFact", "NFCorpus", "SCIDOCS", "ArguAna"])
    parser.add_argument("--docs_per_task", type=int, default=3000)
    parser.add_argument("--doc_top_k", type=int, default=400)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="results_feature_precision")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    resolve_checkpoint_args(args, parser)

    try:
        device = torch.device("cuda")
        rng = np.random.default_rng(args.seed)
        texts = []
        for task in args.tasks:
            _, corpus, _ = load_beir(task)
            docs = list(corpus.values())
            texts += [docs[j] for j in rng.permutation(len(docs))[:args.docs_per_task]]
        print(f"Corpus: {len(texts)} docs from {args.tasks}")

        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        tokenizer.padding_side = "right"
        base, bi = load_models(args, device)
        sae = SAE(args.sae_weights_path, device)
        gain = bias = None
        if args.featgate_path:
            gate_ckpt = torch.load(args.featgate_path, map_location="cpu")
            gain = gate_ckpt["log_gain"].float().exp().to(device)
            bias = gate_ckpt["bias"].float().to(device)
            print(f"Gate {args.featgate_path}: gain mean={gain.mean():.4f} bias mean={bias.mean():.3f}")

        models, mass, hit = collect(texts, tokenizer, base, bi, sae, gain, bias, args, device)

        print(f"\n=== firing precision vs base's gate across {len(texts)} docs "
              f"(top-{args.doc_top_k} retrieval value per doc) ===")
        print(f"{'model':10s}{'features':>10s}{'coverage':>10s}" + "".join(f"{'P_' + w:>9s}" for w in WEIGHTINGS))
        results = {}
        for m in models:
            used = mass[m] > 0
            prec = hit[m][used] / mass[m][used]
            r = {"features": int(used.sum()),
                 "coverage_base_mass": float(mass["base"][used].sum() / mass["base"].sum())}
            for w in WEIGHTINGS:
                wt = mass[m if w == "own" else w][used]
                r[f"precision_{w}"] = float((prec * wt).sum() / wt.sum().clamp(min=1e-12))
            results[m] = r
            print(f"{m:10s}{r['features']:>10d}{r['coverage_base_mass']:>10.3f}"
                  + "".join(f"{r['precision_' + w]:>9.3f}" for w in WEIGHTINGS))

        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "feature_precision.json"), "w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=2)
        print(f"\nSaved to {args.output_dir}/feature_precision.json")
    except Exception:
        traceback.print_exc()
        sys.exit(1)

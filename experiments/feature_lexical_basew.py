"""Mass-weighted feature correlation with a fixed weighting, independent of the compared model.

Variant of feature_lexical.py, part A only. feature_lexical.py weights each pair's per-feature
correlation with the compared model's own retrieval mass, so a model can raise its score by moving
mass onto features that barely change (anchor with lambda_a = 3 exceeded base_half: 0.826 vs 0.806).
Here the same correlations (same corpus, seed, Corr, pooled values) are averaged with three
weightings, over the features compared for that pair:

    own   the compared model's mass (base's for base_half) -- reproduces feature_lexical.py
    base  base's mass, the same weights for every pair
    mntp  mntp's mass, the same weights for every pair

    python experiments/feature_lexical_basew.py --trained_checkpoint_path <root>/output/<group>/<run>/checkpoint-N
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
from feature_lexical import PAIRS, collect, weighted
from mteb_eval_layerwise import resolve_checkpoint_args

WEIGHTINGS = ["own", "base", "mntp"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_checkpoint_path", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--peft_model_name_or_path", type=str)
    parser.add_argument("--sae_weights_path", type=str)
    parser.add_argument("--lora_layers", type=int)
    parser.add_argument("--tasks", type=str, nargs="+", default=["SciFact", "NFCorpus", "SCIDOCS", "ArguAna"])
    parser.add_argument("--docs_per_task", type=int, default=3000)
    parser.add_argument("--k", type=int, default=20, help="Top-K tokens per feature (collect() needs it).")
    parser.add_argument("--min_docs", type=int, default=10)
    parser.add_argument("--doc_top_k", type=int, default=400)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="results_feature_lexical_basew")
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

        corr, _, mass, _ = collect(texts, tokenizer, base, bi, sae, args, device)

        print(f"\n=== mass-weighted correlation with base across {len(texts)} docs, by weighting "
              f"(features active in >= {args.min_docs} docs in both) ===")
        print(f"{'pair':10s}{'features':>10s}{'mean':>8s}" + "".join(f"{'w_' + w:>9s}" for w in WEIGHTINGS))
        results = {}
        for p in PAIRS:
            c, valid = corr[p].corr(args.min_docs)
            cv = c[valid]
            r = {"features": int(valid.sum()), "mean": float(cv.mean())}
            for w in WEIGHTINGS:
                key = ("base" if p == "base_half" else p) if w == "own" else w
                r[f"mass_weighted_{w}"] = weighted(cv, mass[key][valid])
            results[p] = r
            print(f"{p:10s}{r['features']:>10d}{r['mean']:>8.3f}"
                  + "".join(f"{r['mass_weighted_' + w]:>9.3f}" for w in WEIGHTINGS))

        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "feature_lexical_basew.json"), "w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=2)
        print(f"\nSaved to {args.output_dir}/feature_lexical_basew.json")
    except Exception:
        traceback.print_exc()
        sys.exit(1)

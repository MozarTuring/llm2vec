"""Mass-weighted feature correlation with fixed weights, for featgate (and splare) next to mntp.

Variant of feature_lexical_basew.py for the feature-gate model, which feature_lexical_basew.py
cannot load (featgate = mntp SAE pre-activation * gain + bias, from featgate.pt). Uses
feature_lexical_featgate.collect (same corpus, seed, pooled values and correlations as
feature_lexical_featgate.py) and averages each pair's per-feature correlation with base with three
weightings, over the features compared for that pair:

    own   the compared model's mass (base's for base_half) -- reproduces feature_lexical_featgate.py
    base  base's mass, the same weights for every pair
    mntp  mntp's mass, the same weights for every pair

    python experiments/feature_lexical_featgate_basew.py --trained_checkpoint_path <root>/output/layerwise/<run>/checkpoint-N --featgate_path <root>/output/featgate/<run>/checkpoint-N/featgate.pt
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
from feature_lexical import weighted
from feature_lexical_featgate import PAIRS, collect
from mteb_eval_layerwise import resolve_checkpoint_args

WEIGHTINGS = ["own", "base", "mntp"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_checkpoint_path", type=str, required=True,
                        help="LoRA SPLARE checkpoint (the splare model; also gives the train config).")
    parser.add_argument("--featgate_path", type=str, required=True, help="featgate.pt of the gate run.")
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
    parser.add_argument("--output_dir", type=str, default="results_feature_lexical_featgate_basew")
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
        gate_ckpt = torch.load(args.featgate_path, map_location="cpu")
        gain = gate_ckpt["log_gain"].float().exp().to(device)
        bias = gate_ckpt["bias"].float().to(device)
        print(f"Gate {args.featgate_path}: gain mean={gain.mean():.4f} max={gain.max():.2f}  "
              f"bias mean={bias.mean():.3f}")

        corr, _, mass, _, _ = collect(texts, tokenizer, base, bi, sae, gain, bias, args, device)

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
        with open(os.path.join(args.output_dir, "feature_lexical_featgate_basew.json"), "w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=2)
        print(f"\nSaved to {args.output_dir}/feature_lexical_featgate_basew.json")
    except Exception:
        traceback.print_exc()
        sys.exit(1)

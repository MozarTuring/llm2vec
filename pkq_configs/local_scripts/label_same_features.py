"""Label-check scores of one model's top features, judged in every model (same features and labels).

Weights: the reference model's retrieval mass share. Features without a score in a model (no
positive or no negative snippets) are left out for that model.

    python3 -I pkq_configs/local_scripts/label_same_features.py <label_judge_*.json> [--ref splare]
"""

import argparse
import json

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("judge")
parser.add_argument("--ref", default="splare")
args = parser.parse_args()

judge = json.load(open(args.judge))
F = judge["features"]
top = [f for f, r in F[args.ref].items() if r["top"]]
print(f"{args.ref}'s top {len(top)} features")
print(f"{'model':10s}{'n':>5s}{'BA':>9s}{'BA0':>9s}{'TPR':>9s}{'TNR':>9s}{'docs_med':>10s}")
for m, feats in F.items():
    fs = [f for f in top if feats[f].get("BA") is not None]
    w = np.array([F[args.ref][f]["mass_share"] for f in fs])
    avg = lambda k: float(np.dot([feats[f][k] for f in fs], w) / w.sum())
    docs = np.median([feats[f]["positive_docs"] for f in fs])
    print(f"{m:10s}{len(fs):>5d}{avg('BA'):>9.4f}{avg('BA0'):>9.4f}{avg('TPR'):>9.4f}{avg('TNR'):>9.4f}{docs:>10.0f}")

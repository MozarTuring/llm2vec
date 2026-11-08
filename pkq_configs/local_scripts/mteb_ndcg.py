"""
    python pkq_configs/local_scripts/mteb_ndcg.py <results_dir> [<results_dir> ...]
"""

import glob
import json
import os
import sys

TASKS = ["SciFact", "NFCorpus", "ArguAna", "SCIDOCS"]

for root in sys.argv[1:]:
    scores = {}
    for path in glob.glob(os.path.join(root, "**", "*.json"), recursive=True):
        task = os.path.splitext(os.path.basename(path))[0]
        if task not in TASKS:
            continue
        with open(path) as f:
            data = json.load(f)
        scores[task] = data["scores"]["test"][0]["ndcg_at_10"]
    row = [scores.get(t) for t in TASKS]
    mean = sum(row) / len(row) if all(v is not None for v in row) else None
    cells = "  ".join(f"{t}={v * 100:.4f}" if v is not None else f"{t}=NA" for t, v in zip(TASKS, row))
    print(f"{root}\n  {cells}  mean={mean * 100:.4f}" if mean is not None else f"{root}\n  {cells}")

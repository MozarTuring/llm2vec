"""Label-check scores by how many documents a feature fires in (terciles of positive_docs, per model).

    python3 -I pkq_configs/local_scripts/label_by_doc_count.py <label_judge_*.json>
"""

import json
import sys

import numpy as np

judge = json.load(open(sys.argv[1]))
for m, feats in judge["features"].items():
    rows = [r for r in feats.values() if r["top"] and r.get("BA") is not None]
    docs = np.array([r["positive_docs"] for r in rows])
    ba = np.array([r["BA"] for r in rows])
    tpr = np.array([r["TPR"] for r in rows])
    q = np.quantile(docs, [1 / 3, 2 / 3])
    print(f"{m}: positive_docs terciles {q.astype(int).tolist()}")
    for lo, hi, name in [(0, q[0], "few"), (q[0], q[1], "mid"), (q[1], np.inf, "many")]:
        k = (docs >= lo) & (docs < hi)
        print(f"   {name:5s} n={k.sum():3d}  BA {ba[k].mean():.4f}  TPR {tpr[k].mean():.4f}")
    print(f"   corr(log positive_docs, TPR) {np.corrcoef(np.log(docs), tpr)[0, 1]:.4f}")

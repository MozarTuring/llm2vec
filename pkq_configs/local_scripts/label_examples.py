"""Labels, scores and firing snippets of the top features (by retrieval mass) of the given models.

    python3 -I pkq_configs/local_scripts/label_examples.py <label_judge_*.json> <label_snippets_*.json> [--models base splare] [--n 4]
"""

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("judge")
parser.add_argument("snippets")
parser.add_argument("--models", nargs="+", default=["base", "splare"])
parser.add_argument("--n", type=int, default=4)
args = parser.parse_args()

judge = json.load(open(args.judge))
snippets = json.load(open(args.snippets))

for m in args.models:
    rows = sorted([(r["mass_share"], f) for f, r in judge["features"][m].items()
                   if r["top"] and r.get("BA") is not None], reverse=True)[:args.n]
    for _, f in rows:
        r = judge["features"][m][f]
        print(f"\n[{m}] feature {f}  docs {r['positive_docs']}  BA {r['BA']:.4f}  TPR {r['TPR']:.4f}  "
              f"label: {r['label']!r}")
        pos = snippets["models"][m][f]["positives"]
        for x in pos[:3] + pos[5:7]:
            print("   +", repr(x["text"][-80:]))

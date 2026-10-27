"""Label check: do the Neuronpedia labels still describe where each model's features fire?

Reads label_snippets.py's output and Neuronpedia's labels (explanations export of
llama3.1-8b/16-llamascope-res-131k, written by gpt-4o-mini from base's activations). For every model,
feature with a label, and snippet, an LLM judge (default Qwen2.5-72B-Instruct) is asked whether the
label describes the marked token; the answer is Yes if P(Yes) > P(No) for the first answer token.

Per model m and feature i, with the positive (feature in the top-400 retrieval vector) and negative
snippets of label_snippets.py:
    TPR   share of positives judged Yes
    TNR   share of negatives judged No
    BA    balanced accuracy (TPR + TNR) / 2; 0.5 = label carries no information, 1 = label separates
          exactly where the feature fires
    BA0   the same snippets judged with the label of another selected feature (fixed random
          derangement): the floor
Aggregates per model: BA and BA0 averaged over the model's own top features weighted by its own
retrieval mass, and over base's top features weighted by base's mass (same features and weights for
every model).

One process: the judge is spread over all visible GPUs (device_map="auto"); a 72B model does not
fit on one GPU, so this does not use torchrun.

    python experiments/label_judge.py --snippets ${PKQ_DATA_DIR}/backup/<ts>/results_label_snippets/label_snippets_<name>.json
"""

import argparse
import glob
import gzip
import json
import os
import sys
import traceback

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = """A feature of a language model is described by this explanation:
"{label}"

Here is a text excerpt. The token marked as <<token>> is the token being evaluated.

{snippet}

Does the explanation describe the marked token in this context, i.e. would you expect this feature \
to be active on it? Answer Yes or No."""


def load_labels(labels_dir):
    """Feature index -> first label in the export, and the number of labels per feature."""
    labels, counts = {}, {}
    for path in sorted(glob.glob(os.path.join(labels_dir, "batch-*.jsonl.gz"))):
        with gzip.open(path, "rt") as f:
            for line in f:
                e = json.loads(line)
                i = int(e["index"])
                counts[i] = counts.get(i, 0) + 1
                labels.setdefault(i, e["description"].strip())
    return labels, counts


@torch.no_grad()
def p_yes(pairs, tokenizer, judge, yes_id, no_id, batch_size):
    """P(Yes) / (P(Yes) + P(No)) for each (label, snippet)."""
    prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT.format(label=l, snippet=s)}],
        tokenize=False, add_generation_prompt=True) for l, s in pairs]
    device = judge.get_input_embeddings().weight.device
    out = []
    for i in range(0, len(prompts), batch_size):
        batch = tokenizer(prompts[i:i + batch_size], return_tensors="pt", padding=True).to(device)
        logits = judge(**batch).logits[:, -1, [yes_id, no_id]].float()
        out += torch.softmax(logits, dim=-1)[:, 0].tolist()
        if (i // batch_size) % 50 == 0:
            print(f"  judged {i + len(batch['input_ids'])}/{len(prompts)}", flush=True)
    return out


def weighted(x, w):
    w = np.asarray(w, dtype=np.float64)
    return float(np.dot(x, w) / w.sum()) if w.sum() > 0 else float("nan")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snippets", type=str, required=True)
    parser.add_argument("--labels_dir", type=str,
                        default=os.path.join(os.environ.get("PKQ_DATA_DIR", "."),
                                             "neuronpedia/16-llamascope-res-131k/explanations"))
    parser.add_argument("--judge", type=str, default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="results_label_judge")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        with open(args.snippets) as f:
            data = json.load(f)
        labels, counts = load_labels(args.labels_dir)
        print(f"Labels: {len(labels)} features, {sum(c > 1 for c in counts.values())} with more than one")
        feats = [f for f in data["features"] if f in labels]
        print(f"{len(feats)}/{len(data['features'])} selected features have a label")
        rng = np.random.default_rng(args.seed)
        while True:                                    # derangement: no feature keeps its own label
            perm = rng.permutation(len(feats))
            if not (perm == np.arange(len(feats))).any():
                break
        other = {f: labels[feats[k]] for f, k in zip(feats, perm)}

        tokenizer = AutoTokenizer.from_pretrained(args.judge)
        tokenizer.padding_side = "left"
        judge = AutoModelForCausalLM.from_pretrained(args.judge, torch_dtype=torch.bfloat16,
                                                     device_map="auto").eval()
        yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
        no_id = tokenizer.encode("No", add_special_tokens=False)[0]

        # all (model, feature, kind, label type) questions in one list
        keys, pairs = [], []
        for m, rows in data["models"].items():
            for f in feats:
                r = rows[str(f)]
                for kind in ("positives", "negatives"):
                    for s in r[kind]:
                        for lab, text in (("label", labels[f]), ("other", other[f])):
                            keys.append((m, f, kind, lab))
                            pairs.append((text, s["text"]))
        print(f"{len(pairs)} questions")
        probs = p_yes(pairs, tokenizer, judge, yes_id, no_id, args.batch_size)

        per = {}
        for (m, f, kind, lab), p in zip(keys, probs):
            per.setdefault((m, f, lab), {"positives": [], "negatives": []})[kind].append(p)

        results, features = {}, {}
        base_top = [f for f in feats if data["models"]["base"][str(f)]["top"]]
        base_w = [data["models"]["base"][str(f)]["mass_share"] for f in base_top]
        for m, rows in data["models"].items():
            feat_res = {}
            for f in feats:
                r = {"label": labels[f], "mass_share": rows[str(f)]["mass_share"], "top": rows[str(f)]["top"],
                     "positive_docs": rows[str(f)]["positive_docs"]}
                for lab, suffix in (("label", ""), ("other", "0")):
                    pp, pn = per[(m, f, lab)]["positives"], per[(m, f, lab)]["negatives"]
                    if not pp or not pn:
                        r["BA" + suffix] = None
                        continue
                    tpr = float(np.mean(np.array(pp) > 0.5))
                    tnr = float(np.mean(np.array(pn) <= 0.5))
                    r.update({"TPR" + suffix: tpr, "TNR" + suffix: tnr, "BA" + suffix: (tpr + tnr) / 2})
                feat_res[str(f)] = r
            features[m] = feat_res

            own = [f for f in feats if feat_res[str(f)]["top"] and feat_res[str(f)]["BA"] is not None]
            on_base = [f for f in base_top if feat_res[str(f)]["BA"] is not None]
            res = {"features_own": len(own), "features_base": len(on_base)}
            for key in ("BA", "BA0", "TPR", "TNR"):
                res[f"{key}_own"] = weighted([feat_res[str(f)][key] for f in own],
                                             [feat_res[str(f)]["mass_share"] for f in own])
                res[f"{key}_base"] = weighted([feat_res[str(f)][key] for f in on_base],
                                              [data["models"]["base"][str(f)]["mass_share"] for f in on_base])
            results[m] = res

        print(f"\n=== label check, judge {args.judge} ===")
        print(f"{'model':12s}{'n_own':>7s}{'BA_own':>9s}{'BA0_own':>9s}{'TPR_own':>9s}{'TNR_own':>9s}"
              f"{'n_base':>8s}{'BA_base':>9s}{'BA0_base':>10s}")
        for m, r in results.items():
            print(f"{m:12s}{r['features_own']:>7d}{r['BA_own']:>9.4f}{r['BA0_own']:>9.4f}{r['TPR_own']:>9.4f}"
                  f"{r['TNR_own']:>9.4f}{r['features_base']:>8d}{r['BA_base']:>9.4f}{r['BA0_base']:>10.4f}")

        os.makedirs(args.output_dir, exist_ok=True)
        name = os.path.splitext(os.path.basename(args.snippets))[0].replace("label_snippets", "label_judge")
        path = os.path.join(args.output_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump({"args": vars(args), "results": results, "features": features}, f, indent=1,
                      ensure_ascii=False)
        print(f"\nSaved to {path}")
    except Exception:
        traceback.print_exc()
        sys.exit(1)

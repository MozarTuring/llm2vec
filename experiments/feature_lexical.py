"""Tie-robust feature identity and lexicality of SPLARE's features.

Follow-up to feature_contexts.py, whose top-K document overlap is noisy for word features (a feature
firing on every document containing "study" has hundreds of tied documents). Two measurements over
one corpus pass, for base / mntp / splare (as in feature_drift.py):

A. Per-feature correlation with base across documents. Each document's value for a feature is the
   retrieval value log(1+relu(pre)) max-pooled over its tokens (no top-k). Pearson over all documents,
   accumulated streaming. Reference "base_half": base on the same documents truncated to half the
   max_length (same model, slightly different input), as a scale for how high an unchanged feature
   correlates under an input change.
   Reported over features active (> 0) in at least --min_docs documents in both: mean, mass-weighted
   mean (weights = the model's retrieval mass, log(1+relu) max-pool, top doc_top_k per doc, summed),
   quantiles, share > 0.5.

B. Lexicality. Per feature, its top-K tokens over the corpus (by pre-activation). Lexicality = share
   of the top-K tokens that are the feature's most common word (token text lowercased and stripped).
   single_word = lexicality >= --single_word. Reported per model, mass-weighted with the model's own
   retrieval mass, so "share of SPLARE's retrieval mass on single-word features".

    python experiments/feature_lexical.py --trained_checkpoint_path <root>/output/layerwise/<run>/checkpoint-N
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
from feature_contexts import TopK
from feature_drift import MODELS, SAE, load_models
from mteb_eval_layerwise import resolve_checkpoint_args

PAIRS = ["mntp", "splare", "base_half"]


class Corr:
    """Streaming Pearson correlation per feature between x (base) and y."""

    def __init__(self, width, device):
        z = lambda: torch.zeros(width, dtype=torch.float64, device=device)
        self.sx, self.sy, self.sxx, self.syy, self.sxy = z(), z(), z(), z(), z()
        self.nx, self.ny = z(), z()
        self.n = 0

    def update(self, x, y):
        x, y = x.double(), y.double()
        self.sx += x.sum(0); self.sy += y.sum(0)
        self.sxx += (x * x).sum(0); self.syy += (y * y).sum(0); self.sxy += (x * y).sum(0)
        self.nx += (x > 0).sum(0); self.ny += (y > 0).sum(0)
        self.n += x.shape[0]

    def corr(self, min_docs):
        n = self.n
        cov = n * self.sxy - self.sx * self.sy
        var = (n * self.sxx - self.sx ** 2) * (n * self.syy - self.sy ** 2)
        valid = (self.nx >= min_docs) & (self.ny >= min_docs) & (var > 0)
        return cov / var.clamp(min=1e-30).sqrt(), valid


def pooled_values(model_fn, sae, ids, attn, valid):
    pre = sae.pre(model_fn(input_ids=ids, attention_mask=attn)[0].float())
    pooled = (torch.log1p(torch.relu(pre)) * valid.unsqueeze(-1)).max(dim=1)[0]
    return pre, pooled


@torch.no_grad()
def collect(texts, tokenizer, base, bi, sae, args, device):
    W = sae.width
    corr = {p: Corr(W, device) for p in PAIRS}
    tok_top = {m: TopK(args.k, W, device) for m in MODELS}
    mass = {m: torch.zeros(W, dtype=torch.float64, device=device) for m in MODELS}
    token_ids, n_tok = [], 0
    for i in range(0, len(texts), args.batch_size):
        chunk = texts[i:i + args.batch_size]
        batch = tokenizer(chunk, padding=True, truncation=True, max_length=args.max_length,
                          return_tensors="pt").to(device)
        ids, attn = batch["input_ids"], batch["attention_mask"]
        valid = attn.bool() & (ids != tokenizer.bos_token_id)
        tok_global = torch.arange(n_tok, n_tok + int(valid.sum()), device=device)
        token_ids.append(ids[valid].cpu())
        n_tok += int(valid.sum())

        pooled = {}
        for m in MODELS:
            if m == "base":
                fn = base
            elif m == "mntp":
                fn = lambda **kw: _no_adapter(bi, **kw)
            else:
                fn = bi
            pre, pooled[m] = pooled_values(fn, sae, ids, attn, valid)
            tok_top[m].update(pre[valid], tok_global)
            vals, idx = pooled[m].topk(args.doc_top_k, dim=-1)
            mass[m].scatter_add_(0, idx.flatten(), vals.flatten().double())
            del pre

        half = tokenizer(chunk, padding=True, truncation=True, max_length=args.max_length // 2,
                         return_tensors="pt").to(device)
        h_valid = half["attention_mask"].bool() & (half["input_ids"] != tokenizer.bos_token_id)
        _, pooled_half = pooled_values(base, sae, half["input_ids"], half["attention_mask"], h_valid)

        corr["mntp"].update(pooled["base"], pooled["mntp"])
        corr["splare"].update(pooled["base"], pooled["splare"])
        corr["base_half"].update(pooled["base"], pooled_half)
        if (i // args.batch_size) % 100 == 0:
            print(f"  {i + len(chunk)}/{len(texts)} docs, {n_tok} tokens", flush=True)
    return corr, tok_top, mass, torch.cat(token_ids)


def _no_adapter(bi, **kw):
    with bi.disable_adapter():
        return bi(**kw)


def word_ids(tokenizer):
    """Token id -> id of its lowercased, stripped text (so ' Study', 'study' map together)."""
    canon, out = {}, []
    for t in range(len(tokenizer)):
        w = tokenizer.decode([t]).strip().lower()
        out.append(canon.setdefault(w, len(canon)))
    return torch.tensor(out)


def lexicality(top_ids, token_words):
    """top_ids (K, W) global token positions -> (W,) share of the most common word, (W,) that word."""
    words = token_words[top_ids.clamp(min=0)]                # (K, W)
    s, _ = words.sort(dim=0)
    # count of each value in the sorted column = run length; max run length = mode count
    same = (s[1:] == s[:-1]).int()
    run = torch.zeros_like(s, dtype=torch.int32)
    for r in range(1, s.shape[0]):
        run[r] = (run[r - 1] + 1) * same[r - 1]
    best = run.max(dim=0)
    mode = s.gather(0, best.indices[None, :])[0]
    return (best.values + 1).float() / s.shape[0], mode


def weighted(x, w):
    w = w / w.sum().clamp(min=1e-12)
    return float((x.double() * w).sum())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_checkpoint_path", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--peft_model_name_or_path", type=str)
    parser.add_argument("--sae_weights_path", type=str)
    parser.add_argument("--lora_layers", type=int)
    parser.add_argument("--tasks", type=str, nargs="+", default=["SciFact", "NFCorpus", "SCIDOCS", "ArguAna"])
    parser.add_argument("--docs_per_task", type=int, default=3000)
    parser.add_argument("--k", type=int, default=20, help="Top-K tokens per feature for lexicality.")
    parser.add_argument("--single_word", type=float, default=0.8)
    parser.add_argument("--min_docs", type=int, default=10)
    parser.add_argument("--doc_top_k", type=int, default=400)
    parser.add_argument("--num_show", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="results_feature_lexical")
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

        corr, tok_top, mass, token_ids = collect(texts, tokenizer, base, bi, sae, args, device)
        vocab_words = word_ids(tokenizer).to(device)
        token_words = vocab_words[token_ids.to(device)]

        # A. correlation with base
        results = {"correlation": {}, "lexicality": {}}
        print(f"\n=== A. per-feature correlation with base across {len(texts)} docs "
              f"(features active in >= {args.min_docs} docs in both) ===")
        print(f"{'pair':10s}{'features':>10s}{'mean':>8s}{'mass_wt':>9s}{'p10':>8s}{'p50':>8s}{'p90':>8s}{'>0.5':>8s}")
        corr_vals = {}
        for p in PAIRS:
            c, valid = corr[p].corr(args.min_docs)
            corr_vals[p] = (c, valid)
            cv = c[valid]
            w = mass["base" if p == "base_half" else p][valid]
            q = torch.quantile(cv.float(), torch.tensor([0.1, 0.5, 0.9], device=device)).tolist()
            r = {"features": int(valid.sum()), "mean": float(cv.mean()), "mass_weighted": weighted(cv, w),
                 "p10": q[0], "p50": q[1], "p90": q[2], "share_gt_0.5": float((cv > 0.5).float().mean())}
            results["correlation"][p] = r
            print(f"{p:10s}{r['features']:>10d}{r['mean']:>8.3f}{r['mass_weighted']:>9.3f}"
                  f"{r['p10']:>8.3f}{r['p50']:>8.3f}{r['p90']:>8.3f}{r['share_gt_0.5']:>8.3f}")

        # B. lexicality
        print(f"\n=== B. lexicality (share of top-{args.k} tokens that are the feature's most common word; "
              f"single word >= {args.single_word}) ===")
        print(f"{'model':10s}{'mean':>8s}{'mass_wt':>9s}{'single':>8s}{'single_mass':>13s}")
        lex = {}
        for m in MODELS:
            alive = tok_top[m].vals[-1] > 0
            share, mode = lexicality(tok_top[m].ids, token_words)
            lex[m] = (share, mode)
            single = (share >= args.single_word).float()
            w = mass[m][alive]
            r = {"alive": int(alive.sum()), "mean": float(share[alive].mean()),
                 "mass_weighted": weighted(share[alive], w),
                 "single_word_share": float(single[alive].mean()),
                 "single_word_mass_share": weighted(single[alive], w)}
            results["lexicality"][m] = r
            print(f"{m:10s}{r['mean']:>8.3f}{r['mass_weighted']:>9.3f}{r['single_word_share']:>8.3f}"
                  f"{r['single_word_mass_share']:>13.3f}")

        # Top SPLARE features by retrieval mass
        c_s, v_s = corr_vals["splare"]
        order = mass["splare"].argsort(descending=True)[:args.num_show].tolist()
        print(f"\n=== top {args.num_show} SPLARE features by retrieval mass ===")
        print(f"{'feature':>8s}{'mass':>8s}{'corr':>7s}{'lex_base':>9s}{'lex_spl':>8s}  top word base -> splare")
        examples = []
        for f in order:
            words = {}
            for m in ("base", "splare"):
                share, mode = lex[m]
                pos = (token_words == mode[f]).nonzero()
                t = int(token_ids[pos[0, 0]]) if len(pos) else -1
                words[m] = (tokenizer.decode([t]).strip() if t >= 0 else "?", float(share[f]))
            cf = float(c_s[f]) if bool(v_s[f]) else float("nan")
            ms = float(mass["splare"][f] / mass["splare"].sum())
            examples.append({"feature": f, "mass_share": ms, "corr": cf,
                             "base_word": words["base"][0], "base_lex": words["base"][1],
                             "splare_word": words["splare"][0], "splare_lex": words["splare"][1]})
            print(f"{f:>8d}{ms:>8.2%}{cf:>7.2f}{words['base'][1]:>9.2f}{words['splare'][1]:>8.2f}  "
                  f"{words['base'][0]!r} -> {words['splare'][0]!r}")
        results["top_splare_features"] = examples

        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "feature_lexical.json"), "w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {args.output_dir}/feature_lexical.json")
    except Exception:
        traceback.print_exc()
        sys.exit(1)

"""Faithfulness of the feature-gate SPLARE (run_featgate_finetune.py), next to base / mntp / splare.

Same measurements as feature_lexical.py, over the same corpus, with one more model, "featgate": the
mntp SAE pre-activation with the trained per-feature gate, pre * gain + bias (no extra forward pass).

A. Per-feature correlation with base across documents (featgate added to the pairs).
B. Lexicality. The gate is monotone per feature (gain > 0), so featgate's top tokens equal mntp's;
   only the retrieval-mass weighting differs.
C. Newly firing mass: share of featgate's retrieval mass (log(1+relu) max-pool, top doc_top_k per
   doc) that comes from tokens where the mntp pre-activation is <= 0, i.e. firing only because of a
   positive gate bias. This is where featgate is not faithful by construction.
D. Top featgate features by retrieval mass, with gain, bias, correlation and main word.

    python experiments/feature_lexical_featgate.py --trained_checkpoint_path <root>/output/layerwise/<run>/checkpoint-N --featgate_path <root>/output/featgate/<run>/checkpoint-N/featgate.pt
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
from feature_drift import SAE, load_models
from feature_lexical import Corr, _no_adapter, lexicality, pooled_values, weighted, word_ids
from mteb_eval_layerwise import resolve_checkpoint_args

MODELS = ["base", "mntp", "featgate", "splare"]
PAIRS = ["mntp", "featgate", "splare", "base_half"]


@torch.no_grad()
def collect(texts, tokenizer, base, bi, sae, gain, bias, args, device):
    W = sae.width
    corr = {p: Corr(W, device) for p in PAIRS}
    tok_top = {m: TopK(args.k, W, device) for m in MODELS if m != "featgate"}
    mass = {m: torch.zeros(W, dtype=torch.float64, device=device) for m in MODELS}
    new_mass = torch.zeros(W, dtype=torch.float64, device=device)
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
        for m in ("base", "mntp", "splare"):
            if m == "base":
                fn = base
            elif m == "mntp":
                fn = lambda **kw: _no_adapter(bi, **kw)
            else:
                fn = bi
            pre, pooled[m] = pooled_values(fn, sae, ids, attn, valid)
            tok_top[m].update(pre[valid], tok_global)
            if m == "mntp":
                gated = torch.log1p(torch.relu(pre * gain + bias)) * valid.unsqueeze(-1)
                pooled["featgate"] = gated.max(dim=1)[0]
                # max-pool over tokens where mntp itself fires: the faithful part of featgate
                pooled_old = (gated * (pre > 0)).max(dim=1)[0]
                del gated
            del pre
        for m in MODELS:
            vals, idx = pooled[m].topk(args.doc_top_k, dim=-1)
            mass[m].scatter_add_(0, idx.flatten(), vals.flatten().double())
            if m == "featgate":
                new = (vals - pooled_old.gather(1, idx)).clamp(min=0)
                new_mass.scatter_add_(0, idx.flatten(), new.flatten().double())

        half = tokenizer(chunk, padding=True, truncation=True, max_length=args.max_length // 2,
                         return_tensors="pt").to(device)
        h_valid = half["attention_mask"].bool() & (half["input_ids"] != tokenizer.bos_token_id)
        _, pooled_half = pooled_values(base, sae, half["input_ids"], half["attention_mask"], h_valid)

        for p in ("mntp", "featgate", "splare"):
            corr[p].update(pooled["base"], pooled[p])
        corr["base_half"].update(pooled["base"], pooled_half)
        if (i // args.batch_size) % 100 == 0:
            print(f"  {i + len(chunk)}/{len(texts)} docs, {n_tok} tokens", flush=True)
    return corr, tok_top, mass, new_mass, torch.cat(token_ids)


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
    parser.add_argument("--k", type=int, default=20, help="Top-K tokens per feature for lexicality.")
    parser.add_argument("--single_word", type=float, default=0.8)
    parser.add_argument("--min_docs", type=int, default=10)
    parser.add_argument("--doc_top_k", type=int, default=400)
    parser.add_argument("--num_show", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="results_feature_lexical_featgate")
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
              f"bias mean={bias.mean():.3f} min={bias.min():.3f} max={bias.max():.3f}  "
              f"positive bias features={int((bias > 0).sum())}")

        corr, tok_top, mass, new_mass, token_ids = collect(texts, tokenizer, base, bi, sae, gain, bias,
                                                           args, device)
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

        # B. lexicality (featgate uses mntp's top tokens, weighted by featgate's own mass)
        print(f"\n=== B. lexicality (share of top-{args.k} tokens that are the feature's most common word; "
              f"single word >= {args.single_word}) ===")
        print(f"{'model':10s}{'mean':>8s}{'mass_wt':>9s}{'single':>8s}{'single_mass':>13s}")
        lex = {}
        for m in MODELS:
            src = "mntp" if m == "featgate" else m
            alive = tok_top[src].vals[-1] > 0
            if src not in lex:
                lex[src] = lexicality(tok_top[src].ids, token_words)
            share, _ = lex[src]
            single = (share >= args.single_word).float()
            w = mass[m][alive]
            r = {"alive": int(alive.sum()), "mean": float(share[alive].mean()),
                 "mass_weighted": weighted(share[alive], w),
                 "single_word_share": float(single[alive].mean()),
                 "single_word_mass_share": weighted(single[alive], w)}
            results["lexicality"][m] = r
            print(f"{m:10s}{r['mean']:>8.3f}{r['mass_weighted']:>9.3f}{r['single_word_share']:>8.3f}"
                  f"{r['single_word_mass_share']:>13.3f}")

        # C. newly firing mass
        total = mass["featgate"].sum()
        pos = bias > 0
        results["new_firing"] = {
            "mass_share": float(new_mass.sum() / total),
            "mass_share_on_positive_bias_features": float(mass["featgate"][pos].sum() / total),
            "positive_bias_features": int(pos.sum()),
        }
        print(f"\n=== C. featgate mass from tokens where mntp pre <= 0 (positive-bias firing) ===")
        print(f"newly firing mass share: {results['new_firing']['mass_share']:.3f}   "
              f"mass on positive-bias features: {results['new_firing']['mass_share_on_positive_bias_features']:.3f} "
              f"({int(pos.sum())} features)")

        # D. top featgate features by retrieval mass
        c_g, v_g = corr_vals["featgate"]
        share_m, mode_m = lex["mntp"]
        order = mass["featgate"].argsort(descending=True)[:args.num_show].tolist()
        print(f"\n=== D. top {args.num_show} featgate features by retrieval mass ===")
        print(f"{'feature':>8s}{'mass':>8s}{'new':>6s}{'gain':>9s}{'bias':>7s}{'corr':>7s}{'lex':>6s}  word")
        examples = []
        for f in order:
            hit = (token_words == mode_m[f]).nonzero()
            word = tokenizer.decode([int(token_ids[hit[0, 0]])]).strip() if len(hit) else "?"
            ms = float(mass["featgate"][f] / total)
            nf = float(new_mass[f] / mass["featgate"][f].clamp(min=1e-12))
            cf = float(c_g[f]) if bool(v_g[f]) else float("nan")
            examples.append({"feature": f, "mass_share": ms, "new_share": nf, "gain": float(gain[f]),
                             "bias": float(bias[f]), "corr": cf, "lex": float(share_m[f]), "word": word})
            print(f"{f:>8d}{ms:>8.2%}{nf:>6.2f}{float(gain[f]):>9.2f}{float(bias[f]):>7.2f}{cf:>7.2f}"
                  f"{float(share_m[f]):>6.2f}  {word!r}")
        results["top_featgate_features"] = examples

        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "feature_lexical_featgate.json"), "w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {args.output_dir}/feature_lexical_featgate.json")
    except Exception:
        traceback.print_exc()
        sys.exit(1)

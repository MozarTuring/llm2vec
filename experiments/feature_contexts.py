"""Label-free feature faithfulness: does each SAE feature still fire on the same documents / tokens inside SPLARE?

A feature's meaning is what it fires on. For every feature, collect its top-K documents (by max over
tokens of the SAE pre-activation, the ranking SPLARE's log(1+relu) max-pool uses) and its top-K
tokens over a corpus, separately for
    base    causal Llama-3.1-8B (the SAE's training distribution; where Neuronpedia labels come from)
    mntp    bidirectional + merged MNTP adapter, trained LoRA disabled
    splare  mntp + trained retrieval LoRA
and compare each model's top-K sets with base's for the same feature.

Metrics (features with all top-K values > 0 in both models, "alive"):
    doc_overlap        |top-K docs(model) & top-K docs(base)| / K, mean over alive features
    doc_overlap_mass   same, weighted by the feature's share of the model's retrieval mass
                       (log(1+relu) max-pool, top doc_top_k per doc, summed over the corpus)
    doc_overlap_random floor: base feature i vs model feature perm(i)
    tok_overlap        |top-K token ids(model) & top-K token ids(base)| / K (word-level identity)
    self_is_best       among the top --num_matched features by retrieval mass: share whose best
                       matching base feature (max top-K doc overlap) is the same feature
    best_vs_self       mean overlap with the best base match minus overlap with itself
                       (> 0: the feature now behaves like a different base feature)
Also dumps, for the top features by retrieval mass, their top documents and token contexts in base
and in the model, to read by hand.

    python experiments/feature_contexts.py --trained_checkpoint_path <root>/output/layerwise/<run>/checkpoint-N
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
from feature_drift import MODELS, SAE, load_models
from mteb_eval_layerwise import resolve_checkpoint_args


class TopK:
    """Running top-K (values, global ids) per feature, stored as (K, W)."""

    def __init__(self, k, width, device):
        self.vals = torch.full((k, width), -float("inf"), device=device)
        self.ids = torch.full((k, width), -1, dtype=torch.long, device=device)
        self.k = k

    def update(self, vals, ids):
        """vals (N, W), ids (N,) global ids of the rows."""
        top = vals.topk(min(self.k, vals.shape[0]), dim=0)    # shrink the batch first
        all_vals = torch.cat([self.vals, top.values])
        all_ids = torch.cat([self.ids, ids[top.indices]])
        top = all_vals.topk(self.k, dim=0)
        self.vals = top.values
        self.ids = all_ids.gather(0, top.indices)


@torch.no_grad()
def collect(texts, tokenizer, base, bi, sae, args, device):
    W = sae.width
    doc_top = {m: TopK(args.k, W, device) for m in MODELS}
    tok_top = {m: TopK(args.k, W, device) for m in MODELS}
    mass = {m: torch.zeros(W, dtype=torch.float64, device=device) for m in MODELS}
    token_ids, token_doc = [], []
    n_tok = 0
    for i in range(0, len(texts), args.batch_size):
        batch = tokenizer(texts[i:i + args.batch_size], padding=True, truncation=True,
                          max_length=args.max_length, return_tensors="pt").to(device)
        ids, attn = batch["input_ids"], batch["attention_mask"]
        valid = attn.bool() & (ids != tokenizer.bos_token_id)
        hidden = {"base": base(input_ids=ids, attention_mask=attn)[0]}
        with bi.disable_adapter():
            hidden["mntp"] = bi(input_ids=ids, attention_mask=attn)[0]
        hidden["splare"] = bi(input_ids=ids, attention_mask=attn)[0]

        doc_ids = torch.arange(i, i + ids.shape[0], device=device)
        rows = valid.nonzero()
        tok_global = torch.arange(n_tok, n_tok + rows.shape[0], device=device)
        token_ids.append(ids[valid].cpu())
        token_doc.append((rows[:, 0] + i).cpu())
        n_tok += rows.shape[0]

        for m in MODELS:
            pre = sae.pre(hidden[m].float())                     # (B, L, W)
            doc_max = pre.masked_fill(~valid.unsqueeze(-1), -float("inf")).max(dim=1)[0]
            doc_top[m].update(doc_max, doc_ids)
            tok_top[m].update(pre[valid], tok_global)
            pooled = torch.log1p(torch.relu(doc_max))
            vals, idx = pooled.topk(args.doc_top_k, dim=-1)
            mass[m].scatter_add_(0, idx.flatten(), vals.flatten().double())
            del pre, doc_max
        if (i // args.batch_size) % 100 == 0:
            print(f"  {i + ids.shape[0]}/{len(texts)} docs, {n_tok} tokens", flush=True)
    return doc_top, tok_top, mass, torch.cat(token_ids), torch.cat(token_doc)


def set_overlap(a, b):
    """a, b (K, W) ids -> (W,) share of a's ids that are in b."""
    return (a[:, None, :] == b[None, :, :]).any(dim=1).float().mean(dim=0)


def best_base_match(doc_top_m, doc_top_b, alive_b, features, num_docs):
    """For each feature in `features`, the base feature with the largest top-K doc overlap."""
    K, W = doc_top_b.ids.shape
    feat_idx = torch.arange(W, device=alive_b.device)[None, :].expand(K, W)
    keep = alive_b[None, :].expand(K, W)
    A_b = torch.sparse_coo_tensor(
        torch.stack([feat_idx[keep], doc_top_b.ids[keep]]),
        torch.ones(int(keep.sum()), device=alive_b.device), (W, num_docs)).coalesce()
    A_m = torch.zeros(len(features), num_docs, device=alive_b.device)
    A_m.scatter_(1, doc_top_m.ids[:, features].T, 1.0)
    counts = torch.sparse.mm(A_b, A_m.T)                     # (W, F)
    best_count, best_feat = counts.max(dim=0)
    self_count = counts[features, torch.arange(len(features), device=counts.device)]
    return best_feat, best_count / K, self_count / K


def contexts(tok_ids, token_ids, token_doc, tokenizer, width=6):
    out = []
    for t in tok_ids.tolist():
        if t < 0:
            continue
        lo, hi = max(0, t - width), min(len(token_ids), t + width + 1)
        same = token_doc[lo:hi] == token_doc[t]
        span = token_ids[lo:hi][same].tolist()
        center = tokenizer.decode([int(token_ids[t])])
        out.append(f"{tokenizer.decode(span)}  [{center.strip()}]")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_checkpoint_path", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--peft_model_name_or_path", type=str)
    parser.add_argument("--sae_weights_path", type=str)
    parser.add_argument("--lora_layers", type=int)
    parser.add_argument("--tasks", type=str, nargs="+", default=["SciFact", "NFCorpus", "SCIDOCS", "ArguAna"])
    parser.add_argument("--docs_per_task", type=int, default=3000)
    parser.add_argument("--k", type=int, default=20, help="Top-K documents / tokens per feature.")
    parser.add_argument("--doc_top_k", type=int, default=400, help="Pruning for the retrieval mass.")
    parser.add_argument("--num_matched", type=int, default=2000, help="Top features by mass for best-match.")
    parser.add_argument("--num_dump", type=int, default=30)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="results_feature_contexts")
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
            print(f"{task}: {min(len(docs), args.docs_per_task)} docs")
        print(f"Corpus: {len(texts)} docs")

        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        tokenizer.padding_side = "right"
        base, bi = load_models(args, device)
        sae = SAE(args.sae_weights_path, device)

        doc_top, tok_top, mass, token_ids, token_doc = collect(texts, tokenizer, base, bi, sae, args, device)
        token_ids, token_doc = token_ids.numpy(), token_doc.numpy()
        token_ids_gpu = torch.from_numpy(token_ids).to(device)

        alive = {m: doc_top[m].vals[-1] > 0 for m in MODELS}
        print("\nalive (all top-K docs > 0): " + "  ".join(f"{m}={int(alive[m].sum())}" for m in MODELS))
        results = {"alive": {m: int(alive[m].sum()) for m in MODELS}}
        perm_gen = torch.Generator(device="cpu").manual_seed(args.seed)

        print(f"\n{'metric':22s}" + "".join(f"{m:>10s}" for m in MODELS[1:]))
        rows = {}
        for m in MODELS[1:]:
            both = alive[m] & alive["base"]
            feats = both.nonzero().squeeze(-1)
            doc_ov = set_overlap(doc_top[m].ids, doc_top["base"].ids)[feats]
            tok_m = token_ids_gpu[tok_top[m].ids.clamp(min=0)]
            tok_b = token_ids_gpu[tok_top["base"].ids.clamp(min=0)]
            tok_ov = set_overlap(tok_m, tok_b)[feats]
            perm = feats[torch.randperm(len(feats), generator=perm_gen).to(device)]
            rand_ov = (doc_top[m].ids[:, feats][:, None, :] == doc_top["base"].ids[:, perm][None, :, :]) \
                .any(dim=1).float().mean(dim=0)
            w = mass[m][feats]
            w = w / w.sum().clamp(min=1e-12)

            order = feats[mass[m][feats].argsort(descending=True)][:args.num_matched]
            best_feat, best_ov, self_ov = best_base_match(doc_top[m], doc_top["base"], alive["base"],
                                                         order, len(texts))
            rows[m] = {
                "alive_both": int(both.sum()),
                "mass_on_alive_both": float(mass[m][feats].sum() / mass[m].sum()),
                "doc_overlap": float(doc_ov.mean()),
                "doc_overlap_mass": float((doc_ov.double() * w).sum()),
                "doc_overlap_random": float(rand_ov.mean()),
                "tok_overlap": float(tok_ov.mean()),
                "tok_overlap_mass": float((tok_ov.double() * w).sum()),
                "self_is_best": float((best_feat == order).float().mean()),
                "best_vs_self": float((best_ov - self_ov).mean()),
                "self_overlap_top_mass": float(self_ov.mean()),
            }

            dump = []
            for j, f in enumerate(order[:args.num_dump].tolist()):
                dump.append({
                    "feature": f, "mass_share": float(mass[m][f] / mass[m].sum()),
                    "doc_overlap": float(self_ov[j]), "best_base_feature": int(best_feat[j]),
                    "best_base_overlap": float(best_ov[j]),
                    "base_docs": [texts[d][:200] for d in doc_top["base"].ids[:5, f].tolist()],
                    f"{m}_docs": [texts[d][:200] for d in doc_top[m].ids[:5, f].tolist()],
                    "base_tokens": contexts(tok_top["base"].ids[:10, f].cpu(), token_ids, token_doc, tokenizer),
                    f"{m}_tokens": contexts(tok_top[m].ids[:10, f].cpu(), token_ids, token_doc, tokenizer),
                })
            rows[m]["examples"] = dump
        for key in ("alive_both", "mass_on_alive_both", "doc_overlap", "doc_overlap_mass",
                    "doc_overlap_random", "tok_overlap", "tok_overlap_mass", "self_is_best",
                    "best_vs_self", "self_overlap_top_mass"):
            vals = [rows[m][key] for m in MODELS[1:]]
            print(f"{key:22s}" + "".join(f"{v:>10d}" if isinstance(v, int) else f"{v:>10.4f}" for v in vals))
        print(f"(self_is_best / best_vs_self / self_overlap_top_mass over the top {args.num_matched} "
              f"features by each model's retrieval mass; K={args.k})")

        for m in MODELS[1:]:
            print(f"\n--- {m}: top 5 features by retrieval mass ---")
            for ex in rows[m]["examples"][:5]:
                print(f"feature {ex['feature']}  mass={ex['mass_share']:.2%}  doc_overlap={ex['doc_overlap']:.2f}  "
                      f"best base={ex['best_base_feature']} ({ex['best_base_overlap']:.2f})")
                for label in ("base", m):
                    print(f"  {label:6s} tokens: {ex[f'{label}_tokens'][:3]}")
                    print(f"  {label:6s} doc:    {ex[f'{label}_docs'][0][:120]!r}")

        results.update(rows)
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "feature_contexts.json"), "w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {args.output_dir}/feature_contexts.json")
    except Exception:
        traceback.print_exc()
        sys.exit(1)

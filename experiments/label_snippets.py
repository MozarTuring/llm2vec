"""Snippets for the label check: where does each model's feature fire in retrieval, and where not?

Input to label_judge.py, which asks an LLM judge whether a feature's Neuronpedia label (written from
base's activations) describes each snippet. Models: base (causal), mntp (bidirectional + MNTP
adapter, trained LoRA disabled) and the trained checkpoint, named --name (splare, top-layers, anchor).

Features: the union over the three models of each model's top --num_features features by retrieval
mass w_{m,i} = sum_d v_{m,d,i} 1[i in top_400(v_{m,d})], with v_{m,d,i} = max_t log(1 + relu(pre)),
on the 12000-document corpus of feature_lexical.py.

Per model and feature:
    positives  documents where the feature is in the model's top-400 retrieval vector. The
               --num_top with the largest v, plus --num_random sampled from the rest. The marked
               token is the one that gives the max-pooled value.
    negatives  --num_neg documents sampled from those where the feature is not in the top-400. The
               marked token is the one with the largest pre-activation (so both kinds have a mark).
Snippet = --ctx_before tokens, the marked token as <<token>>, --ctx_after tokens.

One process per GPU: torchrun --nproc_per_node=N shards the documents (strided over length-sorted
order); masses are summed with all_reduce, per-document values are gathered; rank 0 samples and
writes results_label_snippets/label_snippets_<name>.json. Plain python runs on one GPU.

    torchrun --nproc_per_node=N experiments/label_snippets.py --trained_checkpoint_path <root>/output/<group>/<run>/checkpoint-N --name splare
"""

import argparse
import json
import os
import sys
import traceback

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from dump_sparse import load_beir
from feature_drift import SAE, load_models
from feature_lexical import _no_adapter, pooled_values
from feature_scale_base import all_sum, setup
from mteb_eval_layerwise import resolve_checkpoint_args


def model_fns(base, bi, name):
    return {"base": base, "mntp": lambda **kw: _no_adapter(bi, **kw), name: bi}


def batches(texts, tokenizer, args, device):
    for i in range(0, len(texts), args.batch_size):
        batch = tokenizer(texts[i:i + args.batch_size], padding=True, truncation=True,
                          max_length=args.max_length, return_tensors="pt").to(device)
        ids, attn = batch["input_ids"], batch["attention_mask"]
        yield i, ids, attn, attn.bool() & (ids != tokenizer.bos_token_id)


@torch.no_grad()
def retrieval_mass(texts, tokenizer, fns, sae, args, device):
    mass = {m: torch.zeros(sae.width, dtype=torch.float64, device=device) for m in fns}
    for i, ids, attn, valid in batches(texts, tokenizer, args, device):
        for m, fn in fns.items():
            pre, pooled = pooled_values(fn, sae, ids, attn, valid)
            del pre
            vals, idx = pooled.topk(args.doc_top_k, dim=-1)
            mass[m].scatter_add_(0, idx.flatten(), vals.flatten().double())
        if (i // args.batch_size) % 100 == 0 and args.rank == 0:
            print(f"  pass 1, rank 0: {i + ids.shape[0]}/{len(texts)} docs", flush=True)
    for t in mass.values():
        all_sum(t)
    return mass


@torch.no_grad()
def feature_values(texts, tokenizer, fns, sae, feats, args, device):
    """Per document and selected feature: retrieval value (0 if not in the top-k) and marked position."""
    F = torch.tensor(feats, device=device)
    out = {m: ([], []) for m in fns}
    for i, ids, attn, valid in batches(texts, tokenizer, args, device):
        for m, fn in fns.items():
            pre, pooled = pooled_values(fn, sae, ids, attn, valid)
            idx = pooled.topk(args.doc_top_k, dim=-1).indices
            in_top = torch.zeros_like(pooled, dtype=torch.bool).scatter_(-1, idx, True)
            val = (pooled[:, F] * in_top[:, F]).float()
            pos = pre[:, :, F].masked_fill(~valid.unsqueeze(-1), float("-inf")).argmax(dim=1)
            out[m][0].append(val.cpu())
            out[m][1].append(pos.cpu())
            del pre, pooled
        if (i // args.batch_size) % 100 == 0 and args.rank == 0:
            print(f"  pass 2, rank 0: {i + ids.shape[0]}/{len(texts)} docs", flush=True)
    return {m: (torch.cat(v).numpy(), torch.cat(p).numpy()) for m, (v, p) in out.items()}


def snippet(ids, p, args, tokenizer):
    left = tokenizer.decode(ids[max(1, p - args.ctx_before):p])
    right = tokenizer.decode(ids[p + 1:p + 1 + args.ctx_after])
    return f"{left}<<{tokenizer.decode(ids[p:p + 1])}>>{right}"


def sample(vals, pos, j, texts, rng, tokenizer, args):
    """Positive and negative snippets of selected feature j for one model."""
    tok = lambda d: tokenizer(texts[d], truncation=True, max_length=args.max_length)["input_ids"]
    v = vals[:, j]
    pos_docs = np.nonzero(v > 0)[0]
    pos_docs = pos_docs[np.argsort(-v[pos_docs], kind="stable")]
    rest = pos_docs[args.num_top:]
    chosen = list(pos_docs[:args.num_top]) + list(rng.permutation(rest)[:args.num_random])
    neg_docs = rng.permutation(np.nonzero(v == 0)[0])[:args.num_neg]
    return ([{"doc": int(d), "value": float(v[d]), "text": snippet(tok(d), int(pos[d, j]), args, tokenizer)}
             for d in chosen],
            [{"doc": int(d), "text": snippet(tok(d), int(pos[d, j]), args, tokenizer)} for d in neg_docs],
            int(len(pos_docs)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_checkpoint_path", type=str, required=True)
    parser.add_argument("--name", type=str, required=True, help="Name of the trained model, e.g. splare.")
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--peft_model_name_or_path", type=str)
    parser.add_argument("--sae_weights_path", type=str)
    parser.add_argument("--lora_layers", type=int)
    parser.add_argument("--tasks", type=str, nargs="+", default=["SciFact", "NFCorpus", "SCIDOCS", "ArguAna"])
    parser.add_argument("--docs_per_task", type=int, default=3000)
    parser.add_argument("--doc_top_k", type=int, default=400)
    parser.add_argument("--num_features", type=int, default=100)
    parser.add_argument("--num_top", type=int, default=5)
    parser.add_argument("--num_random", type=int, default=5)
    parser.add_argument("--num_neg", type=int, default=10)
    parser.add_argument("--ctx_before", type=int, default=24)
    parser.add_argument("--ctx_after", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="results_label_snippets")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    resolve_checkpoint_args(args, parser)

    try:
        args.rank, world, device = setup()
        rng = np.random.default_rng(args.seed)
        texts = []
        for task in args.tasks:
            _, corpus, _ = load_beir(task)
            docs = list(corpus.values())
            texts += [docs[j] for j in rng.permutation(len(docs))[:args.docs_per_task]]
        order = sorted(range(len(texts)), key=lambda j: len(texts[j]))
        shard_idx = order[args.rank::world]
        shard = [texts[j] for j in shard_idx]
        if args.rank == 0:
            print(f"Corpus: {len(texts)} docs from {args.tasks}, {world} process(es)")

        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        tokenizer.padding_side = "right"
        base, bi = load_models(args, device)
        sae = SAE(args.sae_weights_path, device)
        fns = model_fns(base, bi, args.name)

        mass = retrieval_mass(shard, tokenizer, fns, sae, args, device)
        top = {m: mass[m].topk(args.num_features).indices.tolist() for m in fns}
        feats = sorted(set().union(*top.values()))
        if args.rank == 0:
            print(f"{len(feats)} features in the union of each model's top {args.num_features}")

        local = feature_values(shard, tokenizer, fns, sae, feats, args, device)
        gathered = [None] * world
        if dist.is_initialized():
            dist.all_gather_object(gathered, (shard_idx, local))
        else:
            gathered = [(shard_idx, local)]

        if args.rank == 0:
            vals = {m: np.zeros((len(texts), len(feats)), dtype=np.float32) for m in fns}
            pos = {m: np.zeros((len(texts), len(feats)), dtype=np.int64) for m in fns}
            for idx, loc in gathered:
                for m in fns:
                    vals[m][idx], pos[m][idx] = loc[m]

            out = {"features": feats, "models": {}}
            for m in fns:
                total = float(mass[m].sum())
                rows = {}
                for j, f in enumerate(feats):
                    positives, negatives, n_docs = sample(vals[m], pos[m], j, texts, rng, tokenizer, args)
                    rows[str(f)] = {"mass_share": float(mass[m][f]) / total, "top": f in top[m],
                                    "positive_docs": n_docs, "positives": positives, "negatives": negatives}
                out["models"][m] = rows
                n_top = sum(r["top"] for r in rows.values())
                print(f"{m:12s} top features {n_top}, their mass share "
                      f"{sum(r['mass_share'] for r in rows.values() if r['top']):.4f}")

            os.makedirs(args.output_dir, exist_ok=True)
            path = os.path.join(args.output_dir, f"label_snippets_{args.name}.json")
            with open(path, "w") as f:
                json.dump({"args": vars(args), **out}, f, indent=1, ensure_ascii=False)
            print(f"\nSaved to {path}")
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

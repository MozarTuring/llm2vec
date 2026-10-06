"""Dump SPLARE sparse vectors of BEIR queries/corpus for one checkpoint (input to fuse_eval.py).

For each task, saves to {output_dir}/L{lora_layers}/{task}/:
    queries.npz   ids, idx [N, query_top_k] int32, val [N, query_top_k] float32
    corpus.npz    ids, idx [N, doc_top_k]   int32, val [N, doc_top_k]   float32
    qrels.json    {query_id: {doc_id: relevance}}
Features are sorted by value, so smaller budgets are prefixes of the stored ones.

    python experiments/dump_sparse.py --trained_checkpoint_path <ckpt> --task_name SciFact NFCorpus
"""

import argparse
import json
import os

import numpy as np
import torch
from datasets import load_dataset

from mteb_eval_layerwise import LayerwiseEncoder, resolve_checkpoint_args

HF_REPOS = {
    "SciFact": "mteb/scifact",
    "NFCorpus": "mteb/nfcorpus",
    "ArguAna": "mteb/arguana",
    "SCIDOCS": "mteb/scidocs",
    "FiQA2018": "mteb/fiqa",
}


def first_split(repo, config):
    ds = load_dataset(repo, config)
    return ds[list(ds.keys())[0]]


def load_beir(task):
    repo = HF_REPOS[task]
    qrels_ds = load_dataset(repo, "default", split="test")
    qrels = {}
    for row in qrels_ds:
        qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(row["score"])
    queries = {str(r["_id"]): r["text"] for r in first_split(repo, "queries")}
    queries = {qid: queries[qid] for qid in qrels if qid in queries}
    corpus = {str(r["_id"]): ((r.get("title") or "") + " " + r["text"]).strip()
              for r in first_split(repo, "corpus")}
    return queries, corpus, qrels


def encode_sparse(encoder, texts, top_k, slice_size):
    """Top-k (idx, val) per text, sorted by value; encodes in slices to bound host memory."""
    all_idx, all_val = [], []
    for start in range(0, len(texts), slice_size):
        dense = encoder.encode_texts(texts[start:start + slice_size], top_k=top_k)
        val, idx = torch.from_numpy(dense).topk(top_k, dim=-1)
        all_idx.append(idx.numpy().astype(np.int32))
        all_val.append(val.numpy().astype(np.float32))
        del dense
    return np.concatenate(all_idx), np.concatenate(all_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_checkpoint_path", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--peft_model_name_or_path", type=str)
    parser.add_argument("--sae_weights_path", type=str)
    parser.add_argument("--lora_layers", type=int)
    parser.add_argument("--task_name", type=str, nargs="+",
                        default=["SciFact", "NFCorpus", "ArguAna", "SCIDOCS"])
    parser.add_argument("--output_dir", type=str, default="sparse_dumps")
    parser.add_argument("--query_top_k", type=int, default=40)
    parser.add_argument("--doc_top_k", type=int, default=400)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--slice_size", type=int, default=4096)
    args = parser.parse_args()
    resolve_checkpoint_args(args, parser)

    encoder = LayerwiseEncoder(
        model_name_or_path=args.model_name_or_path,
        peft_model_name_or_path=args.peft_model_name_or_path,
        sae_weights_path=args.sae_weights_path,
        lora_layers=args.lora_layers,
        trained_checkpoint_path=args.trained_checkpoint_path,
        max_length=args.max_length,
    )

    for task in args.task_name:
        queries, corpus, qrels = load_beir(task)
        out = os.path.join(args.output_dir, f"L{args.lora_layers}", task)
        os.makedirs(out, exist_ok=True)
        print(f"{task}: {len(queries)} queries, {len(corpus)} docs -> {out}", flush=True)
        for name, texts, k in [("queries", queries, args.query_top_k), ("corpus", corpus, args.doc_top_k)]:
            ids = list(texts)
            idx, val = encode_sparse(encoder, [texts[i] for i in ids], k, args.slice_size)
            np.savez(os.path.join(out, f"{name}.npz"), ids=np.array(ids), idx=idx, val=val)
            nnz = (val > 0).sum(1)
            print(f"  {name}: {len(ids)} x top-{k}, nnz mean={nnz.mean():.0f} min={nnz.min()}", flush=True)
        with open(os.path.join(out, "qrels.json"), "w") as f:
            json.dump(qrels, f)

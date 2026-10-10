"""Reciprocal rank fusion of two single-layer SPLARE checkpoints (CPU only; reads dump_sparse.py output).

Question: do two layers' latent vocabularies retrieve complementary documents?
Every run (top-1000 docs per query) is scored with
pytrec_eval, the evaluator MTEB uses, so numbers are comparable to MTEB results.
For each task, reports nDCG@10 and recall@100 for
    single     each layer alone (q=40, d=400)      sanity check vs. MTEB
    rrf        RRF of the two runs
Fused runs are also saved as TREC run files in {output_dir}/runs/.

    python experiments/fuse_eval.py --dump_dir sparse_dumps --layers 0 26
    python experiments/fuse_eval.py --dump_dir <dir_with_L0> <dir_with_L26> --layers 0 26
"""

import argparse
import json
import os

import numpy as np
import pytrec_eval
import scipy.sparse as sp

DEPTH = 1000
MEASURES = {"ndcg_cut_10": "ndcg@10", "recall_100": "recall@100"}


def load(dump_dir, layer, task, name):
    return dict(np.load(os.path.join(dump_dir, f"L{layer}", task, f"{name}.npz")))


def to_csr(rec, k, width):
    idx, val = rec["idx"][:, :k], rec["val"][:, :k]
    rows = np.repeat(np.arange(len(idx)), idx.shape[1])
    return sp.csr_matrix((val.ravel(), (rows, idx.ravel())), shape=(len(idx), width))


def scores(dump_dir, layer, task, qk, dk):
    """Query x doc scores; a doc with the query's own id is excluded (BEIR/MTEB convention)."""
    q, d = load(dump_dir, layer, task, "queries"), load(dump_dir, layer, task, "corpus")
    width = int(max(q["idx"].max(), d["idx"].max())) + 1
    s = (to_csr(q, qk, width) @ to_csr(d, dk, width).T).toarray()
    qids, dids = [str(x) for x in q["ids"]], [str(x) for x in d["ids"]]
    doc_pos = {doc: i for i, doc in enumerate(dids)}
    for i, qid in enumerate(qids):
        if qid in doc_pos:
            s[i, doc_pos[qid]] = -np.inf
    return s, qids, dids


def rrf(*score_mats, k=60, depth=DEPTH):
    """Reciprocal rank fusion over each run's top-`depth` docs: sum of 1 / (k + rank)."""
    fused = np.zeros_like(score_mats[0])
    rows = np.arange(fused.shape[0])[:, None]
    for s in score_mats:
        depth_ = min(depth, s.shape[1])
        top = np.argpartition(-s, depth_ - 1, axis=1)[:, :depth_]
        ranked = top[rows, np.argsort(-s[rows, top], axis=1, kind="stable")]
        fused[rows, ranked] += 1.0 / (k + np.arange(1, depth_ + 1))
    return fused


def to_run(s, qids, dids, depth=DEPTH):
    """Score matrix -> pytrec_eval run {qid: {doc_id: score}} with each query's top-`depth` docs."""
    depth = min(depth, s.shape[1])
    top = np.argpartition(-s, depth - 1, axis=1)[:, :depth]
    return {qid: {dids[j]: float(s[i, j]) for j in top[i] if np.isfinite(s[i, j])}
            for i, qid in enumerate(qids)}


def evaluate(run, qrels, qids):
    """Mean of each measure over qids (a query with no retrieved docs counts as 0)."""
    per_query = pytrec_eval.RelevanceEvaluator(qrels, set(MEASURES)).evaluate(run)
    return {name: float(np.mean([per_query.get(q, {}).get(m, 0.0) for q in qids]))
            for m, name in MEASURES.items()}


def save_trec(run, path, tag):
    with open(path, "w") as f:
        for qid, docs in run.items():
            for rank, (doc, score) in enumerate(sorted(docs.items(), key=lambda x: -x[1]), 1):
                f.write(f"{qid} Q0 {doc} {rank} {score:.6f} {tag}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", type=str, nargs="+", default=["sparse_dumps"],
                        help="one dir for both layers, or one per layer (in --layers order)")
    parser.add_argument("--layers", type=int, nargs=2, default=[0, 26])
    parser.add_argument("--task_name", type=str, nargs="+",
                        default=["SciFact", "NFCorpus", "ArguAna", "SCIDOCS"])
    parser.add_argument("--query_top_k", type=int, default=40)
    parser.add_argument("--doc_top_k", type=int, default=400)
    parser.add_argument("--rrf_k", type=int, default=60)
    parser.add_argument("--output_dir", type=str, default="results_fusion")
    args = parser.parse_args()

    a, b = args.layers
    dir_a, dir_b = args.dump_dir if len(args.dump_dir) == 2 else args.dump_dir * 2
    qk, dk = args.query_top_k, args.doc_top_k
    os.makedirs(os.path.join(args.output_dir, "runs"), exist_ok=True)
    results = {}
    for task in args.task_name:
        with open(os.path.join(dir_a, f"L{a}", task, "qrels.json")) as f:
            qrels = json.load(f)
        qrels = {q: {d: int(g) for d, g in docs.items()} for q, docs in qrels.items()}
        full_a, qids, dids = scores(dir_a, a, task, qk, dk)
        full_b, qids_b, dids_b = scores(dir_b, b, task, qk, dk)
        assert qids == qids_b and dids == dids_b, "query/doc order differs between dumps"

        runs = {
            f"L{a}": full_a, f"L{b}": full_b,
            "rrf": rrf(full_a, full_b, k=args.rrf_k),
        }
        r = {}
        for name, s in runs.items():
            run = to_run(s, qids, dids)
            r[name] = evaluate(run, qrels, qids)
            if name == "rrf":
                save_trec(run, os.path.join(args.output_dir, "runs", f"{task}.{name}.trec"), name)
        results[task] = r

        print(f"\n=== {task} ({len(qids)} queries, {len(dids)} docs) ===  ndcg@10 / recall@100")
        fmt = lambda n: f"{r[n]['ndcg@10']:.4f} / {r[n]['recall@100']:.4f}"
        print(f"  single (q={qk}, d={dk}):  L{a}={fmt(f'L{a}')}  L{b}={fmt(f'L{b}')}")
        print(f"  rrf:                     {fmt('rrf')}", flush=True)

    out = os.path.join(args.output_dir, "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}")

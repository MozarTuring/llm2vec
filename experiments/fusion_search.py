"""Search for a deployable way to mix two single-layer SPLARE runs that recovers the per-query oracle
headroom (CPU only; reads dump_sparse.py output, reuses fuse_eval.py helpers).

fuse_eval.py showed L16 and L26 win on different queries (oracle max +2.6 to +5.7 nDCG@10 over the
best single layer), but equal-weight RRF recovers none of it. Methods compared per task:
    single          each layer alone (q, d)
    concat_half     score of L_a at (q/2, d/2) + L_b at (q/2, d/2): same feature budget as one layer
    {fam}@0.5       untuned weighted fusion, fam in sum_raw (= concatenation, 2x budget),
                    sum_minmax / sum_zscore (per-query normalization over each run's top-1000),
                    rrf10 / rrf60 (reciprocal rank, k = 10 / 60)
    {fam}@cv        weight w on L_a picked by 5-fold CV over queries (fit on 4 folds, apply to 1)
    {fam}@loto      weight picked on the other tasks (leave one task out)
    qpp_{f}         per query, the run whose query-performance predictor f has the higher within-task
                    percentile (f: top1_z, gap10, cv100; see qpp())
    qppsoft_{f}     z-score fusion with per-query weight pct_a / (pct_a + pct_b)
    sel{2,3}@cv/loto  logistic-regression selector on QPP + overlap features, choosing among
                    {L_a, L_b} or {L_a, L_b, sum_zscore@0.5}; trained on queries where the
                    candidates differ, weighted by the nDCG@10 spread between candidates
    oracle_*        per-query max (upper bounds, not deployable): over {L_a, L_b}, over the 11
                    weights of each family, over the sel3 candidates
Every method gets nDCG@10, recall@100, difference to the best single layer and a paired t-test
p-value. The weight grid curves (test-set means, diagnostic only) are saved too.
Outputs in {output_dir}: results.json, per_query/{task}.npz (per-query nDCG@10 of every method).

    python experiments/fusion_search.py --dump_dir <dir_with_L16> <dir_with_L26> --layers 16 26
"""

import argparse
import json
import os

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fuse_eval import DEPTH, evaluate, rrf, scores, to_run

WEIGHTS = np.round(np.linspace(0, 1, 11), 2)
QPP = ["top1_z", "gap10", "cv100"]
EPS = 1e-12


def top_sorted(s, n=DEPTH):
    n = min(n, s.shape[1])
    top = -np.partition(-s, n - 1, axis=1)[:, :n]
    return -np.sort(-top, axis=1)


def normalize(s, how, top):
    if how == "raw":
        return s
    if how == "minmax":
        lo, hi = top[:, -1:], top[:, :1]
        return (s - lo) / np.maximum(hi - lo, EPS)
    return (s - top.mean(1, keepdims=True)) / np.maximum(top.std(1, keepdims=True), EPS)


def combine(x, y, w, mask):
    """w * x + (1 - w) * y with w a scalar or per-query vector; excluded (self) docs stay at -inf."""
    w = np.asarray(w, dtype=np.float64).reshape(-1, 1) if np.ndim(w) else w
    f = w * np.where(mask, 0.0, x) + (1 - w) * np.where(mask, 0.0, y)
    f[mask] = -np.inf
    return f


def qpp(top):
    """Scale-free query performance predictors from a run's sorted top-1000 scores."""
    t100 = top[:, :100]
    return {
        "top1_z": (top[:, 0] - top.mean(1)) / np.maximum(top.std(1), EPS),
        "gap10": (top[:, 0] - top[:, 9]) / np.maximum(np.abs(top[:, 0]), EPS),
        "cv100": t100.std(1) / np.maximum(np.abs(t100.mean(1)), EPS),
    }


def pct(x):
    return stats.rankdata(x) / len(x)


def overlap(a, b, k):
    ia = np.argpartition(-a, k - 1, axis=1)[:, :k]
    ib = np.argpartition(-b, k - 1, axis=1)[:, :k]
    return np.array([len(set(x) & set(y)) / k for x, y in zip(ia, ib)])


def features(A, B, qa, qb):
    cols = []
    for f in QPP:
        cols += [qa[f], qb[f], pct(qa[f]) - pct(qb[f])]
    cols += [overlap(A, B, 10), overlap(A, B, 100)]
    return np.stack(cols, 1)


def fit_predict(X_tr, nd_tr, X_te):
    """Selector: argmax-candidate classifier, sample weight = nDCG spread among candidates."""
    y, wt = nd_tr.argmax(0), nd_tr.max(0) - nd_tr.min(0)
    keep = wt > 0
    if len(np.unique(y[keep])) < 2:
        return np.full(len(X_te), y[keep][0] if keep.any() else 0)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    clf.fit(X_tr[keep], y[keep], logisticregression__sample_weight=wt[keep])
    return clf.predict(X_te)


def pick(per_metric, choice):
    """Per-query metrics of the candidate chosen per query; per_metric[m] is (n_cand, n_q)."""
    return {m: v[choice, np.arange(v.shape[1])] for m, v in per_metric.items()}


def stack(pq, names):
    return {m: np.stack([pq[n][m] for n in names]) for m in pq[names[0]]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", type=str, nargs="+", default=["sparse_dumps"],
                        help="one dir for both layers, or one per layer (in --layers order)")
    parser.add_argument("--layers", type=int, nargs=2, default=[16, 26])
    parser.add_argument("--task_name", type=str, nargs="+",
                        default=["SciFact", "NFCorpus", "ArguAna", "SCIDOCS"])
    parser.add_argument("--query_top_k", type=int, default=40)
    parser.add_argument("--doc_top_k", type=int, default=400)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="results_fusion_search")
    args = parser.parse_args()

    a, b = args.layers
    la, lb = (f"L{a}", f"L{b}") if a != b else (f"L{a}a", f"L{b}b")
    dir_a, dir_b = args.dump_dir if len(args.dump_dir) == 2 else args.dump_dir * 2
    qk, dk = args.query_top_k, args.doc_top_k
    os.makedirs(os.path.join(args.output_dir, "per_query"), exist_ok=True)
    families = ["sum_raw", "sum_minmax", "sum_zscore", "rrf10", "rrf60"]
    sel_sets = {"sel2": [la, lb], "sel3": [la, lb, "sum_zscore@0.5"]}

    task_data, results = {}, {}
    for task in args.task_name:
        with open(os.path.join(dir_a, f"L{a}", task, "qrels.json")) as f:
            qrels = {q: {d: int(g) for d, g in docs.items()} for q, docs in json.load(f).items()}
        A, qids, dids = scores(dir_a, a, task, qk, dk)
        B, qids_b, dids_b = scores(dir_b, b, task, qk, dk)
        assert qids == qids_b and dids == dids_b, "query/doc order differs between dumps"
        ev = lambda s, depth=100: evaluate(to_run(s, qids, dids, depth), qrels, qids)
        mask = np.isneginf(A)
        print(f"\n=== {task} ({len(qids)} queries, {len(dids)} docs) ===", flush=True)

        pq = {la: ev(A), lb: ev(B)}
        half_a = scores(dir_a, a, task, qk // 2, dk // 2)[0]
        half_b = scores(dir_b, b, task, qk // 2, dk // 2)[0]
        pq["concat_half"] = ev(half_a + half_b)
        del half_a, half_b

        topA, topB = top_sorted(A), top_sorted(B)
        z = (normalize(A, "zscore", topA), normalize(B, "zscore", topB))
        pairs = {
            "sum_raw": (A, B),
            "sum_minmax": (normalize(A, "minmax", topA), normalize(B, "minmax", topB)),
            "sum_zscore": z,
            "rrf10": (rrf(A, k=10), rrf(B, k=10)),
            "rrf60": (rrf(A, k=60), rrf(B, k=60)),
        }
        grid = {}
        for fam, (x, y) in pairs.items():
            grid[fam] = [ev(combine(x, y, w, mask)) for w in WEIGHTS]
            pq[f"{fam}@0.5"] = grid[fam][len(WEIGHTS) // 2]
            print(f"  grid {fam}: " + " ".join(f"{g['ndcg@10'].mean():.4f}" for g in grid[fam]), flush=True)
        del pairs

        qa, qb = qpp(topA), qpp(topB)
        for f in QPP:
            pa, pb = pct(qa[f]), pct(qb[f])
            pq[f"qpp_{f}"] = pick(stack(pq, [la, lb]), (pb > pa).astype(int))
            pq[f"qppsoft_{f}"] = ev(combine(*z, pa / (pa + pb), mask))
        X = features(A, B, qa, qb)
        del A, B, z

        folds = list(KFold(args.folds, shuffle=True, random_state=args.seed).split(X))
        for fam in families:
            cand = stack(dict(enumerate(grid[fam])), list(range(len(WEIGHTS))))
            choice = np.zeros(len(qids), dtype=int)
            for tr, te in folds:
                choice[te] = cand["ndcg@10"][:, tr].mean(1).argmax()
            pq[f"{fam}@cv"] = pick(cand, choice)
            pq[f"oracle_{fam}"] = pick(cand, cand["ndcg@10"].argmax(0))
        for sel, names in sel_sets.items():
            cand = stack(pq, names)
            choice = np.zeros(len(qids), dtype=int)
            for tr, te in folds:
                choice[te] = fit_predict(X[tr], cand["ndcg@10"][:, tr], X[te])
            pq[f"{sel}@cv"] = pick(cand, choice)
            pq[f"oracle_{sel}"] = pick(cand, cand["ndcg@10"].argmax(0))

        task_data[task] = {"pq": pq, "grid": grid, "X": X}
        results[task] = {"n_queries": len(qids), "grid_ndcg@10": {
            fam: {str(w): float(g["ndcg@10"].mean()) for w, g in zip(WEIGHTS, grid[fam])} for fam in families}}

    for task in args.task_name:
        pq, grid = task_data[task]["pq"], task_data[task]["grid"]
        others = [t for t in args.task_name if t != task]
        for fam in families:
            curve = np.mean([[g["ndcg@10"].mean() for g in task_data[t]["grid"][fam]] for t in others], 0)
            w_idx = int(curve.argmax())
            pq[f"{fam}@loto"] = grid[fam][w_idx]
            results[task].setdefault("loto_weight", {})[fam] = float(WEIGHTS[w_idx])
        if others:
            X_tr = np.concatenate([task_data[t]["X"] for t in others])
            for sel, names in sel_sets.items():
                nd_tr = np.concatenate([stack(task_data[t]["pq"], names)["ndcg@10"] for t in others], 1)
                cand = stack(pq, names)
                pq[f"{sel}@loto"] = pick(cand, fit_predict(X_tr, nd_tr, task_data[task]["X"]))

        best = max([la, lb], key=lambda n: pq[n]["ndcg@10"].mean())
        r = results[task]
        r["best_single"] = best
        r["methods"] = {}
        for name, v in pq.items():
            diff = v["ndcg@10"] - pq[best]["ndcg@10"]
            p = float(stats.ttest_rel(v["ndcg@10"], pq[best]["ndcg@10"]).pvalue) if diff.any() else 1.0
            r["methods"][name] = {"ndcg@10": float(v["ndcg@10"].mean()), "recall@100": float(v["recall@100"].mean()),
                                  "delta_ndcg@10": float(diff.mean()), "p_value": p}
        for sel, names in sel_sets.items():
            nd = stack(pq, names)["ndcg@10"]
            differ = nd.max(0) > nd.min(0)
            chosen = {s: pq[f"{sel}@{s}"]["ndcg@10"] for s in ("cv", "loto") if f"{sel}@{s}" in pq}
            r[f"{sel}_hit_rate"] = {s: float((v[differ] == nd.max(0)[differ]).mean()) for s, v in chosen.items()}
            r[f"{sel}_hit_rate"]["always_best_on_average"] = float(
                (nd[nd[:, differ].mean(1).argmax(), differ] == nd.max(0)[differ]).mean())
        np.savez(os.path.join(args.output_dir, "per_query", f"{task}.npz"),
                 **{name: v["ndcg@10"] for name, v in pq.items()})

        print(f"\n=== {task}: nDCG@10 / recall@100 / delta nDCG@10 vs {best} / p ===")
        for name, m in sorted(r["methods"].items(), key=lambda x: -x[1]["ndcg@10"]):
            print(f"  {name:22s} {m['ndcg@10']:.4f} / {m['recall@100']:.4f} / {m['delta_ndcg@10']:+.4f} / "
                  f"{m['p_value']:.4f}")
        for sel in sel_sets:
            print(f"  {sel} hit rate on queries where candidates differ: " +
                  ", ".join(f"{k} {v:.4f}" for k, v in r[f"{sel}_hit_rate"].items()))

    out = os.path.join(args.output_dir, "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}")

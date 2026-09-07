import json
import argparse
import os
import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def rerank_worker(rank, qids_shard, hard_negatives, queries_per_batch, tmp_dir):
    """Score (query, passage) pairs for a shard of queries on one GPU."""
    device = f"cuda:{rank}"
    model_name = os.path.join(os.environ["JWM_DATA_DIR"], "hf_models/naver/trecdl22-crossencoder-debertav3")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval().to(device)
    print(f"[GPU {rank}] Model loaded, processing {len(qids_shard)} queries "
          f"({queries_per_batch} queries/batch)")

    results = {}
    for batch_start in range(0, len(qids_shard), queries_per_batch):
        batch_qids = qids_shard[batch_start : batch_start + queries_per_batch]

        # Collect all pairs across queries in this batch, tracking boundaries
        all_pairs = []
        query_boundaries = []  # (qid, pos_count, neg_count)
        skip_qids = []
        for qid in batch_qids:
            data = hard_negatives[qid]
            negatives = data["hard_negatives"]
            if not negatives:
                skip_qids.append(qid)
                continue
            positives = data["positives"]
            pairs = (
                [(data["query"], pos["text"]) for pos in positives]
                + [(data["query"], neg["text"]) for neg in negatives]
            )
            query_boundaries.append((qid, len(positives), len(negatives)))
            all_pairs.extend(pairs)

        for qid in skip_qids:
            results[qid] = hard_negatives[qid]

        if not all_pairs:
            continue

        inputs = tokenizer(
            all_pairs, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            all_scores = model(**inputs).logits.squeeze(-1).cpu().tolist()

        if not isinstance(all_scores, list):
            all_scores = [all_scores]

        # Distribute scores back to each query
        offset = 0
        for qid, n_pos, n_neg in query_boundaries:
            data = hard_negatives[qid]
            positives = data["positives"]
            negatives = data["hard_negatives"]

            for pos, score in zip(positives, all_scores[offset : offset + n_pos]):
                pos["reranker_score"] = float(score)
            offset += n_pos

            for neg, score in zip(negatives, all_scores[offset : offset + n_neg]):
                if "score" in neg:
                    del neg["score"]
                neg["reranker_score"] = float(score)
            offset += n_neg

            negatives.sort(key=lambda x: x["reranker_score"], reverse=True)

            results[qid] = {
                "query": data["query"],
                "positives": positives,
                "hard_negatives": negatives,
            }

        if batch_start % (queries_per_batch * 100) == 0:
            print(f"[GPU {rank}] {batch_start}/{len(qids_shard)} queries")

    output_path = os.path.join(tmp_dir, f"result_{rank}.json")
    with open(output_path, "w") as f:
        json.dump(results, f)
    print(f"[GPU {rank}] Done. {len(results)} queries saved.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hard_negatives_file", help="Path to the hard negatives JSON file")
    parser.add_argument("--output", default="reranked_hard_negatives.json")
    parser.add_argument(
        "--queries-per-batch", type=int, required=True,
        help="Number of queries to feed to the model in one forward pass.",
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Start index of queries to process (default: 0).",
    )
    parser.add_argument(
        "--end", type=int, default=None,
        help="End index of queries to process (default: all).",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=None,
        help="Number of GPUs to use (default: all available).",
    )
    args = parser.parse_args()

    num_gpus = args.num_gpus or torch.cuda.device_count()
    print(f"Using {num_gpus} GPU(s)")

    # ── Load data (CPU only — no CUDA before fork) ──
    print(f"Loading {args.hard_negatives_file} ...")
    with open(args.hard_negatives_file) as f:
        hard_negatives = json.load(f)

    qids = list(hard_negatives.keys())[args.start : args.end]
    print(f"Processing queries [{args.start}:{args.end}] → {len(qids)} queries")

    tmp_dir = "./"

    if num_gpus <= 1:
        rerank_worker(0, qids, hard_negatives, args.queries_per_batch, tmp_dir)
    else:
        ctx = mp.get_context("fork")
        processes = []
        for rank in range(num_gpus):
            shard = qids[rank::num_gpus]
            p = ctx.Process(
                target=rerank_worker,
                args=(rank, shard, hard_negatives, args.queries_per_batch, tmp_dir),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    # ── Merge results ──
    print("Merging shard results...")
    results = {}
    for rank in range(num_gpus):
        with open(os.path.join(tmp_dir, f"result_{rank}.json")) as f:
            results.update(json.load(f))

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} reranked queries to {args.output}")


if __name__ == "__main__":
    main()

import json
import argparse
import os
import tempfile
import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def rerank_worker(rank, qids_shard, hard_negatives, batch_size, tmp_dir):
    """Score (query, passage) pairs for a shard of queries on one GPU."""
    device = f"cuda:{rank}"
    model_name = "naver/trecdl22-crossencoder-debertav3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval().to(device)
    print(f"[GPU {rank}] Model loaded, processing {len(qids_shard)} queries")

    results = {}
    for idx, qid in enumerate(qids_shard):
        data = hard_negatives[qid]
        query = data["query"]
        negatives = data["hard_negatives"]

        if not negatives:
            results[qid] = data
            continue

        positives = data["positives"]
        pairs = (
            [(query, pos["text"]) for pos in positives]
            + [(query, neg["text"]) for neg in negatives]
        )

        all_scores = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            inputs = tokenizer(
                batch, padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits.squeeze(-1)
            all_scores.extend(logits.cpu().tolist())

        for pos, score in zip(positives, all_scores[: len(positives)]):
            pos["reranker_score"] = float(score)
        for neg, score in zip(negatives, all_scores[len(positives) :]):
            if "score" in neg:
                del neg["score"]
            neg["reranker_score"] = float(score)

        negatives.sort(key=lambda x: x["reranker_score"], reverse=True)

        results[qid] = {
            "query": query,
            "positives": positives,
            "hard_negatives": negatives,
        }

        if idx % 500 == 0:
            print(f"[GPU {rank}] {idx}/{len(qids_shard)} queries")

    output_path = os.path.join(tmp_dir, f"result_{rank}.json")
    with open(output_path, "w") as f:
        json.dump(results, f)
    print(f"[GPU {rank}] Done. {len(results)} queries saved.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hard_negatives_file", help="Path to the hard negatives JSON file")
    parser.add_argument("--output", default="reranked_hard_negatives.json")
    parser.add_argument("--batch_size", type=int, default=64)
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
    print(f"Loaded {len(hard_negatives)} queries")

    qids = list(hard_negatives.keys())
    tmp_dir = tempfile.mkdtemp()

    if num_gpus <= 1:
        rerank_worker(0, qids, hard_negatives, args.batch_size, tmp_dir)
    else:
        # Shard queries across GPUs, fork workers (CUDA not yet initialized)
        ctx = mp.get_context("fork")
        processes = []
        for rank in range(num_gpus):
            shard = qids[rank::num_gpus]
            p = ctx.Process(
                target=rerank_worker,
                args=(rank, shard, hard_negatives, args.batch_size, tmp_dir),
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

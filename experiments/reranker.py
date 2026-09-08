import json
import argparse
import os
import math
import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def split_to_parts(input_file, num_parts, output_dir):
    """Split a JSON dict file into num_parts JSONL files with roughly equal queries."""
    print(f"Loading {input_file} for splitting...")
    with open(input_file) as f:
        data = json.load(f)

    qids = list(data.keys())
    total = len(qids)
    part_size = math.ceil(total / num_parts)
    print(f"Splitting {total} queries into {num_parts} parts (~{part_size} each)")

    os.makedirs(output_dir, exist_ok=True)
    for part_idx in range(num_parts):
        start = part_idx * part_size
        end = min(start + part_size, total)
        part_qids = qids[start:end]
        part_path = os.path.join(output_dir, f"part_{part_idx:02d}.jsonl")
        with open(part_path, "w") as f:
            for qid in part_qids:
                entry = {"qid": qid, **data[qid]}
                f.write(json.dumps(entry) + "\n")
        print(f"  Part {part_idx}: {len(part_qids)} queries → {part_path}")

    print("Split complete.")


def rerank_worker(rank, lines, queries_per_batch, tmp_dir):
    """Score (query, passage) pairs for assigned lines on one GPU."""
    device = f"cuda:{rank}"
    model_name = os.path.join(os.environ["JWM_DATA_DIR"], "hf_models/naver/trecdl22-crossencoder-debertav3")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval().to(device)
    print(f"[GPU {rank}] Model loaded, processing {len(lines)} queries "
          f"({queries_per_batch} queries/batch)")

    results = []
    for batch_start in range(0, len(lines), queries_per_batch):
        batch_lines = lines[batch_start : batch_start + queries_per_batch]

        all_pairs = []
        query_boundaries = []  # (line_idx, pos_count, neg_count)
        skip_indices = []
        for i, entry in enumerate(batch_lines):
            negatives = entry["hard_negatives"]
            if not negatives:
                skip_indices.append(i)
                continue
            positives = entry["positives"]
            pairs = (
                [(entry["query"], pos["text"]) for pos in positives]
                + [(entry["query"], neg["text"]) for neg in negatives]
            )
            query_boundaries.append((i, len(positives), len(negatives)))
            all_pairs.extend(pairs)

        for i in skip_indices:
            results.append(batch_lines[i])

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

        offset = 0
        for i, n_pos, n_neg in query_boundaries:
            entry = batch_lines[i]
            positives = entry["positives"]
            negatives = entry["hard_negatives"]

            for pos, score in zip(positives, all_scores[offset : offset + n_pos]):
                pos["reranker_score"] = float(score)
            offset += n_pos

            for neg, score in zip(negatives, all_scores[offset : offset + n_neg]):
                if "score" in neg:
                    del neg["score"]
                neg["reranker_score"] = float(score)
            offset += n_neg

            negatives.sort(key=lambda x: x["reranker_score"], reverse=True)

            results.append({
                "qid": entry["qid"],
                "query": entry["query"],
                "positives": positives,
                "hard_negatives": negatives,
            })

        if batch_start % (queries_per_batch * 100) == 0:
            print(f"[GPU {rank}] {batch_start}/{len(lines)} queries")

    output_path = os.path.join(tmp_dir, f"result_{rank}.jsonl")
    with open(output_path, "w") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")
    print(f"[GPU {rank}] Done. {len(results)} queries saved.")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── split ──
    sp_split = subparsers.add_parser("split", help="Split JSON into JSONL parts")
    sp_split.add_argument("input_file", help="Path to the hard negatives JSON file")
    sp_split.add_argument("--num-parts", type=int, required=True,
                          help="Number of parts to split into.")
    sp_split.add_argument("--output-dir", type=str, required=True,
                          help="Directory to write part_XX.jsonl files.")

    # ── rerank ──
    sp_rerank = subparsers.add_parser("rerank", help="Rerank all JSONL parts in a directory")
    sp_rerank.add_argument("input_dir", help="Directory containing part_XX.jsonl files")
    sp_rerank.add_argument("--output-dir", required=True,
                           help="Directory to write reranked_XX.jsonl files.")
    sp_rerank.add_argument("--queries-per-batch", type=int, required=True,
                           help="Number of queries per forward pass.")
    sp_rerank.add_argument("--num-gpus", type=int, default=None,
                           help="Number of GPUs to use (default: all available).")

    args = parser.parse_args()

    if args.command == "split":
        split_to_parts(args.input_file, args.num_parts, args.output_dir)
        return

    # ── rerank ──
    num_gpus = args.num_gpus or torch.cuda.device_count()
    print(f"Using {num_gpus} GPU(s)")

    os.makedirs(args.output_dir, exist_ok=True)
    part_files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".jsonl"))
    print(f"Found {len(part_files)} JSONL files in {args.input_dir}")

    for part_file in part_files:
        part_path = os.path.join(args.input_dir, part_file)
        out_name = "reranked_" + part_file
        out_path = os.path.join(args.output_dir, out_name)

        if os.path.exists(out_path):
            print(f"Skipping {part_file} — {out_name} already exists")
            continue

        print(f"\n{'='*60}")
        print(f"Loading {part_file} ...")
        with open(part_path) as f:
            lines = [json.loads(line) for line in f]
        print(f"Loaded {len(lines)} queries")

        tmp_dir = "./"

        if num_gpus <= 1:
            rerank_worker(0, lines, args.queries_per_batch, tmp_dir)
        else:
            ctx = mp.get_context("fork")
            processes = []
            for rank in range(num_gpus):
                shard = lines[rank::num_gpus]
                p = ctx.Process(
                    target=rerank_worker,
                    args=(rank, shard, args.queries_per_batch, tmp_dir),
                )
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

        # Merge GPU shards for this part
        with open(out_path, "w") as fout:
            for rank in range(num_gpus):
                shard_path = os.path.join(tmp_dir, f"result_{rank}.jsonl")
                with open(shard_path) as fin:
                    for line in fin:
                        fout.write(line)
                os.remove(shard_path)
        count = sum(1 for _ in open(out_path))
        print(f"Saved {count} reranked queries to {out_path}")


if __name__ == "__main__":
    main()

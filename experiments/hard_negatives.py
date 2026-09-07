import os
import argparse
import random
import tempfile
import json
import torch
import torch.multiprocessing as mp
import ir_datasets
from sentence_transformers import SparseEncoder
from collections import defaultdict


def load_msmarco_data(max_queries=None, max_passages=None):
    """Load MS MARCO passage ranking data via ir_datasets.

    Args:
        max_queries: Maximum number of queries to load (None for all ~503K).
        max_passages: Maximum number of passages to load (None for all ~8.8M).
    """
    print("Loading MS MARCO passage ranking dataset...")
    ds = ir_datasets.load("msmarco-passage/train")

    # Load passages
    print(f"Loading passages{f' (max {max_passages:,})' if max_passages else ' (all ~8.8M)'}...")
    all_passages = {}
    for i, doc in enumerate(ds.docs_iter()):
        if max_passages and i >= max_passages:
            break
        all_passages[doc.doc_id] = doc.text
    passage_id_set = set(all_passages.keys())

    # Load qrels, keeping only those whose passages are in our collection
    print("Loading relevance judgments...")
    positives = defaultdict(list)
    for qrel in ds.qrels_iter():
        if qrel.relevance > 0 and qrel.doc_id in passage_id_set:
            positives[qrel.query_id].append(qrel.doc_id)

    # Load queries that have at least one positive in our passage set
    print(f"Loading queries{f' (max {max_queries:,})' if max_queries else ' (all)'}...")
    queries = {}
    for query in ds.queries_iter():
        if query.query_id in positives:
            queries[query.query_id] = query.text
            if max_queries and len(queries) >= max_queries:
                break

    # Trim positives to only the queries we kept
    positives = {qid: pids for qid, pids in positives.items() if qid in queries}

    print(f"Loaded {len(queries):,} queries with positives, {len(all_passages):,} unique passages")
    return queries, positives, all_passages


def gpu_worker(rank, num_gpus, query_ids, query_texts, passage_texts,
               cache_dir, passage_chunk_size, top_k, query_batch_size,
               num_chunks, needs_encoding, tmp_dir, barrier):
    """Worker that runs on one GPU: encodes passages (if needed) then mines top-k."""
    device = f"cuda:{rank}"
    model = SparseEncoder("naver/splade-cocondenser-selfdistil", device=device)

    # Phase 1: encode passages in parallel (each GPU handles its assigned chunks)
    if needs_encoding:
        os.makedirs(cache_dir, exist_ok=True)
        for chunk_idx in range(rank, num_chunks, num_gpus):
            chunk_path = os.path.join(cache_dir, f"chunk_{chunk_idx}.pt")
            if os.path.exists(chunk_path):
                print(f"[GPU {rank}] Chunk {chunk_idx}/{num_chunks} cached, skipping")
                continue
            p_start = chunk_idx * passage_chunk_size
            p_end = min(p_start + passage_chunk_size, len(passage_texts))
            emb = model.encode_document(passage_texts[p_start:p_end])
            torch.save(emb.cpu(), chunk_path)
            print(f"[GPU {rank}] Encoded chunk {chunk_idx}/{num_chunks}")
        barrier.wait()

    # Phase 2: mine top-k passage indices for this GPU's query shard
    shard_qids = query_ids[rank::num_gpus]
    shard_texts = query_texts[rank::num_gpus]

    shard_results = {}
    for q_start in range(0, len(shard_qids), query_batch_size):
        q_end = min(q_start + query_batch_size, len(shard_qids))
        batch_texts = shard_texts[q_start:q_end]
        num_q = q_end - q_start

        batch_emb = model.encode_query(batch_texts)

        top_scores = torch.full((num_q, top_k), float("-inf"), device=device)
        top_indices = torch.zeros((num_q, top_k), dtype=torch.long, device=device)

        for chunk_idx in range(num_chunks):
            chunk_path = os.path.join(cache_dir, f"chunk_{chunk_idx}.pt")
            chunk_emb = torch.load(chunk_path, weights_only=True).to(device)
            p_start = chunk_idx * passage_chunk_size

            chunk_scores = model.similarity(batch_emb, chunk_emb)
            chunk_k = min(top_k, chunk_scores.shape[1])
            chunk_top_scores, chunk_top_idx = torch.topk(chunk_scores, k=chunk_k, dim=1)
            chunk_top_idx += p_start

            combined_scores = torch.cat([top_scores, chunk_top_scores], dim=1)
            combined_indices = torch.cat([top_indices, chunk_top_idx], dim=1)
            final_k = min(top_k, combined_scores.shape[1])
            best_scores, best_pos = torch.topk(combined_scores, k=final_k, dim=1)
            top_scores = best_scores
            top_indices = combined_indices.gather(1, best_pos)

            del chunk_emb, chunk_scores
            torch.cuda.empty_cache()

        top_scores_cpu = top_scores.cpu()
        top_indices_cpu = top_indices.cpu()

        for i in range(num_q):
            qid = shard_qids[q_start + i]
            shard_results[qid] = [
                (top_indices_cpu[i, j].item(), top_scores_cpu[i, j].item())
                for j in range(top_k)
                if top_scores_cpu[i, j].item() > float("-inf")
            ]

        if q_start % (query_batch_size * 10) == 0:
            print(f"[GPU {rank}] {q_start}/{len(shard_qids)} queries")

    output_path = os.path.join(tmp_dir, f"shard_{rank}.pt")
    torch.save(shard_results, output_path)
    print(f"[GPU {rank}] Done. {len(shard_results)} queries saved.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mine hard negatives from MS MARCO using SPLADE. "
        "Follows SPLADE-v3 strategy: retrieve top-K, keep N from top + M random from remainder."
    )
    parser.add_argument(
        "--max-queries", type=int, default=None,
        help="Max number of queries to process (default: all ~503K).",
    )
    parser.add_argument(
        "--max-passages", type=int, default=None,
        help="Max number of passages to load from the collection (default: all ~8.8M).",
    )
    parser.add_argument(
        "--top-k", type=int, required=True,
        help="Number of top candidates to retrieve per query from SPLADE.",
    )
    parser.add_argument(
        "--num-top", type=int, required=True,
        help="Number of top-ranked negatives to keep (e.g. 50).",
    )
    parser.add_argument(
        "--num-random", type=int, required=True,
        help="Number of random negatives to sample from rank num_top+1 to top_k (e.g. 50).",
    )
    parser.add_argument(
        "--passage-chunk-size", type=int, default=50000,
        help="Chunk size for encoding passages to disk (default: 50000).",
    )
    parser.add_argument(
        "--query-batch-size", type=int, default=256,
        help="Batch size for query encoding (default: 256).",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="passage_embeddings_cache",
        help="Directory to cache passage embeddings (default: passage_embeddings_cache).",
    )
    parser.add_argument(
        "--output", type=str, default="msmarco_hard_negatives.json",
        help="Output JSON file (default: msmarco_hard_negatives.json).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling (default: 42).",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=None,
        help="Number of GPUs to use (default: all available).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    num_gpus = args.num_gpus or torch.cuda.device_count()
    print(f"Using {num_gpus} GPU(s)")

    # ── Load data (CPU only — no CUDA before fork) ──
    queries, positives, all_passages = load_msmarco_data(
        max_queries=args.max_queries,
        max_passages=args.max_passages,
    )
    passage_ids = list(all_passages.keys())
    passage_texts = list(all_passages.values())
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]

    num_passages = len(passage_ids)
    num_chunks = (num_passages + args.passage_chunk_size - 1) // args.passage_chunk_size
    needs_encoding = any(
        not os.path.exists(os.path.join(args.cache_dir, f"chunk_{i}.pt"))
        for i in range(num_chunks)
    )
    if needs_encoding:
        print(f"Passage cache incomplete — will encode {num_chunks} chunks across {num_gpus} GPU(s)")
    else:
        print(f"Passage cache complete ({num_chunks} chunks)")

    # ── Mine top-k indices ──
    tmp_dir = tempfile.mkdtemp()

    if num_gpus <= 1:
        # Single-GPU: run worker directly (no fork needed)
        gpu_worker(
            0, 1, query_ids, query_texts, passage_texts,
            args.cache_dir, args.passage_chunk_size, args.top_k,
            args.query_batch_size, num_chunks, needs_encoding,
            tmp_dir, None,
        )
    else:
        ctx = mp.get_context("fork")
        barrier = ctx.Barrier(num_gpus) if needs_encoding else None
        processes = []
        for rank in range(num_gpus):
            p = ctx.Process(
                target=gpu_worker,
                args=(rank, num_gpus, query_ids, query_texts, passage_texts,
                      args.cache_dir, args.passage_chunk_size, args.top_k,
                      args.query_batch_size, num_chunks, needs_encoding,
                      tmp_dir, barrier),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    # ── Merge shards and resolve indices → PIDs/texts ──
    print("Merging shards and resolving passages...")
    raw_topk = {}
    for rank in range(num_gpus):
        shard = torch.load(
            os.path.join(tmp_dir, f"shard_{rank}.pt"), weights_only=False,
        )
        raw_topk.update(shard)

    rng = random.Random(args.seed)
    hard_negatives = {}
    for qid in query_ids:
        if qid not in raw_topk:
            continue
        positive_pids = set(positives.get(qid, []))

        # Resolve global indices to PIDs + texts, filtering positives
        candidates = []
        for global_idx, score in raw_topk[qid]:
            pid = passage_ids[global_idx]
            if pid not in positive_pids:
                candidates.append(
                    {"pid": pid, "score": score, "text": all_passages[pid]}
                )

        # SPLADE-v3 strategy: top-N + random from remainder
        top_part = candidates[: args.num_top]
        remainder = candidates[args.num_top :]
        random_part = rng.sample(remainder, min(args.num_random, len(remainder)))

        hard_negatives[qid] = {
            "query": queries[qid],
            "positives": [
                {"pid": pid, "text": all_passages[pid]} for pid in positive_pids
            ],
            "hard_negatives": top_part + random_part,
        }

    # ── Save ──
    with open(args.output, "w") as f:
        json.dump(hard_negatives, f, indent=2)
    print(f"\nSaved {len(hard_negatives):,} query hard-negative sets to {args.output}")

    sample_qid = next(iter(hard_negatives))
    sample = hard_negatives[sample_qid]
    print(f"\n{'='*80}")
    print(f"Sample query: {sample['query']}")
    print(f"Positive: {sample['positives'][0]['text'][:120]}...")
    print(f"Num negatives: {len(sample['hard_negatives'])}")
    for i, neg in enumerate(sample["hard_negatives"][:3]):
        print(f"  {i+1}. [score={neg['score']:.4f}] {neg['text'][:120]}...")


if __name__ == "__main__":
    main()

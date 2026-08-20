import os
import argparse
import torch
import json
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


def encode_passages_to_disk(model, passage_texts, cache_dir, chunk_size=50000):
    """Encode all passages in chunks and save each chunk to disk."""
    os.makedirs(cache_dir, exist_ok=True)
    num_chunks = (len(passage_texts) + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        chunk_path = os.path.join(cache_dir, f"chunk_{chunk_idx}.pt")
        if os.path.exists(chunk_path):
            print(f"  Chunk {chunk_idx}/{num_chunks} already cached, skipping")
            continue

        p_start = chunk_idx * chunk_size
        p_end = min(p_start + chunk_size, len(passage_texts))
        chunk_texts = passage_texts[p_start:p_end]

        embeddings = []
        batch = chunk_texts[ : chunk_size]
        emb = model.encode_document(batch)
        embeddings.append(emb.cpu())

        chunk_emb = torch.cat(embeddings, dim=0)
        torch.save(chunk_emb, chunk_path)
        print(f"  Chunk {chunk_idx}/{num_chunks}: encoded {p_end - p_start} passages, saved to {chunk_path}")

    return num_chunks


def mine_hard_negatives(
    model, queries, positives, all_passages, cache_dir,
    top_k=30, num_hard_negatives=10, query_batch_size=256, passage_chunk_size=10000,
):
    """
    Mine hard negatives: passages that score highly with the query
    but are NOT positives.
    """
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    passage_ids = list(all_passages.keys())
    num_passages = len(passage_ids)
    num_chunks = (num_passages + passage_chunk_size - 1) // passage_chunk_size

    print(f"\nMining hard negatives for {len(query_texts)} queries against {num_passages} passages...")

    hard_negatives = {}

    for q_start in range(0, len(query_ids), query_batch_size):
        q_end = min(q_start + query_batch_size, len(query_ids))
        batch_query_texts = query_texts[q_start:q_end]
        num_queries_in_batch = q_end - q_start

        batch_query_emb = model.encode_query(batch_query_texts)
        device = batch_query_emb.device

        top_scores = torch.full((num_queries_in_batch, top_k), float("-inf"), device=device)
        top_indices = torch.zeros((num_queries_in_batch, top_k), dtype=torch.long, device=device)

        for chunk_idx in range(num_chunks):
            chunk_path = os.path.join(cache_dir, f"chunk_{chunk_idx}.pt")
            passage_chunk_emb = torch.load(chunk_path, weights_only=True).to(device)
            p_start = chunk_idx * passage_chunk_size

            chunk_scores = model.similarity(batch_query_emb, passage_chunk_emb)

            chunk_k = min(top_k, chunk_scores.shape[1])
            chunk_top_scores, chunk_top_idx = torch.topk(chunk_scores, k=chunk_k, dim=1)
            chunk_top_idx += p_start

            combined_scores = torch.cat([top_scores, chunk_top_scores], dim=1)
            combined_indices = torch.cat([top_indices, chunk_top_idx], dim=1)
            final_k = min(top_k, combined_scores.shape[1])
            best_scores, best_pos = torch.topk(combined_scores, k=final_k, dim=1)
            top_scores = best_scores
            top_indices = combined_indices.gather(1, best_pos)

            del passage_chunk_emb, chunk_scores
            torch.cuda.empty_cache()

        top_scores = top_scores.cpu()
        top_indices = top_indices.cpu()

        for i in range(num_queries_in_batch):
            qid = query_ids[q_start + i]
            positive_pids = set(positives.get(qid, []))

            negatives = []
            for j in range(top_k):
                pid = passage_ids[top_indices[i, j].item()]
                if pid not in positive_pids:
                    negatives.append(
                        {"pid": pid, "score": top_scores[i, j].item(), "text": all_passages[pid]}
                    )
                    if len(negatives) >= num_hard_negatives:
                        break

            hard_negatives[qid] = {
                "query": queries[qid],
                "positives": [
                    {"pid": pid, "text": all_passages[pid]} for pid in positive_pids
                ],
                "hard_negatives": negatives,
            }

        print(f"  Processed queries {q_start}-{q_end}/{len(query_ids)}")

    return hard_negatives


def parse_args():
    parser = argparse.ArgumentParser(description="Mine hard negatives from MS MARCO using SPLADE.")
    parser.add_argument(
        "--max-queries", type=int, default=None,
        help="Max number of queries to process (default: all ~503K).",
    )
    parser.add_argument(
        "--max-passages", type=int, default=None,
        help="Max number of passages to load from the collection (default: all ~8.8M).",
    )
    parser.add_argument(
        "--top-k", type=int, default=30,
        help="Number of top candidates to retrieve per query (default: 30).",
    )
    parser.add_argument(
        "--num-hard-negatives", type=int, default=10,
        help="Number of hard negatives to keep per query (default: 10).",
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
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading SPLADE CoCondenser SelfDistil model...")
    model = SparseEncoder("naver/splade-cocondenser-selfdistil")

    queries, positives, all_passages = load_msmarco_data(
        max_queries=args.max_queries,
        max_passages=args.max_passages,
    )

    passage_texts = list(all_passages.values())

    print(f"\nEncoding {len(passage_texts):,} passages to disk...")
    encode_passages_to_disk(model, passage_texts, args.cache_dir, chunk_size=args.passage_chunk_size)

    hard_negatives = mine_hard_negatives(
        model,
        queries,
        positives,
        all_passages,
        args.cache_dir,
        top_k=args.top_k,
        num_hard_negatives=args.num_hard_negatives,
        query_batch_size=args.query_batch_size,
        passage_chunk_size=args.passage_chunk_size,
    )

    # Save results
    with open(args.output, "w") as f:
        json.dump(hard_negatives, f, indent=2)
    print(f"\nSaved {len(hard_negatives):,} query hard-negative sets to {args.output}")

    # Print a sample
    sample_qid = next(iter(hard_negatives))
    sample = hard_negatives[sample_qid]
    print(f"\n{'='*80}")
    print(f"Sample query: {sample['query']}")
    print(f"\nPositive passage(s):")
    for p in sample["positives"]:
        print(f"  - {p['text'][:120]}...")
    print(f"\nTop hard negatives:")
    for i, neg in enumerate(sample["hard_negatives"][:5]):
        print(f"  {i+1}. [score={neg['score']:.4f}] {neg['text'][:120]}...")


if __name__ == "__main__":
    main()

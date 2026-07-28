import torch
import json
from datasets import load_dataset
from sentence_transformers import SparseEncoder
from collections import defaultdict


def load_msmarco_data(num_queries=500, num_passages=50000):
    """Load a subset of MS MARCO for hard negative mining."""
    print("Loading MS MARCO dataset...")
    dataset = load_dataset("microsoft/ms_marco", "v1.1", split="train")

    queries = {}
    positives = defaultdict(list)
    all_passages = {}
    passage_id = 0

    for i, example in enumerate(dataset):
        # if len(queries) >= num_queries:
        #     break

        query = example["query"]
        query_id = str(i)

        if query_id in queries:
            continue

        queries[query_id] = query

        for j, (passage_text, is_selected) in enumerate(
            zip(example["passages"]["passage_text"], example["passages"]["is_selected"])
        ):
            pid = f"p_{passage_id}"
            all_passages[pid] = passage_text
            passage_id += 1

            if is_selected == 1:
                positives[query_id].append(pid)

    # Keep only queries that have at least one positive
    queries = {qid: q for qid, q in queries.items() if qid in positives}

    print(f"Loaded {len(queries)} queries with positives, {len(all_passages)} passages")
    return queries, positives, all_passages


def encode_in_batches(model, texts, batch_size=64, is_query=True):
    """Encode texts in batches to avoid OOM."""
    all_embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        if is_query:
            emb = model.encode_query(batch)
        else:
            emb = model.encode_document(batch)
        all_embeddings.append(emb)
        if (start // batch_size) % 10 == 0:
            print(f"  Encoded {start + len(batch)}/{len(texts)}")

    return torch.cat(all_embeddings, dim=0)


def mine_hard_negatives(
    model, queries, positives, all_passages, top_k=30, num_hard_negatives=10
):
    """
    Mine hard negatives: passages that score highly with the query
    but are NOT positives.
    """
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    passage_ids = list(all_passages.keys())
    passage_texts = [all_passages[pid] for pid in passage_ids]

    print(f"\nEncoding {len(passage_texts)} passages...")
    passage_embeddings = encode_in_batches(model, passage_texts[:100], batch_size=256, is_query=False)

    print(f"\nMining hard negatives for {len(query_texts)} queries...")
    hard_negatives = {}
    batch_size = 256

    for batch_start in range(0, len(query_ids), batch_size):
        batch_end = min(batch_start + batch_size, len(query_ids))
        batch_query_texts = query_texts[batch_start:batch_end]

        batch_query_emb = model.encode_query(batch_query_texts)
        scores = model.similarity(batch_query_emb, passage_embeddings)

        _, top_indices = torch.topk(scores, k=min(top_k, scores.shape[1]), dim=1)

        for i in range(batch_end - batch_start):
            qid = query_ids[batch_start + i]
            positive_pids = set(positives.get(qid, []))

            negatives = []
            for idx in top_indices[i].tolist():
                pid = passage_ids[idx]
                if pid not in positive_pids:
                    negatives.append(
                        {"pid": pid, "score": scores[i, idx].item(), "text": all_passages[pid]}
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

        print(f"  Processed {batch_end}/{len(query_ids)} queries")

    return hard_negatives


def main():
    print("Loading SPLADE CoCondenser SelfDistil model...")
    model = SparseEncoder("naver/splade-cocondenser-selfdistil")

    queries, positives, all_passages = load_msmarco_data(
        num_queries=500, num_passages=50000
    )

    hard_negatives = mine_hard_negatives(
        model,
        queries,
        positives,
        all_passages,
        top_k=30,
        num_hard_negatives=10,
    )

    # Save results
    output_file = "msmarco_hard_negatives.json"
    with open(output_file, "w") as f:
        json.dump(hard_negatives, f, indent=2)
    print(f"\nSaved {len(hard_negatives)} query hard-negative sets to {output_file}")

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

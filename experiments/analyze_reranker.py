import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reranked_file", help="Path to the reranked hard negatives JSON file")
    args = parser.parse_args()

    with open(args.reranked_file) as f:
        data = json.load(f)

    total = 0
    has_harder_negative = 0

    for qid, entry in data.items():
        positives = entry["positives"]
        negatives = entry["hard_negatives"]

        if not positives or not negatives:
            continue

        min_pos_score = min(p["reranker_score"] for p in positives)
        max_neg_score = max(n["reranker_score"] for n in negatives)

        total += 1
        if max_neg_score > min_pos_score:
            has_harder_negative += 1

    pct = has_harder_negative / total * 100 if total else 0
    print(f"Total queries: {total}")
    print(f"Queries with a hard negative scoring higher than at least one positive: {has_harder_negative} ({pct:.1f}%)")


if __name__ == "__main__":
    main()

import json
import argparse
from sentence_transformers import CrossEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hard_negatives_file", help="Path to the hard negatives JSON file")
    parser.add_argument("--output", default="reranked_hard_negatives.json")
    parser.add_argument("--top_k", type=int, default=8)
    args = parser.parse_args()

    with open(args.hard_negatives_file) as f:
        hard_negatives = json.load(f)

    model = CrossEncoder("naver/trecdl22-crossencoder-debertav3", trust_remote_code=True)

    results = {}
    for idx, (qid, data) in enumerate(hard_negatives.items()):
        query = data["query"]
        negatives = data["hard_negatives"][:args.top_k]

        if not negatives:
            results[qid] = data
            continue

        pairs = [(query, neg["text"]) for neg in negatives]
        scores = model.predict(pairs)

        for neg, score in zip(negatives, scores):
            del neg["score"]
            neg["reranker_score"] = float(score)

        negatives.sort(key=lambda x: x["reranker_score"], reverse=True)

        results[qid] = {
            "query": query,
            "positives": data["positives"],
            "hard_negatives": negatives,
        }

        if idx % 500 == 0:
            print(f"  Processed {idx}/{len(hard_negatives)} queries")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} reranked queries to {args.output}")


if __name__ == "__main__":
    main()

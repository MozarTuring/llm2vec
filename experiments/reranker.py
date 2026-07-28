import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hard_negatives_file", help="Path to the hard negatives JSON file")
    parser.add_argument("--output", default="reranked_hard_negatives.json")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    with open(args.hard_negatives_file) as f:
        hard_negatives = json.load(f)

    model_name = "naver/trecdl22-crossencoder-debertav3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    results = {}
    for idx, (qid, data) in enumerate(hard_negatives.items()):
        query = data["query"]
        negatives = data["hard_negatives"][:args.top_k]

        if not negatives:
            results[qid] = data
            continue

        pairs = [(query, neg["text"]) for neg in negatives]

        all_scores = []
        for start in range(0, len(pairs), args.batch_size):
            batch = pairs[start : start + args.batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**inputs).logits.squeeze(-1)
            all_scores.extend(logits.cpu().tolist())

        for neg, score in zip(negatives, all_scores):
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

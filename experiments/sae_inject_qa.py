"""Training-free knowledge injection: retrieved document as SAE features added to the generator's residual stream.

Each document is encoded by a SPLARE checkpoint exactly as for retrieval (layer-N residual -> frozen
Llama Scope SAE -> log(1+relu) -> max-pool, BOS excluded -> top-k). The pooled values are mapped back
to SAE activations (expm1), decoded with the SAE decoder (no bias) into a residual-stream direction,
rescaled to alpha * activation_norm, and added at every position to the output of layer N of the base
causal Llama during generation. No context tokens, no trained projector.

Data: SQuAD v1.1 validation questions whose answer is a year that occurs in the paragraph.
  orig  the original paragraph
  cf    counterfactual paragraph: every occurrence of the answer year replaced by another year,
        so a correct "cf" answer can only come from the document, not from parametric memory.

Conditions (each on orig and cf):
  closed        no document
  text          document as text in the prompt (upper bound)
  sae@a         decoded SAE features, norm a * activation_norm
  random@a      same top-k values on random feature indices, decoded, same norm (control)
  dense@a       mean layer-N residual of the document (encoder), same norm (is the SAE needed?)

Metrics: orig -> answer contains the gold year; cf -> answer contains the cf year (follow_cf) or the
original year (follow_orig, i.e. parametric memory or topic prior).

    python experiments/sae_inject_qa.py --trained_checkpoint_path <root>/output/layerwise/<run>/checkpoint-N --output_dir results_sae_inject
"""

import argparse
import gc
import json
import os
import random
import re
import sys
import traceback

import numpy as np
import torch
from datasets import load_dataset
from safetensors.torch import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from mteb_eval_layerwise import LayerwiseEncoder, resolve_checkpoint_args

YEAR_RE = re.compile(r"^\d{4}$")

FEW_SHOT = (
    "Answer each question with a short answer.\n\n"
    "Question: What is the capital of France?\nAnswer: Paris\n\n"
    "Question: In what year did the Second World War end?\nAnswer: 1945\n\n"
    "Question: Who wrote the play Hamlet?\nAnswer: William Shakespeare\n\n"
)


def build_examples(num_examples, seed):
    """SQuAD validation questions with a year answer present in the paragraph, plus a cf paragraph."""
    rng = random.Random(seed)
    data = load_dataset("rajpurkar/squad", split="validation")
    examples, seen = [], set()
    for row in data:
        year = row["answers"]["text"][0].strip()
        if not YEAR_RE.match(year) or not 1000 <= int(year) <= 2025:
            continue
        pattern = re.compile(rf"\b{year}\b")
        if row["question"] in seen or year in row["question"] or not pattern.search(row["context"]):
            continue
        while True:
            cf_year = str(int(year) + rng.choice([-1, 1]) * rng.randint(3, 50))
            if int(cf_year) <= 2025 and not re.search(rf"\b{cf_year}\b", row["context"]):
                break
        seen.add(row["question"])
        examples.append({
            "id": row["id"], "question": row["question"], "year": year, "cf_year": cf_year,
            "orig": row["context"], "cf": pattern.sub(cf_year, row["context"]),
        })
    rng.shuffle(examples)
    print(f"SQuAD validation: {len(examples)} year questions, using {min(num_examples, len(examples))}")
    return examples[:num_examples]


@torch.no_grad()
def encode_documents(encoder, texts, top_k, max_length, batch_size):
    """Pooled SAE vectors (as in retrieval, top-k pruned) and mean layer-N residuals, on one GPU."""
    module = encoder.encode_module
    device = next(module.parameters()).device
    pooled_all, mean_all = [], []
    for i in range(0, len(texts), batch_size):
        batch = encoder.tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                                  max_length=max_length, return_tensors="pt").to(device)
        hidden = module.backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])[0]
        pool_mask = (batch["attention_mask"] * (batch["input_ids"] != encoder.tokenizer.bos_token_id)).unsqueeze(-1)
        sae_out = torch.log(1 + torch.relu(module.sae(hidden * module.sae_norm_scale))) * pool_mask
        pooled = sae_out.max(dim=1)[0]
        vals, idx = pooled.topk(top_k, dim=-1)
        pooled = torch.zeros_like(pooled).scatter_(-1, idx, vals)
        mean = (hidden.float() * pool_mask).sum(1) / pool_mask.sum(1).clamp(min=1)
        pooled_all.append(pooled.float().cpu())
        mean_all.append(mean.cpu())
    return torch.cat(pooled_all), torch.cat(mean_all)


def decode_features(pooled, W_dec, sae_norm_scale, rng):
    """Pooled log1p values -> SAE activations -> residual directions; also the random-index control."""
    acts = torch.expm1(pooled)
    sae_dir = (acts.to(W_dec) @ W_dec.T).float() / sae_norm_scale
    rand = torch.zeros_like(acts)
    for r in range(acts.shape[0]):
        nz = acts[r].nonzero().squeeze(-1)
        perm = torch.from_numpy(rng.choice(acts.shape[1], size=len(nz), replace=False))
        rand[r, perm] = acts[r, nz]
    rand_dir = (rand.to(W_dec) @ W_dec.T).float() / sae_norm_scale
    return sae_dir, rand_dir


class ResidualInjector:
    """Forward hook adding a per-example vector to a decoder layer's output at every position."""

    def __init__(self, layer):
        self.delta = None
        layer.register_forward_hook(self.hook)

    def hook(self, module, inputs, output):
        if self.delta is None:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden + self.delta[:, None, :].to(hidden.dtype)
        return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden


def make_prompt(question, doc=None):
    context = f"Context: {doc}\n\n" if doc is not None else ""
    return f"{FEW_SHOT}{context}Question: {question}\nAnswer:"


@torch.no_grad()
def generate(model, tokenizer, injector, prompts, deltas, batch_size, max_new_tokens):
    answers = []
    for i in range(0, len(prompts), batch_size):
        batch = tokenizer(prompts[i:i + batch_size], padding=True, return_tensors="pt").to(model.device)
        injector.delta = None if deltas is None else deltas[i:i + batch_size].to(model.device)
        out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        texts = tokenizer.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
        answers.extend(t.strip().split("\n")[0] for t in texts)
    injector.delta = None
    return answers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_checkpoint_path", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--peft_model_name_or_path", type=str)
    parser.add_argument("--sae_weights_path", type=str)
    parser.add_argument("--lora_layers", type=int)
    parser.add_argument("--output_dir", type=str, default="results_sae_inject")
    parser.add_argument("--num_examples", type=int, default=500)
    parser.add_argument("--doc_top_k", type=int, default=400)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0],
                        help="Injection norm in units of the SAE's dataset-average activation norm.")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    resolve_checkpoint_args(args, parser)

    try:
        torch.manual_seed(args.seed)
        examples = build_examples(args.num_examples, args.seed)
        n = len(examples)

        # 1. Encode orig + cf documents with the retrieval encoder, then free it.
        encoder = LayerwiseEncoder(
            model_name_or_path=args.model_name_or_path,
            peft_model_name_or_path=args.peft_model_name_or_path,
            sae_weights_path=args.sae_weights_path,
            lora_layers=args.lora_layers,
            trained_checkpoint_path=args.trained_checkpoint_path,
            max_length=args.max_length,
        )
        sae_norm_scale = encoder.sae_norm_scale
        activation_norm = encoder.encode_module.sae.in_features ** 0.5 / sae_norm_scale
        docs = [ex["orig"] for ex in examples] + [ex["cf"] for ex in examples]
        pooled, dense_dir = encode_documents(encoder, docs, args.doc_top_k, args.max_length, args.batch_size)
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Encoded {len(docs)} documents: nnz mean={(pooled != 0).sum(1).float().mean():.0f}, "
              f"activation_norm={activation_norm:.3f}")

        # 2. Decode SAE features into residual directions.
        with safe_open(args.sae_weights_path, framework="pt") as f:
            W_dec = f.get_tensor("decoder.weight").to("cuda", torch.float32)  # (d_model, d_sae)
        sae_dir, rand_dir = decode_features(pooled, W_dec, sae_norm_scale,
                                            np.random.default_rng(args.seed))
        del W_dec
        torch.cuda.empty_cache()
        sae_dir, rand_dir = sae_dir.cpu(), rand_dir.cpu()
        directions = {}
        for name, vecs in (("sae", sae_dir), ("random", rand_dir), ("dense", dense_dir)):
            norms = vecs.norm(dim=-1, keepdim=True)
            print(f"[{name}] raw direction norm: mean={norms.mean():.2f} min={norms.min():.2f} max={norms.max():.2f}")
            directions[name] = vecs / norms.clamp(min=1e-8)
        cos = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()
        print(f"mean cosine: sae vs dense={cos(directions['sae'], directions['dense']):.3f}  "
              f"sae vs random={cos(directions['sae'], directions['random']):.3f}  "
              f"sae orig vs cf={cos(directions['sae'][:n], directions['sae'][n:]):.3f}")

        # 3. Generate with the base causal model, injecting after layer N.
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16)
        model.to("cuda").eval()
        injector = ResidualInjector(model.model.layers[args.lora_layers])
        print(f"Generator {args.model_name_or_path}, injecting at output of layer {args.lora_layers}")

        conditions = [("closed", None, None), ("text", None, None)]
        conditions += [(f"{name}@{a}", name, a) for a in args.alphas for name in ("sae", "random", "dense")]
        results, generations = {}, {ex["id"]: dict(ex) for ex in examples}
        for split, offset in (("orig", 0), ("cf", n)):
            for cond, name, alpha in conditions:
                docs_split = [ex[split] for ex in examples]
                prompts = [make_prompt(ex["question"], d if cond == "text" else None)
                           for ex, d in zip(examples, docs_split)]
                deltas = None if name is None else directions[name][offset:offset + n] * alpha * activation_norm
                answers = generate(model, tokenizer, injector, prompts, deltas, args.batch_size,
                                   args.max_new_tokens)
                follow_orig = np.mean([ex["year"] in a for ex, a in zip(examples, answers)])
                follow_cf = np.mean([ex["cf_year"] in a for ex, a in zip(examples, answers)])
                key = f"{split}/{cond}"
                results[key] = {"follow_orig": follow_orig, "follow_cf": follow_cf}
                for ex, a in zip(examples, answers):
                    generations[ex["id"]][key] = a
                shown = f"acc={follow_orig:.3f}" if split == "orig" else \
                    f"follow_cf={follow_cf:.3f} follow_orig={follow_orig:.3f}"
                print(f"[{split:4s}] {cond:12s} {shown}   e.g. {answers[:3]}", flush=True)

        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump({"args": vars(args), "num_examples": n, "results": results}, f, indent=2)
        with open(os.path.join(args.output_dir, "generations.jsonl"), "w") as f:
            for g in generations.values():
                f.write(json.dumps(g) + "\n")
        print(f"Saved results to {args.output_dir}")
    except Exception:
        traceback.print_exc()
        sys.exit(1)

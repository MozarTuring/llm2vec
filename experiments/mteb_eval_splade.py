"""Evaluate a SPLADE (MLM-head) model with the same MTEB harness as
mteb_eval_layerwise.py, to check the harness against published numbers
(e.g. naver/splade-v3 = 50.7 on MTEB(eng, v2) retrieval in the SPLARE paper).
"""
import argparse
import sys
import traceback

import mteb
import numpy as np
import torch
from torch import nn
from mteb.models.model_meta import ModelMeta
from transformers import AutoModelForMaskedLM, AutoTokenizer

from mteb_eval_layerwise import MTEBWrapper

MTEB_ENG_V2_RETRIEVAL = [
    "ArguAna",
    "CQADupstackGamingRetrieval",
    "CQADupstackUnixRetrieval",
    "ClimateFEVERHardNegatives",
    "FEVERHardNegatives",
    "FiQA2018",
    "HotpotQAHardNegatives",
    "SCIDOCS",
    "TRECCOVID",
    "Touche2020Retrieval.v3",
]


class SpladePoolModule(nn.Module):
    """MLM head + SPLADE max pooling, so DataParallel gathers only pooled vectors."""

    def __init__(self, mlm_model):
        super().__init__()
        self.mlm_model = mlm_model

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        logits = self.mlm_model(
            input_ids=input_ids, attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).logits
        reps = torch.log1p(torch.relu(logits)) * attention_mask.unsqueeze(-1)
        pooled, _ = reps.max(dim=1)
        return pooled


class SpladeEncoder:
    def __init__(self, model_name_or_path, max_length=512, batch_size_per_gpu=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        mlm_model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
        self.vocab_size = mlm_model.config.vocab_size
        module = SpladePoolModule(mlm_model).to(self.device).eval()
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if num_gpus > 1:
            module = nn.DataParallel(module)
            print(f"Using DataParallel with {num_gpus} GPUs")
        self.module = module
        self.max_length = max_length
        self.batch_size = batch_size_per_gpu * num_gpus

    @torch.no_grad()
    def encode_texts(self, texts, top_k=None):
        # Sort by length to minimise padding, then restore the original order.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
        out = np.zeros((len(texts), self.vocab_size), dtype=np.float32)
        for start in range(0, len(texts), self.batch_size):
            idx = order[start:start + self.batch_size]
            inputs = self.tokenizer(
                [texts[i] for i in idx], padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            ).to(self.device)
            pooled = self.module(
                inputs["input_ids"], inputs["attention_mask"],
                inputs.get("token_type_ids"),
            )
            if top_k is not None:
                vals, top_idx = pooled.topk(top_k, dim=-1)
                pooled = torch.zeros_like(pooled).scatter_(-1, top_idx, vals)
            out[idx] = pooled.float().cpu().numpy()
        return out


class SpladeMTEBWrapper(MTEBWrapper):
    def __init__(self, encoder, model_name, query_top_k, doc_top_k, max_length):
        super().__init__(encoder, query_top_k, doc_top_k, max_length=max_length)
        self._mteb_model_meta = ModelMeta(
            name=model_name,
            revision="0.0.1",
            release_date="2026-09-25",
            languages=["eng-Latn"],
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=max_length,
            embed_dim=encoder.vocab_size,
            license=None,
            open_weights=True,
            public_training_code=None,
            public_training_data=None,
            framework=["PyTorch"],
            similarity_fn_name="dot",
            use_instructions=False,
            training_datasets=None,
            loader=None,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="naver/splade-v3")
    parser.add_argument("--task_name", type=str, nargs="*")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--query_top_k", type=int, default=None,
                        help="Omit for no pruning.")
    parser.add_argument("--doc_top_k", type=int, default=None,
                        help="Omit for no pruning.")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Per-GPU batch size.")
    args = parser.parse_args()

    try:
        encoder = SpladeEncoder(args.model_name_or_path, args.max_length, args.batch_size)
        model = SpladeMTEBWrapper(
            encoder, args.model_name_or_path, args.query_top_k, args.doc_top_k,
            args.max_length,
        )
        task_names = args.task_name or MTEB_ENG_V2_RETRIEVAL
        tasks = mteb.get_tasks(tasks=task_names)
        evaluation = mteb.MTEB(tasks=tasks)
        print(f"Starting evaluation on {task_names}", flush=True)
        results = evaluation.run(model, output_folder=args.output_dir, overwrite_results=True)
        print(f"Evaluation complete. Results: {results}", flush=True)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

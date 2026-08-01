import argparse
import glob
import json
import os
import sys
from typing import Any

import mteb
from mteb.models.model_meta import ModelMeta
import numpy as np
import torch
from torch import nn

from transformers import AutoConfig, AutoTokenizer
from peft import PeftModel
from safetensors.torch import safe_open

from llm2vec.models import LlamaBiModel, MistralBiModel, GemmaBiModel, Qwen2BiModel


def get_model_class(config):
    name = config.__class__.__name__
    if name == "LlamaConfig":
        return LlamaBiModel
    elif name == "MistralConfig":
        return MistralBiModel
    elif name == "GemmaConfig":
        return GemmaBiModel
    elif name == "Qwen2Config":
        return Qwen2BiModel
    else:
        raise ValueError(f"Model class {name} not supported.")


class SqrtDNorm(nn.Module):
    def forward(self, hidden_states):
        dim = hidden_states.shape[-1]
        return hidden_states * (dim ** 0.5) / hidden_states.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)


class LayerwiseEncoder:
    def __init__(self, model_name_or_path, peft_model_name_or_path, sae_weights_path,
                 lora_layers, trained_checkpoint_path=None, torch_dtype=torch.bfloat16,
                 attn_implementation="sdpa"):
        config = AutoConfig.from_pretrained(model_name_or_path)
        model_class = get_model_class(config)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        backbone = model_class.from_pretrained(
            model_name_or_path,
            config=config,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )

        if peft_model_name_or_path is not None:
            backbone = PeftModel.from_pretrained(backbone, peft_model_name_or_path)
            backbone = backbone.merge_and_unload()

        num_active = lora_layers + 1
        backbone.layers = backbone.layers[:num_active]

        if trained_checkpoint_path is not None:
            backbone = PeftModel.from_pretrained(backbone, trained_checkpoint_path)

        self.backbone = backbone
        self.pre_sae_norm = SqrtDNorm()

        with safe_open(sae_weights_path, framework="pt") as f:
            encoder_weight = f.get_tensor("encoder.weight")
            encoder_bias = f.get_tensor("encoder.bias")
        sae = nn.Linear(encoder_weight.shape[1], encoder_weight.shape[0])
        with torch.no_grad():
            sae.weight.copy_(encoder_weight)
            sae.bias.copy_(encoder_bias)
        sae.to(torch_dtype)
        sae.requires_grad_(False)
        self.sae = sae

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone.to(self.device)
        self.sae.to(self.device)
        self.backbone.eval()

    @torch.no_grad()
    def encode_texts(self, texts, batch_size=64, save_dir=None):
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            meta_path = os.path.join(save_dir, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                if meta["total"] == len(texts):
                    chunks = sorted(glob.glob(os.path.join(save_dir, "chunk_*.npy")))
                    if chunks:
                        return np.concatenate([np.load(p) for p in chunks], axis=0)

        chunk_idx = 0
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            try:
                inputs = self.tokenizer(
                    batch, padding=True, truncation=True, max_length=1024, return_tensors="pt"
                ).to(self.device)

                outputs = self.backbone(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                hidden_states = outputs[0]
                hidden_states = self.pre_sae_norm(hidden_states)
                sae_out = self.sae(hidden_states)
                sae_out = torch.log(1 + torch.relu(sae_out))
                pooled, _ = sae_out.max(dim=1)
            except torch.cuda.OutOfMemoryError:
                print(f"CUDA OOM at batch {start}-{start+batch_size} / {len(texts)}, batch_size={batch_size}", file=sys.stderr)
                sys.exit(1)

            if save_dir is not None:
                np.save(os.path.join(save_dir, f"chunk_{chunk_idx}.npy"),
                        pooled.cpu().float().numpy())
            chunk_idx += 1

            del inputs, outputs, hidden_states, sae_out, pooled

        if save_dir is not None:
            with open(os.path.join(save_dir, "meta.json"), "w") as f:
                json.dump({"total": len(texts)}, f)
            chunks = sorted(glob.glob(os.path.join(save_dir, "chunk_*.npy")))
            return np.concatenate([np.load(p) for p in chunks], axis=0)

        return np.concatenate([], axis=0)


class MTEBWrapper:
    def __init__(self, encoder, cache_dir):
        self.encoder = encoder
        self.cache_dir = cache_dir
        self._mteb_model_meta = ModelMeta(
            name="custom/layerwise-sparse-encoder",
            revision="0.0.1",
            release_date="2026-07-29",
            languages=["eng-Latn"],
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=1024,
            embed_dim=32768,
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

    @property
    def mteb_model_meta(self):
        return self._mteb_model_meta

    def encode(self, inputs, *, task_metadata=None, hf_split=None, hf_subset=None,
               prompt_type=None, **kwargs):
        task_name = "unknown"
        if task_metadata is not None:
            if hasattr(task_metadata, "metadata"):
                task_name = task_metadata.metadata.name
            elif hasattr(task_metadata, "name"):
                task_name = task_metadata.name

        enc_type = "unknown"
        if prompt_type is not None:
            enc_type = prompt_type.value if hasattr(prompt_type, "value") else str(prompt_type)

        save_dir = os.path.join(self.cache_dir, task_name, enc_type)

        all_sentences = []
        for batch in inputs:
            sentences = batch["text"] if isinstance(batch, dict) else batch
            if isinstance(sentences, str):
                all_sentences.append(sentences)
            else:
                all_sentences.extend(sentences)

        return self.encoder.encode_texts(all_sentences, save_dir=save_dir)

    def similarity(self, embeddings1, embeddings2):
        if isinstance(embeddings1, np.ndarray):
            embeddings1 = torch.from_numpy(embeddings1)
        if isinstance(embeddings2, np.ndarray):
            embeddings2 = torch.from_numpy(embeddings2)
        return torch.mm(embeddings1, embeddings2.T)

    def similarity_pairwise(self, embeddings1, embeddings2):
        if isinstance(embeddings1, np.ndarray):
            embeddings1 = torch.from_numpy(embeddings1)
        if isinstance(embeddings2, np.ndarray):
            embeddings2 = torch.from_numpy(embeddings2)
        return (embeddings1 * embeddings2).sum(dim=-1)


MTEB_ENG_V2_RETRIEVAL = [
    "ArguAna",
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
    "ClimateFEVER",
    "DBPedia",
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "MSMARCO",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--peft_model_name_or_path", type=str, default=None)
    parser.add_argument("--sae_weights_path", type=str, required=True)
    parser.add_argument("--trained_checkpoint_path", type=str, default=None)
    parser.add_argument("--lora_layers", type=int, required=True)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--task_type", type=str, default="retrieval",
                        choices=["retrieval", "all"])
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--cache_dir", type=str, default="embedding_cache")
    args = parser.parse_args()

    encoder = LayerwiseEncoder(
        model_name_or_path=args.model_name_or_path,
        peft_model_name_or_path=args.peft_model_name_or_path,
        sae_weights_path=args.sae_weights_path,
        lora_layers=args.lora_layers,
        trained_checkpoint_path=args.trained_checkpoint_path,
    )

    model = MTEBWrapper(encoder, cache_dir=args.cache_dir)

    if args.task_name:
        task_names = [args.task_name]
    else:
        task_names = MTEB_ENG_V2_RETRIEVAL

    tasks = mteb.get_tasks(tasks=task_names)
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, output_folder=args.output_dir)

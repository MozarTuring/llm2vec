#!/usr/bin/env python
# coding=utf-8
"""
Layer-wise LoRA finetuning with sparse autoencoder.

Architecture:
    Input → Embedding → Layers[0..N] → RMSNorm → SparseAutoencoder → TaskHead → Loss

    - N = lora_layers (0-indexed, inclusive). Layers after N are discarded.
    - Prior LoRA (e.g. from MNTP) is loaded and merged before truncation.
    - A new LoRA adapter is trained on the truncated layers.
    - SparseAutoencoder and TaskHead are abstract — implement their forward().

Data:
    MS MARCO queries + positive passages + 8 hard negatives per query.
    Hard negatives are pre-mined using experiments/hard_negatives.py:
        python experiments/hard_negatives.py
    which produces msmarco_hard_negatives.json.

Each training step encodes 10 text groups through backbone+SAE:
    [query, positive, neg1, neg2, ..., neg8]
"""

import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)
from transformers.trainer_utils import seed_worker

from peft import LoraConfig, get_peft_model, PeftModel
from safetensors.torch import safe_open

from llm2vec.models import (
    MistralBiModel,
    LlamaBiModel,
    GemmaBiModel,
    Qwen2BiModel,
)

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════
#  Abstract components — implement these for your task
# ═════════════════════════════════════════════════════════════════


class SparseAutoencoder(nn.Module):
    """Sparse autoencoder applied to the truncated backbone output.

    forward(hidden_states):
        hidden_states — (batch_size, seq_len, hidden_size)
        returns       — transformed representation (shape depends on impl)
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implement SparseAutoencoder.forward()")


class TaskHead(nn.Module):
    """Downstream task head.

    forward(encoded_groups):
        encoded_groups — list of 10 tensors (SAE outputs), one per text group:
            [0]  query representations     (batch_size, ...)
            [1]  positive representations  (batch_size, ...)
            [2:] hard negative reps        (8 groups, each batch_size, ...)
        returns — scalar loss (torch.Tensor)
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(
        self, pred_probs: torch.Tensor, target_probs: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError("Implement TaskHead.forward()")


# ═════════════════════════════════════════════════════════════════
#  Normalization
# ═════════════════════════════════════════════════════════════════


class SqrtDNorm(nn.Module):
    """Normalize each vector so that its L2 norm equals sqrt(dim).

    For a vector v of dimension d, the output is:
        v_out = sqrt(d) * v / ||v||_2
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dim = hidden_states.shape[-1]
        return (
            hidden_states
            * (dim**0.5)
            / hidden_states.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        )


# ═════════════════════════════════════════════════════════════════
#  Model
# ═════════════════════════════════════════════════════════════════


class LayerwiseModel(nn.Module):
    """Truncated backbone + normalization + SAE + task head.

    encode()  — single text group through backbone → norm → SAE
    forward() — all 10 text groups → norm → SAE → task head → loss
    """

    def __init__(
        self, config, backbone, sae, task_head, temperature, lambda_q, lambda_d,
        jump_relu_threshold=0.9609375, sae_top_k=50, sae_norm_scale=1.0,
    ):
        super().__init__()
        self.config = config
        self.backbone = backbone
        self.sae = sae
        self.task_head = task_head
        self.temperature = temperature
        self.lambda_q = lambda_q
        self.lambda_d = lambda_d
        self.jump_relu_threshold = jump_relu_threshold
        self.sae_top_k = sae_top_k
        self.sae_norm_scale = sae_norm_scale

    def encode(self, sentence_feature: Dict[str, torch.Tensor]):
        outputs = self.backbone(
            input_ids=sentence_feature["input_ids"],
            attention_mask=sentence_feature["attention_mask"],
        )
        hidden_states = outputs[0]
        # Dataset-wise normalization for Llama Scope SAE:
        # scale = sqrt(d_model) / activation_norm (constant, not per-sample)
        hidden_states = hidden_states * self.sae_norm_scale
        sae_out = self.sae(hidden_states)
        # JumpReLU activation (threshold from SAE hyperparams.json)
        sae_out = torch.where(sae_out > self.jump_relu_threshold, sae_out, torch.zeros_like(sae_out))
        # SAE TopK: keep only top-k features per token
        topk_vals, topk_idx = sae_out.topk(self.sae_top_k, dim=-1)
        sae_out = torch.zeros_like(sae_out).scatter_(-1, topk_idx, topk_vals)
        sae_out = torch.log(1 + sae_out)
        pooled, _ = sae_out.max(dim=1)
        return pooled, sae_out

    @staticmethod
    def flops_loss(sae_out):
        return torch.sum(sae_out.mean(dim=0) ** 2)

    def forward(
        self, features: List[Dict[str, torch.Tensor]], reranker_scores: torch.Tensor
    ):
        results = [self.encode(f) for f in features]
        pooled = [r[0] for r in results]
        sae_outs = [r[1] for r in results]

        query = pooled[0]
        positive = pooled[1]
        negatives = pooled[2:]

        pos_score = (query * positive).sum(dim=-1, keepdim=True)
        neg_scores = torch.stack(
            [(query * neg).sum(dim=-1) for neg in negatives], dim=1
        )
        scores = torch.cat([pos_score, neg_scores], dim=1)

        log_pred = torch.log_softmax(scores / self.temperature, dim=1)
        target_probs = torch.softmax(reranker_scores.to(scores.device), dim=1)
        kl_loss = nn.functional.kl_div(log_pred, target_probs, reduction="batchmean")

        query_flops = self.flops_loss(pooled[0])
        doc_flops = sum(self.flops_loss(p) for p in pooled[1:]) / len(pooled[1:])

        loss = kl_loss + self.lambda_q * query_flops + self.lambda_d * doc_flops
        return (loss,)

    def save_peft_model(self, path):
        self.backbone.save_pretrained(path)
        torch.save(self.sae.state_dict(), os.path.join(path, "sae.pt"))
        torch.save(self.task_head.state_dict(), os.path.join(path, "task_head.pt"))

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )


# ═════════════════════════════════════════════════════════════════
#  Data
# ═════════════════════════════════════════════════════════════════


class TrainSample:
    """One training example with multiple texts."""

    def __init__(
        self, texts: List[str], reranker_scores: List[float] = None, label: float = 1.0
    ):
        self.texts = texts
        self.reranker_scores = reranker_scores
        self.label = label


class MSMARCOHardNegDataset(torch.utils.data.Dataset):
    """MS MARCO dataset with pre-mined hard negatives.

    Each sample has: [query, positive, neg1, neg2, ..., neg_k]

    Expects the JSON output from experiments/hard_negatives.py:
        {
            "query_id": {
                "query": "...",
                "positives": [{"pid": "...", "text": "..."}],
                "hard_negatives": [{"pid": "...", "score": ..., "text": "..."}, ...]
            }, ...
        }
    """

    def __init__(self, file_path: str, num_hard_negatives: int = 8):
        with open(file_path) as f:
            raw = json.load(f)

        self.samples = []
        skipped = 0
        for qid, item in raw.items():
            if not item["positives"]:
                skipped += 1
                continue
            query = item["query"]
            pos = item["positives"][0]
            positive = pos["text"]
            neg_items = item["hard_negatives"][:num_hard_negatives]
            negs = [n["text"] for n in neg_items]
            if len(negs) < num_hard_negatives:
                skipped += 1
                continue
            reranker_scores = [pos["reranker_score"]] + [
                n["reranker_score"] for n in neg_items
            ]
            self.samples.append((query, positive, negs, reranker_scores))

        print(
            f"Loaded {len(self.samples)} training samples from {file_path} "
            f"({skipped} skipped, {num_hard_negatives} hard negatives each)"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        query, positive, negs, reranker_scores = self.samples[idx]
        texts = [query, positive] + negs
        return TrainSample(texts=texts, reranker_scores=reranker_scores)


class ContrastiveCollator:
    """Tokenizes each text group separately.

    Input:  list of TrainSample (each has 10 texts)
    Output: list of 10 dicts, each dict = {input_ids, attention_mask}
            tokenized for one text group across the batch
    """

    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: List[TrainSample]):
        num_texts = len(features[0].texts)
        text_groups: List[List[str]] = [[] for _ in range(num_texts)]

        for sample in features:
            for i, text in enumerate(sample.texts):
                text_groups[i].append(text)

        tokenized_groups = []
        for group in text_groups:
            tokenized = self.tokenizer(
                group,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokenized_groups.append(tokenized)

        reranker_scores = torch.tensor([s.reranker_scores for s in features])
        return {"features": tokenized_groups, "reranker_scores": reranker_scores}


# ═════════════════════════════════════════════════════════════════
#  LoRA setup
# ═════════════════════════════════════════════════════════════════


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


def initialize_peft(
    model,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_modules: Optional[List[str]] = None,
):
    if lora_modules is None and model.config.__class__.__name__ in [
        "LlamaConfig",
        "MistralConfig",
        "GemmaConfig",
        "Qwen2Config",
    ]:
        lora_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    elif lora_modules is None:
        raise ValueError("lora_modules must be specified for this model.")

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=None,
    )

    model = get_peft_model(model, config)
    print("Model's LoRA trainable parameters:")
    model.print_trainable_parameters()
    return model


# ═════════════════════════════════════════════════════════════════
#  Trainer
# ═════════════════════════════════════════════════════════════════


class StopTrainingCallback(TrainerCallback):
    def __init__(self, stop_after_n_steps: int):
        self.stop_after_n_steps = stop_after_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_after_n_steps:
            control.should_training_stop = True


class LayerwiseTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        features = inputs["features"]
        reranker_scores = inputs["reranker_scores"]
        output = model(features, reranker_scores)
        loss = output[0] if isinstance(output, (tuple, list)) else output
        return (loss, output) if return_outputs else loss

    def _remove_unused_columns(self, dataset, description=None):
        return dataset

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.save_peft_model(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


# ═════════════════════════════════════════════════════════════════
#  Arguments
# ═════════════════════════════════════════════════════════════════


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Base model checkpoint."},
    )
    peft_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Prior PEFT/LoRA checkpoint to load and merge (e.g. MNTP output)."
        },
    )
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    token: str = field(default=None)
    trust_remote_code: bool = field(default=False)
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={"choices": ["auto", "bfloat16", "float16", "float32"]},
    )
    attn_implementation: Optional[str] = field(
        default="sdpa",
        metadata={"choices": ["eager", "sdpa", "flash_attention_2"]},
    )
    low_cpu_mem_usage: bool = field(default=False)
    sae_weights_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the safetensors file containing SAE encoder weights."
        },
    )


@dataclass
class DataArguments:
    hard_negatives_file: str = field(
        default="msmarco_hard_negatives.json",
        metadata={
            "help": "Path to the hard negatives JSON produced by experiments/hard_negatives.py."
        },
    )
    num_hard_negatives: int = field(
        default=8,
        metadata={"help": "Number of hard negatives per query to use."},
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length for tokenization."},
    )


@dataclass
class CustomArguments:
    temperature: float = field(
        metadata={"help": "Temperature for softmax on predicted scores."}
    )
    lambda_q: float = field(
        metadata={"help": "FLOPS regularization weight for queries."}
    )
    lambda_d: float = field(
        metadata={"help": "FLOPS regularization weight for documents."}
    )
    lora_r: int = field(default=16, metadata={"help": "LoRA rank."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})
    lora_layers: int = field(
        default=31,
        metadata={
            "help": (
                "Last layer index to keep (0-based, inclusive). "
                "E.g. 10 → layers 0-10; 17 → layers 0-17. "
                "Layers after this index are discarded."
            )
        },
    )
    stop_after_n_steps: int = field(
        default=10000, metadata={"help": "Stop training after n steps."}
    )


# ═════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, CustomArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        json_file = os.path.abspath(sys.argv[1])
        with open(json_file) as f:
            config_dict = json.load(f)

        # Infer sae_weights_path and output_dir from lora_layers if not specified
        lora_layers = config_dict.get("lora_layers")
        config_dict["sae_weights_path"] = (
            f"../remote_data/llm2vec/"
            f"Llama3_1-8B-Base-L{lora_layers}R-8x/checkpoints/final.safetensors"
        )
        print(
            f"Inferred sae_weights_path from lora_layers={lora_layers}: "
            f"{config_dict['sae_weights_path']}"
        )
        config_dict["output_dir"] = (
            f"output/layerwise/Meta-Llama-3.1-8B-msmarco-mntp-L{lora_layers}"
        )
        print(
            f"Inferred output_dir from lora_layers={lora_layers}: "
            f"{config_dict['output_dir']}"
        )

        model_args, data_args, training_args, custom_args = parser.parse_dict(
            config_dict
        )
    else:
        (
            model_args,
            data_args,
            training_args,
            custom_args,
        ) = parser.parse_args_into_dataclasses()

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    # ── Logging ───────────────────────────────────────────────
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    set_seed(training_args.seed)

    # ── Load tokenizer ────────────────────────────────────────
    tok_name = model_args.tokenizer_name or model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tok_name,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ── Load base bidirectional model ─────────────────────────
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model_class = get_model_class(config)
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        attn_implementation=model_args.attn_implementation,
    )

    # ── Load & merge prior LoRA (e.g. MNTP) ───────────────────
    if model_args.peft_model_name_or_path is not None:
        logger.info(
            f"Loading prior PEFT adapter from {model_args.peft_model_name_or_path}"
        )
        model = PeftModel.from_pretrained(model, model_args.peft_model_name_or_path)
        model = model.merge_and_unload()
        logger.info("Prior PEFT adapter merged into base model")

    # ── Truncate to layers 0..lora_layers ─────────────────────
    num_active = custom_args.lora_layers + 1
    total_layers = len(model.layers)
    assert num_active <= total_layers, (
        f"lora_layers={custom_args.lora_layers} but model only has "
        f"{total_layers} layers (indices 0-{total_layers - 1})"
    )
    model.layers = model.layers[:num_active]
    # The SAE was trained on the residual stream BEFORE the model's final
    # RMSNorm (hook_point = "blocks.N.hook_resid_post").  HuggingFace's
    # LlamaModel.forward() applies self.norm() to the last hidden state
    # before returning it, which distorts the distribution the SAE expects.
    # Replace the final norm with Identity so outputs[0] is the raw
    # residual stream that the SAE encoder was trained on.
    model.norm = nn.Identity()
    import gc; gc.collect(); torch.cuda.empty_cache()
    print(
        f"Truncated model: kept layers 0-{custom_args.lora_layers} "
        f"({num_active}/{total_layers}), discarded layers "
        f"{custom_args.lora_layers + 1}-{total_layers - 1}"
    )

    # ── Apply LoRA to all kept layers ─────────────────────────
    model = initialize_peft(
        model,
        lora_r=custom_args.lora_r,
        lora_alpha=custom_args.lora_r,
        lora_dropout=custom_args.lora_dropout,
    )

    # ── Build full model: backbone → SAE → task ───────────────
    hidden_size = config.hidden_size
    with safe_open(model_args.sae_weights_path, framework="pt") as f:
        encoder_weight = f.get_tensor("encoder.weight")  # Shape: [32768, 4096]
        encoder_bias = f.get_tensor("encoder.bias")  # Shape: [32768]
    sae = nn.Linear(4096, 32768)
    with torch.no_grad():
        sae.weight.copy_(encoder_weight)
        sae.bias.copy_(encoder_bias)
    sae.to(torch_dtype)
    sae.requires_grad_(False)

    # ── Load SAE hyperparams (JumpReLU threshold, TopK) ──────
    sae_dir = os.path.dirname(os.path.dirname(model_args.sae_weights_path))
    sae_hyperparams_path = os.path.join(sae_dir, "hyperparams.json")
    with open(sae_hyperparams_path) as f:
        sae_hyperparams = json.load(f)
    jump_relu_threshold = sae_hyperparams["jump_relu_threshold"]
    sae_top_k = sae_hyperparams["top_k"]
    activation_norm = sae_hyperparams["dataset_average_activation_norm"]["in"]
    sae_norm_scale = (hidden_size ** 0.5) / activation_norm
    print(f"SAE hyperparams from {sae_hyperparams_path}:")
    print(f"  jump_relu_threshold: {jump_relu_threshold}")
    print(f"  top_k: {sae_top_k}")
    print(f"  activation_norm: {activation_norm}")
    print(f"  norm_activation: {sae_hyperparams.get('norm_activation', 'unknown')}")
    print(f"  sae_norm_scale: {sae_norm_scale:.4f}")

    task_head = TaskHead(hidden_size)

    layerwise_model = LayerwiseModel(
        config=config,
        backbone=model,
        sae=sae,
        task_head=task_head,
        temperature=custom_args.temperature,
        lambda_q=custom_args.lambda_q,
        lambda_d=custom_args.lambda_d,
        jump_relu_threshold=jump_relu_threshold,
        sae_top_k=sae_top_k,
        sae_norm_scale=sae_norm_scale,
    )

    print(f"\nLayerwiseModel ready:")
    print(f"  Backbone: {num_active} layers (0-{custom_args.lora_layers})")
    print(f"  Hidden size: {hidden_size}")
    print(f"  LoRA rank: {custom_args.lora_r}")

    # ── Load dataset ──────────────────────────────────────────
    train_dataset = MSMARCOHardNegDataset(
        file_path=data_args.hard_negatives_file,
        num_hard_negatives=data_args.num_hard_negatives,
    )

    data_collator = ContrastiveCollator(
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length,
    )

    # ── Train ─────────────────────────────────────────────────
    trainer = LayerwiseTrainer(
        model=layerwise_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.add_callback(StopTrainingCallback(custom_args.stop_after_n_steps))

    trainer.train()


if __name__ == "__main__":
    main()

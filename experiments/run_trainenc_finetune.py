"""
    torchrun --nproc_per_node=4 experiments/run_trainenc_finetune.py --config train_configs/trainenc/MetaLlama3.1-mntp-trainenc.json --hard_negatives_file <reranker_parts>/
"""

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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

from peft import LoraConfig, get_peft_model, PeftModel
from safetensors.torch import safe_open

from llm2vec.models import (
    MistralBiModel,
    LlamaBiModel,
    GemmaBiModel,
    Qwen2BiModel,
)

logger = logging.getLogger(__name__)


class TrainableEncoder(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight.float())
        self.bias = nn.Parameter(bias.float())
        self.out_features = weight.shape[0]
        self._w0 = None
        self._b0 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype))

    @torch.no_grad()
    def drift(self):
        if self._w0 is None:
            self._w0 = self.weight.detach().clone()
            self._b0 = self.bias.detach().clone()
        w = (self.weight - self._w0).norm() / self._w0.norm()
        b = (self.bias - self._b0).norm() / self._b0.norm()
        return w.item(), b.item()


class LayerwiseModel(nn.Module):
    def __init__(
        self, config, backbone, sae, temperature, lambda_q, lambda_d, bos_token_id,
        sae_norm_scale, flops_chunk_size=0,
    ):
        super().__init__()
        self.config = config
        self.bos_token_id = bos_token_id
        self.flops_chunk_size = flops_chunk_size
        self.backbone = backbone
        self.sae = sae
        self.temperature = temperature
        self.lambda_q = lambda_q
        self.lambda_d = lambda_d
        self.sae_norm_scale = sae_norm_scale
        self.global_step = 0

    def encode(self, sentence_feature: Dict[str, torch.Tensor]):
        outputs = self.backbone(
            input_ids=sentence_feature["input_ids"],
            attention_mask=sentence_feature["attention_mask"],
        )
        sae_pre = self.sae(outputs[0] * self.sae_norm_scale)
        sae_out = torch.log(1 + torch.relu(sae_pre))
        pool_mask = sentence_feature["attention_mask"] * (
            sentence_feature["input_ids"] != self.bos_token_id
        )
        sae_out = sae_out * pool_mask.unsqueeze(-1)
        pooled, _ = sae_out.max(dim=1)
        return pooled

    @staticmethod
    def flops_loss(sae_out):
        return torch.sum(sae_out.mean(dim=0) ** 2)

    def chunked_flops(self, query, docs):
        n = query.shape[0]
        chunk = self.flops_chunk_size if 0 < self.flops_chunk_size < n else n
        q_flops, d_flops = [], []
        for start in range(0, n, chunk):
            end = start + chunk
            q_flops.append(self.flops_loss(query[start:end]))
            d_flops.append(self.flops_loss(torch.cat([d[start:end] for d in docs], dim=0)))
        return torch.stack(q_flops).mean(), torch.stack(d_flops).mean()

    def forward(
        self, features: List[Dict[str, torch.Tensor]], reranker_scores: torch.Tensor
    ):
        pooled = [self.encode(f) for f in features]
        query, positive, negatives = pooled[0], pooled[1], pooled[2:]

        pos_score = (query * positive).sum(dim=-1, keepdim=True)
        neg_scores = torch.stack(
            [(query * neg).sum(dim=-1) for neg in negatives], dim=1
        )
        scores = torch.cat([pos_score, neg_scores], dim=1)

        log_pred = torch.log_softmax(scores / self.temperature, dim=1)
        target_probs = torch.softmax(reranker_scores.to(scores.device), dim=1)
        kl_loss = nn.functional.kl_div(log_pred, target_probs, reduction="batchmean")

        query_flops, doc_flops = self.chunked_flops(pooled[0], pooled[1:])

        if self.global_step < 20 or self.global_step % 200 == 0:
            w_drift, b_drift = self.sae.drift()
            print(f"[loss step={self.global_step}] kl={kl_loss.item():.4f} "
                  f"flops_q={query_flops.item():.1f} flops_d={doc_flops.item():.1f} "
                  f"lq={self.lambda_q:.6f} ld={self.lambda_d:.6f} "
                  f"W_enc_drift={w_drift:.6f} b_enc_drift={b_drift:.6f}")

        loss = kl_loss + self.lambda_q * query_flops + self.lambda_d * doc_flops
        self.global_step += 1
        return (loss,)

    def save_peft_model(self, path):
        self.backbone.save_pretrained(path)
        torch.save({"weight": self.sae.weight.detach().cpu(), "bias": self.sae.bias.detach().cpu()},
                   os.path.join(path, "sae.pt"))

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )


class TrainSample:
    def __init__(self, texts: List[str], reranker_scores: List[float] = None):
        self.texts = texts
        self.reranker_scores = reranker_scores


class MSMARCOHardNegDataset(torch.utils.data.Dataset):
    def __init__(self, file_path: str, num_hard_negatives: int = 8):
        self.samples = []
        skipped = 0

        if os.path.isdir(file_path):
            jsonl_files = sorted(
                f for f in os.listdir(file_path) if f.endswith(".jsonl")
            )
            print(f"Loading {len(jsonl_files)} JSONL files from {file_path}")
            for fname in jsonl_files:
                with open(os.path.join(file_path, fname)) as f:
                    for line in f:
                        skipped += self._add_sample(json.loads(line), num_hard_negatives)
        else:
            with open(file_path) as f:
                raw = json.load(f)
            for item in raw.values():
                skipped += self._add_sample(item, num_hard_negatives)

        print(
            f"Loaded {len(self.samples)} training samples from {file_path} "
            f"({skipped} skipped, {num_hard_negatives} hard negatives each)"
        )

    def _add_sample(self, item, num_hard_negatives):
        if not item["positives"]:
            return 1
        pos = item["positives"][0]
        if len(item["hard_negatives"]) < num_hard_negatives:
            return 1
        neg_items = random.sample(item["hard_negatives"], num_hard_negatives)
        negs = [n["text"] for n in neg_items]
        reranker_scores = [pos["reranker_score"]] + [
            n["reranker_score"] for n in neg_items
        ]
        self.samples.append((item["query"], pos["text"], negs, reranker_scores))
        return 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        query, positive, negs, reranker_scores = self.samples[idx]
        return TrainSample(texts=[query, positive] + negs, reranker_scores=reranker_scores)


class ContrastiveCollator:
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: List[TrainSample]):
        num_texts = len(features[0].texts)
        text_groups: List[List[str]] = [[] for _ in range(num_texts)]
        for sample in features:
            for i, text in enumerate(sample.texts):
                text_groups[i].append(text)

        tokenized_groups = [
            self.tokenizer(
                group,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            for group in text_groups
        ]
        reranker_scores = torch.tensor([s.reranker_scores for s in features])
        return {"features": tokenized_groups, "reranker_scores": reranker_scores}


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


def initialize_peft(model, lora_r, lora_alpha, lora_dropout, layers_to_transform):
    lora_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    layer_alt = "|".join(str(i) for i in layers_to_transform)
    module_alt = "|".join(lora_modules)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=rf"(.*\.)?layers\.({layer_alt})\.(self_attn|mlp)\.({module_alt})",
        lora_dropout=lora_dropout,
        bias="none",
        task_type=None,
    )
    model = get_peft_model(model, config)
    print("Model's LoRA trainable parameters:")
    model.print_trainable_parameters()
    return model


class StopTrainingCallback(TrainerCallback):
    def __init__(self, stop_after_n_steps: int):
        self.stop_after_n_steps = stop_after_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_after_n_steps:
            control.should_training_stop = True


class LayerwiseTrainer(Trainer):
    def __init__(self, *args, sae_learning_rate, **kwargs):
        super().__init__(*args, **kwargs)
        self.sae_learning_rate = sae_learning_rate

    def create_optimizer(self):
        if self.optimizer is None:
            named = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
            sae_params = [p for n, p in named if n.startswith("sae.")]
            other_params = [p for n, p in named if not n.startswith("sae.")]
            print(f"Optimizer: {sum(p.numel() for p in other_params)} LoRA parameters at lr "
                  f"{self.args.learning_rate}, {sum(p.numel() for p in sae_params)} SAE encoder "
                  f"parameters at lr {self.sae_learning_rate}")
            self.optimizer = torch.optim.AdamW(
                [{"params": other_params},
                 {"params": sae_params, "lr": self.sae_learning_rate}],
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,
            )
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        output = model(inputs["features"], inputs["reranker_scores"])
        loss = output[0]
        return (loss, output) if return_outputs else loss

    def _remove_unused_columns(self, dataset, description=None):
        return dataset

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        self.accelerator.unwrap_model(self.model).save_peft_model(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    peft_model_name_or_path: Optional[str] = field(default=None)
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
    sae_weights_path: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    hard_negatives_file: str = field(default=None)
    num_hard_negatives: int = field(default=8)
    max_seq_length: int = field(default=512)


@dataclass
class CustomArguments:
    lora_r: int = field(default=16)
    lora_alpha: Optional[int] = field(default=None)
    lora_dropout: float = field(default=0.05)
    lora_layers: int = field(default=31)
    lora_first_layer: int = field(default=0)
    stop_after_n_steps: int = field(default=10000)


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, required=True)
    pre_args, remaining_argv = pre_parser.parse_known_args()

    with open(os.path.abspath(pre_args.config)) as f:
        config_dict = json.load(f)

    lora_layers = config_dict.get("lora_layers")
    data_dir = os.environ["PKQ_DATA_DIR"]
    val = config_dict.get("peft_model_name_or_path")
    if val and not os.path.isabs(val):
        resolved = os.path.join(data_dir, val)
        if os.path.exists(resolved):
            print(f"Resolved peft_model_name_or_path: {val} -> {resolved}")
            config_dict["peft_model_name_or_path"] = resolved
    from huggingface_hub import hf_hub_download
    sae_expansion = config_dict.get("sae_expansion", 8)
    config_dict["sae_weights_path"] = hf_hub_download(
        f"OpenMOSS-Team/Llama3_1-8B-Base-LXR-{sae_expansion}x",
        f"Llama3_1-8B-Base-L{lora_layers}R-{sae_expansion}x/checkpoints/final.safetensors",
        local_files_only=True,
    )
    print(f"sae_weights_path: {config_dict['sae_weights_path']}")

    temperature = config_dict.pop("temperature")
    lambda_q = config_dict.pop("lambda_q")
    lambda_d = config_dict.pop("lambda_d")
    sae_learning_rate = config_dict.pop("sae_learning_rate")
    flops_chunk_size = config_dict.pop("flops_chunk_size", 0)
    config_dict.pop("flops_warmup_steps", None)
    config_dict.pop("sae_expansion", None)

    lora_first_layer = config_dict.get("lora_first_layer", 0)
    output_dir = (f"output/trainenc/Meta-Llama-3.1-8B-msmarco-mntp-L{lora_layers}"
                  f"-from{lora_first_layer}-saelr{sae_learning_rate:g}")
    config_dict["output_dir"] = output_dir
    print(f"output_dir: {output_dir}")

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, CustomArguments)
    )
    parser.set_defaults(**config_dict)
    for action in parser._actions:
        if action.dest in config_dict and action.required:
            action.required = False
    model_args, data_args, training_args, custom_args = (
        parser.parse_args_into_dataclasses(args=remaining_argv)
    )
    training_args.output_dir = output_dir

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

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

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model = get_model_class(config).from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        attn_implementation=model_args.attn_implementation,
    )

    if model_args.peft_model_name_or_path is not None:
        logger.info(f"Loading prior PEFT adapter from {model_args.peft_model_name_or_path}")
        model = PeftModel.from_pretrained(model, model_args.peft_model_name_or_path)
        model = model.merge_and_unload()

    num_active = custom_args.lora_layers + 1
    total_layers = len(model.layers)
    assert num_active <= total_layers
    model.layers = model.layers[:num_active]
    model.norm = nn.Identity()
    import gc; gc.collect(); torch.cuda.empty_cache()
    print(f"Truncated model: kept layers 0-{custom_args.lora_layers} ({num_active}/{total_layers})")

    lora_layer_ids = list(range(custom_args.lora_first_layer, custom_args.lora_layers + 1))
    assert lora_layer_ids
    model = initialize_peft(
        model,
        lora_r=custom_args.lora_r,
        lora_alpha=custom_args.lora_alpha or custom_args.lora_r,
        lora_dropout=custom_args.lora_dropout,
        layers_to_transform=lora_layer_ids,
    )
    lora_wrapped = sorted({int(n.split("layers.")[1].split(".")[0])
                           for n, p in model.named_parameters() if "lora_" in n})
    print(f"LoRA on layers {lora_wrapped}")
    assert lora_wrapped == lora_layer_ids

    with safe_open(model_args.sae_weights_path, framework="pt") as f:
        sae = TrainableEncoder(f.get_tensor("encoder.weight"), f.get_tensor("encoder.bias"))

    sae_dir = os.path.dirname(os.path.dirname(model_args.sae_weights_path))
    with open(os.path.join(sae_dir, "hyperparams.json")) as f:
        sae_hyperparams = json.load(f)
    activation_norm = sae_hyperparams["dataset_average_activation_norm"]["in"]
    sae_norm_scale = (config.hidden_size ** 0.5) / activation_norm
    print(f"SAE: width {sae.out_features}, activation_norm {activation_norm}, "
          f"sae_norm_scale {sae_norm_scale}, sae_learning_rate {sae_learning_rate}")

    layerwise_model = LayerwiseModel(
        config=config,
        backbone=model,
        sae=sae,
        temperature=temperature,
        lambda_q=lambda_q,
        lambda_d=lambda_d,
        bos_token_id=tokenizer.bos_token_id,
        sae_norm_scale=sae_norm_scale,
        flops_chunk_size=flops_chunk_size,
    )

    train_dataset = MSMARCOHardNegDataset(
        file_path=data_args.hard_negatives_file,
        num_hard_negatives=data_args.num_hard_negatives,
    )
    data_collator = ContrastiveCollator(tokenizer=tokenizer, max_length=data_args.max_seq_length)

    trainer = LayerwiseTrainer(
        model=layerwise_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        sae_learning_rate=sae_learning_rate,
    )
    trainer.add_callback(StopTrainingCallback(custom_args.stop_after_n_steps))
    trainer.train()


if __name__ == "__main__":
    main()

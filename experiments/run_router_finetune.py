"""
    torchrun --nproc_per_node=4 experiments/run_router_finetune.py --config train_configs/router/MetaLlama3.1-mntp-router-L16_26.json --hard_negatives_file <reranker_parts>/
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List

import torch
from torch import nn

import transformers
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, TrainerCallback, TrainingArguments, set_seed

from peft import PeftModel
from safetensors.torch import safe_open

from run_trainenc_finetune import (
    ContrastiveCollator,
    DataArguments,
    LayerwiseTrainer as BaseTrainer,
    MSMARCOHardNegDataset,
    ModelArguments,
    StopTrainingCallback,
    get_model_class,
    initialize_peft,
)

logger = logging.getLogger(__name__)


def router_input(hidden, mask):
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return nn.functional.layer_norm(pooled.float(), pooled.shape[-1:])


class RouterModel(nn.Module):
    def __init__(self, config, backbone, saes, router, sae_layers, sae_norm_scales, temperature,
                 lambda_q, lambda_d, bos_token_id, tau_start, tau_end, balance_weight):
        super().__init__()
        self.config = config
        self.backbone = backbone
        self.saes = saes
        self.router = router
        self.sae_layers = sae_layers
        self.sae_norm_scales = sae_norm_scales
        self.temperature = temperature
        self.lambda_q = lambda_q
        self.lambda_d = lambda_d
        self.bos_token_id = bos_token_id
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.balance_weight = balance_weight
        self.progress = 0.0
        self.global_step = 0

    @property
    def tau(self):
        return self.tau_start + (self.tau_end - self.tau_start) * self.progress

    def encode(self, feature: Dict[str, torch.Tensor]):
        outputs = self.backbone(input_ids=feature["input_ids"], attention_mask=feature["attention_mask"],
                                output_hidden_states=True)
        pool_mask = feature["attention_mask"] * (feature["input_ids"] != self.bos_token_id)
        vectors = []
        for layer, sae, norm_scale in zip(self.sae_layers, self.saes, self.sae_norm_scales):
            sae_out = torch.log(1 + torch.relu(sae(outputs.hidden_states[layer + 1] * norm_scale)))
            vectors.append((sae_out * pool_mask.unsqueeze(-1)).max(dim=1)[0])
        logits = self.router(router_input(outputs.hidden_states[-1], pool_mask))
        return torch.stack(vectors, dim=1), logits

    @staticmethod
    def flops_loss(vectors):
        return torch.sum(vectors.mean(dim=0) ** 2)

    def forward(self, features: List[Dict[str, torch.Tensor]], reranker_scores: torch.Tensor):
        encoded = [self.encode(f) for f in features]
        q_vec, q_logits = encoded[0]
        d_vecs = torch.stack([e[0] for e in encoded[1:]], dim=1)
        d_logits = torch.stack([e[1] for e in encoded[1:]], dim=1)

        layer_scores = (q_vec.unsqueeze(1) * d_vecs).sum(dim=-1).float()
        pair_logits = q_logits.unsqueeze(1) + d_logits
        gates = nn.functional.gumbel_softmax(pair_logits, tau=self.tau, dim=-1)
        scores = (gates * layer_scores).sum(dim=-1)

        log_pred = torch.log_softmax(scores / self.temperature, dim=1)
        target_probs = torch.softmax(reranker_scores.to(scores.device), dim=1)
        kl_loss = nn.functional.kl_div(log_pred, target_probs, reduction="batchmean")

        n_layers = pair_logits.shape[-1]
        probs = torch.softmax(pair_logits, dim=-1).reshape(-1, n_layers)
        routed = nn.functional.one_hot(probs.argmax(dim=-1), n_layers).float().mean(dim=0)
        balance_loss = n_layers * (routed * probs.mean(dim=0)).sum()

        query_flops = self.flops_loss(q_vec.flatten(1))
        doc_flops = self.flops_loss(d_vecs.flatten(0, 1).flatten(1))

        if self.global_step < 20 or self.global_step % 200 == 0:
            share = " ".join(f"L{layer}={routed[i].item():.3f}" for i, layer in enumerate(self.sae_layers))
            entropy = -(probs * probs.clamp(min=1e-9).log()).sum(dim=-1).mean()
            print(f"[loss step={self.global_step}] kl={kl_loss.item():.4f} balance={balance_loss.item():.4f} "
                  f"flops_q={query_flops.item():.1f} flops_d={doc_flops.item():.1f} tau={self.tau:.3f} "
                  f"routed {share} router_entropy={entropy.item():.4f}")

        loss = (kl_loss + self.lambda_q * query_flops + self.lambda_d * doc_flops
                + self.balance_weight * balance_loss)
        self.global_step += 1
        return (loss,)

    def save_peft_model(self, path):
        self.backbone.save_pretrained(path)
        torch.save({"state_dict": self.router.state_dict(), "sae_layers": self.sae_layers},
                   os.path.join(path, "router.pt"))

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)


class ProgressCallback(TrainerCallback):
    def __init__(self, model):
        self.model = model

    def on_step_begin(self, args, state, control, **kwargs):
        self.model.progress = min(state.global_step / max(state.max_steps, 1), 1.0)


class RouterTrainer(BaseTrainer):
    def create_optimizer(self):
        if self.optimizer is None:
            named = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
            router_params = [p for n, p in named if n.startswith("router.")]
            other_params = [p for n, p in named if not n.startswith("router.")]
            print(f"Optimizer: {sum(p.numel() for p in other_params)} LoRA parameters at lr "
                  f"{self.args.learning_rate}, {sum(p.numel() for p in router_params)} router "
                  f"parameters at lr {self.sae_learning_rate}")
            self.optimizer = torch.optim.AdamW(
                [{"params": other_params}, {"params": router_params, "lr": self.sae_learning_rate}],
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,
            )
        return self.optimizer


@dataclass
class CustomArguments:
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=None)
    lora_dropout: float = field(default=0.05)
    stop_after_n_steps: int = field(default=10000)


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, required=True)
    pre_args, remaining_argv = pre_parser.parse_known_args()

    with open(os.path.abspath(pre_args.config)) as f:
        config_dict = json.load(f)

    sae_layers = sorted(config_dict.pop("sae_layers"))
    data_dir = os.environ["PKQ_DATA_DIR"]
    val = config_dict.get("peft_model_name_or_path")
    if val and not os.path.isabs(val):
        resolved = os.path.join(data_dir, val)
        if os.path.exists(resolved):
            print(f"Resolved peft_model_name_or_path: {val} -> {resolved}")
            config_dict["peft_model_name_or_path"] = resolved
    from huggingface_hub import hf_hub_download
    sae_expansion = config_dict.pop("sae_expansion", 8)
    sae_weights_paths = [
        hf_hub_download(
            f"OpenMOSS-Team/Llama3_1-8B-Base-LXR-{sae_expansion}x",
            f"Llama3_1-8B-Base-L{layer}R-{sae_expansion}x/checkpoints/final.safetensors",
            local_files_only=True,
        )
        for layer in sae_layers
    ]

    temperature = config_dict.pop("temperature")
    lambda_q = config_dict.pop("lambda_q")
    lambda_d = config_dict.pop("lambda_d")
    router_learning_rate = config_dict.pop("router_learning_rate")
    tau_start = config_dict.pop("router_tau_start")
    tau_end = config_dict.pop("router_tau_end")
    balance_weight = config_dict.pop("router_balance_weight")
    config_dict.pop("flops_chunk_size", None)
    config_dict.pop("flops_warmup_steps", None)

    layers_tag = "+".join(str(layer) for layer in sae_layers)
    output_dir = f"output/router/Meta-Llama-3.1-8B-msmarco-mntp-L{layers_tag}"
    config_dict["output_dir"] = output_dir
    print(f"output_dir: {output_dir}")

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, CustomArguments))
    parser.set_defaults(**config_dict)
    for action in parser._actions:
        if action.dest in config_dict and action.required:
            action.required = False
    model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses(args=remaining_argv)
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

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name or model_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    torch_dtype = (model_args.torch_dtype if model_args.torch_dtype in ["auto", None]
                   else getattr(torch, model_args.torch_dtype))
    model = get_model_class(config).from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        attn_implementation=model_args.attn_implementation,
    )
    if model_args.peft_model_name_or_path is not None:
        model = PeftModel.from_pretrained(model, model_args.peft_model_name_or_path)
        model = model.merge_and_unload()

    num_active = sae_layers[-1] + 1
    total_layers = len(model.layers)
    model.layers = model.layers[:num_active]
    model.norm = nn.Identity()
    import gc; gc.collect(); torch.cuda.empty_cache()
    print(f"Truncated model: kept layers 0-{sae_layers[-1]} ({num_active}/{total_layers})")

    model = initialize_peft(
        model,
        lora_r=custom_args.lora_r,
        lora_alpha=custom_args.lora_alpha or custom_args.lora_r,
        lora_dropout=custom_args.lora_dropout,
        layers_to_transform=list(range(num_active)),
    )

    saes, sae_norm_scales = nn.ModuleList(), []
    for path in sae_weights_paths:
        with safe_open(path, framework="pt") as f:
            weight, bias = f.get_tensor("encoder.weight"), f.get_tensor("encoder.bias")
        sae = nn.Linear(weight.shape[1], weight.shape[0])
        with torch.no_grad():
            sae.weight.copy_(weight)
            sae.bias.copy_(bias)
        sae.to(torch_dtype)
        sae.requires_grad_(False)
        saes.append(sae)
        with open(os.path.join(os.path.dirname(os.path.dirname(path)), "hyperparams.json")) as f:
            activation_norm = json.load(f)["dataset_average_activation_norm"]["in"]
        sae_norm_scales.append((config.hidden_size ** 0.5) / activation_norm)
        print(f"SAE {path}: activation_norm {activation_norm}, sae_norm_scale {sae_norm_scales[-1]}")

    router = nn.Linear(config.hidden_size, len(sae_layers))
    nn.init.zeros_(router.weight)
    nn.init.zeros_(router.bias)
    print(f"Router over layers {sae_layers}: lr {router_learning_rate}, tau {tau_start} -> {tau_end}, "
          f"balance weight {balance_weight}")

    router_model = RouterModel(
        config=config,
        backbone=model,
        saes=saes,
        router=router,
        sae_layers=sae_layers,
        sae_norm_scales=sae_norm_scales,
        temperature=temperature,
        lambda_q=lambda_q,
        lambda_d=lambda_d,
        bos_token_id=tokenizer.bos_token_id,
        tau_start=tau_start,
        tau_end=tau_end,
        balance_weight=balance_weight,
    )

    train_dataset = MSMARCOHardNegDataset(
        file_path=data_args.hard_negatives_file,
        num_hard_negatives=data_args.num_hard_negatives,
    )
    data_collator = ContrastiveCollator(tokenizer=tokenizer, max_length=data_args.max_seq_length)

    trainer = RouterTrainer(
        model=router_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        sae_learning_rate=router_learning_rate,
    )
    trainer.add_callback(StopTrainingCallback(custom_args.stop_after_n_steps))
    trainer.add_callback(ProgressCallback(router_model))
    trainer.train()


if __name__ == "__main__":
    main()

"""MTEB eval of a feature-gate checkpoint (trained with run_featgate_finetune.py).

The encoder is the frozen backbone (base + merged MNTP adapter, no trained LoRA) and frozen SAE, with
the trained per-feature gate applied to the SAE pre-activation: log(1 + relu(g * pre + b)), then the
usual BOS-excluded max-pool and top-k pruning.

    python experiments/mteb_eval_featgate.py --trained_checkpoint_path <root>/output/featgate/<run>/checkpoint-N --task_name SciFact NFCorpus ArguAna SCIDOCS
"""

import argparse
import os
import sys
import traceback

import mteb
import torch
from torch.nn.parallel import replicate

from mteb_eval_layerwise import EncodeModule, LayerwiseEncoder, MTEBWrapper, resolve_checkpoint_args


class FeatGateEncodeModule(EncodeModule):
    def __init__(self, module, log_gain, bias):
        super().__init__(module.backbone, module.sae, module.sae_norm_scale, module.bos_token_id,
                         module.jump_relu_threshold, module.sae_top_k)
        dtype = module.sae.weight.dtype
        self.register_buffer("gain", log_gain.exp().to(dtype))
        self.register_buffer("gate_bias", bias.to(dtype))

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        sae_pre = self.sae(outputs[0] * self.sae_norm_scale) * self.gain + self.gate_bias
        sae_out = torch.log(1 + torch.relu(sae_pre))
        pool_mask = attention_mask * (input_ids != self.bos_token_id)
        sae_out = sae_out * pool_mask.unsqueeze(-1)
        return sae_out.max(dim=1)[0]


def add_gate(encoder, gate_path):
    """Swap the encoder's module for the gated one, re-replicate and recalibrate memory."""
    gate = torch.load(gate_path, map_location="cpu")
    gain = gate["log_gain"].exp()
    print(f"Gate from {gate_path}: gain mean={gain.mean():.4f} min={gain.min():.4f} max={gain.max():.4f}  "
          f"bias mean={gate['bias'].mean():.3f} min={gate['bias'].min():.3f} max={gate['bias'].max():.3f}")
    module = FeatGateEncodeModule(encoder.encode_module, gate["log_gain"], gate["bias"])
    module.to(encoder.device).eval()
    encoder.encode_module = module
    encoder.replicas = None
    if encoder.num_gpus > 1:
        with torch.no_grad():
            encoder.replicas = replicate(module, encoder.device_ids, detach=True)
    encoder.gpu_modules = encoder.replicas if encoder.replicas is not None else [module]
    torch.cuda.empty_cache()
    encoder.bytes_per_token = encoder._calibrate_bytes_per_token()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_checkpoint_path", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--peft_model_name_or_path", type=str)
    parser.add_argument("--sae_weights_path", type=str)
    parser.add_argument("--lora_layers", type=int)
    parser.add_argument("--task_name", type=str, nargs="+",
                        default=["SciFact", "NFCorpus", "ArguAna", "SCIDOCS"])
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--query_top_k", type=int, default=40, help="0 disables pruning.")
    parser.add_argument("--doc_top_k", type=int, default=400, help="0 disables pruning.")
    parser.add_argument("--max_length", type=int, default=1024)
    args = parser.parse_args()
    args.query_top_k = args.query_top_k or None
    args.doc_top_k = args.doc_top_k or None
    resolve_checkpoint_args(args, parser)

    try:
        encoder = LayerwiseEncoder(
            model_name_or_path=args.model_name_or_path,
            peft_model_name_or_path=args.peft_model_name_or_path,
            sae_weights_path=args.sae_weights_path,
            lora_layers=args.lora_layers,
            trained_checkpoint_path=None,
            max_length=args.max_length,
        )
        add_gate(encoder, os.path.join(args.trained_checkpoint_path, "featgate.pt"))
        model = MTEBWrapper(encoder, query_top_k=args.query_top_k, doc_top_k=args.doc_top_k,
                            max_length=args.max_length)
        tasks = mteb.get_tasks(tasks=args.task_name)
        evaluation = mteb.MTEB(tasks=tasks)
        print(f"Starting evaluation on {args.task_name}", flush=True)
        results = evaluation.run(model, output_folder=args.output_dir, overwrite_results=True)
        print(f"Evaluation complete. Results: {results}", flush=True)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

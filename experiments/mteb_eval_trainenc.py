"""
    python experiments/mteb_eval_trainenc.py --trained_checkpoint_path <root>/output/trainenc/<run>/checkpoint-N --task_name SciFact NFCorpus ArguAna SCIDOCS
"""

import argparse
import os
import sys
import traceback

import mteb
import torch
from torch.nn.parallel import replicate

from mteb_eval_layerwise import LayerwiseEncoder, MTEBWrapper, resolve_checkpoint_args


@torch.no_grad()
def load_encoder(encoder, sae_path):
    trained = torch.load(sae_path, map_location="cpu")
    sae = encoder.encode_module.sae
    w_drift = (trained["weight"] - sae.weight.float().cpu()).norm() / sae.weight.float().cpu().norm()
    b_drift = (trained["bias"] - sae.bias.float().cpu()).norm() / sae.bias.float().cpu().norm()
    print(f"SAE encoder from {sae_path}: W_enc_drift={w_drift:.6f} b_enc_drift={b_drift:.6f}")
    sae.weight.copy_(trained["weight"].to(sae.weight.dtype))
    sae.bias.copy_(trained["bias"].to(sae.bias.dtype))
    encoder.replicas = None
    if encoder.num_gpus > 1:
        encoder.replicas = replicate(encoder.encode_module, encoder.device_ids, detach=True)
    encoder.gpu_modules = encoder.replicas if encoder.replicas is not None else [encoder.encode_module]
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
    parser.add_argument("--output_dir", type=str, default="results")
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
            trained_checkpoint_path=args.trained_checkpoint_path,
            max_length=args.max_length,
        )
        load_encoder(encoder, os.path.join(args.trained_checkpoint_path, "sae.pt"))
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

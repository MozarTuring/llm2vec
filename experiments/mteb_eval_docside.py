"""MTEB eval of a document-side-only checkpoint (trained with run_docside_finetune.py).

Queries are encoded by the base model (+ merged prior MNTP LoRA, no trained LoRA), documents by
the trained checkpoint; both through the same frozen SAE. One encoder is loaded with the trained
LoRA, which is switched off on every GPU replica for query batches (as disable_adapter() in training).

    python experiments/mteb_eval_docside.py --trained_checkpoint_path <root>/output/docside/<run>/checkpoint-N --task_name SciFact NFCorpus ArguAna SCIDOCS
"""

import argparse
import sys
import traceback

import mteb
from peft.tuners.tuners_utils import BaseTunerLayer

from mteb_eval_layerwise import LayerwiseEncoder, MTEBWrapper, resolve_checkpoint_args


def set_lora_enabled(encoder, enabled):
    """Toggle the trained LoRA on every per-GPU module (replicas do not share module state)."""
    n = 0
    for module in encoder.gpu_modules:
        for m in module.modules():
            if isinstance(m, BaseTunerLayer):
                m.enable_adapters(enabled)
                n += 1
    return n


class DocsideMTEBWrapper(MTEBWrapper):
    """Encodes queries with the trained LoRA off and documents with it on."""

    def encode(self, inputs, *, prompt_type=None, **kwargs):
        enc_type = prompt_type.value if hasattr(prompt_type, "value") else str(prompt_type)
        lora_on = enc_type != "query"
        n = set_lora_enabled(self.encoder, lora_on)
        print(f"[docside] enc_type={enc_type} trained LoRA {'on' if lora_on else 'off'} "
              f"({n} layers over {len(self.encoder.gpu_modules)} GPU modules)", flush=True)
        return super().encode(inputs, prompt_type=prompt_type, **kwargs)


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
            trained_checkpoint_path=args.trained_checkpoint_path,
            max_length=args.max_length,
        )
        model = DocsideMTEBWrapper(encoder, query_top_k=args.query_top_k,
                                   doc_top_k=args.doc_top_k, max_length=args.max_length)
        tasks = mteb.get_tasks(tasks=args.task_name)
        evaluation = mteb.MTEB(tasks=tasks)
        print(f"Starting evaluation on {args.task_name}", flush=True)
        results = evaluation.run(model, output_folder=args.output_dir, overwrite_results=True)
        print(f"Evaluation complete. Results: {results}", flush=True)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

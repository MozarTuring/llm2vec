"""MTEB eval of a document-side-only checkpoint (trained with run_docside_finetune.py).

Queries are encoded by the base model (+ merged prior MNTP LoRA, no trained LoRA), documents by
the trained checkpoint; both through the same frozen SAE. Two encoders are loaded (one with,
one without the trained LoRA).

    python experiments/mteb_eval_docside.py --trained_checkpoint_path <root>/output/docside/<run>/checkpoint-N --task_name SciFact NFCorpus ArguAna SCIDOCS
"""

import argparse
import sys
import traceback

import mteb

from mteb_eval_layerwise import LayerwiseEncoder, MTEBWrapper, resolve_checkpoint_args


class DocsideMTEBWrapper(MTEBWrapper):
    """Routes queries to the base encoder and documents to the trained encoder."""

    def __init__(self, query_encoder, doc_encoder, query_top_k, doc_top_k, max_length=1024):
        super().__init__(doc_encoder, query_top_k, doc_top_k, max_length)
        self.query_encoder = query_encoder
        self.doc_encoder = doc_encoder

    def encode(self, inputs, *, prompt_type=None, **kwargs):
        enc_type = prompt_type.value if hasattr(prompt_type, "value") else str(prompt_type)
        self.encoder = self.query_encoder if enc_type == "query" else self.doc_encoder
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
        common = dict(
            model_name_or_path=args.model_name_or_path,
            peft_model_name_or_path=args.peft_model_name_or_path,
            sae_weights_path=args.sae_weights_path,
            lora_layers=args.lora_layers,
            max_length=args.max_length,
        )
        print("Loading query encoder (base model, no trained LoRA)", flush=True)
        query_encoder = LayerwiseEncoder(trained_checkpoint_path=None, **common)
        print("Loading document encoder (trained checkpoint)", flush=True)
        doc_encoder = LayerwiseEncoder(trained_checkpoint_path=args.trained_checkpoint_path, **common)

        model = DocsideMTEBWrapper(query_encoder, doc_encoder, query_top_k=args.query_top_k,
                                   doc_top_k=args.doc_top_k, max_length=args.max_length)
        tasks = mteb.get_tasks(tasks=args.task_name)
        evaluation = mteb.MTEB(tasks=tasks)
        print(f"Starting evaluation on {args.task_name}", flush=True)
        results = evaluation.run(model, output_folder=args.output_dir, overwrite_results=True)
        print(f"Evaluation complete. Results: {results}", flush=True)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

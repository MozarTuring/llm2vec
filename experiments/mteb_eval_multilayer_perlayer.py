"""
    python experiments/mteb_eval_multilayer_perlayer.py --trained_checkpoint_path <root>/output/multilayer/<run>/checkpoint-N --task_name SciFact NFCorpus ArguAna SCIDOCS
"""

import argparse
import sys
import traceback

import mteb
import torch

from mteb_eval_multilayer import MultilayerEncoder, MultilayerMTEBWrapper, resolve_multilayer_args


class PerLayerEncoder(MultilayerEncoder):
    def __init__(self, *args, sae_layers, **kwargs):
        super().__init__(*args, sae_layers=sae_layers, **kwargs)
        self.n_layers = len(sae_layers)

    def encode_texts(self, texts, top_k=None, chunk_size=512, **kwargs):
        embeddings = super().encode_texts(texts, top_k=None, **kwargs)
        if top_k is None:
            return embeddings
        for start in range(0, embeddings.shape[0], chunk_size):
            rows = torch.from_numpy(embeddings[start:start + chunk_size])
            for i in range(self.n_layers):
                block = rows[:, i * self.d_sae:(i + 1) * self.d_sae]
                vals, idx = block.topk(top_k, dim=-1)
                block.zero_()
                block.scatter_(-1, idx, vals)
        return embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_checkpoint_path", type=str, required=True)
    parser.add_argument("--task_name", type=str, nargs="+",
                        default=["SciFact", "NFCorpus", "ArguAna", "SCIDOCS"])
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--query_top_k", type=int, default=40, help="Per layer; 0 disables pruning.")
    parser.add_argument("--doc_top_k", type=int, default=400, help="Per layer; 0 disables pruning.")
    parser.add_argument("--max_length", type=int, default=1024)
    args = parser.parse_args()
    args.query_top_k = args.query_top_k or None
    args.doc_top_k = args.doc_top_k or None
    resolve_multilayer_args(args, parser)

    try:
        encoder = PerLayerEncoder(
            model_name_or_path=args.model_name_or_path,
            peft_model_name_or_path=args.peft_model_name_or_path,
            sae_weights_paths=args.sae_weights_paths,
            sae_layers=args.sae_layers,
            trained_checkpoint_path=args.trained_checkpoint_path,
            max_length=args.max_length,
        )
        model = MultilayerMTEBWrapper(encoder, args.sae_layers, query_top_k=args.query_top_k,
                                      doc_top_k=args.doc_top_k, max_length=args.max_length)
        tasks = mteb.get_tasks(tasks=args.task_name)
        evaluation = mteb.MTEB(tasks=tasks)
        print(f"Starting evaluation on {args.task_name}", flush=True)
        results = evaluation.run(model, output_folder=args.output_dir, overwrite_results=True)
        print(f"Evaluation complete. Results: {results}", flush=True)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

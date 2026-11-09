"""
    python experiments/mteb_eval_qrouter.py --trained_checkpoint_path <root>/output/qrouter/<run>/checkpoint-N --task_name SciFact NFCorpus ArguAna SCIDOCS
"""

import argparse
import os
import sys
import traceback

import mteb
import torch
from torch.nn.parallel import replicate

from mteb_eval_multilayer import resolve_multilayer_args
from mteb_eval_router import RouterEncodeModule, RouterEncoder, RouterMTEBWrapper
from run_router_finetune import router_input


class QueryRouterEncodeModule(RouterEncodeModule):
    def __init__(self, module, router_layer):
        super().__init__(module, module.router)
        self.router_layer = router_layer

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        pool_mask = attention_mask * (input_ids != self.bos_token_id)
        pooled = []
        for layer, sae, norm_scale in zip(self.sae_layers, self.saes, self.sae_norm_scales):
            sae_out = torch.log(1 + torch.relu(sae(outputs.hidden_states[layer + 1] * norm_scale)))
            pooled.append((sae_out * pool_mask.unsqueeze(-1)).max(dim=1)[0])
        logits = self.router(router_input(outputs.hidden_states[self.router_layer + 1], pool_mask))
        return torch.cat(pooled + [logits.to(pooled[0].dtype)], dim=-1)


class QueryRouterEncoder(RouterEncoder):
    def __init__(self, *args, trained_checkpoint_path, **kwargs):
        super().__init__(*args, trained_checkpoint_path=trained_checkpoint_path, **kwargs)
        saved = torch.load(os.path.join(trained_checkpoint_path, "router.pt"), map_location="cpu")
        router_layer = saved["router_layer"]
        print(f"Router reads layer {router_layer}")
        module = QueryRouterEncodeModule(self.encode_module, router_layer).to(self.device).eval()
        self.encode_module = module
        self.replicas = None
        if self.num_gpus > 1:
            with torch.no_grad():
                self.replicas = replicate(module, self.device_ids, detach=True)
        self.gpu_modules = self.replicas if self.replicas is not None else [module]
        torch.cuda.empty_cache()
        self.bytes_per_token = self._calibrate_bytes_per_token()


class QueryRouterMTEBWrapper(RouterMTEBWrapper):
    def similarity(self, embeddings1, embeddings2):
        (q_parts, q_logits), (d_parts, _) = self._parts(embeddings1), self._parts(embeddings2)
        layer_scores = torch.stack([torch.mm(q, d.T) for q, d in zip(q_parts, d_parts)], dim=-1)
        choice = q_logits.argmax(dim=-1)
        index = choice.view(-1, 1, 1).expand(-1, layer_scores.shape[1], 1)
        scores = layer_scores.gather(-1, index).squeeze(-1)
        share = "  ".join(f"L{layer}={(choice == i).float().mean().item():.1%}"
                          for i, layer in enumerate(self.sae_layers))
        print(f"[DEBUG similarity] queries routed: {share}", flush=True)
        return scores

    def similarity_pairwise(self, embeddings1, embeddings2):
        (q_parts, q_logits), (d_parts, _) = self._parts(embeddings1), self._parts(embeddings2)
        layer_scores = torch.stack([(q * d).sum(dim=-1) for q, d in zip(q_parts, d_parts)], dim=-1)
        choice = q_logits.argmax(dim=-1)
        return layer_scores.gather(-1, choice.unsqueeze(-1)).squeeze(-1)


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
        encoder = QueryRouterEncoder(
            model_name_or_path=args.model_name_or_path,
            peft_model_name_or_path=args.peft_model_name_or_path,
            sae_weights_paths=args.sae_weights_paths,
            sae_layers=args.sae_layers,
            trained_checkpoint_path=args.trained_checkpoint_path,
            max_length=args.max_length,
        )
        model = QueryRouterMTEBWrapper(encoder, args.sae_layers, query_top_k=args.query_top_k,
                                       doc_top_k=args.doc_top_k, max_length=args.max_length)
        tasks = mteb.get_tasks(tasks=args.task_name)
        evaluation = mteb.MTEB(tasks=tasks)
        print(f"Starting evaluation on {args.task_name}", flush=True)
        results = evaluation.run(model, output_folder=args.output_dir, overwrite_results=True)
        print(f"Evaluation complete. Results: {results}", flush=True)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

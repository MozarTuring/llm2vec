"""
    python experiments/mteb_eval_router.py --trained_checkpoint_path <root>/output/router/<run>/checkpoint-N --task_name SciFact NFCorpus ArguAna SCIDOCS
"""

import argparse
import os
import sys
import traceback

import mteb
import numpy as np
import torch
from torch import nn
from torch.nn.parallel import replicate

from mteb_eval_multilayer import MultilayerEncoder, MultilayerMTEBWrapper, resolve_multilayer_args
from run_router_finetune import router_input


class RouterEncodeModule(nn.Module):
    def __init__(self, module, router):
        super().__init__()
        self.backbone = module.backbone
        self.saes = module.saes
        self.sae_layers = module.sae_layers
        self.sae_norm_scales = module.sae_norm_scales
        self.bos_token_id = module.bos_token_id
        self.router = router

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        pool_mask = attention_mask * (input_ids != self.bos_token_id)
        pooled = []
        for layer, sae, norm_scale in zip(self.sae_layers, self.saes, self.sae_norm_scales):
            sae_out = torch.log(1 + torch.relu(sae(outputs.hidden_states[layer + 1] * norm_scale)))
            pooled.append((sae_out * pool_mask.unsqueeze(-1)).max(dim=1)[0])
        logits = self.router(router_input(outputs.hidden_states[-1], pool_mask))
        return torch.cat(pooled + [logits.to(pooled[0].dtype)], dim=-1)


class RouterEncoder(MultilayerEncoder):
    def __init__(self, *args, sae_layers, trained_checkpoint_path, **kwargs):
        super().__init__(*args, sae_layers=sae_layers, trained_checkpoint_path=trained_checkpoint_path, **kwargs)
        self.n_layers = len(sae_layers)
        saved = torch.load(os.path.join(trained_checkpoint_path, "router.pt"), map_location="cpu")
        assert saved["sae_layers"] == sae_layers, (saved["sae_layers"], sae_layers)
        router = nn.Linear(self.encode_module.backbone.config.hidden_size, self.n_layers)
        router.load_state_dict(saved["state_dict"])
        print(f"Router from {trained_checkpoint_path}: bias {router.bias.tolist()}")
        module = RouterEncodeModule(self.encode_module, router).to(self.device).eval()
        self.encode_module = module
        self.replicas = None
        if self.num_gpus > 1:
            with torch.no_grad():
                self.replicas = replicate(module, self.device_ids, detach=True)
        self.gpu_modules = self.replicas if self.replicas is not None else [module]
        torch.cuda.empty_cache()
        self.bytes_per_token = self._calibrate_bytes_per_token()

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


class RouterMTEBWrapper(MultilayerMTEBWrapper):
    def _parts(self, emb):
        emb = torch.from_numpy(emb) if isinstance(emb, np.ndarray) else emb
        n = len(self.sae_layers)
        return [emb[:, i * self.d_sae:(i + 1) * self.d_sae] for i in range(n)], emb[:, n * self.d_sae:]

    def encode(self, inputs, **kwargs):
        embeddings = super(MultilayerMTEBWrapper, self).encode(inputs, **kwargs)
        n = len(self.sae_layers)
        choice = embeddings[:, n * self.d_sae:].argmax(axis=1)
        shares = "  ".join(f"L{layer}={(choice == i).mean():.1%}" for i, layer in enumerate(self.sae_layers))
        print(f"[DEBUG encode] per-text router argmax share: {shares}", flush=True)
        return embeddings

    def similarity(self, embeddings1, embeddings2):
        (q_parts, q_logits), (d_parts, d_logits) = self._parts(embeddings1), self._parts(embeddings2)
        layer_scores = torch.stack([torch.mm(q, d.T) for q, d in zip(q_parts, d_parts)], dim=-1)
        choice = (q_logits.unsqueeze(1) + d_logits.unsqueeze(0)).argmax(dim=-1)
        scores = layer_scores.gather(-1, choice.unsqueeze(-1)).squeeze(-1)
        k = min(10, scores.shape[1])
        top = scores.topk(k, dim=1).indices
        all_share = "  ".join(f"L{layer}={(choice == i).float().mean().item():.1%}"
                              for i, layer in enumerate(self.sae_layers))
        top_share = "  ".join(f"L{layer}={(choice.gather(1, top) == i).float().mean().item():.1%}"
                              for i, layer in enumerate(self.sae_layers))
        mixed = (choice.gather(1, top).float().std(dim=1) > 0).float().mean().item()
        print(f"[DEBUG similarity] pairs routed: {all_share}; top-{k} pairs routed: {top_share}; "
              f"queries whose top-{k} mixes layers: {mixed:.1%}", flush=True)
        return scores

    def similarity_pairwise(self, embeddings1, embeddings2):
        (q_parts, q_logits), (d_parts, d_logits) = self._parts(embeddings1), self._parts(embeddings2)
        layer_scores = torch.stack([(q * d).sum(dim=-1) for q, d in zip(q_parts, d_parts)], dim=-1)
        choice = (q_logits + d_logits).argmax(dim=-1)
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
        encoder = RouterEncoder(
            model_name_or_path=args.model_name_or_path,
            peft_model_name_or_path=args.peft_model_name_or_path,
            sae_weights_paths=args.sae_weights_paths,
            sae_layers=args.sae_layers,
            trained_checkpoint_path=args.trained_checkpoint_path,
            max_length=args.max_length,
        )
        model = RouterMTEBWrapper(encoder, args.sae_layers, query_top_k=args.query_top_k,
                                  doc_top_k=args.doc_top_k, max_length=args.max_length)
        tasks = mteb.get_tasks(tasks=args.task_name)
        evaluation = mteb.MTEB(tasks=tasks)
        print(f"Starting evaluation on {args.task_name}", flush=True)
        results = evaluation.run(model, output_folder=args.output_dir, overwrite_results=True)
        print(f"Evaluation complete. Results: {results}", flush=True)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

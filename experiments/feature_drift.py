"""Do SAE features keep their meaning inside SPLARE? Feature drift of a frozen SAE under MNTP + retrieval fine-tuning.

The Llama Scope SAE was trained on residuals of the causal base model. SPLARE feeds it residuals of
a bidirectional MNTP model with a retrieval LoRA. On the same texts, compare the layer-N residuals of
    base    causal Llama-3.1-8B (the SAE's training distribution)
    mntp    bidirectional + merged MNTP adapter (trained LoRA disabled)
    splare  mntp + trained retrieval LoRA (the checkpoint)

Per model (tokens, BOS and padding excluded):
    fvu        SAE reconstruction error / residual variance (JumpReLU SAE as released)
    l0         active features per token, JumpReLU gate
    relu_pos   share of features with positive pre-activation (what SPLARE's log(1+relu) keeps)
    res_norm   mean residual L2 norm
Against base (same tokens):
    res_cos    cosine of residuals
    act_cos    cosine of JumpReLU activation vectors
    jaccard    Jaccard of active feature sets
    top10      overlap of the 10 largest activations
Pooled, per text (the retrieval vector: log(1+relu) max-pooled, top-k):
    topk_vs_base   overlap of the top-k feature set with base's
    mass_off_base  share of the top-k pooled mass on features the base SAE never activates
                   (JumpReLU) anywhere in that text. For base itself this is the share that comes
                   only from sub-threshold relu activity, the reference level.
Feature frequency over the corpus (JumpReLU): log-frequency correlation with base, and features
active in a model but never in base.

JumpReLU gate on raw weights (see Llama Scope notes): pre = W_enc (s x) + b_enc, active if
pre * ||d_i|| / s > threshold, with s = sqrt(d) / activation_norm. Reconstruction:
x_hat = (W_dec (pre * gate) + b_dec) / s.

    python experiments/feature_drift.py --trained_checkpoint_path <root>/output/layerwise/<run>/checkpoint-N
"""

import argparse
import json
import os
import sys
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from safetensors.torch import safe_open
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from dump_sparse import load_beir
from mteb_eval_layerwise import get_model_class, resolve_checkpoint_args

MODELS = ["base", "mntp", "splare"]


def truncate(backbone, num_active):
    backbone.layers = backbone.layers[:num_active]
    backbone.norm = nn.Identity()
    return backbone


def load_models(args, device):
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    num_active = args.lora_layers + 1
    base = truncate(AutoModel.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16,
                                              attn_implementation="sdpa"), num_active)
    bi = get_model_class(config).from_pretrained(args.model_name_or_path, config=config,
                                                 torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    if args.peft_model_name_or_path is not None:
        bi = PeftModel.from_pretrained(bi, args.peft_model_name_or_path).merge_and_unload()
    bi = PeftModel.from_pretrained(truncate(bi, num_active), args.trained_checkpoint_path)
    print(f"base: {type(base).__name__}, bi: {type(bi.get_base_model()).__name__}, "
          f"kept {num_active} layers")
    return base.to(device).eval(), bi.to(device).eval()


class SAE:
    def __init__(self, path, device):
        with safe_open(path, framework="pt") as f:
            self.W_enc = f.get_tensor("encoder.weight").to(device, torch.bfloat16)  # (W, d)
            self.b_enc = f.get_tensor("encoder.bias").to(device, torch.bfloat16)
            self.W_dec = f.get_tensor("decoder.weight").to(device, torch.bfloat16)  # (d, W)
            self.b_dec = f.get_tensor("decoder.bias").to(device, torch.float32)
        with open(os.path.join(os.path.dirname(os.path.dirname(path)), "hyperparams.json")) as f:
            hyp = json.load(f)
        self.s = self.W_enc.shape[1] ** 0.5 / hyp["dataset_average_activation_norm"]["in"]
        self.threshold = hyp["jump_relu_threshold"]
        self.dec_norm = self.W_dec.float().norm(dim=0)
        self.width = self.W_enc.shape[0]
        print(f"SAE {path}: width={self.width} s={self.s:.4f} threshold={self.threshold} "
              f"decoder norm mean={self.dec_norm.mean():.3f}")

    def pre(self, x):
        return F.linear((x * self.s).to(torch.bfloat16), self.W_enc, self.b_enc).float()

    def gate(self, pre):
        return pre * self.dec_norm / self.s > self.threshold

    def decode(self, acts):
        return (F.linear(acts.to(torch.bfloat16), self.W_dec).float() + self.b_dec) / self.s


class Stats:
    """Accumulators for one text group (queries or documents)."""

    def __init__(self, width, device):
        self.sums = {m: {} for m in MODELS}
        self.x_sum = {m: torch.zeros(4096, dtype=torch.float64, device=device) for m in MODELS}
        self.freq = {m: torch.zeros(width, dtype=torch.float64, device=device) for m in MODELS}
        self.tokens = 0
        self.texts = 0

    def add(self, m, key, value):
        self.sums[m][key] = self.sums[m].get(key, 0.0) + float(value)


@torch.no_grad()
def process_batch(texts, tokenizer, base, bi, sae, stats, top_k, max_length, device):
    batch = tokenizer(texts, padding=True, truncation=True, max_length=max_length,
                      return_tensors="pt").to(device)
    ids, attn = batch["input_ids"], batch["attention_mask"]
    valid = attn.bool() & (ids != tokenizer.bos_token_id)
    hidden = {"base": base(input_ids=ids, attention_mask=attn)[0]}
    with bi.disable_adapter():
        hidden["mntp"] = bi(input_ids=ids, attention_mask=attn)[0]
    hidden["splare"] = bi(input_ids=ids, attention_mask=attn)[0]

    tok, pooled, any_gate = {}, {}, {}
    for m in MODELS:
        x = hidden[m].float()
        pre = sae.pre(x)                                     # (B, L, W)
        gate = sae.gate(pre) & valid.unsqueeze(-1)
        p = (torch.log1p(torch.relu(pre)) * valid.unsqueeze(-1)).max(dim=1)[0]
        vals, idx = p.topk(top_k, dim=-1)
        pooled[m] = torch.zeros_like(p).scatter_(-1, idx, vals)
        any_gate[m] = gate.any(dim=1)

        x_t, pre_t, gate_t = x[valid], pre[valid], gate[valid]
        acts = pre_t * gate_t
        recon = sae.decode(acts)
        stats.add(m, "sse", ((x_t - recon) ** 2).sum())
        stats.add(m, "x_sq", (x_t.double() ** 2).sum())
        stats.x_sum[m] += x_t.double().sum(0)
        stats.add(m, "l0", gate_t.sum())
        stats.add(m, "relu_pos", (pre_t > 0).float().mean(-1).sum())
        stats.add(m, "res_norm", x_t.norm(dim=-1).sum())
        stats.freq[m] += gate_t.sum(0).double()
        tok[m] = (x_t, acts, gate_t)
        del pre, gate

    x_b, a_b, g_b = tok["base"]
    top_b = a_b.topk(10, dim=-1).indices
    for m in MODELS[1:]:
        x_m, a_m, g_m = tok[m]
        stats.add(m, "res_cos", F.cosine_similarity(x_m, x_b, dim=-1).sum())
        stats.add(m, "act_cos", F.cosine_similarity(a_m, a_b, dim=-1).sum())
        union = (g_m | g_b).sum(-1).clamp(min=1)
        stats.add(m, "jaccard", ((g_m & g_b).sum(-1) / union).sum())
        top_m = a_m.topk(10, dim=-1).indices
        stats.add(m, "top10", ((top_m[:, :, None] == top_b[:, None, :]).any(-1).float().mean(-1)).sum())
        overlap = ((pooled[m] > 0) & (pooled["base"] > 0)).sum(-1).float() / top_k
        stats.add(m, "topk_vs_base", overlap.sum())
    for m in MODELS:
        mass = (pooled[m] * ~any_gate["base"]).sum(-1) / pooled[m].sum(-1).clamp(min=1e-8)
        stats.add(m, "mass_off_base", mass.sum())

    stats.tokens += int(valid.sum())
    stats.texts += len(texts)


def summarize(stats):
    n_tok, n_txt = stats.tokens, stats.texts
    out = {}
    for m in MODELS:
        s = stats.sums[m]
        mean_x = stats.x_sum[m] / n_tok
        sst = s["x_sq"] - n_tok * float((mean_x ** 2).sum())
        r = {"fvu": s["sse"] / sst, "l0": s["l0"] / n_tok, "relu_pos": s["relu_pos"] / n_tok,
             "res_norm": s["res_norm"] / n_tok, "mass_off_base": s["mass_off_base"] / n_txt}
        for key in ("res_cos", "act_cos", "jaccard", "top10"):
            if key in s:
                r[key] = s[key] / n_tok
        if "topk_vs_base" in s:
            r["topk_vs_base"] = s["topk_vs_base"] / n_txt
        f_m, f_b = stats.freq[m], stats.freq["base"]
        r["alive"] = int((f_m > 0).sum())
        r["alive_not_in_base"] = int(((f_m > 0) & (f_b == 0)).sum())
        r["logfreq_corr_base"] = float(np.corrcoef(torch.log1p(f_m).cpu().numpy(),
                                                   torch.log1p(f_b).cpu().numpy())[0, 1])
        out[m] = r
    return out


def print_table(name, n_txt, n_tok, res):
    cols = ["fvu", "l0", "relu_pos", "res_norm", "res_cos", "act_cos", "jaccard", "top10",
            "topk_vs_base", "mass_off_base", "alive", "alive_not_in_base", "logfreq_corr_base"]
    print(f"\n=== {name}: {n_txt} texts, {n_tok} tokens ===")
    print(f"{'metric':20s}" + "".join(f"{m:>10s}" for m in MODELS))
    for c in cols:
        row = []
        for m in MODELS:
            v = res[m].get(c)
            row.append(f"{'-':>10s}" if v is None else (f"{v:>10d}" if isinstance(v, int) else f"{v:>10.4f}"))
        print(f"{c:20s}" + "".join(row))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_checkpoint_path", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--peft_model_name_or_path", type=str)
    parser.add_argument("--sae_weights_path", type=str)
    parser.add_argument("--lora_layers", type=int)
    parser.add_argument("--task", type=str, default="SciFact")
    parser.add_argument("--num_queries", type=int, default=300)
    parser.add_argument("--num_docs", type=int, default=1000)
    parser.add_argument("--query_top_k", type=int, default=40)
    parser.add_argument("--doc_top_k", type=int, default=400)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="results_feature_drift")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    resolve_checkpoint_args(args, parser)

    try:
        device = torch.device("cuda")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        tokenizer.padding_side = "right"
        base, bi = load_models(args, device)
        sae = SAE(args.sae_weights_path, device)

        queries, corpus, _ = load_beir(args.task)
        rng = np.random.default_rng(args.seed)
        q_texts = list(queries.values())
        d_texts = list(corpus.values())
        q_texts = [q_texts[i] for i in rng.permutation(len(q_texts))[:args.num_queries]]
        d_texts = [d_texts[i] for i in rng.permutation(len(d_texts))[:args.num_docs]]

        results = {}
        for name, texts, top_k in (("queries", q_texts, args.query_top_k),
                                   ("documents", d_texts, args.doc_top_k)):
            stats = Stats(sae.width, device)
            for i in range(0, len(texts), args.batch_size):
                process_batch(texts[i:i + args.batch_size], tokenizer, base, bi, sae, stats,
                              top_k, args.max_length, device)
            results[name] = summarize(stats)
            print_table(f"{args.task} {name} (top-k {top_k})", stats.texts, stats.tokens, results[name])

        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, f"{args.task}.json"), "w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=2)
        print(f"\nSaved to {args.output_dir}/{args.task}.json")
    except Exception:
        traceback.print_exc()
        sys.exit(1)

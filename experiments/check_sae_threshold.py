"""Check SAE encoder bias stats and empirical L0 to determine
whether jump_relu_threshold needs to be scaled by sae_norm_scale.

Run on remote:
    python experiments/check_sae_threshold.py \
        --sae_weights_path /home/jinma/project_remote_jwm/remote_data/llm2vec/Llama3_1-8B-Base-L0R-8x/checkpoints/final.safetensors
"""
import argparse
import json
import os

import torch
from safetensors.torch import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae_weights_path", required=True)
    parser.add_argument("--model_name_or_path", default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--num_samples", type=int, default=8,
                        help="Number of sample texts to run through the model")
    args = parser.parse_args()

    # ── Load SAE hyperparams ──
    sae_dir = os.path.dirname(os.path.dirname(args.sae_weights_path))
    hyp_path = os.path.join(sae_dir, "hyperparams.json")
    with open(hyp_path) as f:
        hyp = json.load(f)

    threshold_raw = hyp["jump_relu_threshold"]
    activation_norm = hyp["dataset_average_activation_norm"]["in"]
    d_model = hyp["d_model"]
    norm_scale = (d_model ** 0.5) / activation_norm
    threshold_scaled = threshold_raw * norm_scale

    print(f"=== SAE Hyperparams ===")
    print(f"  d_model:          {d_model}")
    print(f"  d_sae:            {hyp['d_sae']}")
    print(f"  activation_norm:  {activation_norm}")
    print(f"  norm_scale:       {norm_scale:.4f}")
    print(f"  threshold (raw):  {threshold_raw}")
    print(f"  threshold*scale:  {threshold_scaled:.6f}")
    print()

    # ── Load SAE weights ──
    with safe_open(args.sae_weights_path, framework="pt") as f:
        keys = list(f.keys())
        print(f"Checkpoint keys: {keys}")
        w_enc = f.get_tensor("encoder.weight")   # [d_sae, d_model]
        b_enc = f.get_tensor("encoder.bias")      # [d_sae]

    print(f"\n=== Encoder Bias (b_enc) ===")
    print(f"  shape: {b_enc.shape}, dtype: {b_enc.dtype}")
    b_enc_f = b_enc.float()
    print(f"  min:   {b_enc_f.min():.6f}")
    print(f"  max:   {b_enc_f.max():.6f}")
    print(f"  mean:  {b_enc_f.mean():.6f}")
    print(f"  std:   {b_enc_f.std():.6f}")
    print(f"  >0:    {(b_enc_f > 0).sum()}/{b_enc_f.numel()}")

    print(f"\n=== Encoder Weight (W_enc) ===")
    print(f"  shape: {w_enc.shape}")
    row_norms = w_enc.float().norm(dim=1)  # norm of each [d_model] row
    print(f"  row norms: min={row_norms.min():.4f}, max={row_norms.max():.4f}, mean={row_norms.mean():.4f}")

    # ── Check L0 with a few real hidden states ──
    print(f"\n=== Loading model for L0 check ({args.num_samples} samples) ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Sparse autoencoders decompose neural network activations into interpretable features.",
        "Information retrieval systems use inverted indexes for efficient search.",
        "The Eiffel Tower is located in Paris, France.",
        "Machine learning models require large amounts of training data.",
        "Climate change is one of the most pressing issues of our time.",
        "Python is a popular programming language for data science.",
        "The mitochondria is the powerhouse of the cell.",
    ][:args.num_samples]

    inputs = tokenizer(sample_texts, padding=True, truncation=True,
                       max_length=128, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        output_hidden_states=True)
        # Layer 0 residual: after embedding + first transformer block
        # hook_point_in: blocks.0.hook_resid_post = hidden_states[1]
        hidden = outputs.hidden_states[1]  # [batch, seq, d_model]
        print(f"  Hidden states shape: {hidden.shape}")
        norms = hidden.float().norm(dim=-1)
        print(f"  Hidden state L2 norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")

    # ── Compute encoder output and check L0 ──
    sae = torch.nn.Linear(d_model, hyp["d_sae"], dtype=torch.bfloat16, device=hidden.device)
    with torch.no_grad():
        sae.weight.copy_(w_enc.to(hidden.device))
        sae.bias.copy_(b_enc.to(hidden.device))

    # Flatten to [total_tokens, d_model], mask padding
    mask = inputs["attention_mask"].bool()  # [batch, seq]
    hidden_flat = hidden[mask]  # [num_real_tokens, d_model]
    print(f"\n  Real tokens (excl padding): {hidden_flat.shape[0]}")

    with torch.no_grad():
        # Without normalization
        h_raw = sae(hidden_flat)  # [tokens, d_sae]
        # With normalization
        h_norm = sae(hidden_flat * norm_scale)

    print(f"\n=== Encoder output (h = x @ W_enc^T + b_enc) ===")
    print(f"  WITHOUT norm_scale:")
    h_raw_f = h_raw.float()
    print(f"    stats: min={h_raw_f.min():.4f}, max={h_raw_f.max():.4f}, mean={h_raw_f.mean():.6f}, std={h_raw_f.std():.4f}")
    l0_raw_orig = (h_raw_f > threshold_raw).float().sum(dim=-1)
    l0_raw_scaled = (h_raw_f > threshold_scaled).float().sum(dim=-1)
    print(f"    L0 (threshold={threshold_raw}):           mean={l0_raw_orig.mean():.1f}, std={l0_raw_orig.std():.1f}")
    print(f"    L0 (threshold={threshold_scaled:.4f}):  mean={l0_raw_scaled.mean():.1f}, std={l0_raw_scaled.std():.1f}")

    print(f"\n  WITH norm_scale ({norm_scale:.4f}):")
    h_norm_f = h_norm.float()
    print(f"    stats: min={h_norm_f.min():.4f}, max={h_norm_f.max():.4f}, mean={h_norm_f.mean():.6f}, std={h_norm_f.std():.4f}")
    l0_norm_orig = (h_norm_f > threshold_raw).float().sum(dim=-1)
    l0_norm_scaled = (h_norm_f > threshold_scaled).float().sum(dim=-1)
    print(f"    L0 (threshold={threshold_raw}):           mean={l0_norm_orig.mean():.1f}, std={l0_norm_orig.std():.1f}")
    print(f"    L0 (threshold={threshold_scaled:.4f}):  mean={l0_norm_scaled.mean():.1f}, std={l0_norm_scaled.std():.1f}")

    print(f"\n=== Summary ===")
    print(f"  Paper expects L0 ~ 100 per token (SAEs with L0 closest to 100)")
    print(f"  Current code: norm_scale=YES, threshold={threshold_raw} => L0={l0_norm_orig.mean():.1f}")
    print(f"  Alt 1: norm_scale=YES, threshold={threshold_scaled:.4f} => L0={l0_norm_scaled.mean():.1f}")
    print(f"  Alt 2: norm_scale=NO,  threshold={threshold_raw} => L0={l0_raw_orig.mean():.1f}")
    print(f"  Alt 3: norm_scale=NO,  threshold={threshold_scaled:.4f} => L0={l0_raw_scaled.mean():.1f}")
    print(f"\n  Whichever gives L0 closest to ~100 is the correct combination.")


if __name__ == "__main__":
    main()

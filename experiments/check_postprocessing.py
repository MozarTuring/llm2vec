"""Check if the Llama Scope SAE checkpoint has been post-processed.

Post-processing (paper Section 3.3.4) does two things:
  Step 1: b_enc /= S_in  (fold input normalization into bias)
  Step 2: Normalize decoder columns to unit norm, rescale encoder accordingly

If post-processed:
  - Decoder column norms should all be ~1.0
  - b_enc should be very small (divided by S_in ≈ 59)
  - SAE expects RAW input (no normalization)

If NOT post-processed:
  - Decoder column norms will vary
  - b_enc retains training-scale values
  - SAE expects NORMALIZED input (multiply by S_in)
"""
import json
import os
import sys
import torch
from safetensors.torch import safe_open

sae_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.environ.get("PKQ_DATA_DIR", "."), "Llama3_1-8B-Base-L0R-8x"
)
ckpt_path = os.path.join(sae_dir, "checkpoints", "final.safetensors")
hp_path = os.path.join(sae_dir, "hyperparams.json")

with open(hp_path) as f:
    hp = json.load(f)

d_model = hp["d_model"]
activation_norm = hp["dataset_average_activation_norm"]["in"]
S_in = (d_model ** 0.5) / activation_norm

print(f"S_in (norm_scale) = sqrt({d_model}) / {activation_norm} = {S_in:.4f}")
print(f"norm_activation = {hp.get('norm_activation', 'unknown')}")
print()

with safe_open(ckpt_path, framework="pt") as f:
    W_enc = f.get_tensor("encoder.weight")   # (d_sae, d_model)
    b_enc = f.get_tensor("encoder.bias")      # (d_sae,)
    W_dec = f.get_tensor("decoder.weight")    # (d_model, d_sae)
    b_dec = f.get_tensor("decoder.bias")      # (d_model,)

print(f"W_enc shape: {W_enc.shape}")
print(f"W_dec shape: {W_dec.shape}")
print()

# Check decoder column norms (if post-processed step 2, should be ~1.0)
dec_col_norms = W_dec.float().norm(dim=0)  # norm of each column (d_sae columns)
print(f"Decoder column norms (if post-processed, should all be ~1.0):")
print(f"  mean={dec_col_norms.mean():.6f}  std={dec_col_norms.std():.6f}")
print(f"  min={dec_col_norms.min():.6f}  max={dec_col_norms.max():.6f}")
print()

# Check encoder bias (if post-processed step 1, should be divided by S_in ≈ 59)
print(f"Encoder bias stats:")
print(f"  mean={b_enc.float().mean():.6f}  std={b_enc.float().std():.6f}")
print(f"  min={b_enc.float().min():.6f}  max={b_enc.float().max():.6f}")
print(f"  If post-processed, these would be ~{1/S_in:.4f}x of training values")
print()

# Check decoder bias
print(f"Decoder bias stats:")
print(f"  mean={b_dec.float().mean():.6f}  std={b_dec.float().std():.6f}")
print()

# Summary
if dec_col_norms.std() < 0.01 and (dec_col_norms - 1.0).abs().max() < 0.05:
    print("CONCLUSION: Checkpoint IS post-processed (decoder columns have unit norm)")
    print("  -> Do NOT apply sae_norm_scale to input")
else:
    print("CONCLUSION: Checkpoint is NOT post-processed (decoder columns vary)")
    print("  -> DO apply sae_norm_scale to input")

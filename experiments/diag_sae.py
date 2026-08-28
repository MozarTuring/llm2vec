"""Diagnostic: SAE activation statistics at layer 26 after MNTP merge."""
import torch
import sys, os
from transformers import AutoConfig, AutoTokenizer
from safetensors.torch import safe_open
from peft import PeftModel
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm2vec.models import LlamaBiModel

device = "cuda:1"
dtype = torch.bfloat16

# Load model
config = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
tokenizer.pad_token = tokenizer.eos_token

model = LlamaBiModel.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B",
    config=config, torch_dtype=dtype, attn_implementation="sdpa",
)

# Merge MNTP
model = PeftModel.from_pretrained(model, "output/mntp/Meta-Llama-3.1-8B-msmarco")
model = model.merge_and_unload()
print("MNTP merged")

# Truncate to layer 26
model.layers = model.layers[:27]
model.norm = nn.Identity()
model.to(device)
model.eval()
print("Truncated to 27 layers")

# Load SAE encoder
sae_path = "../remote_data/llm2vec/Llama3_1-8B-Base-L26R-8x/checkpoints/final.safetensors"
with safe_open(sae_path, framework="pt") as f:
    enc_w = f.get_tensor("encoder.weight")
    enc_b = f.get_tensor("encoder.bias")
    dec_b = f.get_tensor("decoder.bias")

sae = nn.Linear(4096, 32768, bias=True)
with torch.no_grad():
    sae.weight.copy_(enc_w)
    sae.bias.copy_(enc_b)
sae.to(dtype).to(device)
sae.requires_grad_(False)
print("SAE loaded")

# Sample texts
texts = [
    "What is the capital of France?",
    "Paris is the capital and most populous city of France.",
    "The Eiffel Tower is a wrought-iron lattice tower in Paris.",
    "Berlin is the capital of Germany.",
    "Machine learning is a subset of artificial intelligence.",
    "The quick brown fox jumps over the lazy dog.",
    "Retrieval augmented generation combines search with LLMs.",
    "Sparse representations enable efficient inverted index search.",
]

inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    hidden = outputs[0]

    print(f"\n=== Hidden state statistics ===")
    print(f"Shape: {hidden.shape}")
    norms = hidden.float().norm(dim=-1)
    print(f"L2 norm: mean={norms.mean():.1f}, std={norms.std():.1f}, min={norms.min():.1f}, max={norms.max():.1f}")
    print(f"sqrt(D) = {4096**0.5:.1f}")

    # SAE pre-activations
    pre_act = sae(hidden).float()
    print(f"\n=== SAE pre-activation (W_enc @ h + b_enc) ===")
    print(f"Mean: {pre_act.mean():.4f}, Std: {pre_act.std():.4f}")
    print(f"Min: {pre_act.min():.4f}, Max: {pre_act.max():.4f}")
    frac_pos = (pre_act > 0).float().mean()
    print(f"Fraction positive: {frac_pos:.4f} ({frac_pos*32768:.0f} / 32768 features)")

    # After ReLU
    post_relu = torch.relu(pre_act)
    pos_vals = post_relu[post_relu > 0]
    print(f"\n=== After ReLU ===")
    print(f"Positive values: mean={pos_vals.mean():.4f}, std={pos_vals.std():.4f}")

    # After log(1 + ReLU)
    splade = torch.log(1 + post_relu)

    # Max-pooled
    pooled, _ = splade.max(dim=1)
    nnz = (pooled > 0).float().sum(dim=1)
    print(f"\n=== Max-pooled representations ===")
    print(f"Non-zero features per sample: mean={nnz.mean():.0f}, min={nnz.min():.0f}, max={nnz.max():.0f}")
    pooled_pos = pooled[pooled > 0]
    print(f"Positive values: mean={pooled_pos.mean():.4f}, max={pooled_pos.max():.4f}")

    # Dot products
    dots = pooled @ pooled.T
    print(f"\n=== Dot products (scores) ===")
    for i in range(8):
        print(f"  [{' '.join(f'{dots[i,j]:8.1f}' for j in range(8))}]")
    offdiag = dots[~torch.eye(8, dtype=bool, device=device)]
    print(f"Cross-scores: mean={offdiag.mean():.1f}, std={offdiag.std():.1f}")
    print(f"score/80 = {offdiag.mean()/80:.1f}  (softmax input at tau=80)")

    # With SqrtDNorm
    print(f"\n=== With SqrtDNorm before SAE ===")
    D = hidden.shape[-1]
    h_norm = hidden.float() * (D**0.5) / hidden.float().norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
    pre_act_n = sae(h_norm.to(dtype)).float()
    frac_n = (pre_act_n > 0).float().mean()
    print(f"Fraction positive: {frac_n:.4f} ({frac_n*32768:.0f} / 32768 features)")
    splade_n = torch.log(1 + torch.relu(pre_act_n))
    pooled_n, _ = splade_n.max(dim=1)
    nnz_n = (pooled_n > 0).float().sum(dim=1)
    print(f"Non-zero features per sample: mean={nnz_n.mean():.0f}")
    dots_n = pooled_n @ pooled_n.T
    offdiag_n = dots_n[~torch.eye(8, dtype=bool, device=device)]
    print(f"Cross-scores: mean={offdiag_n.mean():.1f}")
    print(f"score/80 = {offdiag_n.mean()/80:.1f}")

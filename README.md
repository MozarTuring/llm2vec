## Overview

This repo reproduces paper https://openreview.net/pdf?id=TuFjICawSc which combines a truncated LLM backbone (Llama 3.1 8B) with a frozen Sparse Autoencoder (SAE). The model learns to produce sparse query/document representations whose dot product scores approximate a cross-encoder reranker's relevance judgments.

**Stage 0 — MNTP Finetuning:** Convert the causal LLM (Llama 3.1 8B) into a bidirectional encoder using LoRA with masked next-token prediction on MS MARCO passages.

**Stage 1 — Hard Negative Mining:** Use SPLADE to retrieve the most similar non-positive passages for each MS MARCO query.

**Stage 2 — Cross-Encoder Reranking:** Score all positives and hard negatives with a DeBERTa v3 cross-encoder to produce high-quality relevance labels.

**Stage 3 — Layerwise LoRA Training:** Merge the MNTP LoRA into the base model, truncate to layer 0, attach a frozen SAE, and train new LoRA adapters with KL divergence + FLOPS sparsity loss against the cross-encoder scores.

**Stage 4 — MTEB Evaluation:** Evaluate the trained sparse retriever on all 26 MTEB English v2 retrieval tasks.

**environment setup:** Clone this repo and cd into it, then refer to jwm_configs/env.sh

## Stage 0: MNTP Finetuning (Masked Next Token Prediction)
**Run:**
```bash
python experiments/run_mntp.py train_configs/mntp/MetaLlama3.1-msmarco.json
```
**Purpose:** Convert a causal (left-to-right) LLM into a **bidirectional** encoder. Standard LLMs use causal attention masks — each token can only attend to previous tokens. For retrieval, we need each token to attend to the full sequence. MNTP finetunes the model with bidirectional attention using a masked language modeling objective, teaching it to use future context.

**How it works:**

1. **Load Llama 3.1 8B** using LLM2Vec's bidirectional model class (`LlamaBiForMNTP`), which replaces the causal attention mask with a full (bidirectional) attention mask.
2. **Apply LoRA** (rank 16, alpha 32) to all attention and MLP projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). Only LoRA parameters are trained; base weights are frozen.
3. **Mask tokens** — randomly mask 20% of tokens by replacing them with a blank token (`_`). The model predicts the next token at each masked position (next token prediction, not same-position prediction like BERT).
4. **Train on MS MARCO passages** (`Tevatron/msmarco-passage-corpus`) for 10,000 steps.

### Masking Strategy

Unlike BERT's `[MASK]` token, LLMs don't have a native mask token. The script supports three strategies:
- `blank` (default): use `_` as the mask token
- `eos`: use the EOS token
- `mask`: add a new `<mask>` token to the vocabulary

A variant `DataCollatorForLanguageModelingWithFullMasking` replaces 100% of selected tokens with the mask token (no random or identity substitution), controlled by `data_collator_type`.

**Output:** LoRA adapter saved to `output/mntp/Meta-Llama-3.1-8B-msmarco/`



## Stage 1: Hard Negative Mining
**Run:**
```bash
python experiments/hard_negatives.py
```
**Purpose:** For each MS MARCO query, find passages that are highly similar (by a sparse retriever) but are NOT annotated positives. These "hard negatives" are the most useful training signals — the model must learn to distinguish them from true positives.

**How it works:**

1. **Encode passages to disk** — use SPLADE (`naver/splade-cocondenser-selfdistil`) to encode all passages in chunks of 50K, saved as `.pt` files to `passage_embeddings_cache/`.
2. **Mine negatives** — for each batch of queries:
   - Encode queries with SPLADE
   - Load passage chunks from disk one at a time, compute similarity on GPU
   - Maintain a running top-30 across all chunks
   - Filter out positive passages, keep top-10 non-positive passages as hard negatives

**Output:** `msmarco_hard_negatives.json`


## Stage 2: Cross-Encoder Reranking

**Run:**
```bash
python experiments/reranker.py msmarco_hard_negatives.json --output reranked_hard_negatives.json --top_k 8
```

**Purpose:** Score every positive and hard negative passage with a cross-encoder to get high-quality relevance scores. These scores become the training targets (ground truth distribution).

**How it works:**

1. Load `msmarco_hard_negatives.json`
2. For each query, form (query, passage) pairs for all positives and top-8 hard negatives
3. Score each pair with `naver/trecdl22-crossencoder-debertav3` (DeBERTa v3 cross-encoder)
4. Replace the original SPLADE retrieval scores with `reranker_score`
5. Sort negatives by reranker score (descending)

**Output:** `reranked_hard_negatives.json`



## Stage 3: Layerwise LoRA Training
**Run:**
```bash
python experiments/run_layerwise_finetune.py train_configs/layerwise/MetaLlama3.1-mntp-layerwise.json
```

**Purpose:** Train a sparse retrieval model that produces sparse vectors whose dot product similarity approximates the cross-encoder's relevance scores.

### Architecture

The forward pass proceeds through these stages in order:

1. **Embedding** — token embeddings from the LLM's embedding layer.
2. **Layer 0 + RMSNorm** — a single transformer layer with trainable LoRA adapters (rank 64, alpha 128).
3. **SqrtDNorm** — normalization: `v * sqrt(d) / ||v||₂`, where d is the hidden dimension.
4. **SAE Linear (4096 → 32768)** — a frozen pretrained sparse autoencoder encoder layer.
5. **Sparse activation** — `log(1 + ReLU(x))`, producing non-negative sparse values.
6. **Max pooling (dim=1)** — aggregate over the sequence dimension, yielding a (batch, 32768) sparse embedding.

**Key components:**

- **Backbone:** Llama 3.1 8B, truncated to only layer 0 (embedding + 1 transformer layer). Prior MNTP LoRA is merged before truncation.
- **LoRA:** New LoRA adapters (rank 64, alpha 128) applied to the single kept layer. Only these are trained.
- **SAE:** Pretrained sparse autoencoder (`Llama3_1-8B-Base-L0R-8x`), maps 4096-dim hidden states to 32768-dim sparse vectors. Frozen during training.
- **Activation:** `log(1 + ReLU(y))` produces non-negative sparse activations.
- **Pooling:** Max over the sequence dimension — each dimension keeps its strongest activation across all tokens.

### Loss Function

Each training sample has 1 query + 1 positive + 8 hard negatives (10 texts total). All are encoded through the same model.

```
L = L_KL + λ_q · FLOPS_q + λ_d · FLOPS_d
```

**Output:** Checkpoint saved to `output/layerwise/Meta-Llama-3.1-8B-msmarco-mntp-L0/`


## Stage 4: MTEB Evaluation
**Run:**
```bash
python experiments/mteb_eval_layerwise.py --model_name_or_path meta-llama/Meta-Llama-3.1-8B --peft_model_name_or_path output/mntp/Meta-Llama-3.1-8B-msmarco --sae_weights_path /home/jinma/project_remote_jwm/remote_data/splare/Llama3_1-8B-Base-L0R-8x/checkpoints/final.safetensors --trained_checkpoint_path output/layerwise/Meta-Llama-3.1-8B-msmarco-mntp-L0/checkpoint-3930 --lora_layers 0 --task_name SciFact --output_dir results --query_top_k 40 --doc_top_k 400 --hard_negatives_file reranked_hard_negatives.json --num_hard_negatives 8 --temperature 80 --lambda_q 1e-4 --lambda_d 1e-4 --max_seq_length 128 --max_length 1024
```

**Purpose:** Evaluate the trained sparse retrieval model on the MTEB English v2 retrieval benchmark (26 tasks).

### Model Loading (mirrors training)

1. Load base Llama 3.1 8B
2. Load and merge MNTP LoRA
3. Truncate to layer 0
4. Load trained LoRA checkpoint (not merged — applied during forward pass)
5. Load frozen SAE from pretrained weights

### Inference

- Max sequence length: 1024 (longer than training's 128 to handle longer evaluation documents)
- Top-k sparsification: keep only top-40 non-zero values for queries, top-400 for documents
- Similarity: dot product between sparse query and document vectors
- Embeddings cached to disk per task, separated by query/document

### Verification Step

Before evaluation, computes the training loss on a batch of training data to verify the loaded model matches what was trained. Prints KL loss, FLOPS loss, total loss, and score statistics.

### MTEB English v2 Retrieval Tasks (26)

ArguAna, CQADupstack (12 subtopics), ClimateFEVER, DBPedia, FEVER, FiQA2018, HotpotQA, MSMARCO, NFCorpus, NQ, QuoraRetrieval, SCIDOCS, SciFact, Touche2020, TRECCOVID

**Output:** Results saved to `results/custom__layerwise-sparse-encoder/0.0.1/{TaskName}.json`


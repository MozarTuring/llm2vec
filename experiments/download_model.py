#!/usr/bin/env python
"""Download model, tokenizer, and dataset required for MNTP training.

Usage:
    python experiments/download_model.py train_configs/mntp/MetaLlama3.1-msmarco.json
    python experiments/download_model.py --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct
    python experiments/download_model.py --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
        --dataset_name Tevatron/msmarco-passage-corpus --cache_dir ./cache
"""

import argparse
import json
import os
import sys

from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer
from huggingface_hub import snapshot_download


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download model, tokenizer, and dataset for MNTP training."
    )
    parser.add_argument(
        "config_file",
        nargs="?",
        default=None,
        help="Path to a JSON config file (same format as run_mntp.py configs). "
        "CLI arguments override values from the config file.",
    )
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token for gated models (e.g. Llama).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow custom code from the model repo.",
    )
    parser.add_argument(
        "--skip_model", action="store_true", help="Skip downloading the model."
    )
    parser.add_argument(
        "--skip_dataset", action="store_true", help="Skip downloading the dataset."
    )

    args = parser.parse_args()

    if args.config_file is not None:
        with open(args.config_file) as f:
            config = json.load(f)
        if args.model_name_or_path is None:
            args.model_name_or_path = config.get("model_name_or_path")
        if args.dataset_name is None:
            args.dataset_name = config.get("dataset_name")
        if args.dataset_config_name is None:
            args.dataset_config_name = config.get("dataset_config_name")
        if args.cache_dir is None:
            args.cache_dir = config.get("cache_dir")
        if args.token is None:
            args.token = config.get("token")

    if args.model_name_or_path is None and not args.skip_model:
        parser.error(
            "Provide --model_name_or_path or a config file with model_name_or_path."
        )

    return args


def download_model(model_name_or_path, cache_dir=None, token=None, trust_remote_code=False):
    print(f"\n{'='*60}")
    print(f"Downloading model weights: {model_name_or_path}")
    print(f"{'='*60}")
    path = snapshot_download(
        model_name_or_path,
        cache_dir=cache_dir,
        token=token,
    )
    print(f"Model weights cached at: {path}")

    print(f"\nDownloading config: {model_name_or_path}")
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        token=token,
        trust_remote_code=trust_remote_code,
    )
    print(f"Config type: {config.__class__.__name__}")

    print(f"\nDownloading tokenizer: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        token=token,
        trust_remote_code=trust_remote_code,
    )
    print(f"Tokenizer type: {tokenizer.__class__.__name__}")
    print(f"Vocab size: {len(tokenizer)}")

    return config, tokenizer


def download_dataset(dataset_name, dataset_config_name=None, cache_dir=None, token=None):
    print(f"\n{'='*60}")
    print(f"Downloading dataset: {dataset_name}", end="")
    if dataset_config_name:
        print(f" (config: {dataset_config_name})")
    else:
        print()
    print(f"{'='*60}")

    ds = load_dataset(
        dataset_name,
        dataset_config_name,
        cache_dir=cache_dir,
        token=token,
    )
    print(f"Dataset splits: {list(ds.keys())}")
    for split, data in ds.items():
        print(f"  {split}: {len(data)} rows, columns: {data.column_names}")

    return ds


def main():
    args = parse_args()

    if not args.skip_model:
        download_model(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            token=args.token,
            trust_remote_code=args.trust_remote_code,
        )

    if not args.skip_dataset and args.dataset_name:
        download_dataset(
            args.dataset_name,
            dataset_config_name=args.dataset_config_name,
            cache_dir=args.cache_dir,
            token=args.token,
        )
    elif not args.skip_dataset:
        print("\nNo dataset_name specified, skipping dataset download.")

    print(f"\n{'='*60}")
    print("Done.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

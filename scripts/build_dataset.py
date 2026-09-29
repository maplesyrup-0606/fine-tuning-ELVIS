#!/usr/bin/env python3
"""
Build ms-swift chat-format JSONL for fine-tuning.

For each pattern in data/continuity/rules.json, emits 12 rows:
  6 positive samples (images 00003..00008 from train/<pattern>/positive/)
  6 negative samples (images 00003..00008 from train/<pattern>/negative/)

Each row's prompt matches ELVIS Mode 1 Stage 2 (evaluate_llm) verbatim so the
model sees the same input format during training as at evaluation:

    Using the following reasoning rules: <rule>. Classify this image as
    Positive or Negative. Only answer with positive or negative.

Response: lowercase "positive" or "negative".

Total rows: 432 patterns * 2 sides * 6 samples = 5,184. Runs in seconds
on the login node (no GPU, no model loading).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


STAGE2_PROMPT_TEMPLATE = (
    "Using the following reasoning rules: {rule}. "
    "Classify this image as Positive or Negative. "
    "Only answer with positive or negative."
)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--rules", default=None,
                        help="override rules.cache_path from config")
    parser.add_argument("--output", default=None,
                        help="override training.dataset_path from config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    principle = cfg["data"]["principle"]
    n_stage1 = cfg["data"]["stage1_examples_per_side"]
    n_train = cfg["data"]["finetuning_samples_per_side"]
    rules_path = args.rules or cfg["rules"]["cache_path"]
    output_path = args.output or cfg["training"]["dataset_path"]

    elvis_data = os.environ.get(cfg["paths"]["elvis_data_env"])
    if not elvis_data:
        sys.exit(f"ERROR: env var {cfg['paths']['elvis_data_env']} is not set")
    train_root = Path(elvis_data) / principle / "train"

    with open(rules_path) as f:
        rules_payload = json.load(f)
    rules = rules_payload["rules"]
    print(f"Loaded {len(rules)} rules from {rules_path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    missing_images = []
    with open(output_path, "w") as out:
        for pattern, rule in rules.items():
            prompt = STAGE2_PROMPT_TEMPLATE.format(rule=rule.strip())
            for side in ("positive", "negative"):
                for i in range(n_stage1, n_stage1 + n_train):
                    img_path = train_root / pattern / side / f"{i:05d}.png"
                    if not img_path.exists():
                        missing_images.append(str(img_path))
                        continue
                    row = {
                        "messages": [
                            {"role": "user", "content": "<image>" + prompt},
                            {"role": "assistant", "content": side},
                        ],
                        "images": [str(img_path.resolve())],
                    }
                    out.write(json.dumps(row) + "\n")
                    rows_written += 1

    expected = len(rules) * 2 * n_train
    print(f"Wrote {rows_written} rows to {output_path} (expected {expected})")
    if missing_images:
        print(f"WARNING: {len(missing_images)} images not found. First 5:")
        for p in missing_images[:5]:
            print(f"  {p}")
        sys.exit(1)


if __name__ == "__main__":
    main()

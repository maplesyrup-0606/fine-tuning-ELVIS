#!/usr/bin/env python3
"""
Pre-generate Stage 1 rules for all continuity patterns with base Qwen3-VL-4B.

For each pattern, feeds the first N positive + N negative training images (Stage 1
demonstration examples, matching ELVIS Mode 1) through the base model via ELVIS's
`infer_logic_rules`, and caches the returned rule text to data/rules.json.

Downstream: build_dataset.py consumes rules.json to build the ms-swift training
JSONL. Rules are pre-generated once because ms-swift can't run arbitrary Python
between training steps; distribution drift is expected to be small for LoRA rank 4.

Runtime: ~15-30 min on MIG 3g.40gb for 432 patterns. GPU required.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
from tqdm import tqdm


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default=None,
                        help="override rules.cache_path from config")
    parser.add_argument("--patterns", default=None,
                        help="comma-separated substrings; process only matching patterns (debug)")
    parser.add_argument("--dry-run", action="store_true",
                        help="list patterns without loading the model")
    args = parser.parse_args()

    cfg = load_config(args.config)
    principle = cfg["data"]["principle"]
    n_stage1 = cfg["data"]["stage1_examples_per_side"]
    output_path = args.output or cfg["rules"]["cache_path"]
    base_id = cfg["model"]["base_id"]

    elvis_root = os.environ.get(cfg["paths"]["elvis_root_env"])
    if not elvis_root:
        sys.exit(f"ERROR: env var {cfg['paths']['elvis_root_env']} is not set")
    sys.path.insert(0, elvis_root)
    from scripts.baseline_models.qwen import load_qwen_model, infer_logic_rules, load_images

    elvis_data = os.environ.get(cfg["paths"]["elvis_data_env"])
    if not elvis_data:
        sys.exit(f"ERROR: env var {cfg['paths']['elvis_data_env']} is not set")
    train_root = Path(elvis_data) / principle / "train"
    if not train_root.is_dir():
        sys.exit(f"ERROR: train dir not found at {train_root}")

    all_patterns = sorted(p.name for p in train_root.iterdir() if p.is_dir())
    if args.patterns:
        wanted = tuple(args.patterns.split(","))
        patterns = [p for p in all_patterns if any(w in p for w in wanted)]
    else:
        patterns = all_patterns
    print(f"Discovered {len(all_patterns)} patterns in {train_root}; processing {len(patterns)}.")

    if args.dry_run:
        for p in patterns:
            print(p)
        return

    print(f"Loading base model {base_id} (this will take a minute)...")
    model, processor = load_qwen_model(base_id)

    rules = {}
    for pattern in tqdm(patterns, desc="rules"):
        pos_dir = train_root / pattern / "positive"
        neg_dir = train_root / pattern / "negative"
        pos_imgs = load_images(str(pos_dir), img_size=224, num_samples=n_stage1)
        neg_imgs = load_images(str(neg_dir), img_size=224, num_samples=n_stage1)
        if len(pos_imgs) < n_stage1 or len(neg_imgs) < n_stage1:
            sys.exit(f"ERROR: pattern {pattern} has fewer than {n_stage1} images per side")
        rules[pattern] = infer_logic_rules(model, processor, pos_imgs, neg_imgs, principle)

    payload = {
        "meta": {
            "source_model": base_id,
            "principle": principle,
            "stage1_examples_per_side": n_stage1,
            "num_patterns": len(rules),
            "elvis_data_path": str(train_root),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "rules": rules,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {len(rules)} rules to {output_path}")


if __name__ == "__main__":
    main()

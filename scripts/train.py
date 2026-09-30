#!/usr/bin/env python3
"""
LoRA fine-tune Qwen3-VL-4B on ELVIS continuity via ms-swift.

Reads every hyperparameter from config.yaml. Wraps `swift sft` via subprocess
because ms-swift's Python API changes between minor versions but the CLI is
stable across releases.

Trains for `training.max_epochs` in one process, saving one LoRA adapter per
epoch. The config's early-stopping rules (target 0.67, regression patience 1)
are applied POST-HOC by scripts/eval_continuity.py at checkpoint-selection
time, not during training. Rationale: 3 epochs × 5,184 samples on a 3g.40gb
MIG slice is <2h; in-loop early stop would require a custom TrainerCallback
that's fragile against ms-swift version drift.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_swift_args(cfg, variant, output_dir, resume_from=None):
    """Convert config.yaml + variant selection → `swift sft` CLI arguments.

    variant: one of the keys under cfg["variants"] (llm, projector, both).
    """
    train_cfg = cfg["training"]
    lora_cfg = cfg["lora"]
    variant_cfg = cfg["variants"][variant]

    # 224*224 = 50176. Matches rule-gen (generate_rules.py, img_size=224) and
    # eval (Mode 1 in ELVIS uses 224 by default). All ELVIS-generated images
    # are exactly 224*224 so this is a no-op cap in practice, but locks
    # behavior against future data-gen changes.
    max_pixels = cfg["data"]["regeneration"]["img_size"] ** 2

    args = [
        "swift", "sft",
        "--model", cfg["model"]["base_id"],
        "--torch_dtype", "bfloat16",
        "--tuner_type", "lora",
        "--dataset", train_cfg["dataset_path"],
        "--output_dir", str(output_dir),

        # LoRA — shared hyperparameters
        "--lora_rank", str(lora_cfg["rank"]),
        "--lora_alpha", str(lora_cfg["alpha"]),
        "--lora_dropout", str(lora_cfg["dropout"]),

        # Per-variant routing. target_modules is a List[str] in ms-swift
        # (nargs='+'), so unpack the list as separate CLI values.
        "--target_modules", *variant_cfg["target_modules"],
        "--freeze_llm", str(variant_cfg["freeze_llm"]).lower(),
        "--freeze_vit", str(variant_cfg["freeze_vit"]).lower(),
        "--freeze_aligner", str(variant_cfg["freeze_aligner"]).lower(),

        # Training
        "--num_train_epochs", str(train_cfg["max_epochs"]),
        "--per_device_train_batch_size", str(train_cfg["per_device_batch_size"]),
        "--gradient_accumulation_steps", str(train_cfg["gradient_accumulation_steps"]),
        "--learning_rate", str(train_cfg["learning_rate"]),
        "--lr_scheduler_type", train_cfg["lr_scheduler"],
        "--warmup_ratio", str(train_cfg["warmup_ratio"]),

        # Checkpoint every epoch, keep all so eval_continuity.py can pick best
        "--save_strategy", "epoch",
        "--save_total_limit", str(train_cfg["max_epochs"]),
        "--logging_steps", "5",

        # Image budget — match rule-gen + eval resolution
        "--max_pixels", str(max_pixels),

        # Reproducibility
        "--seed", str(cfg["experiment"]["seed"]),

        # WandB
        "--report_to", "wandb",
    ]

    if resume_from:
        args.extend(["--resume_from_checkpoint", str(resume_from)])

    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--variant", required=True,
                        choices=["llm", "projector", "both"],
                        help="which LoRA target scope to train")
    parser.add_argument("--output-dir", default=None,
                        help="override results/checkpoints dir (default: <results>/checkpoints/<variant>)")
    parser.add_argument("--resume-from", default=None,
                        help="path to checkpoint to resume from")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the swift command without running")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.variant not in cfg.get("variants", {}):
        parser.error(f"variant {args.variant!r} not found in {args.config} variants: block")

    # One WandB project, three runs — comparable in the UI.
    os.environ.setdefault("WANDB_PROJECT", cfg["wandb"]["project"])
    os.environ.setdefault("WANDB_NAME", f"continuity-{args.variant}")

    results_env = cfg["paths"]["results_env"]
    results_root = Path(os.environ.get(results_env, "results"))
    output_dir = Path(args.output_dir) if args.output_dir else results_root / "checkpoints" / args.variant
    output_dir.mkdir(parents=True, exist_ok=True)

    swift_args = build_swift_args(cfg, args.variant, output_dir, args.resume_from)

    print("=" * 60)
    print(f"Variant: {args.variant}")
    print(f"Output: {output_dir}")
    print("Running:")
    print(" \\\n  ".join(swift_args))
    print("=" * 60)

    if args.dry_run:
        return 0

    return subprocess.run(swift_args).returncode


if __name__ == "__main__":
    sys.exit(main())

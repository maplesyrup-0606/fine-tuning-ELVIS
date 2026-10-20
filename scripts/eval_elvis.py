#!/usr/bin/env python3
"""
Evaluate a fine-tuned LoRA adapter on ELVIS Mode 1 across Gestalt principles.

Wraps ELVIS's own `_run_qwen_baseline` (via peft PeftModel loading), so
in-distribution accuracy (continuity) and negative-transfer to sibling
principles (proximity/similarity/closure/symmetry) are measured with the
exact same protocol as ELVIS's published baselines.

Usage:
    python scripts/eval_elvis.py \\
        --variant llm \\
        --checkpoint results/checkpoints/llm/v1-YYYYMMDD-.../checkpoint-486 \\
        --principles all

Requires env: ELVIS_ROOT, ELVIS_DATA.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml


ALL_PRINCIPLES = ["continuity", "proximity", "similarity", "closure", "symmetry"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--variant", required=True,
                        choices=["llm", "projector", "both", "base"],
                        help="'base' evaluates the un-fine-tuned model (no adapter) — "
                             "use to populate sibling-principle baseline numbers.")
    parser.add_argument("--checkpoint", default=None,
                        help="path to LoRA adapter dir (required unless --variant base)")
    parser.add_argument("--principles", nargs="+", default=["all"],
                        help="one or more of continuity/proximity/similarity/closure/symmetry, or 'all'")
    parser.add_argument("--img-num", type=int, default=3,
                        help="Stage 1 few-shot examples per side (must match training img_num)")
    parser.add_argument("--task-num", default="full",
                        help="how many pattern folders to eval (int) or 'full'")
    parser.add_argument("--start-num", type=int, default=0)
    parser.add_argument("--output-dir", default=None,
                        help="root output dir; per-principle results at <dir>/<variant>/<principle>/ "
                             "(default: results/<evaluation.results_subdir> from config)")
    parser.add_argument("--rubrics", action="store_true",
                        help="v3 rubric evaluation: skip Stage 1 and classify with the pattern's ground-truth "
                             "rubric from data/<principle>/rubrics.json (default output: "
                             "results/<evaluation.rubric_results_subdir>)")
    args = parser.parse_args()

    if args.principles == ["all"]:
        args.principles = ALL_PRINCIPLES

    unknown = set(args.principles) - set(ALL_PRINCIPLES)
    if unknown:
        parser.error(f"unknown principle(s): {sorted(unknown)}. Choose from {ALL_PRINCIPLES}")

    if args.variant != "base" and not args.checkpoint:
        parser.error("--checkpoint is required unless --variant base")
    if args.checkpoint and not Path(args.checkpoint).is_dir():
        parser.error(f"--checkpoint path is not a directory: {args.checkpoint}")

    elvis_root = os.environ.get("ELVIS_ROOT")
    elvis_data = os.environ.get("ELVIS_DATA")
    if not elvis_root:
        parser.error("ELVIS_ROOT env var must be set (import path for ELVIS package)")
    if not elvis_data:
        parser.error("ELVIS_DATA env var must be set (root dir containing per-principle subdirs)")
    if elvis_root not in sys.path:
        sys.path.insert(0, elvis_root)

    from scripts.baseline_models.qwen import _run_qwen_baseline

    cfg = yaml.safe_load(open(args.config))
    model_id = cfg["model"]["base_id"]
    img_size = cfg["data"]["regeneration"]["img_size"]
    rule_max_new_tokens = cfg["evaluation"]["rule_max_new_tokens"]
    # v3 checkpoints live under <results>/<v3.checkpoint_subdir>/<run>/<variant>/<vN-ts>/<checkpoint>.
    ckpt = Path(args.checkpoint) if args.checkpoint and args.variant != "base" else None
    v3_run = ckpt.parents[2].name if ckpt and cfg["v3"]["checkpoint_subdir"] in ckpt.parts else None
    label = f"{v3_run}-{args.variant}" if v3_run else args.variant     # e.g. all-both
    if args.output_dir is None:
        key = ("rubric_results_subdir" if args.rubrics else
               "v3_results_subdir" if v3_run else "results_subdir")
        results_root = Path(os.environ.get(cfg["paths"]["results_env"], "results"))
        args.output_dir = str(results_root / cfg["evaluation"][key])

    # Rubric mode: load every principle's rubrics up front and make sure each pattern
    # ELVIS will evaluate (the <principle>/train folders) has one — fail before loading the model.
    rubrics = {}
    if args.rubrics:
        for principle in args.principles:
            path = Path(f"data/{principle}/rubrics.json")
            if not path.exists():
                parser.error(f"{path} missing — run generate_programmatic_rules.py --principle {principle} --rubrics")
            table = json.load(open(path))["rubrics"]
            names = sorted(p.name for p in Path(elvis_data, principle, "train").iterdir() if p.is_dir())
            missing = [n for n in names if n not in table]
            if missing:
                parser.error(f"{principle}: {len(missing)} ELVIS patterns have no rubric, e.g. {missing[0]}")
            rubrics[principle] = table

    adapter_path = args.checkpoint if args.variant != "base" else None

    # Bucket every run into <variant>/<version-tag>/<checkpoint-name>/ so
    # multiple checkpoints of the same variant don't overwrite each other.
    # For --variant base (no adapter) we substitute a wall-clock timestamp.
    if adapter_path:
        ckpt_path = Path(adapter_path)
        run_key = Path(label) / ckpt_path.parent.name / ckpt_path.name
    else:
        run_key = Path(label) / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output_dir) / run_key
    run_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for principle in args.principles:
        data_path = os.path.join(elvis_data, principle)
        principle_out = run_dir / principle
        principle_out.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"Evaluating variant={args.variant} principle={principle}")
        print(f"Checkpoint: {adapter_path or '(base model, no adapter)'}")
        print(f"Run dir:    {run_dir}")
        print(f"{'='*60}")
        ret = _run_qwen_baseline(
            model_id=model_id,
            model_name=f"Qwen3-VL-4B-{args.variant}",
            data_path=data_path,
            img_size=img_size,
            principle=principle,
            batch_size=1,
            img_num=args.img_num,
            start_num=args.start_num,
            task_num=args.task_num,
            adapter_path=adapter_path,
            output_dir=str(principle_out),
            rule_max_new_tokens=rule_max_new_tokens,
            rules=rubrics.get(principle),
        )
        if ret is None:
            print(f"[warn] {principle}: no pattern folders found under {data_path}/train — skipping")
            continue
        acc, f1 = ret
        results[principle] = {"accuracy": acc, "f1": f1}

    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "variant": args.variant,
            "label": label,
            "checkpoint": adapter_path,
            "img_num": args.img_num,
            "task_num": args.task_num,
            "rule_max_new_tokens": rule_max_new_tokens,
            "rule_source": "rubric" if args.rubrics else "model",
            "principles": results,
        }, f, indent=2)

    print(f"\n{'='*60}\nSummary: {summary_path}\n{'='*60}")
    for p in args.principles:
        if p in results:
            r = results[p]
            print(f"  {p:12s}: acc={r['accuracy']:.2f}% f1={r['f1']:.4f}")
        else:
            print(f"  {p:12s}: (skipped)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Summarize finished training runs from ms-swift's logging.jsonl files.

For each run under results/<training.checkpoint_subdir>/<variant>/v*/ prints:
  - train loss / token_acc at the first and last logged step, max grad_norm
  - one line per validation pass (eval_loss, eval_token_acc, ...)
  - final step and checkpoint paths

Usage:
  python scripts/summarize_training.py                      # v2 runs (checkpoints_v2)
  python scripts/summarize_training.py --root results/checkpoints   # v1 runs

Note: with per-row loss_scale the losses are weighted, so they are not on the
same scale as v1's. token_acc is ~94% rule-text tokens, not classification.
"""

import argparse
import json
from pathlib import Path

import yaml


def fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--root", default=None,
                        help="default: results/<training.checkpoint_subdir> from config")
    args = parser.parse_args()

    if args.root is None:
        with open(args.config) as f:
            args.root = f"results/{yaml.safe_load(f)['training']['checkpoint_subdir']}"

    logs = sorted(Path(args.root).glob("*/v*/logging.jsonl"))
    if not logs:
        raise SystemExit(f"no logging.jsonl found under {args.root}/<variant>/v*/")

    for path in logs:
        rows = []
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        train = [r for r in rows if "loss" in r and "log_history" not in r]
        evals = [r for r in rows if "eval_loss" in r and "log_history" not in r]
        final = next((r for r in reversed(rows) if "log_history" in r or "last_model_checkpoint" in r), {})

        print(f"\n=== {path.parent.parent.name}  ({path.parent.name}) ===")
        if train:
            first, last = train[0], train[-1]
            grad = [r["grad_norm"] for r in train if isinstance(r.get("grad_norm"), (int, float))]
            print(f"  train loss      {fmt(first['loss'])} -> {fmt(last['loss'])}")
            if "token_acc" in first:
                print(f"  train token_acc {fmt(first['token_acc'])} -> {fmt(last.get('token_acc'))}")
            if grad:
                print(f"  max grad_norm   {fmt(max(grad))}")
        if evals:
            print("  validation:")
            for r in evals:
                fields = {k: v for k, v in r.items()
                          if k.startswith("eval_") and k not in ("eval_runtime", "eval_samples_per_second",
                                                                 "eval_steps_per_second")}
                where = f"epoch {fmt(r['epoch'])}" if "epoch" in r else f"step {r.get('global_step', '?')}"
                print(f"    {where:>14}: " + "  ".join(f"{k}={fmt(v)}" for k, v in fields.items()))
        else:
            print("  validation: none logged")
        print(f"  global_step     {final.get('global_step', train[-1].get('global_step') if train else '?')}")
        print(f"  last checkpoint {final.get('last_model_checkpoint')}")


if __name__ == "__main__":
    main()

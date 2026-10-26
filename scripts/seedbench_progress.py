#!/usr/bin/env python3
"""
One-line-per-model progress of running VLMEvalKit SEED-Bench-2 evaluations.

For each model key: answers saved so far (largest partial .pkl under
$VLMEVALKIT_DIR/outputs/<key>/), % of the dataset, runaway answers (> --long chars,
i.e. generation that ran to the token limit), whether a score file exists (done),
and the job's Slurm state.

Usage:
  python scripts/seedbench_progress.py                       # the five v3 SEED-Bench keys
  python scripts/seedbench_progress.py Qwen3-VL-4B-Instruct-t0 --dataset SEEDBench2
"""

import argparse
import os
import pickle
import subprocess
from pathlib import Path

KEYS = ["Qwen3-VL-4B-Instruct-t0",
        "Qwen3-VL-4B-Instruct-v3-continuity-llm-e3", 
        "Qwen3-VL-4B-Instruct-v3-continuity-both-e3",
        "Qwen3-VL-4B-Instruct-v3-all-llm-e3", 
        "Qwen3-VL-4B-Instruct-v3-all-both-e3",
        "Qwen3-VL-4B-Instruct-t0p",
        "Qwen3-VL-4B-Instruct-v3-continuity-llm-e3-t0p",
        "Qwen3-VL-4B-Instruct-v3-continuity-both-e3-t0p",
        "Qwen3-VL-4B-Instruct-v3-all-llm-e3-t0p",
        "Qwen3-VL-4B-Instruct-v3-all-both-e3-t0p"]

def dataset_size(dataset):
    tsv = Path(os.environ.get("LMUData", Path.home() / "LMUData")) / f"{dataset}.tsv"
    if not tsv.exists():
        return None
    with open(tsv) as f:
        return sum(1 for _ in f) - 1


def slurm_states():
    """{job name: 'RUNNING 1:23:45' / 'PENDING'} for the current user."""
    try:
        out = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j|%T|%M"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    return {j: f"{s} {t}" if s == "RUNNING" else s for j, s, t in (l.split("|") for l in out.split("\n") if l)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("keys", nargs="*", default=KEYS)
    parser.add_argument("--dataset", default="SEEDBench2")
    parser.add_argument("--long", type=int, default=10000, help="chars above which an answer counts as runaway")
    args = parser.parse_args()

    root = Path(os.environ.get("VLMEVALKIT_DIR", Path(os.environ.get("SCRATCH", "."), "VLMEvalKit"))) / "outputs"
    total = dataset_size(args.dataset)
    jobs = slurm_states()
    print(f"{'model':44s} {'answers':>14s} {'%':>6s} {'runaway':>8s}  {'done':5s} slurm")
    for key in args.keys:
        out = root / key
        best = {}
        for pkl in out.glob(f"**/*{args.dataset}*.pkl"):
            if "result" in pkl.name:          # scoring output, not partial predictions
                continue
            try:
                d = pickle.load(open(pkl, "rb"))
            except Exception as e:            # e.g. being written right now
                print(f"  [warn] cannot read {pkl}: {type(e).__name__}: {e}")
                continue
            if not isinstance(d, dict):
                print(f"  [warn] {pkl} holds a {type(d).__name__}, not a dict; skipped")
                continue
            if isinstance(d, dict) and len(d) > len(best):
                best = d
        n = len(best)
        runaway = sum(len(str(v)) > args.long for v in best.values())
        done = any(out.glob(f"**/*{args.dataset}*acc*")) if out.exists() else False
        pct = f"{100 * n / total:.1f}" if total else "?"
        short = key.replace("Qwen3-VL-4B-Instruct-", "")
        state = jobs.get(f"sb2-{short}", "-")
        print(f"{key:44s} {n:>6d}/{total or '?':<7} {pct:>6s} {runaway:>8d}  {'yes' if done else 'no':5s} {state}")


if __name__ == "__main__":
    main()

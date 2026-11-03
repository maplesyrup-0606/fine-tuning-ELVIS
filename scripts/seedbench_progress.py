#!/usr/bin/env python3
"""
One-line-per-model progress of running VLMEvalKit SEED-Bench-2 evaluations.

For each model key: answers saved so far (final <key>_<dataset>.xlsx if finished, else the
largest partial .pkl under $VLMEVALKIT_DIR/outputs/<key>/), % of the dataset, runaway answers
(>= --long chars, i.e. generation that ran to the token limit), whether the run is done
(score file exists), and the job's Slurm state.

Usage:
  python scripts/seedbench_progress.py                # current runs (the -cp set)
  python scripts/seedbench_progress.py --pending      # ... only those not finished yet
  python scripts/seedbench_progress.py --past         # earlier runs (t0 / t0p / t07)
  python scripts/seedbench_progress.py --all          # every run found on disk
  python scripts/seedbench_progress.py Qwen3-VL-4B-Instruct-t0 --dataset SEEDBench2
"""

import argparse
import os
import pickle
import subprocess
from pathlib import Path

PREFIX = "Qwen3-VL-4B-Instruct"
MODELS = ["", "-v3-continuity-llm-e3", "-v3-continuity-both-e3", "-v3-all-llm-e3", "-v3-all-both-e3"]

# Current runs: Qwen's own MCQ prompt (use_custom_prompt=True), T0+pen and T0.7+pen.
KEYS = [f"{PREFIX}{m}-{t}" for t in ("t0p-cp", "t07-cp") for m in MODELS]

# Earlier runs (generic prompt): T0 (no suffix for v3), T0+pen (-t0p), T0.7+pen (-t07; base = old run).
KEYS_PAST = ([f"{PREFIX}-t0"] + [f"{PREFIX}{m}" for m in MODELS[1:]]
             + [f"{PREFIX}{m}-t0p" for m in MODELS]
             + [PREFIX] + [f"{PREFIX}{m}-t07" for m in MODELS[1:]])


def short_name(key):
    s = key[len(PREFIX):].lstrip("-") if key.startswith(PREFIX) else key
    return s if s.startswith("v3-") else f"base-{s}" if s else "base"


def dataset_size(dataset):
    tsv = Path(os.environ.get("LMUData", Path.home() / "LMUData")) / f"{dataset}.tsv"
    if not tsv.exists():
        return None
    # Count rows, not lines: some questions contain line breaks inside quoted fields.
    import pandas as pd
    return len(pd.read_csv(tsv, sep="\t", usecols=["index"]))


def slurm_states():
    """{job name: 'RUNNING 1:23:45' / 'PENDING'} for the current user."""
    try:
        out = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j|%T|%M"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    return {j: f"{s} {t}" if s == "RUNNING" else s for j, s, t in (l.split("|") for l in out.split("\n") if l)}


def progress(key, root, results, dataset, long):
    """(answers, runaway, done, source) for one model key."""
    out = root / key
    # A finished run keeps all answers in <key>_<dataset>.xlsx (the in-progress .pkl is
    # deleted at the end); look in VLMEvalKit's outputs and in our results/seedbench_*/ copies.
    final = sorted(out.glob(f"**/{key}_{dataset}.xlsx")) + \
        sorted(Path(results).glob(f"seedbench_*/{key}_{dataset}.xlsx"))
    if final:
        import pandas as pd
        best = dict(enumerate(pd.read_excel(final[0], usecols=["prediction"])["prediction"].astype(str)))
        source = "xlsx"
    else:
        best, source = {}, "pkl"
    for pkl in ([] if final else out.glob(f"**/*{dataset}*.pkl")):
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
        if len(d) > len(best):
            best = d
    runaway = sum(len(str(v)) >= long for v in best.values())
    done = bool(final) or (any(out.glob(f"**/*{dataset}*acc*")) if out.exists() else False)
    return len(best), runaway, done, source


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("keys", nargs="*", default=None, help="model keys (default: current -cp runs)")
    parser.add_argument("--past", action="store_true", help="earlier runs (t0 / t0p / t07) instead of the current ones")
    parser.add_argument("--all", action="store_true",
                        help="every run found: results/seedbench_*/<key>_<dataset>.xlsx and outputs/<key>/ "
                             "folders with <dataset> files")
    parser.add_argument("--pending", action="store_true", help="hide runs that are already done")
    parser.add_argument("--dataset", default="SEEDBench2")
    parser.add_argument("--long", type=int, default=10000, help="answers with at least this many chars count as runaway")
    parser.add_argument("--results", default="results", help="folder with our seedbench_<key>/ copies")
    args = parser.parse_args()

    root = Path(os.environ.get("VLMEVALKIT_DIR", Path(os.environ.get("SCRATCH", "."), "VLMEvalKit"))) / "outputs"
    keys = args.keys or (KEYS_PAST if args.past else KEYS)
    if args.all:
        found = {p.name[:-len(f"_{args.dataset}.xlsx")]
                 for p in Path(args.results).glob(f"seedbench_*/*_{args.dataset}.xlsx")}
        found |= {d.name for d in root.iterdir() if d.is_dir() and any(d.glob(f"**/*{args.dataset}*"))} \
            if root.is_dir() else set()
        keys = sorted(found | set(KEYS) | set(KEYS_PAST))

    total = dataset_size(args.dataset)
    jobs = slurm_states()
    print(f"{'run':34s} {'answers':>14s} {'%':>6s} {'runaway':>14s}  {'done':5s} slurm")
    hidden = 0
    for key in keys:
        n, runaway, done, source = progress(key, root, args.results, args.dataset, args.long)
        if args.pending and done:
            hidden += 1
            continue
        pct = f"{100 * n / total:.1f}" if total else "?"
        run_pct = f"{runaway} ({100 * runaway / n:.1f}%)" if n else "0"
        state = jobs.get(f"sb2-{key[len(PREFIX):].lstrip('-')}", "-") if key.startswith(PREFIX) else "-"
        print(f"{short_name(key):34s} {n:>6d}/{total or '?':<7} {pct:>6s} {run_pct:>14s}  "
              f"{'yes' if done else 'no':5s} {state}")
    if hidden:
        print(f"({hidden} finished runs hidden)")


if __name__ == "__main__":
    main()

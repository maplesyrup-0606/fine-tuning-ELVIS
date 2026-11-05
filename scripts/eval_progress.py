#!/usr/bin/env python3
"""
Progress (and optionally scores) of VLMEvalKit runs, one line per model, for any dataset.

For each model key: answers saved so far (final <key>_<dataset>.xlsx if finished, else the
largest partial .pkl under $VLMEVALKIT_DIR/outputs/<key>/), % of the dataset, runaway answers
(>= --long chars), whether the run is done (score file exists), the Slurm state of its job,
and with --scores the numbers in VLMEvalKit's *_acc file.

Run sets (--set) map to config.py keys:
  t0      base <P>-t0,  v3 <P>-v3-<m>-e3            T0, no penalty, generic prompt
  t0p     base <P>-t0p, v3 <P>-v3-<m>-e3-t0p        T0 + penalty 1.5, generic prompt
  t07     base <P>,     v3 <P>-v3-<m>-e3-t07        T0.7 + penalty 1.5, generic prompt
  t0p-cp  <P>[-v3-<m>-e3]-t0p-cp                    T0 + penalty, Qwen MCQ prompt, 200 tokens
  t07-cp  <P>[-v3-<m>-e3]-t07-cp                    T0.7 + penalty, Qwen MCQ prompt, 200 tokens

Usage:
  python scripts/eval_progress.py                                  # SEEDBench2, -cp sets
  python scripts/eval_progress.py --set t0p t07 --scores           # earlier SEED-Bench runs + scores
  python scripts/eval_progress.py --dataset VisFactor --set t07 --scores
  python scripts/eval_progress.py --dataset VisFactor --set t07 --models base all-both
  python scripts/eval_progress.py --pending                        # hide finished runs
  python scripts/eval_progress.py --all                            # every run found for the dataset
  python scripts/eval_progress.py Qwen3-VL-4B-Instruct-t0p         # explicit keys
"""

import argparse
import os
import pickle
import subprocess
from pathlib import Path

PREFIX = "Qwen3-VL-4B-Instruct"
MODELS = {"base": "", "continuity-llm": "-v3-continuity-llm-e3", "continuity-both": "-v3-continuity-both-e3",
          "all-llm": "-v3-all-llm-e3", "all-both": "-v3-all-both-e3"}
# (base suffix, v3 suffix) per run set
SETS = {"t0": ("-t0", ""), "t0p": ("-t0p", "-t0p"), "t07": ("", "-t07"),
        "t0p-cp": ("-t0p-cp", "-t0p-cp"), "t07-cp": ("-t07-cp", "-t07-cp")}


def keys_for(sets, models):
    keys = []
    for s in sets:
        base_sfx, v3_sfx = SETS[s]
        for m in models:
            keys.append(PREFIX + (base_sfx if m == "base" else MODELS[m] + v3_sfx))
    return keys


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


def slurm_jobs():
    """[(job name, 'RUNNING 1:23:45' / 'PENDING')] for the current user."""
    try:
        out = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j|%T|%M"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [(j, f"{s} {t}" if s == "RUNNING" else s) for j, s, t in (l.split("|") for l in out.split("\n") if l)]


def job_state(key, jobs, job_prefix):
    """Job names are <prefix>-<key without PREFIX-> (e.g. sb2-v3-all-both-e3-t07, vf-t0p);
    extra dashes after the prefix are ignored (vf--v3-..., and plain vf / vf- for the base key)."""
    rest = key[len(PREFIX):].lstrip("-") if key.startswith(PREFIX) else key
    for name, state in jobs:
        pre, _, tail = name.partition("-")
        if tail.lstrip("-") == rest and (job_prefix is None or pre == job_prefix):
            return state
    return "-"


def progress(key, root, results, dataset, long):
    """(answers, runaway, done, score files) for one model key."""
    out = root / key
    # A finished run keeps all answers in <key>_<dataset>.xlsx (the in-progress .pkl is deleted
    # at the end); look in VLMEvalKit's outputs and in our results/*/ copies.
    final = sorted(out.glob(f"**/{key}_{dataset}.xlsx")) + sorted(Path(results).glob(f"*/{key}_{dataset}.xlsx"))
    best = {}
    if final:
        import pandas as pd
        best = dict(enumerate(pd.read_excel(final[0], usecols=["prediction"])["prediction"].astype(str)))
    for pkl in ([] if final else out.glob(f"**/*_{dataset}.pkl")):
        try:
            d = pickle.load(open(pkl, "rb"))
        except Exception as e:  # e.g. being written right now
            print(f"  [warn] cannot read {pkl}: {type(e).__name__}: {e}")
            continue
        if isinstance(d, dict) and len(d) > len(best):
            best = d
    runaway = sum(len(str(v)) >= long for v in best.values())
    # results/ first: rescore_vlmeval.py re-scores there (outputs/ keeps the original scoring); skip _old<date> backups.
    acc = [f for f in sorted(Path(results).glob(f"*/{key}_{dataset}*acc*")) + sorted(out.glob(f"**/{key}_{dataset}*acc*"))
           if "_old" not in f.stem]
    return len(best), runaway, bool(final) or bool(acc), acc


def scores(acc_file):
    """Numbers from the first row of VLMEvalKit's *_acc file (csv or xlsx; columns differ per dataset)."""
    import pandas as pd
    df = pd.read_csv(acc_file) if acc_file.suffix == ".csv" else pd.read_excel(acc_file)
    vals = []
    for col, v in df.iloc[0].items():
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue  # e.g. the 'split' column
        vals.append(f"{col}={100 * v:.1f}" if v <= 1 else f"{col}={v:.1f}")  # fractions shown as %
    return " ".join(vals)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("keys", nargs="*", help="explicit model keys (overrides --set/--models)")
    parser.add_argument("--dataset", default="SEEDBench2", help="VLMEvalKit dataset name (SEEDBench2, VisFactor, ...)")
    parser.add_argument("--set", nargs="+", default=["t0p-cp", "t07-cp"], choices=list(SETS), help="run sets")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--all", action="store_true", help="every key that has results for this dataset")
    parser.add_argument("--pending", action="store_true", help="hide runs that are already done")
    parser.add_argument("--scores", action="store_true", help="also print the *_acc scores of finished runs")
    parser.add_argument("--job-prefix", default=None,
                        help="Slurm job-name prefix to match (default: sb2 for SEEDBench2, vf for VisFactor, else any)")
    parser.add_argument("--long", type=int, default=10000, help="answers with at least this many chars count as runaway")
    parser.add_argument("--results", default="results", help="folder with our <dataset>_<key>/ copies")
    args = parser.parse_args()

    root = Path(os.environ.get("VLMEVALKIT_DIR", Path(os.environ.get("SCRATCH", "."), "VLMEvalKit"))) / "outputs"
    keys = args.keys or keys_for(args.set, args.models)
    if args.all:
        found = {p.name[:-len(f"_{args.dataset}.xlsx")] for p in Path(args.results).glob(f"*/*_{args.dataset}.xlsx")}
        if root.is_dir():
            found |= {d.name for d in root.iterdir() if d.is_dir() and any(d.glob(f"**/*_{args.dataset}*"))}
        keys = sorted(found)
    job_prefix = args.job_prefix or {"SEEDBench2": "sb2", "VisFactor": "vf"}.get(args.dataset)

    total = dataset_size(args.dataset)
    jobs = slurm_jobs()
    print(f"{args.dataset} | sets: {' '.join(args.set) if not (args.keys or args.all) else '-'}")
    print(f"{'run':34s} {'answers':>14s} {'%':>6s} {'runaway':>14s}  {'done':5s} slurm")
    hidden = 0
    for key in keys:
        n, runaway, done, acc = progress(key, root, args.results, args.dataset, args.long)
        if args.pending and done:
            hidden += 1
            continue
        pct = f"{100 * n / total:.1f}" if total else "?"
        run_pct = f"{runaway} ({100 * runaway / n:.1f}%)" if n else "0"
        print(f"{short_name(key):34s} {n:>6d}/{total or '?':<7} {pct:>6s} {run_pct:>14s}  "
              f"{'yes' if done else 'no':5s} {job_state(key, jobs, job_prefix)}")
        if args.scores and acc:
            try:
                print(f"{'':34s} scores: {scores(acc[0])}")
            except Exception as e:
                print(f"{'':34s} scores: (could not read {acc[0].name}: {type(e).__name__})")
    if hidden:
        print(f"({hidden} finished runs hidden)")


if __name__ == "__main__":
    main()

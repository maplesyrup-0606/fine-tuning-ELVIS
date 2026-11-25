#!/usr/bin/env python3
"""
Export every benchmark question as text (no images) with every model's answer and right/wrong, one CSV per
benchmark, for offline analysis (e.g. reading the questions' text priors). Small enough to copy to a laptop.

Columns: index, category columns (category / l2-category / subpart), question, hint, A..E, answer,
  <model>_hit (1/0) and <model>_pred (first 80 characters) for every model,
  n_diff = how many fine-tuned models got a different right/wrong than base on that question.
Source: each model's judged results of --set (default t0p-cp, the single greedy run); MMMU: validation split.
Rows are aligned by position (all runs list the questions in the same order); the question text is checked.

Usage (on Fir, from fine-tuning-ELVIS):
  python scripts/export_questions.py                       # -> results/question_export/<dataset>.csv
  python scripts/export_questions.py --datasets MMStar --set t0p-cp-r224 --models base continuity-vit
"""

import argparse
from pathlib import Path

import pandas as pd

from eval_progress import MODELS, SETS, keys_for

DATASETS = ["SEEDBench2", "MMMU_DEV_VAL", "MMStar", "MMBench_DEV_EN_V11", "AI2D_TEST"]
META = ["index", "category", "l2-category", "subpart", "question", "hint", "A", "B", "C", "D", "E", "answer"]


def result_file(key, ds, results):
    files = [f for f in Path(results).glob(f"*/{key}_{ds}_*_result.xlsx") if "_old" not in f.stem]
    return max(files, key=lambda f: f.stat().st_mtime) if files else None


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--set", default="t0p-cp", choices=list(SETS))
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/question_export")
    args = parser.parse_args()
    base = "base" if "base" in args.models else args.models[0]
    Path(args.out).mkdir(parents=True, exist_ok=True)

    for ds in args.datasets:
        runs = {}
        for m, key in zip(args.models, keys_for([args.set], args.models)):
            f = result_file(key, ds, args.results)
            if f is None:
                print(f"  [skip] {ds}/{m}: no result file")
                continue
            d = pd.read_excel(f)
            if "split" in d and (d["split"] == "validation").any():
                d = d[d["split"] == "validation"]
            runs[m] = d.reset_index(drop=True)
        if base not in runs:
            print(f"[skip] {ds}: no base results")
            continue
        b = runs[base]
        out = b[[c for c in META if c in b]].copy()
        for m, d in runs.items():
            if len(d) != len(b) or not (d["question"].astype(str).values == b["question"].astype(str).values).all():
                print(f"  [skip] {ds}/{m}: questions do not line up with base")
                continue
            out[f"{m}_hit"] = d["hit"].astype(int).values
            out[f"{m}_pred"] = d["prediction"].astype(str).str.replace(r"\s+", " ", regex=True).str[:80].values
        ft = [c for c in out if c.endswith("_hit") and c != f"{base}_hit"]
        out["n_diff"] = (out[ft].ne(out[f"{base}_hit"], axis=0)).sum(axis=1) if ft else 0
        path = Path(args.out) / f"{ds}.csv"
        out.to_csv(path, index=False)
        print(f"{ds}: {len(out)} questions, {len(ft) + 1} models -> {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

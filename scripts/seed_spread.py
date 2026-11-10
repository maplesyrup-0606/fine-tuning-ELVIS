#!/usr/bin/env python3
"""
Seed runs (t07-cp-s0..s9: Qwen MCQ prompt, T0.7, vLLM seed 0..9): verify every run, then summarize the
spread of Overall accuracy across seeds per model and benchmark, and each v3 model's mean vs base's mean.

Verification is the same as general_bench.py (all answers, scored, no answer scored without the judge).
Per benchmark x model: mean, SD, min, max over the seeds found, the greedy t0p-cp score for reference, and
diff = mean(model) - mean(base) with +-2 SE, SE = sqrt(sd_model^2/k + sd_base^2/k) over k seeds each.
This measures decoding (sampling) spread only; question-sampling noise is not in it (see general_bench).

Usage (on Fir, from fine-tuning-ELVIS):
  python scripts/seed_spread.py
  python scripts/seed_spread.py --datasets MMStar --seeds 0 1 2 3 4 --csv results/seed_spread.csv
"""

import argparse
import math
from pathlib import Path

import pandas as pd

from eval_progress import MODELS, dataset_size, keys_for
from general_bench import DATASETS, LABELS, check


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--results", default="results")
    parser.add_argument("--csv", default="results/seed_spread.csv", help="per-seed accuracies (long format)")
    args = parser.parse_args()
    base = "base" if "base" in args.models else args.models[0]

    rows, bad = [], []
    for ds in args.datasets:
        total = dataset_size(ds)
        for s in args.seeds:
            for m, key in zip(args.models, keys_for([f"t07-cp-s{s}"], args.models)):
                r = check(key, ds, args.results, total)
                if not r["ok"]:
                    bad.append(f"{ds}/{m}/s{s} (answers {r['answers']}/{total}, scored {r['scored']}, unscored {r['unscored']})")
                acc = (r["acc"] or {}).get("Overall")
                rows.append(dict(dataset=ds, model=m, seed=s, acc=acc, ok=r["ok"], random=r["random"]))
    n_runs = len(rows)
    print(f"verification: {n_runs - len(bad)}/{n_runs} runs OK")
    for b in bad:
        print(f"  !! {b}")

    df = pd.DataFrame(rows)
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)
    ok = df[df.ok & df.acc.notna()]

    for ds in args.datasets:
        greedy = {m: (check(k, ds, args.results, None)["acc"] or {}).get("Overall")
                  for m, k in zip(args.models, keys_for(["t0p-cp"], args.models))}
        g = ok[ok.dataset == ds].groupby("model")["acc"]
        st = pd.DataFrame({"k": g.size(), "mean": g.mean(), "sd": g.std(ddof=1), "min": g.min(), "max": g.max()})
        print(f"\n{LABELS.get(ds, ds)}  (Overall accuracy over seeds; diff = mean - base mean, ±2 SE)")
        print(f"  {'model':16s} {'k':>2s} {'mean':>7s} {'sd':>5s} {'min':>7s} {'max':>7s} {'greedy':>7s}   diff vs base")
        for m in args.models:
            if m not in st.index:
                print(f"  {m:16s}  no OK runs")
                continue
            r = st.loc[m]
            gr = greedy.get(m)
            line = (f"  {m:16s} {int(r.k):>2d} {r['mean']:7.2f} {r.sd:5.2f} {r['min']:7.2f} {r['max']:7.2f} "
                    f"{'-' if gr is None else f'{gr:7.2f}':>7s}")
            if m != base and base in st.index:
                b = st.loc[base]
                se = math.sqrt(r.sd ** 2 / r.k + b.sd ** 2 / b.k)
                d = r["mean"] - b["mean"]
                line += f"   {d:+6.2f} ± {2 * se:4.2f}" + ("  *" if abs(d) > 2 * se else "")
            print(line)
    print(f"\n* = |diff| > 2 SE of the seed means (beyond decoding noise). Per-seed numbers: {args.csv}")


if __name__ == "__main__":
    main()

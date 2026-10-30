#!/usr/bin/env python3
"""
Paired (McNemar exact) test of each model vs base on SEED-Bench-2, per skill.

All models answered the same questions, so compare them question by question:
  only_model = questions only the model got right, only_base = only base got right.
Under "no difference" these are a fair coin split; p = exact two-sided binomial test.
Reads results/seedbench_<key>/*_enriched.xlsx (enrich_seedbench.py output).

Usage:
  python scripts/seedbench_mcnemar.py --skills 19 20 21 22
  python scripts/seedbench_mcnemar.py --suffix ""            # the no-penalty (t0) runs
"""

import argparse
import glob
import math

import pandas as pd

PREFIX = "Qwen3-VL-4B-Instruct"
MODELS = ["v3-continuity-llm-e3", "v3-continuity-both-e3", "v3-all-llm-e3", "v3-all-both-e3"]


def binom_two_sided(k, n):
    """Exact two-sided binomial test p-value for k successes out of n at p = 0.5."""
    if n == 0:
        return 1.0
    pmf = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
    return min(1.0, sum(p for p in pmf if p <= pmf[k] * (1 + 1e-9)))


def load(results, key):
    files = glob.glob(f"{results}/seedbench_{key}/*_enriched.xlsx")
    if not files:
        raise SystemExit(f"no *_enriched.xlsx for {key} (run enrich_seedbench.py first)")
    return pd.read_excel(files[0], usecols=["index", "question_type_id", "hit"]).set_index("index")


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--skills", nargs="+", type=int, default=None, help="question_type_id(s); default all")
    parser.add_argument("--suffix", default="-t0p", help="key suffix of the run set (-t0p, or '' for t0)")
    parser.add_argument("--base", default=None, help="base key (default: <prefix>-t0p / <prefix>-t0)")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()

    base_key = args.base or f"{PREFIX}-{'t0p' if args.suffix == '-t0p' else 't0'}"
    base = load(args.results, base_key)
    skills = args.skills or sorted(base["question_type_id"].unique())
    print(f"base = {base_key}\n")
    print(f"{'skill':>5} {'model':24s} {'n':>5} {'base%':>6} {'model%':>7} {'diff':>6} "
          f"{'only_model':>10} {'only_base':>9} {'p':>8}")
    for m in MODELS:
        d = base.join(load(args.results, f"{PREFIX}-{m}{args.suffix}"), rsuffix="_m", how="inner")
        for s in skills:
            x = d[d["question_type_id"] == s]
            only_m = int(((x["hit"] == 0) & (x["hit_m"] == 1)).sum())
            only_b = int(((x["hit"] == 1) & (x["hit_m"] == 0)).sum())
            p = binom_two_sided(only_m, only_m + only_b)
            star = " *" if p < 0.05 else ""
            print(f"{s:>5} {m:24s} {len(x):>5} {100 * x['hit'].mean():6.1f} {100 * x['hit_m'].mean():7.1f} "
                  f"{100 * (x['hit_m'].mean() - x['hit'].mean()):+6.1f} {only_m:>10} {only_b:>9} {p:8.4f}{star}")
        print()
    print("* p < 0.05 (no correction for testing many skills x models at once)")


if __name__ == "__main__":
    main()

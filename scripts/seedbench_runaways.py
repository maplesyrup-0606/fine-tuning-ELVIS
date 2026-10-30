#!/usr/bin/env python3
"""
Runaway-answer analysis across SEED-Bench-2 runs (answers that keep generating far past the
choice, typically looping to the token limit).

Per run (from results/<dir>/*_enriched.xlsx, else the result xlsx):
  1. answer length: median, share >= --long-1k chars, runaways (>= --long chars)
  2. accuracy on runaway vs normal answers (what runaways cost)
  3. runaway rate per SEED-Bench-2 part
  4. overlap: do runs run away on the same questions (rows paired by position)
  5. --examples N: start/end of N runaway answers per run -> --out text file

Usage:
  python scripts/seedbench_runaways.py
  python scripts/seedbench_runaways.py --run results/seedbench_base:base-T0.7 --run <dir>:<label>
"""

import argparse
import glob
from pathlib import Path

import pandas as pd

P = "results/seedbench_Qwen3-VL-4B-Instruct"
V3 = [("continuity-llm", "cont-llm"), ("continuity-both", "cont-both"), ("all-llm", "all-llm"), ("all-both", "all-both")]
# Keys: T0 runs are <P>-t0 (base) and <P>-v3-<m>-e3 (no suffix); T0+pen runs end in -t0p.
DEFAULT_RUNS = (["results/seedbench_base:base T0.7+pen", f"{P}-t0:base T0"]
                + [f"{P}-v3-{m}-e3:{lab} T0" for m, lab in V3]
                + [f"{P}-t0p:base T0+pen"]
                + [f"{P}-v3-{m}-e3-t0p:{lab} T0+pen" for m, lab in V3])


def load(d):
    files = (sorted(glob.glob(f"{d}/*_enriched.xlsx")) or sorted(glob.glob(f"{d}/*_gpt-4o-mini_result.xlsx"))
             or sorted(glob.glob(f"{d}/*_result.xlsx")))
    if not files:
        return None
    df = pd.read_excel(files[0])
    df["len"] = df["prediction"].astype(str).str.len()
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--run", action="append", default=None, help="<results dir>:<label>, repeatable")
    parser.add_argument("--long", type=int, default=10000, help="runaway threshold (chars)")
    parser.add_argument("--long-1k", type=int, default=1000, help="'long answer' threshold (chars)")
    parser.add_argument("--examples", type=int, default=3, help="runaway examples per run to write out")
    parser.add_argument("--out", default="results/seedbench_runaways.txt")
    args = parser.parse_args()

    runs = {}
    for spec in args.run or DEFAULT_RUNS:
        d, label = spec.rsplit(":", 1)
        df = load(d)
        if df is None:
            print(f"[skip] {label}: no result xlsx in {d}")
            continue
        runs[label] = df
    if not runs:
        raise SystemExit("no runs found")

    print(f"\n1-2. Answer length and what runaways cost (runaway = >= {args.long} chars)\n")
    print(f"{'run':20s} {'n':>6s} {'median':>6s} {'>=1k':>6s} {'runaway':>14s} {'acc':>6s} "
          f"{'acc normal':>10s} {'acc runaway':>11s}")
    for label, df in runs.items():
        r = df["len"] >= args.long
        acc_r = f"{100 * df.loc[r, 'hit'].mean():.1f}" if r.any() else "-"
        print(f"{label:20s} {len(df):6d} {int(df['len'].median()):6d} "
              f"{100 * (df['len'] >= args.long_1k).mean():5.1f}% {int(r.sum()):6d} ({100 * r.mean():4.1f}%) "
              f"{100 * df['hit'].mean():6.1f} {100 * df.loc[~r, 'hit'].mean():10.1f} {acc_r:>11s}")

    if all("subpart" in df.columns for df in runs.values()):
        print("\n3. Runaway rate (%) per SEED-Bench-2 part\n")
        table = pd.DataFrame({label: df.assign(r=df["len"] >= args.long).groupby("subpart")["r"].mean() * 100
                              for label, df in runs.items()})
        table.insert(0, "n", next(iter(runs.values())).groupby("subpart").size())
        with pd.option_context("display.width", 250, "display.max_columns", 50):
            print(table.round(1).to_string())

    labels = list(runs)
    ref = runs[labels[0]]
    aligned = [l for l in labels if len(runs[l]) == len(ref) and (runs[l]["index"].values == ref["index"].values).all()]
    if len(aligned) > 1:
        print("\n4. Overlap of runaway questions (Jaccard = shared / in either; 1 = same questions)\n")
        sets = {l: set(runs[l].index[runs[l]["len"] >= args.long]) for l in aligned}
        print(" " * 20 + "".join(f"{l[:12]:>13s}" for l in aligned))
        for a in aligned:
            row = "".join(f"{(len(sets[a] & sets[b]) / len(sets[a] | sets[b]) if sets[a] | sets[b] else 0):13.2f}"
                          for b in aligned)
            print(f"{a:20s}{row}")
        if set(labels) - set(aligned):
            print(f"(left out, rows not in the same order: {sorted(set(labels) - set(aligned))})")

    if args.examples:
        lines = []
        for label, df in runs.items():
            for _, row in df[df["len"] >= args.long].head(args.examples).iterrows():
                p = str(row["prediction"])
                what = row["question_type"] if "question_type" in row else row.get("subpart", "")
                lines += [f"=== {label} | index {row['index']} | {what} | {len(p)} chars | "
                          f"correct {row.get('answer', '?')} | hit {row['hit']}",
                          "START: " + p[:400].replace("\n", " "), "END:   " + p[-300:].replace("\n", " "), ""]
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(lines))
        print(f"\n5. Runaway examples: {args.out}")


if __name__ == "__main__":
    main()

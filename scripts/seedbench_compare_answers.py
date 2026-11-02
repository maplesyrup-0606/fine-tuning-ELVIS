#!/usr/bin/env python3
"""
Side-by-side SEED-Bench-2 answers of base and the four v3 models for random questions.

For each sampled question: the question, options and correct answer, then every model's
raw answer (prediction) marked correct/wrong. Reads the saved result files only.

Run sets (decoding settings):
  t0p  temperature 0 + presence penalty 1.5 (greedy)      base: <P>-t0p,  v3: <P>-v3-<m>-e3-t0p
  t0   temperature 0, no penalty (greedy)                  base: <P>-t0,   v3: <P>-v3-<m>-e3
  t07  temperature 0.7 + penalty (Qwen's recommended)      base: results/seedbench_base, v3: <P>-v3-<m>-e3-t07

Usage:
  python scripts/seedbench_compare_answers.py                         # 10 random questions: greedy (t0p) vs T0.7
  python scripts/seedbench_compare_answers.py --disagree              # only questions where the greedy models differ
  python scripts/seedbench_compare_answers.py --also none             # greedy answers only
  python scripts/seedbench_compare_answers.py --subpart "Video & Text Comprehension" -n 5 --out results/compare.md
"""

import argparse
import glob
import random

import pandas as pd

P = "Qwen3-VL-4B-Instruct"
V3 = ["continuity-llm", "continuity-both", "all-llm", "all-both"]
SUFFIX = {"t0p": "-t0p", "t0": "", "t07": "-t07"}


def run_dirs(rset):
    base = "results/seedbench_base" if rset == "t07" else f"results/seedbench_{P}-{rset}"
    dirs = {"base": base}
    dirs.update({m: f"results/seedbench_{P}-v3-{m}-e3{SUFFIX[rset]}" for m in V3})
    return dirs


def load(d):
    f = glob.glob(f"{d}/*_gpt-4o-mini_result.xlsx") or glob.glob(f"{d}/*_result.xlsx")
    if not f:
        return None
    # Rows are paired by position, not `index`: the saved index is lossy (~3,099 pairs of
    # different questions share one value), but every result file lists the questions in the
    # same order. main() checks the question text matches row by row.
    return pd.read_excel(f[0]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--set", dest="rset", default="t0p", choices=SUFFIX, help="main run set")
    parser.add_argument("--also", choices=list(SUFFIX) + ["none"], default="t07",
                        help="second run set shown under each question (default t07: greedy vs sampled)")
    parser.add_argument("-n", type=int, default=10, help="number of questions")
    parser.add_argument("--subpart", default=None, help="restrict to one SEED-Bench-2 part")
    parser.add_argument("--disagree", action="store_true", help="only questions where the main set's models differ")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=160, help="max characters shown per answer")
    parser.add_argument("--out", default=None, help="also write the output to this file (e.g. .md)")
    args = parser.parse_args()

    sets = [args.rset] + ([args.also] if args.also not in (None, "none", args.rset) else [])
    runs = {}
    for s in sets:
        for name, d in run_dirs(s).items():
            df = load(d)
            if df is None:
                print(f"[skip] {name} ({s}): no result file in {d}")
                continue
            runs[(s, name)] = df
    main_runs = {k: v for k, v in runs.items() if k[0] == args.rset}
    if not main_runs:
        raise SystemExit("no results for the main run set")

    ref = next(iter(main_runs.values()))
    for (s, name), df in runs.items():
        if len(df) != len(ref) or not (df["question"].astype(str).values == ref["question"].astype(str).values).all():
            raise SystemExit(f"{name} ({s}): rows do not line up with {args.rset} base (length or question order differs)")
    idx = ref.index
    if args.subpart:
        idx = idx[ref.loc[idx, "subpart"] == args.subpart]
    if args.disagree:
        idx = [i for i in idx if len({int(df.at[i, "hit"]) for df in main_runs.values()}) > 1]
    picked = random.Random(args.seed).sample(list(idx), min(args.n, len(idx)))

    lines = []
    for i in picked:
        q = ref.loc[i]
        lines += ["=" * 100, f"row {i} (index {q['index']}) | {q['subpart']}", f"Q: {q['question']}"]
        lines += [f"   {c}. {q[c]}" for c in "ABCDE" if c in q and pd.notna(q[c])]
        lines.append(f"correct: {q['answer']}")
        for s in sets:
            lines.append(f"-- {s}")
            for (rs, name), df in runs.items():
                if rs != s:
                    continue
                pred = " ".join(str(df.at[i, "prediction"]).split())
                more = f" … [{len(pred)} chars]" if len(pred) > args.width else ""
                mark = "✓" if int(df.at[i, "hit"]) == 1 else "✗"
                lines.append(f"   {mark} {name:16s} {pred[:args.width]}{more}")
    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()

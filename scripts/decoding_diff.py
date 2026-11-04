#!/usr/bin/env python3
"""
Diagnose the accuracy difference between two decoding settings (default: T0+pen vs T0.7+pen,
generic prompt) for base and the v3 models, question by question, from saved VLMEvalKit results.

Per model:
  1. consistency checks (bug?): same questions, same order, same correct answers in both runs;
     most frequent answers (one string repeated over many questions = degenerate decoding)
  2. accuracy A vs B, questions that flip each way, exact McNemar p, % identical answers
  3. answer format per setting: letter only / letter + text / other text / runaway (>= 10k chars)
  4. how answers were scored (VLMEvalKit `log`): rule / judge letter / judge Z (no answer) / random
  5. on questions right in B but wrong in A: format + scoring path of the A answer (clear wrong letter
     = real difference; no readable answer = format/scoring problem), and SEED-Bench part
Then, on base's B-right/A-wrong questions: how each v3 model answered them in A.

Usage:
  python scripts/decoding_diff.py                                    # A = t0p, B = t07
  python scripts/decoding_diff.py --a t0 --b t07 --examples 5 --out results/decoding_diff.md
"""

import argparse
import glob
import math
import re
from collections import Counter

import pandas as pd

from eval_progress import MODELS, SETS, keys_for, short_name

LETTER_ONLY = re.compile(r"\(?[A-E]\)?[.:]?")
LETTER_FIRST = re.compile(r"\(?[A-E][\).:]?\s")


def load(key, results, dataset):
    f = sorted(glob.glob(f"{results}/*/{key}_{dataset}_gpt-4o-mini_result.xlsx")) or \
        sorted(glob.glob(f"{results}/*/{key}_{dataset}_*_result.xlsx"))
    if not f:
        return None
    df = pd.read_excel(f[0]).reset_index(drop=True)
    df["prediction"] = df["prediction"].astype(str).str.strip()
    df["log"] = df["log"].astype(str) if "log" in df else ""
    return df


def fmt(p):
    if len(p) >= 10000:
        return "runaway"
    if LETTER_ONLY.fullmatch(p):
        return "letter only"
    if LETTER_FIRST.match(p):
        return "letter + text"
    return "other text"


def path(row):
    """VLMEvalKit logs the prediction itself when rules matched, else the judge's output."""
    log, pred = row["log"], row["prediction"]
    if "randomly generate" in log:
        return "random"
    if log.startswith(f"Match Log: {pred}"):
        return "rule"
    m = re.search(r"Match Log:\s*\(?([A-Z])\)?", log)
    if m and m.group(1) == "Z":
        return "judge: Z (no answer)"
    return "judge: letter"


def mcnemar(n_a_only, n_b_only):
    n, k = n_a_only + n_b_only, min(n_a_only, n_b_only)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def pct_table(series):
    c = series.value_counts(normalize=True) * 100
    return "  ".join(f"{k} {v:.1f}%" for k, v in c.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--a", default="t0p", choices=list(SETS), help="setting A (default greedy t0p)")
    parser.add_argument("--b", default="t07", choices=list(SETS), help="setting B (default sampled t07)")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--dataset", default="SEEDBench2")
    parser.add_argument("--results", default="results")
    parser.add_argument("--examples", type=int, default=0, help="example flips per model written to --out")
    parser.add_argument("--out", default="results/decoding_diff.md")
    args = parser.parse_args()

    ka, kb = keys_for([args.a], args.models), keys_for([args.b], args.models)
    runs, examples = {}, []
    for m, key_a, key_b in zip(args.models, ka, kb):
        A, B = load(key_a, args.results, args.dataset), load(key_b, args.results, args.dataset)
        if A is None or B is None:
            print(f"[skip] {m}: missing result file ({key_a if A is None else key_b})")
            continue
        print("=" * 100)
        print(f"{m}:  A = {short_name(key_a)}   B = {short_name(key_b)}")

        # 1. consistency
        same_len = len(A) == len(B)
        same_q = same_len and (A["question"].astype(str).values == B["question"].astype(str).values).all()
        same_gt = same_len and (A["answer"].astype(str).values == B["answer"].astype(str).values).all()
        print(f"  checks: same length {same_len} | same questions/order {same_q} | same correct answers {same_gt}")
        if not (same_q and same_gt):
            print("  !! rows do not line up; skipping this model (possible bug in the files)")
            continue
        top = Counter(A["prediction"]).most_common(3)
        print("  most frequent A answers: " + " | ".join(f"{p[:40]!r} x{n}" for p, n in top))

        # 2. accuracy + flips
        a_only = int(((A.hit == 1) & (B.hit == 0)).sum())
        b_only = int(((A.hit == 0) & (B.hit == 1)).sum())
        same_pred = (A.prediction == B.prediction).mean() * 100
        print(f"  accuracy A {100 * A.hit.mean():.2f}  B {100 * B.hit.mean():.2f}"
              f"  (B-A {100 * (B.hit.mean() - A.hit.mean()):+.2f})"
              f" | only A right {a_only} | only B right {b_only} | McNemar p {mcnemar(a_only, b_only):.2g}"
              f" | identical answers {same_pred:.1f}%")

        # 3-4. format and scoring path per setting
        for name, D in (("A", A), ("B", B)):
            D["fmt"], D["path"] = D.prediction.map(fmt), D.apply(path, axis=1)
            print(f"  {name} format : {pct_table(D.fmt)} | median {int(D.prediction.str.len().median())} chars")
            print(f"  {name} scoring: {pct_table(D.path)}")

        # 5. flips: right in B, wrong in A
        flips = (A.hit == 0) & (B.hit == 1)
        if flips.any():
            print(f"  on the {int(flips.sum())} questions right in B but wrong in A:")
            print(f"    A answer format : {pct_table(A.fmt[flips])}")
            print(f"    A scoring path  : {pct_table(A.path[flips])}")
            print(f"    B answer format : {pct_table(B.fmt[flips])}")
            print(f"    SEED-Bench part : {pct_table(A.subpart[flips])}")
        runs[m] = (A, B, flips)
        for i in A.index[flips][:args.examples]:
            examples += [f"### {m} | row {i} | {A.at[i, 'subpart']}", f"**Q:** {A.at[i, 'question']}",
                         " ".join(f"{c}. {A.at[i, c]}" for c in "ABCDE" if c in A and pd.notna(A.at[i, c])),
                         f"**correct:** {A.at[i, 'answer']}",
                         f"- **A** ({A.at[i, 'path']}): {A.at[i, 'prediction'][:400]}",
                         f"- **B** ({B.at[i, 'path']}): {B.at[i, 'prediction'][:400]}", ""]

    if "base" in runs and len(runs) > 1:
        A0, _, flips = runs["base"]
        print("=" * 100)
        print(f"On base's {int(flips.sum())} B-right/A-wrong questions, the other models in setting A:")
        for m, (A, _, _) in runs.items():
            if m == "base":
                continue
            print(f"  {m:16s} correct {100 * A.hit[flips].mean():.1f}% | format: {pct_table(A.fmt[flips])}")

    if args.examples and examples:
        with open(args.out, "w") as f:
            f.write("\n".join(examples) + "\n")
        print(f"\nexamples written to {args.out}")


if __name__ == "__main__":
    main()

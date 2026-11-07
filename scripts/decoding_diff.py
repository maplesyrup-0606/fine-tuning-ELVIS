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
  python scripts/decoding_diff.py --examples 10 --out results/decoding_diff.html   # + readable HTML examples
  python scripts/decoding_diff.py --a t0 --b t07 --models base
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
    if "no GPT-based answer matching" in log:  # rules failed and no GPT judge was available -> scored as wrong
        return "no judge (exact matching) -> wrong"
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
    parser.add_argument("--examples", type=int, default=0, help="example flips per model written to --out (HTML)")
    parser.add_argument("--out", default="results/decoding_diff.html")
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
            # Same answer text but different score = the difference comes from scoring, not the model.
            same = A.prediction == B.prediction
            n_same = int((flips & same).sum())
            rev_same = int(((A.hit == 1) & (B.hit == 0) & same).sum())
            print(f"    identical answer text in A and B: {n_same} ({100 * n_same / flips.sum():.1f}%)"
                  f"  [reverse direction, A right/B wrong with identical text: {rev_same}]")
            if n_same:
                print(f"      their scoring  A: {pct_table(A.path[flips & same])}")
                print(f"      their scoring  B: {pct_table(B.path[flips & same])}")
        runs[m] = (A, B, flips)
        same = A.prediction == B.prediction
        order = list(A.index[flips & same]) + list(A.index[flips & ~same])  # identical-answer flips first
        for i in order[:args.examples]:
            examples.append(dict(
                model=m, row=i, part=A.at[i, "subpart"], q=A.at[i, "question"], answer=A.at[i, "answer"],
                options=[(c, A.at[i, c]) for c in "ABCDE" if c in A and pd.notna(A.at[i, c])],
                same=bool(same[i]), runs=[(f"A = {short_name(key_a)}", A.loc[i]), (f"B = {short_name(key_b)}", B.loc[i])]))

    if "base" in runs and len(runs) > 1:
        A0, _, flips = runs["base"]
        print("=" * 100)
        print(f"On base's {int(flips.sum())} B-right/A-wrong questions, the other models in setting A:")
        for m, (A, _, _) in runs.items():
            if m == "base":
                continue
            print(f"  {m:16s} correct {100 * A.hit[flips].mean():.1f}% | format: {pct_table(A.fmt[flips])}")

    if args.examples and examples:
        write_html(examples, args.out)
        print(f"\nexamples written to {args.out}")


def write_html(examples, out):
    """Readable report: answers in boxes (model Markdown shown as plain text), judge decision beside them."""
    from html import escape
    css = ("body{font:14px/1.45 system-ui,sans-serif;max-width:1200px;margin:0 auto;padding:16px;background:#fcfcfb;color:#111}"
           ".card{border:1px solid #ddd;border-radius:8px;padding:12px;margin:16px 0;background:#fff}"
           ".meta{color:#555;font-size:12px}.q{font-weight:600;margin:6px 0}.opt{margin:2px 0}"
           ".runs{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:8px}"
           ".run{border:1px solid #eee;border-radius:6px;padding:8px}"
           "pre{white-space:pre-wrap;max-height:320px;overflow:auto;background:#f6f6f4;padding:8px;border-radius:4px;font-size:12px}"
           ".ok{color:#1a7f4b;font-weight:600}.bad{color:#c0392b;font-weight:600}"
           ".badge{background:#fff3cd;border:1px solid #e0c36a;border-radius:4px;padding:1px 6px;font-size:12px}")
    parts = [f"<!doctype html><meta charset='utf-8'><title>Decoding diff examples</title><style>{css}</style>",
             "<h2>Questions right in B but wrong in A</h2><p class='meta'>Identical-answer cases first: the same text "
             "scored differently points to the scoring, not the model.</p>"]
    for e in examples:
        badge = " <span class='badge'>identical answer text</span>" if e["same"] else ""
        opts = "".join(f"<div class='opt'>{escape(c)}. {escape(str(v))}</div>" for c, v in e["options"])
        runs = ""
        for label, r in e["runs"]:
            mark = "<span class='ok'>✓ right</span>" if int(r["hit"]) == 1 else "<span class='bad'>✗ wrong</span>"
            runs += (f"<div class='run'><b>{escape(label)}</b> {mark} <span class='meta'>scored by: {escape(r['path'])}</span>"
                     f"<pre>{escape(str(r['prediction'])[:3000])}</pre>"
                     f"<div class='meta'>VLMEvalKit log:</div><pre>{escape(str(r['log'])[:600])}</pre></div>")
        parts.append(f"<div class='card'><div class='meta'>{escape(e['model'])} · row {e['row']} · {escape(str(e['part']))}"
                     f"{badge}</div><div class='q'>{escape(str(e['q']))}</div>{opts}"
                     f"<div>correct: <b>{escape(str(e['answer']))}</b></div><div class='runs'>{runs}</div></div>")
    with open(out, "w") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()

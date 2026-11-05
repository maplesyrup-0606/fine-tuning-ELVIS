#!/usr/bin/env python3
"""
VisFactor CS1-CS3 answers of every model, side by side, as one Markdown file.

Reads each run's saved answers (<key>_VisFactor.xlsx in $VLMEVALKIT_DIR/outputs/<key>/ or results/*/) and scores
them the way VisFactor does: the last {"answer": ...} JSON in the output, exact match (case-insensitive)
against the comma-separated accepted answers; no JSON = empty = wrong.

Output: a summary table (correct per model and subtest), then per question the prompt, the accepted
answers, and each model's extracted answer, verdict and raw output.

Usage:
  python scripts/visfactor_cs_answers.py                          # t07 set -> results/visfactor_cs_answers.md
  python scripts/visfactor_cs_answers.py --set t0p --out results/visfactor_cs_t0p.md
"""

import argparse
import os
import re
from pathlib import Path

import pandas as pd

from eval_progress import MODELS, SETS, keys_for, short_name

SUBTESTS = ["CS1", "CS2", "CS3"]


def last_json_answer(s):
    """Same as VisFactor.extract_last_json_answer."""
    m = list(re.finditer(r'\{\s*"answer"\s*:\s*(.+?)\s*\}', s))
    if not m:
        return ""
    v = m[-1].group(1).strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        v = v[1:-1]
    return v


def find_answers(key, root, results):
    files = sorted((root / key).glob(f"**/{key}_VisFactor.xlsx")) + sorted(Path(results).glob(f"*/{key}_VisFactor.xlsx"))
    return files[0] if files else None


def cell(s, n=None):
    """One Markdown table cell: no pipes or line breaks; optionally truncated."""
    s = str(s)
    if n and len(s) > n:
        s = s[:n] + " …"
    return s.replace("|", "\\|").replace("\r", "").replace("\n", " ⏎ ") or "∅"


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("keys", nargs="*", help="explicit model keys (overrides --set/--models)")
    parser.add_argument("--set", default="t07", choices=list(SETS))
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--results", default="results")
    parser.add_argument("--raw-chars", type=int, default=300, help="truncate raw outputs to this many chars (0 = full)")
    parser.add_argument("--out", default="results/visfactor_cs_answers.md")
    args = parser.parse_args()

    root = Path(os.environ.get("VLMEVALKIT_DIR", Path(os.environ.get("SCRATCH", "."), "VLMEvalKit"))) / "outputs"
    runs = {}
    for key in args.keys or keys_for([args.set], args.models):
        f = find_answers(key, root, args.results)
        if f is None:
            print(f"[skip] {short_name(key)}: no {key}_VisFactor.xlsx")
            continue
        d = pd.read_excel(f)
        d = d[d["category_id"].astype(str).isin(SUBTESTS)].reset_index(drop=True)
        d["prediction"] = d["prediction"].astype(str)
        d["pred"] = d["prediction"].map(last_json_answer)
        d["ok"] = [p.lower() in [a.strip().lower() for a in str(ans).split(",")] for p, ans in zip(d["pred"], d["answer"])]
        runs[short_name(key)] = d
        print(f"{short_name(key)}: {f}")
    if not runs:
        raise SystemExit("no VisFactor answer files found")

    names = list(runs)
    ref = runs[names[0]]
    for n in names[1:]:  # rows must be the same questions in the same order
        if list(runs[n]["index"]) != list(ref["index"]):
            raise SystemExit(f"{n}: question order differs from {names[0]}")

    lines = [f"# VisFactor CS1–CS3 answers ({args.set if not args.keys else 'selected runs'})", "",
             "Scored like VisFactor: last `{\"answer\": ...}` JSON in the output, exact match (case-insensitive) "
             "against the accepted answers; ∅ = no JSON found (counts as wrong). Each question is its own item.", "",
             "| model | " + " | ".join(SUBTESTS) + " | total |", "|---|" + "---|" * (len(SUBTESTS) + 1)]
    for n, d in runs.items():
        per = [f"{int(d.ok[d.category_id == s].sum())}/{int((d.category_id == s).sum())}" for s in SUBTESTS]
        lines.append(f"| {n} | " + " | ".join(per) + f" | {int(d.ok.sum())}/{len(d)} |")

    for s in SUBTESTS:
        rows = ref.index[ref["category_id"].astype(str) == s]
        if not len(rows):
            continue
        first = str(ref.at[rows[0], "question"]).replace("<br>", "\n")
        lines += ["", f"## {s} ({len(rows)} questions)", ""]
        # The instruction text is usually the same for every question of a subtest: show it once.
        lines += ["<details><summary>prompt</summary>", "", "```", first, "```", "", "</details>"]
        for i in rows:
            n_ok = sum(bool(runs[n].at[i, "ok"]) for n in names)
            q = str(ref.at[i, "question"]).replace("<br>", "\n")
            lines += ["", f"### {s} · item {ref.at[i, 'eval_index']} · index {ref.at[i, 'index']} — {n_ok}/{len(names)} right", "",
                      f"**accepted:** `{ref.at[i, 'answer']}`", ""]
            if q != first:
                lines += ["<details><summary>prompt (differs)</summary>", "", "```", q, "```", "", "</details>", ""]
            lines += ["| model | answer | | raw output |", "|---|---|---|---|"]
            for n in names:
                r = runs[n].loc[i]
                lines.append(f"| {n} | {cell(r['pred'])} | {'✓' if r['ok'] else '✗'} | {cell(r['prediction'], args.raw_chars)} |")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Walk results/eval/ and gather per-principle accuracy/f1 into one table.

Rows: base + each (variant, checkpoint-step).
Cols: continuity | proximity | similarity | closure | symmetry.

Prints a markdown summary table + a delta-vs-base table (for overfitting view).
Optionally writes CSV.

Usage:
    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --metric f1
    python scripts/aggregate_results.py --csv results/eval/summary_matrix.csv
"""

import argparse
import csv
import json
import re
from pathlib import Path

PRINCIPLES = ["continuity", "proximity", "similarity", "closure", "symmetry"]
VARIANT_ORDER = {"base": 0, "llm": 1, "projector": 2, "both": 3}


def parse_summary(summary_path, root):
    rel = summary_path.relative_to(root)
    parts = rel.parts[:-1]  # drop 'summary.json'
    variant = parts[0]
    step = None
    if variant == "base":
        row_key = "base" if len(parts) == 1 else f"base/{parts[-1]}"
    else:
        ckpt = parts[-1]
        m = re.search(r"checkpoint-(\d+)", ckpt)
        if m:
            step = int(m.group(1))
            row_key = f"{variant}/e{step}"
        else:
            row_key = f"{variant}/{ckpt}"

    with open(summary_path) as f:
        data = json.load(f)
    return {
        "row_key": row_key,
        "variant": variant,
        "step": step,
        "principles": data.get("principles", {}),
        "checkpoint": data.get("checkpoint"),
    }


def sort_key(row):
    return (VARIANT_ORDER.get(row["variant"], 99), row["step"] or 0, row["row_key"])


def fmt_val(v):
    return f"{v:6.2f}" if isinstance(v, (int, float)) else "   —  "


def fmt_delta(v, b):
    if not isinstance(v, (int, float)) or not isinstance(b, (int, float)):
        return "   —  "
    return f"{v - b:+6.2f}"


def print_table(rows, metric, deltas_vs=None):
    col_w = 16
    val_w = 8
    header = ["run".ljust(col_w)] + [p.ljust(val_w) for p in PRINCIPLES]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["-" * (len(h) + 2) for h in header]) + "|")
    for r in rows:
        if deltas_vs is not None and r["row_key"] == deltas_vs["row_key"]:
            continue
        cells = [r["row_key"].ljust(col_w)]
        for p in PRINCIPLES:
            v = r["principles"].get(p, {}).get(metric)
            if deltas_vs is None:
                cells.append(fmt_val(v).ljust(val_w))
            else:
                b = deltas_vs["principles"].get(p, {}).get(metric)
                cells.append(fmt_delta(v, b).ljust(val_w))
        print("| " + " | ".join(cells) + " |")


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--results-root", default="results/eval")
    parser.add_argument("--metric", choices=["accuracy", "f1"], default="accuracy")
    parser.add_argument("--csv", default=None, help="optional path to also write CSV")
    args = parser.parse_args()

    root = Path(args.results_root)
    if not root.is_dir():
        parser.error(f"results root not found: {root}")

    summaries = sorted(root.glob("**/summary.json"))
    if not summaries:
        parser.error(f"no summary.json found under {root}")

    rows = [parse_summary(p, root) for p in summaries]
    rows.sort(key=sort_key)

    print(f"\n=== Accuracy matrix ({args.metric}) ===\n")
    print_table(rows, args.metric)

    base = next((r for r in rows if r["row_key"] == "base"), None)
    if base is None:
        base = next((r for r in rows if r["variant"] == "base"), None)
    if base is not None:
        print(f"\n=== Deltas vs {base['row_key']} ({args.metric}) ===")
        print("Positive on continuity = in-dist learning; negative on siblings = negative transfer.\n")
        print_table([r for r in rows if r["variant"] != "base"], args.metric, deltas_vs=base)

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "variant", "step", "checkpoint"] +
                       [f"{p}_{args.metric}" for p in PRINCIPLES])
            for r in rows:
                w.writerow([
                    r["row_key"], r["variant"], r["step"] or "", r["checkpoint"] or "",
                ] + [r["principles"].get(p, {}).get(args.metric, "") for p in PRINCIPLES])
        print(f"\nCSV written: {csv_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Walk results/eval/ and gather per-principle accuracy/f1 into one table.

Rows: base + each (variant, checkpoint-step).
Cols: continuity | proximity | similarity | closure | symmetry.

Prints a markdown summary table + a delta-vs-base table, optionally writes
CSV, and optionally renders an overfitting plot (one subplot per variant,
x=training step, y=metric, 5 lines per subplot, continuity bolded).

The base `continuity` cell is injected from `config.yaml`
(`evaluation.baseline_reference.continuity_mode1_base`) since the base
eval only ran the 4 sibling principles.

Usage:
    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --metric f1
    python scripts/aggregate_results.py --csv results/eval/summary_matrix.csv
    python scripts/aggregate_results.py --plot results/eval/overfitting.png
"""

import argparse
import csv
import json
import re
from pathlib import Path

PRINCIPLES = ["continuity", "proximity", "similarity", "closure", "symmetry"]
VARIANT_ORDER = {"base": 0, "llm": 1, "projector": 2, "both": 3}
TRAIN_VARIANTS = ["llm", "projector", "both"]


def parse_summary(summary_path, root):
    rel = summary_path.relative_to(root)
    parts = rel.parts[:-1]
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
    col_w, val_w = 16, 8
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


def load_config_baseline(config_path):
    """Return the known ELVIS base continuity accuracy, or None."""
    try:
        import yaml
    except ImportError:
        return None
    if not Path(config_path).is_file():
        return None
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return (cfg.get("evaluation", {})
               .get("baseline_reference", {})
               .get("continuity_mode1_base"))


def inject_base_continuity(base_row, base_cont, observed_scale_pct):
    """Add base.continuity in the same scale as the observed accuracy values."""
    if base_row is None or base_cont is None:
        return None
    existing = base_row["principles"].get("continuity", {}).get("accuracy")
    if isinstance(existing, (int, float)):
        return existing
    val = base_cont * 100 if observed_scale_pct else base_cont
    base_row["principles"]["continuity"] = {"accuracy": val, "f1": None}
    return val


def detect_scale_pct(rows, metric):
    """True if observed values look like percent (max > 1.5)."""
    vals = [
        r["principles"].get(p, {}).get(metric)
        for r in rows for p in PRINCIPLES
        if isinstance(r["principles"].get(p, {}).get(metric), (int, float))
    ]
    return bool(vals) and max(vals) > 1.5


def plot_overfitting(rows, base_row, metric, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    base_p = base_row["principles"] if base_row else {}
    colors = {
        "continuity": "#d62728",
        "proximity":  "#1f77b4",
        "similarity": "#2ca02c",
        "closure":    "#9467bd",
        "symmetry":   "#ff7f0e",
    }

    for ax, variant in zip(axes, TRAIN_VARIANTS):
        variant_rows = sorted(
            [r for r in rows if r["variant"] == variant],
            key=lambda x: x["step"] or 0,
        )
        for p in PRINCIPLES:
            xs, ys = [], []
            b = base_p.get(p, {}).get(metric)
            if isinstance(b, (int, float)):
                xs.append(0)
                ys.append(b)
            for r in variant_rows:
                v = r["principles"].get(p, {}).get(metric)
                if isinstance(v, (int, float)):
                    xs.append(r["step"])
                    ys.append(v)
            if not xs:
                continue
            is_target = p == "continuity"
            ax.plot(
                xs, ys, "o-", label=p,
                color=colors[p],
                linewidth=3 if is_target else 1.6,
                markersize=7 if is_target else 5,
                zorder=5 if is_target else 3,
            )

        ax.set_title(f"variant: {variant}")
        ax.set_xlabel("training step")
        ax.set_xticks([0, 162, 324, 486])
        ax.set_xticklabels(["base", "e1", "e2", "e3"])
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color="gray", linestyle=":", alpha=0.5)

    axes[0].set_ylabel(metric)
    axes[-1].legend(loc="best", fontsize=9)
    fig.suptitle(
        "Overfitting view — continuity (bold) is the fine-tuning target; "
        "siblings should ideally hold flat"
    )
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot written: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--results-root", default="results/eval")
    parser.add_argument("--config", default="config.yaml",
                        help="source of baseline_reference.continuity_mode1_base")
    parser.add_argument("--metric", choices=["accuracy", "f1"], default="accuracy")
    parser.add_argument("--csv", default=None, help="path to also write CSV")
    parser.add_argument("--plot", default="results/eval/overfitting.png",
                        help="path to write overfitting plot (set empty to skip)")
    args = parser.parse_args()

    root = Path(args.results_root)
    if not root.is_dir():
        parser.error(f"results root not found: {root}")

    summaries = sorted(root.glob("**/summary.json"))
    if not summaries:
        parser.error(f"no summary.json found under {root}")

    rows = [parse_summary(p, root) for p in summaries]
    rows.sort(key=sort_key)

    base = next((r for r in rows if r["variant"] == "base"), None)
    scale_pct = detect_scale_pct(rows, args.metric)
    base_cont = load_config_baseline(args.config)
    injected = inject_base_continuity(base, base_cont, scale_pct) if base_cont is not None else None
    if injected is not None:
        print(f"[info] base.continuity={injected:.2f} injected from "
              f"{args.config}:evaluation.baseline_reference.continuity_mode1_base "
              f"(scale={'percent' if scale_pct else 'fraction'})")

    print(f"\n=== Accuracy matrix ({args.metric}) ===\n")
    print_table(rows, args.metric)

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
                w.writerow(
                    [r["row_key"], r["variant"], r["step"] or "", r["checkpoint"] or ""]
                    + [r["principles"].get(p, {}).get(args.metric, "") for p in PRINCIPLES]
                )
        print(f"\nCSV written: {csv_path}")

    if args.plot:
        plot_overfitting(rows, base, args.metric, args.plot)


if __name__ == "__main__":
    main()

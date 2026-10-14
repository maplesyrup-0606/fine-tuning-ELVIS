#!/usr/bin/env python3
"""
Per-principle ELVIS accuracy curves: base -> epoch 1 -> 2 -> 3, one line per variant.

One panel per principle. v2 (results/eval_v2) as solid lines, v1 (results/eval)
as dashed lines in the same color for comparison. Each line starts at its own
run's base model (v1 base continuity is filled from config, as in
aggregate_results.py, since v1's base run skipped continuity).

Usage:
  python scripts/plot_principle_curves.py
  python scripts/plot_principle_curves.py --metric f1 --no-v1

Writes <v2-root>/curves_by_principle.png and prints the plotted numbers.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aggregate_results import PRINCIPLES, TRAIN_VARIANTS, load_config_baseline, parse_summary

# Categorical slots 1-3 of the dataviz reference palette (validated: CVD-safe on
# the light surface; slot 3 needs non-color relief -> markers + legend + table).
COLORS = {"llm": "#2a78d6", "projector": "#eb6834", "both": "#1baf7a"}
MARKERS = {"llm": "o", "projector": "s", "both": "^"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def load(root, metric, config, fill_base_continuity):
    """{principle: {variant: {x: value}}}, x = 0 for base, 1..n for epochs."""
    rows = [parse_summary(p, root) for p in sorted(root.glob("**/summary.json"))]
    out = {p: {} for p in PRINCIPLES}
    base = next((r for r in rows if r["variant"] == "base"), None)
    base_vals = {}
    if base:
        for p in PRINCIPLES:
            v = base["principles"].get(p, {}).get(metric)
            if v is None and p == "continuity" and fill_base_continuity and metric == "accuracy":
                ref = load_config_baseline(config)
                v = ref * 100 if ref is not None else None
            base_vals[p] = v
    for variant in TRAIN_VARIANTS:
        steps = sorted({r["step"] for r in rows if r["variant"] == variant and r["step"]})
        for p in PRINCIPLES:
            series = {0: base_vals[p]} if base_vals.get(p) is not None else {}
            for epoch, step in enumerate(steps, start=1):
                r = next(r for r in rows if r["variant"] == variant and r["step"] == step)
                v = r["principles"].get(p, {}).get(metric)
                if v is not None:
                    series[epoch] = v
            out[p][variant] = series
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--v2-root", default="results/eval_v2")
    parser.add_argument("--v1-root", default="results/eval")
    parser.add_argument("--no-v1", action="store_true", help="plot v2 only")
    parser.add_argument("--metric", choices=["accuracy", "f1"], default="accuracy")
    parser.add_argument("--out", default=None, help="default: <v2-root>/curves_by_principle.png")
    args = parser.parse_args()

    v2_root, v1_root = Path(args.v2_root), Path(args.v1_root)
    if not v2_root.is_dir():
        raise SystemExit(f"v2 results not found: {v2_root}")
    v2 = load(v2_root, args.metric, args.config, fill_base_continuity=False)
    v1 = None if args.no_v1 or not v1_root.is_dir() else load(v1_root, args.metric, args.config, True)

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig, axes = plt.subplots(1, len(PRINCIPLES), figsize=(4 * len(PRINCIPLES), 4.2), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, p in zip(axes, PRINCIPLES):
        ax.set_facecolor(SURFACE)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for variant in TRAIN_VARIANTS:
            for data, style, label in ((v2, "-", f"{variant} (v2)"), (v1, "--", f"{variant} (v1)")):
                if data is None or not data[p].get(variant):
                    continue
                xs = sorted(data[p][variant])
                ax.plot(xs, [data[p][variant][x] for x in xs], linestyle=style, linewidth=2,
                        color=COLORS[variant], marker=MARKERS[variant], markersize=8 if style == "-" else 6,
                        markeredgecolor=SURFACE, markeredgewidth=1.5,
                        alpha=1.0 if style == "-" else 0.55, label=label)
        ax.set_title(p, color=INK, fontsize=12, fontweight="bold" if p == "continuity" else "normal")
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["base", "ep 1", "ep 2", "ep 3"])
    axes[0].set_ylabel(f"{args.metric} (%)" if args.metric == "accuracy" else args.metric)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"ELVIS Mode 1 {args.metric} by principle — v2 solid, v1 dashed", color=INK, fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    out = Path(args.out or v2_root / "curves_by_principle.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"Wrote {out}\n")

    # Table view (same numbers as the plot).
    for name, data in (("v2", v2), ("v1", v1)):
        if data is None:
            continue
        print(f"{name} {args.metric}:  " + "  ".join(f"{p[:10]:>10}" for p in PRINCIPLES))
        base = {p: next((s[0] for s in data[p].values() if 0 in s), None) for p in PRINCIPLES}
        print("  base         " + "  ".join(f"{base[p]:10.2f}" if base[p] is not None else f"{'—':>10}"
                                         for p in PRINCIPLES))
        for variant in TRAIN_VARIANTS:
            n = max((len([x for x in data[p].get(variant, {}) if x > 0]) for p in PRINCIPLES), default=0)
            for e in range(1, n + 1):
                vals = [data[p].get(variant, {}).get(e) for p in PRINCIPLES]
                print(f"  {variant:>9} e{e}  " + "  ".join(f"{v:10.2f}" if v is not None else f"{'—':>10}"
                                                           for v in vals))
        print()


if __name__ == "__main__":
    main()

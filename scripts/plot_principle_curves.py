#!/usr/bin/env python3
"""
Per-principle ELVIS accuracy curves: base -> epoch 1 -> 2 -> 3, one line per variant.

One panel per principle. --root as solid lines, --compare-root as dashed lines in the
same color (e.g. v2 vs v1, or v3 rubric evaluation vs v3 standard evaluation). Each line
starts at its own root's base model (a missing base continuity value is filled from
config, as in aggregate_results.py, since v1's base run skipped continuity).
Variants are discovered from the results: llm / projector / both (v1, v2) or
<run>-<variant> labels such as continuity-both, all-llm (v3).

Usage:
  python scripts/plot_principle_curves.py                                   # v2 vs v1
  python scripts/plot_principle_curves.py --root results/eval_v3_rubric --label rubric \\
      --compare-root results/eval_v3 --compare-label "own rule"           # v3
  python scripts/plot_principle_curves.py --metric f1 --no-compare

Writes <root>/curves_by_principle.png and prints the plotted numbers.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aggregate_results import PRINCIPLES, load_config_baseline, parse_summary

# Categorical slots of the dataviz reference palette (validated: CVD-safe on the light
# surface; low-contrast slots need non-color relief -> markers + legend + table).
VARIANT_ORDER = ["llm", "projector", "both", "continuity-llm", "continuity-both", "all-llm", "all-both"]
COLORS = {"llm": "#2a78d6", "projector": "#eb6834", "both": "#1baf7a",
          "continuity-llm": "#2a78d6", "continuity-both": "#eb6834",
          "all-llm": "#1baf7a", "all-both": "#eda100"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def marker(variant):
    return {"llm": "o", "projector": "s", "both": "^"}[variant.rsplit("-", 1)[-1]]


def load(root, metric, config, fill_base_continuity):
    """({principle: {variant: {x: value}}}, variants), x = 0 for base, 1..n for epochs."""
    rows = [parse_summary(p, root) for p in sorted(root.glob("**/summary.json"))]
    variants = sorted({r["variant"] for r in rows if r["variant"] != "base"},
                      key=lambda v: VARIANT_ORDER.index(v) if v in VARIANT_ORDER else 99)
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
    for variant in variants:
        steps = sorted({r["step"] for r in rows if r["variant"] == variant and r["step"]})
        for p in PRINCIPLES:
            series = {0: base_vals[p]} if base_vals.get(p) is not None else {}
            for epoch, step in enumerate(steps, start=1):
                r = next(r for r in rows if r["variant"] == variant and r["step"] == step)
                v = r["principles"].get(p, {}).get(metric)
                if v is not None:
                    series[epoch] = v
            out[p][variant] = series
    return out, variants


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--root", default="results/eval_v2", help="solid lines")
    parser.add_argument("--compare-root", default="results/eval", help="dashed lines")
    parser.add_argument("--label", default=None, help="legend tag for --root (default: its folder name)")
    parser.add_argument("--compare-label", default=None, help="legend tag for --compare-root")
    parser.add_argument("--no-compare", action="store_true", help="plot --root only")
    parser.add_argument("--metric", choices=["accuracy", "f1"], default="accuracy")
    parser.add_argument("--out", default=None, help="default: <root>/curves_by_principle.png")
    args = parser.parse_args()

    root, croot = Path(args.root), Path(args.compare_root)
    if not root.is_dir():
        raise SystemExit(f"results not found: {root}")
    label, clabel = args.label or root.name, args.compare_label or croot.name
    main_data, main_vars = load(root, args.metric, args.config, fill_base_continuity=False)
    cmp_data, cmp_vars = (None, []) if args.no_compare or not croot.is_dir() else \
        load(croot, args.metric, args.config, fill_base_continuity=True)
    variants = sorted(set(main_vars) | set(cmp_vars),
                      key=lambda v: VARIANT_ORDER.index(v) if v in VARIANT_ORDER else 99)

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
        for variant in variants:
            for data, style, tag in ((main_data, "-", label), (cmp_data, "--", clabel)):
                if data is None or not data[p].get(variant):
                    continue
                xs = sorted(data[p][variant])
                ax.plot(xs, [data[p][variant][x] for x in xs], linestyle=style, linewidth=2,
                        color=COLORS.get(variant, INK_2), marker=marker(variant),
                        markersize=8 if style == "-" else 6, markeredgecolor=SURFACE, markeredgewidth=1.5,
                        alpha=1.0 if style == "-" else 0.55, label=f"{variant} ({tag})")
        ax.set_title(p, color=INK, fontsize=12, fontweight="bold" if p == "continuity" else "normal")
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["base", "ep 1", "ep 2", "ep 3"])
    axes[0].set_ylabel(f"{args.metric} (%)" if args.metric == "accuracy" else args.metric)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), frameon=False,
               bbox_to_anchor=(0.5, -0.12 if len(labels) > 4 else -0.06))
    title = f"ELVIS {args.metric} by principle — {label} solid" + ("" if cmp_data is None else f", {clabel} dashed")
    fig.suptitle(title, color=INK, fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    out = Path(args.out or root / "curves_by_principle.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"Wrote {out}\n")

    # Table view (same numbers as the plot).
    for name, data, vs in ((label, main_data, main_vars), (clabel, cmp_data, cmp_vars)):
        if data is None:
            continue
        print(f"{name} {args.metric}:".ljust(22) + "  ".join(f"{p[:10]:>10}" for p in PRINCIPLES))
        base = {p: next((s[0] for s in data[p].values() if 0 in s), None) for p in PRINCIPLES}
        print("  base".ljust(22) + "  ".join(f"{base[p]:10.2f}" if base[p] is not None else f"{'—':>10}"
                                            for p in PRINCIPLES))
        for variant in vs:
            n = max((len([x for x in data[p].get(variant, {}) if x > 0]) for p in PRINCIPLES), default=0)
            for e in range(1, n + 1):
                vals = [data[p].get(variant, {}).get(e) for p in PRINCIPLES]
                print(f"  {variant} e{e}".ljust(22) + "  ".join(f"{v:10.2f}" if v is not None else f"{'—':>10}"
                                                               for v in vals))
        print()


if __name__ == "__main__":
    main()

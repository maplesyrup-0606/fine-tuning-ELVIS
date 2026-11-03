#!/usr/bin/env python3
"""
Grouped bar chart of VLMEvalKit *_acc scores for any dataset: subtests on x, one bar per model.

Picks runs the same way as eval_progress.py (--dataset / --set / --models, or explicit keys)
and reads each finished run's <key>_<dataset>*acc* file (first row, numeric columns; values
<= 1 are shown as %). Writes a PNG and a CSV of the plotted numbers.

Usage:
  python scripts/plot_acc.py --dataset VisFactor --set t07
  python scripts/plot_acc.py --dataset VisFactor --set t07 --columns CS1 CS2 CS3 ALL --out results/plots/visfactor_cs.png
  python scripts/plot_acc.py --dataset VisFactor --set t0p t07          # both decoding settings side by side
  python scripts/plot_acc.py --dataset VisFactor --set t07 --heatmap    # models x subtests heat map
"""

import argparse
import os
from pathlib import Path

import pandas as pd

from eval_progress import MODELS, SETS, keys_for, progress, short_name

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#7f7f7f",
          "#bcbd22", "#17becf"]


def read_scores(acc_file):
    df = pd.read_csv(acc_file) if acc_file.suffix == ".csv" else pd.read_excel(acc_file)
    out = {}
    for col, v in df.iloc[0].items():
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        out[str(col)] = 100 * v if v <= 1 else v
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("keys", nargs="*", help="explicit model keys (overrides --set/--models)")
    parser.add_argument("--dataset", default="VisFactor")
    parser.add_argument("--set", nargs="+", default=["t07"], choices=list(SETS))
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--columns", nargs="+", default=None, help="subtests to plot (default: all)")
    parser.add_argument("--heatmap", action="store_true", help="heat map (models x subtests) instead of bars")
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default=None, help="PNG path (default: results/plots/<dataset>_<sets>.png)")
    args = parser.parse_args()

    root = Path(os.environ.get("VLMEVALKIT_DIR", Path(os.environ.get("SCRATCH", "."), "VLMEvalKit"))) / "outputs"
    keys = args.keys or keys_for(args.set, args.models)
    table = {}
    for key in keys:
        _, _, _, acc = progress(key, root, args.results, args.dataset, 10000)
        if not acc:
            print(f"[skip] {short_name(key)}: no score file yet")
            continue
        table[short_name(key)] = read_scores(acc[0])
    if not table:
        raise SystemExit("no finished runs to plot")

    df = pd.DataFrame(table).T                       # rows = models, columns = subtests
    cols = args.columns or list(df.columns)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"unknown columns {missing}; available: {list(df.columns)}")
    df = df[cols]
    print(df.round(1).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    title = f"{args.dataset}: {' / '.join(args.set) if not args.keys else 'selected runs'}"
    if args.heatmap:
        # Single-hue sequential scale: lighter = lower, darker = higher; numbers in every cell.
        vals = df.values.astype(float)
        fig, ax = plt.subplots(figsize=(max(6, 0.75 * len(cols) + 3), max(2.5, 0.55 * len(df) + 1.5)))
        im = ax.imshow(vals, cmap="Blues", vmin=0, vmax=max(100, float(np.nanmax(vals))), aspect="auto")
        for r in range(vals.shape[0]):
            for c in range(vals.shape[1]):
                v = vals[r, c]
                ax.text(c, r, "-" if np.isnan(v) else f"{v:.0f}", ha="center", va="center", fontsize=9,
                        color="white" if v > 60 else "#1a1a1a")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, fontsize=10)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df.index, fontsize=10)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="score (%)")
        ax.set_title(title, loc="left")
        fig.tight_layout()
        out = Path(args.out or f"{args.results}/plots/{args.dataset}_{'_'.join(args.set)}_heatmap.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        df.round(2).to_csv(out.with_suffix(".csv"))
        print(f"\nwritten: {out}  and  {out.with_suffix('.csv')}")
        return

    n_models = len(df)
    width = 0.8 / n_models
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(cols) * max(1, n_models / 3) + 3), 5.5))
    x = np.arange(len(cols))
    for i, (model, row) in enumerate(df.iterrows()):
        bars = ax.bar(x + (i - (n_models - 1) / 2) * width, row.values, width, label=model,
                      color=COLORS[i % len(COLORS)])
        if len(cols) * n_models <= 60:
            for b, v in zip(bars, row.values):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.0f}", ha="center", va="bottom",
                        fontsize=7, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(cols, fontsize=10)
    ax.set_ylabel("score (%)")
    ax.set_ylim(0, min(105, max(10, float(np.nanmax(df.values)) * 1.15)))
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left")
    ax.legend(ncol=min(n_models, 5), fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.tight_layout()

    out = Path(args.out or f"{args.results}/plots/{args.dataset}_{'_'.join(args.set)}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    df.round(2).to_csv(out.with_suffix(".csv"))
    print(f"\nwritten: {out}  and  {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()

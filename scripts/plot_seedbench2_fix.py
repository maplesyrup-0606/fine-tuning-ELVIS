#!/usr/bin/env python3
"""
Plots for the fixed SEED-Bench-2 runs (SEEDBench2_Fix, all 24,371 questions asked), base + 8 v3 models.

  overall.png   left: accuracy per model, fixed (filled) and old collided run (hollow, if present)
                right: FT - base with a paired 95% band (+-2 SE over the same questions)
  per_type.png  27 question types x 8 FT models: FT - base in points; cells outside the paired
                +-2 SE band are outlined. Left column: base accuracy and n.
  base_old_new.png  per type, base old (collided) vs fixed accuracy: what the fix changed

Reads results/*/<key>_SEEDBench2_Fix_*_result.xlsx (and the old SEEDBench2 ones) via
seedbench2_fix_compare.load(). Usage:
  python scripts/plot_seedbench2_fix.py [--results results] [--out results/seedbench2_fix_plots]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from eval_progress import MODELS, keys_for  # noqa: E402
from seedbench2_fix_compare import TYPES, load  # noqa: E402

INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
BASE_C, FT_C, OLD_C = "#4b5563", "#2563eb", "#9ca3af"


def paired(h, hb):
    """FT - base in points and its paired 2 SE (same questions)."""
    d = h.astype(float) - hb.astype(float)
    n = len(d)
    return d.mean() * 100, 2 * d.std(ddof=1) / np.sqrt(n) * 100 if n > 1 else np.nan


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="t0p-cp")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/seedbench2_fix_plots")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    new, old = {}, {}
    for m, key in zip(MODELS, keys_for([a.set], list(MODELS))):
        n = load(key, "SEEDBench2_Fix", a.results)
        if n is None or len(n) != 24371:
            print(f"[skip] {m}: fixed result missing or incomplete ({None if n is None else len(n)} rows)")
            continue
        new[m] = n
        o = load(key, "SEEDBench2", a.results)
        if o is not None:
            old[m] = o
    if "base" not in new:
        raise SystemExit("no fixed base result")

    # align every model on the base row order
    b = new["base"]
    order = b["match"]
    for m in new:
        new[m] = new[m].set_index("match").loc[order].reset_index()
    for m in old:
        old[m] = old[m].drop_duplicates("match").set_index("match").reindex(order).reset_index()
    tid = b["index"].astype(str).str.split("_").str[0].astype(int).values
    hb = new["base"]["hit"].values
    fts = [m for m in new if m != "base"]

    # ---------- overall ----------
    models = list(new)
    y = np.arange(len(models))[::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.1]})
    for yi, m in zip(y, models):
        acc = new[m]["hit"].mean() * 100
        c = BASE_C if m == "base" else FT_C
        if m in old:
            oa = old[m]["hit"].mean() * 100
            a1.plot([oa, acc], [yi, yi], color=GRID, lw=2, zorder=1)
            a1.scatter(oa, yi, s=60, facecolor="white", edgecolor=OLD_C, lw=2, zorder=2)
        a1.scatter(acc, yi, s=70, color=c, zorder=3, edgecolor="white", lw=1.5)
        a1.annotate(f"{acc:.1f}", (acc, yi), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK)
    a1.set_yticks(y, models)
    a1.set_xlabel("accuracy (%)")
    a1.set_title("Overall accuracy: fixed (filled)" + (" vs old collided run (hollow)" if old else ""),
                 fontsize=10, loc="left", color=INK)
    style(a1)

    yf = np.arange(len(fts))[::-1]
    for yi, m in zip(yf, fts):
        d, se2 = paired(new[m]["hit"].values, hb)
        a2.errorbar(d, yi, xerr=se2, fmt="o", color=FT_C, ecolor=FT_C, elinewidth=2, capsize=0, ms=7,
                    mec="white", mew=1.5)
        a2.text(max(d + se2, 0) + 0.05, yi, f"{d:+.2f}", va="center", fontsize=9, color=INK)
    a2.axvline(0, color=MUTED, lw=1)
    a2.set_yticks(yf, fts)
    lim = max(1.0, max(abs(paired(new[m]["hit"].values, hb)[0]) + paired(new[m]["hit"].values, hb)[1]
                       for m in fts) * 1.4)
    a2.set_xlim(-lim, lim)
    a2.set_xlabel("FT − base (points), bar = paired ±2 SE")
    a2.set_title(f"FT − base (base {hb.mean() * 100:.1f}%, n = {len(hb):,})",
                 fontsize=10, loc="left", color=INK)
    style(a2)
    fig.tight_layout()
    fig.savefig(out / "overall.png", dpi=160)
    plt.close(fig)

    # ---------- per type heatmap ----------
    types = sorted(set(tid))
    D = np.full((len(types), len(fts)), np.nan)
    S = np.zeros_like(D, dtype=bool)
    nt, base_acc = [], []
    for i, t in enumerate(types):
        sel = tid == t
        nt.append(int(sel.sum()))
        base_acc.append(hb[sel].mean() * 100)
        for j, m in enumerate(fts):
            d, se2 = paired(new[m]["hit"].values[sel], hb[sel])
            D[i, j] = d
            S[i, j] = abs(d) > se2
    vmax = max(5.0, np.nanpercentile(np.abs(D), 98))
    fig, ax = plt.subplots(figsize=(1.0 * len(fts) + 5.2, 0.36 * len(types) + 1.8))
    im = ax.imshow(D, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(len(types)):
        for j in range(len(fts)):
            v = D[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v) > vmax * 0.6 else INK,
                    fontweight="bold" if S[i, j] else "normal")
            if S[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.4))
    ax.set_xticks(range(len(fts)), fts, rotation=35, ha="right")
    ax.set_yticks(range(len(types)),
                  [f"{t:>2}. {TYPES[t - 1]}  (n={n}, base {ba:.0f}%)" for t, n, ba in zip(types, nt, base_acc)],
                  fontsize=8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("FT − base (points)")
    ax.set_title(f"Fixed SEED-Bench-2, per type: FT − base. Outlined = outside paired ±2 SE "
                 f"({int(S.sum())} of {S.size}; ~{S.size * 0.05:.0f} expected by chance)",
                 fontsize=10, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(out / "per_type.png", dpi=160)
    plt.close(fig)

    # ---------- base old vs fixed per type ----------
    if "base" in old:
        ho = old["base"]["hit"].values.astype(float)
        nint = b["index"].astype(str).str.replace("_", "").astype(int)
        coll = (nint.map(nint.value_counts()) > 1).values
        fig, ax = plt.subplots(figsize=(9, 0.32 * len(types) + 1.6))
        yy = np.arange(len(types))[::-1]
        for yi, t in zip(yy, types):
            sel = tid == t
            o, nw = np.nanmean(ho[sel]) * 100, hb[sel].mean() * 100
            ax.plot([o, nw], [yi, yi], color=GRID, lw=2, zorder=1)
            ax.scatter(o, yi, s=45, facecolor="white", edgecolor=OLD_C, lw=2, zorder=2)
            ax.scatter(nw, yi, s=50, color=BASE_C, zorder=3)
            ax.text(101, yi, f"{coll[sel].mean() * 100:.0f}%", va="center", fontsize=8, color=MUTED)
        ax.text(101, yy[0] + 1, "collided", fontsize=8, color=MUTED)
        ax.set_yticks(yy, [f"{t:>2}. {TYPES[t - 1]}" for t in types], fontsize=8)
        ax.set_xlim(0, 100)
        ax.set_xlabel("base accuracy (%)")
        ax.set_title(f"Base: old collided run (hollow, {np.nanmean(ho) * 100:.1f}%) vs fixed "
                     f"(filled, {hb.mean() * 100:.1f}%)", fontsize=10, loc="left", color=INK)
        style(ax)
        fig.tight_layout()
        fig.savefig(out / "base_old_new.png", dpi=160)
        plt.close(fig)

    print(f"models: {list(new)}; old runs found: {list(old)}")
    print(f"wrote {sorted(p.name for p in out.glob('*.png'))} to {out}")


if __name__ == "__main__":
    main()

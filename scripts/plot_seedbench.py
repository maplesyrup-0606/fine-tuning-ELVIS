#!/usr/bin/env python3
"""
Plot SEED-Bench-2 accuracy from VLMEvalKit result xlsx files.

Two viz modes, chosen with `--group-by`:

- `subpart` (default): 6-axis radar at the coarse L1/L2/L3 subpart level.
  Works from VLMEvalKit's native `*_gpt-4o-mini_result.xlsx`.
- `skill`: 6-panel faceted horizontal bar chart, one panel per subpart,
  bars per fine-grained skill (27 total from the paper's taxonomy).
  Requires `*_enriched.xlsx` produced by `scripts/enrich_seedbench.py`.

Supports one or more models — pass each `--result <dir>[:label]`.

Usage (default radar):
    python scripts/plot_seedbench.py --result results/seedbench_base:base

Usage (faceted skill bars — requires enriched xlsx):
    python scripts/enrich_seedbench.py --input results/seedbench_base/Qwen3-VL-4B-Instruct_SEEDBench2_gpt-4o-mini_result.xlsx
    python scripts/plot_seedbench.py --result results/seedbench_base:base --group-by skill
"""

import argparse
from pathlib import Path

import pandas as pd


SUBPART_ORDER = [
    "Single-Image & Text Comprehension",
    "Multiple-Images & Text Comprehension",
    "Video & Text Comprehension",
    "Interleaved Image & Text Comprehension",
    "Image Generation",
    "Image & Text Generation",
]
LEVEL_OF = {
    "Single-Image & Text Comprehension": "L1",
    "Multiple-Images & Text Comprehension": "L1",
    "Video & Text Comprehension": "L1",
    "Interleaved Image & Text Comprehension": "L2",
    "Image Generation": "L3",
    "Image & Text Generation": "L3",
}
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]


def load_result(result_dir: Path, label: str):
    """Prefer *_enriched.xlsx (has question_type), fall back to raw result xlsx."""
    enriched = sorted(result_dir.glob("*_enriched.xlsx"))
    raw = sorted(result_dir.glob("*_gpt-4o-mini_result.xlsx"))
    source = enriched[0] if enriched else (raw[0] if raw else None)
    if source is None:
        raise FileNotFoundError(
            f"no *_enriched.xlsx or *_gpt-4o-mini_result.xlsx in {result_dir}")

    df = pd.read_excel(source)
    for col in ("hit", "subpart", "level"):
        if col not in df.columns:
            raise ValueError(f"{source} missing column: {col}")

    per_subpart = (df.groupby("subpart")
                     .agg(n=("hit", "size"), acc=("hit", "mean"))
                     .reset_index())
    per_skill = None
    if "question_type" in df.columns and "question_type_id" in df.columns:
        per_skill = (df.groupby(["subpart", "question_type_id", "question_type"])
                       .agg(n=("hit", "size"), acc=("hit", "mean"))
                       .reset_index())
    return {
        "label": label,
        "overall": df["hit"].mean(),
        "n_total": len(df),
        "per_subpart": per_subpart,
        "per_skill": per_skill,
        "source": str(source),
        "enriched": enriched != [],
    }


def parse_result_arg(arg: str):
    if ":" in arg:
        d, lab = arg.rsplit(":", 1)
    else:
        d, lab = arg, Path(arg).name
    return Path(d), lab


def plot_radar(results, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    subparts = SUBPART_ORDER
    n = len(subparts)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 9), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for i, r in enumerate(results):
        subpart_acc = {row.subpart: row.acc * 100
                       for row in r["per_subpart"].itertuples()}
        vals = [subpart_acc.get(s, 0) for s in subparts]
        vals += vals[:1]
        color = COLORS[i % len(COLORS)]
        ax.plot(angles, vals, "o-", color=color, linewidth=2,
                label=f"{r['label']} (overall {r['overall']*100:.2f}%)")
        ax.fill(angles, vals, color=color, alpha=0.15)
        ax.plot(angles, [r["overall"] * 100] * (n + 1),
                color=color, linestyle=":", linewidth=1, alpha=0.7)

    labels = [f"{s.replace(' & ', '\n& ')}\n[{LEVEL_OF[s]}]" for s in subparts]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=8)
    ax.set_rlabel_position(135)
    ax.grid(True, alpha=0.4)

    if results:
        r = results[0]
        subpart_acc = {row.subpart: row.acc * 100
                       for row in r["per_subpart"].itertuples()}
        subpart_n = {row.subpart: row.n for row in r["per_subpart"].itertuples()}
        for a, s in zip(angles[:-1], subparts):
            v = subpart_acc.get(s, 0)
            ax.annotate(f"{v:.1f}%\n(n={subpart_n.get(s,0)})",
                        xy=(a, v), xytext=(a, v + 7),
                        ha="center", fontsize=7, color=COLORS[0])

    ax.set_title("SEED-Bench-2 per-subpart accuracy\n"
                 "(dotted ring = each model's overall accuracy)",
                 pad=28, fontsize=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot written: {out_path}")


def plot_faceted_bars(results, out_path: Path):
    """6-panel faceted horizontal bars, one panel per subpart, bar per skill."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    for r in results:
        if r["per_skill"] is None:
            raise RuntimeError(
                f"{r['label']}: skill-level data unavailable. "
                f"Run `enrich_seedbench.py --input {r['source']}` first "
                f"to produce *_enriched.xlsx.")

    subpart_skills = {}
    for sp in SUBPART_ORDER:
        skills_in_sp = {}
        for r in results:
            sub = r["per_skill"][r["per_skill"]["subpart"] == sp]
            for row in sub.itertuples():
                skills_in_sp[(row.question_type_id, row.question_type)] = True
        subpart_skills[sp] = sorted(skills_in_sp.keys())

    counts = [max(1, len(subpart_skills[sp])) for sp in SUBPART_ORDER]
    total = sum(counts)
    fig = plt.figure(figsize=(13, max(10, 0.45 * total + 4)))
    gs = fig.add_gridspec(len(SUBPART_ORDER), 1, height_ratios=counts,
                          hspace=0.9)

    n_models = len(results)
    bar_h = 0.8 / max(1, n_models)
    xmax = 100

    for pi, sp in enumerate(SUBPART_ORDER):
        ax = fig.add_subplot(gs[pi])
        skills = subpart_skills[sp]
        y = np.arange(len(skills))

        for mi, r in enumerate(results):
            sub = r["per_skill"][r["per_skill"]["subpart"] == sp]
            qid_to = {(row.question_type_id, row.question_type):
                      (row.acc * 100, row.n) for row in sub.itertuples()}
            vals = [qid_to.get(k, (0, 0))[0] for k in skills]
            ns = [qid_to.get(k, (0, 0))[1] for k in skills]
            offset = (mi - (n_models - 1) / 2) * bar_h
            color = COLORS[mi % len(COLORS)]
            bars = ax.barh(y + offset, vals, bar_h, color=color,
                           label=r["label"] if pi == 0 else None)
            for b, v, nn in zip(bars, vals, ns):
                ax.text(min(v + 1.0, xmax - 0.5),
                        b.get_y() + b.get_height() / 2,
                        f"{v:.1f}%  (n={nn})", va="center", fontsize=7,
                        color=color)

        for mi, r in enumerate(results):
            row = r["per_subpart"][r["per_subpart"]["subpart"] == sp]
            if len(row):
                ax.axvline(row.iloc[0]["acc"] * 100,
                           color=COLORS[mi % len(COLORS)],
                           linestyle=":", linewidth=1, alpha=0.6)

        ax.set_yticks(y)
        ax.set_yticklabels([f"{qid}. {qname}" for qid, qname in skills],
                           fontsize=9)
        ax.set_xlim(0, xmax)
        ax.invert_yaxis()
        ax.set_title(f"[{LEVEL_OF[sp]}]  {sp}", loc="left", fontsize=10,
                     fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        if pi < len(SUBPART_ORDER) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("accuracy (%)")

    handles, labels = [], []
    for mi, r in enumerate(results):
        handles.append(plt.Rectangle((0, 0), 1, 1,
                                     color=COLORS[mi % len(COLORS)]))
        labels.append(f"{r['label']} (overall {r['overall']*100:.2f}%)")
    fig.legend(handles, labels, loc="upper center", ncol=min(n_models, 3),
               bbox_to_anchor=(0.5, 0.995), fontsize=10)

    fig.suptitle("SEED-Bench-2 per-skill accuracy "
                 "(dotted = each model's subpart overall)",
                 y=1.005, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot written: {out_path}")


def plot_per_subpart(results, out_dir: Path):
    """One PNG per subpart (6 total), each sized for its own skill count."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import re

    for r in results:
        if r["per_skill"] is None:
            raise RuntimeError(
                f"{r['label']}: skill-level data unavailable. "
                f"Run `enrich_seedbench.py --input {r['source']}` first.")

    n_models = len(results)
    bar_h = 0.8 / max(1, n_models)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for sp in SUBPART_ORDER:
        skills_seen = {}
        for r in results:
            sub = r["per_skill"][r["per_skill"]["subpart"] == sp]
            for row in sub.itertuples():
                skills_seen[(row.question_type_id, row.question_type)] = True
        skills = sorted(skills_seen.keys())
        if not skills:
            continue

        fig_h = max(3.5, 0.55 * len(skills) * n_models + 2.0)
        fig, ax = plt.subplots(figsize=(13, fig_h))
        y = np.arange(len(skills))

        overalls = []
        for mi, r in enumerate(results):
            sub = r["per_skill"][r["per_skill"]["subpart"] == sp]
            qid_to = {(row.question_type_id, row.question_type):
                      (row.acc * 100, row.n) for row in sub.itertuples()}
            vals = [qid_to.get(k, (0, 0))[0] for k in skills]
            ns = [qid_to.get(k, (0, 0))[1] for k in skills]
            offset = (mi - (n_models - 1) / 2) * bar_h
            color = COLORS[mi % len(COLORS)]

            subrow = r["per_subpart"][r["per_subpart"]["subpart"] == sp]
            sp_overall = (subrow.iloc[0]["acc"] * 100) if len(subrow) else None
            overalls.append((r["label"], sp_overall, color))

            legend_lbl = (f"{r['label']} (subpart overall "
                          f"{sp_overall:.1f}%)" if sp_overall is not None
                          else r["label"])
            bars = ax.barh(y + offset, vals, bar_h, color=color,
                           label=legend_lbl)
            for b, v, nn in zip(bars, vals, ns):
                ax.text(min(v + 0.7, 100 - 0.5),
                        b.get_y() + b.get_height() / 2,
                        f"{v:.1f}%  (n={nn})",
                        va="center", fontsize=10, color=color)

        for lbl, ov, color in overalls:
            if ov is not None:
                ax.axvline(ov, color=color, linestyle=":", linewidth=1.2,
                           alpha=0.65)

        ax.set_yticks(y)
        ax.set_yticklabels([f"{qid}. {qname}" for qid, qname in skills],
                           fontsize=11)
        ax.set_xlim(0, 100)
        ax.invert_yaxis()
        ax.set_xlabel("accuracy (%)", fontsize=11)
        ax.set_title(f"[{LEVEL_OF[sp]}]  {sp}\n"
                     f"SEED-Bench-2 per-skill accuracy "
                     f"(dotted = each model's subpart overall)",
                     fontsize=12, loc="left")
        ax.grid(axis="x", alpha=0.3)
        ax.legend(loc="lower right", fontsize=10)

        fig.tight_layout()
        slug = re.sub(r"[^a-z0-9]+", "_", sp.lower()).strip("_")
        out_path = out_dir / f"seedbench_skill_{slug}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)

    for p in written:
        print(f"Plot written: {p}")


def write_csv(results, csv_path: Path, group_by: str):
    rows = []
    for r in results:
        if group_by == "skill" and r["per_skill"] is not None:
            for row in r["per_skill"].itertuples():
                rows.append({
                    "label": r["label"],
                    "level": LEVEL_OF.get(row.subpart, "?"),
                    "subpart": row.subpart,
                    "question_type_id": row.question_type_id,
                    "question_type": row.question_type,
                    "n": row.n,
                    "accuracy": row.acc,
                })
        else:
            for row in r["per_subpart"].itertuples():
                rows.append({
                    "label": r["label"],
                    "level": LEVEL_OF.get(row.subpart, "?"),
                    "subpart": row.subpart,
                    "n": row.n,
                    "accuracy": row.acc,
                })
        rows.append({"label": r["label"], "level": "ALL", "subpart": "Overall",
                     "n": r["n_total"], "accuracy": r["overall"]})
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"CSV written: {csv_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--result", action="append", required=True,
                   help="<dir>[:label], repeat for each model")
    p.add_argument("--group-by", choices=["subpart", "skill"], default="subpart",
                   help="radar at subpart level (default) or "
                        "faceted bars at skill level (requires enriched xlsx)")
    p.add_argument("--out", default=None,
                   help="PNG output path (default: <first-result-dir>/"
                        "seedbench_<group>.png)")
    p.add_argument("--csv", default=None,
                   help="optional CSV summary path")
    p.add_argument("--write-csv", action="store_true",
                   help="write CSV alongside the PNG using the default path")
    p.add_argument("--per-subpart", action="store_true",
                   help="skill mode only: emit one PNG per subpart (6 total) "
                        "into the first result dir, instead of a single combined "
                        "faceted PNG. Each PNG sized for its own skill count.")
    args = p.parse_args()

    results = []
    first_dir = None
    for arg in args.result:
        d, lab = parse_result_arg(arg)
        if not d.is_dir():
            p.error(f"not a directory: {d}")
        if first_dir is None:
            first_dir = d
        results.append(load_result(d, lab))

    default_stem = f"seedbench_{args.group_by}"
    if args.out is None:
        args.out = str(first_dir / f"{default_stem}.png")
    if args.csv is None and args.write_csv:
        args.csv = str(first_dir / f"{default_stem}.csv")

    print("\n=== Per-subpart accuracy ===")
    print(f"{'label':<16}{'level':<6}{'subpart':<42}{'n':>6}  acc")
    for r in results:
        for row in r["per_subpart"].itertuples():
            print(f"{r['label']:<16}{LEVEL_OF.get(row.subpart,'?'):<6}"
                  f"{row.subpart:<42}{row.n:>6}  {row.acc*100:6.2f}%")
        print(f"{r['label']:<16}{'ALL':<6}{'Overall':<42}{r['n_total']:>6}"
              f"  {r['overall']*100:6.2f}%")
        print()

    if args.group_by == "subpart":
        plot_radar(results, Path(args.out))
    elif args.per_subpart:
        plot_per_subpart(results, first_dir)
    else:
        plot_faceted_bars(results, Path(args.out))

    if args.csv:
        write_csv(results, Path(args.csv), args.group_by)


if __name__ == "__main__":
    main()

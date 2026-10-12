#!/usr/bin/env python3
"""
Plot OCRBench_v2 per-subtask accuracy from VLMEvalKit `_score.json` files.

Schema (confirmed from a real score.json):
  - English subtasks (8): en_text_recognition, en_text_detection,
    en_text_spotting, en_relationship_extraction, en_element_parsing,
    en_mathematical_calculation, en_visual_text_understanding,
    en_knowledge_reasoning
  - Chinese subtasks (5): cn_text_recognition, cn_relationship_extraction,
    cn_element_parsing, cn_visual_text_understanding, cn_knowledge_reasoning
  - Overall: "English Overall Score", "Chinese Overall Score"

Chinese has fewer subtasks than English (no text_detection/spotting/math).

Usage:
    python scripts/plot_ocrbench_v2.py \\
        --result results/OCRBench_v2_Qwen3-VL-4B-Instruct:base \\
        --result results/OCRBench_v2_Qwen3-VL-4B-Instruct-llm-e2:llm/e2 \\
        --result results/OCRBench_v2_Qwen3-VL-4B-Instruct-both-e2:both/e2 \\
        --write-csv
"""

import argparse
import csv
import json
from pathlib import Path


EN_TASKS = [
    "text_recognition",
    "text_detection",
    "text_spotting",
    "relationship_extraction",
    "element_parsing",
    "mathematical_calculation",
    "visual_text_understanding",
    "knowledge_reasoning",
]
CN_TASKS = [
    "text_recognition",
    "relationship_extraction",
    "element_parsing",
    "visual_text_understanding",
    "knowledge_reasoning",
]

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]


def parse_result_arg(arg: str):
    if ":" in arg:
        d, lab = arg.rsplit(":", 1)
    else:
        d, lab = arg, Path(arg).name
    return Path(d), lab


def load_scores(result_dir: Path, label: str):
    jsons = sorted(result_dir.glob("*_OCRBench_v2_score.json"))
    if not jsons:
        raise FileNotFoundError(
            f"no *_OCRBench_v2_score.json in {result_dir}")
    with open(jsons[0]) as f:
        d = json.load(f)
    return {
        "label": label,
        "source": str(jsons[0]),
        "english": {t: d.get(f"en_{t}") for t in EN_TASKS},
        "chinese": {t: d.get(f"cn_{t}") for t in CN_TASKS},
        "english_overall": d.get("English Overall Score"),
        "chinese_overall": d.get("Chinese Overall Score"),
    }


def plot(results, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n_models = len(results)
    bar_h = 0.8 / max(1, n_models)

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[len(EN_TASKS) + 1,
                                               len(CN_TASKS) + 1],
                          hspace=0.4)
    ax_en = fig.add_subplot(gs[0])
    ax_cn = fig.add_subplot(gs[1])

    def _plot_panel(ax, tasks, lang_key, overall_key, title):
        y = np.arange(len(tasks) + 1)
        labels = tasks + ["OVERALL"]

        for mi, r in enumerate(results):
            vals = []
            for t in tasks:
                v = r[lang_key].get(t)
                vals.append((v * 100) if isinstance(v, (int, float)) else 0)
            overall = r[overall_key]
            vals.append((overall * 100) if isinstance(overall, (int, float)) else 0)

            offset = (mi - (n_models - 1) / 2) * bar_h
            color = COLORS[mi % len(COLORS)]
            ov_label = (f"{r['label']} ({overall*100:.1f}%)"
                        if isinstance(overall, (int, float))
                        else r["label"])
            bars = ax.barh(y + offset, vals, bar_h, color=color,
                           label=ov_label)
            for b, v in zip(bars, vals):
                ax.text(min(v + 0.8, 100 - 0.5),
                        b.get_y() + b.get_height() / 2,
                        f"{v:.1f}", va="center", fontsize=8,
                        color=color)

        ax.axhline(len(tasks) - 0.5, color="gray", linestyle="--",
                   alpha=0.5, linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        for tick, lbl in zip(ax.get_yticklabels(), labels):
            if lbl == "OVERALL":
                tick.set_fontweight("bold")
        ax.set_xlim(0, 100)
        ax.invert_yaxis()
        ax.set_xlabel("score (%)")
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
        ax.grid(axis="x", alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)

    _plot_panel(ax_en, EN_TASKS, "english", "english_overall",
                "English OCR subtasks")
    _plot_panel(ax_cn, CN_TASKS, "chinese", "chinese_overall",
                "Chinese OCR subtasks")

    fig.suptitle("OCRBench_v2 per-subtask accuracy", fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot written: {out_path}")


def write_csv(results, csv_path: Path):
    rows = []
    for r in results:
        for t in EN_TASKS:
            rows.append({"label": r["label"], "lang": "en",
                         "task": t, "score": r["english"].get(t)})
        rows.append({"label": r["label"], "lang": "en",
                     "task": "OVERALL", "score": r["english_overall"]})
        for t in CN_TASKS:
            rows.append({"label": r["label"], "lang": "cn",
                         "task": t, "score": r["chinese"].get(t)})
        rows.append({"label": r["label"], "lang": "cn",
                     "task": "OVERALL", "score": r["chinese_overall"]})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "lang", "task", "score"])
        w.writeheader()
        w.writerows(rows)
    print(f"CSV written: {csv_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--result", action="append", required=True,
                   help="<dir>[:label], repeat for each model")
    p.add_argument("--out", default=None,
                   help="PNG output path "
                        "(default: <first-result-dir>/ocrbench_v2_breakdown.png)")
    p.add_argument("--csv", default=None, help="optional CSV summary path")
    p.add_argument("--write-csv", action="store_true",
                   help="write CSV alongside the PNG using the default path")
    args = p.parse_args()

    results = []
    first_dir = None
    for arg in args.result:
        d, lab = parse_result_arg(arg)
        if not d.is_dir():
            p.error(f"not a directory: {d}")
        if first_dir is None:
            first_dir = d
        results.append(load_scores(d, lab))

    if args.out is None:
        args.out = str(first_dir / "ocrbench_v2_breakdown.png")
    if args.csv is None and args.write_csv:
        args.csv = str(first_dir / "ocrbench_v2_breakdown.csv")

    print("\n=== OCRBench_v2 scores ===")
    header = f"{'label':<16}{'lang':<6}{'task':<30}{'score':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        for lang_key, tasks, overall_key in [
                ("english", EN_TASKS, "english_overall"),
                ("chinese", CN_TASKS, "chinese_overall")]:
            lang = lang_key[:2]
            for t in tasks:
                v = r[lang_key].get(t)
                vs = f"{v*100:6.2f}%" if isinstance(v, (int, float)) else "   —  "
                print(f"{r['label']:<16}{lang:<6}{t:<30}{vs:>8}")
            v = r[overall_key]
            vs = f"{v*100:6.2f}%" if isinstance(v, (int, float)) else "   —  "
            print(f"{r['label']:<16}{lang:<6}{'OVERALL':<30}{vs:>8}")
        print()

    plot(results, Path(args.out))
    if args.csv:
        write_csv(results, Path(args.csv))


if __name__ == "__main__":
    main()

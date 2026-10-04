#!/usr/bin/env python3
"""
Render N random continuity training examples as a contact-sheet PNG with
rule + answer annotations, for auditing training data quality (shortcuts,
rule drift, image/rule mismatch).

Each cell shows:
  - the training image
  - a positive/negative badge (ground-truth from the jsonl)
  - the pattern name (abbreviated)
  - the Stage 1 rule text (truncated)

The question text (``Classify this image as Positive or Negative``) is
constant across all 5,184 examples, so it goes in the figure title rather
than per-cell to save space.

Usage:
    # On Fir, images in place
    python scripts/inspect_training_sample.py \\
        --images-root $ELVIS_DATA/continuity/train \\
        --num 50 --out data/training_sample.png

    # Locally, after pulling an elvis_sample.tgz
    python scripts/inspect_training_sample.py \\
        --images-root ~/elvis_sample \\
        --num 50 --out data/training_sample.png
"""

import argparse
import json
import random
import re
import textwrap
from pathlib import Path


FIR_PREFIX_PATTERNS = [
    "/scratch/merc0606/ELVIS/data/continuity/train/",
    "/scratch/merc0606/ELVIS/data/continuity/train",
]

RULE_RE = re.compile(
    r"Using the following reasoning rules:\s*(.+?)\s*Classify this image",
    re.DOTALL,
)


def rewrite_image_path(abs_path: str, images_root: Path) -> Path:
    for prefix in FIR_PREFIX_PATTERNS:
        if abs_path.startswith(prefix):
            tail = abs_path[len(prefix):].lstrip("/")
            return images_root / tail
    parts = Path(abs_path).parts
    if len(parts) >= 3:
        return images_root / Path(*parts[-3:])
    return images_root / Path(abs_path).name


def extract_rule(user_content: str) -> str:
    m = RULE_RE.search(user_content)
    if not m:
        return "(rule parse failed)"
    return m.group(1).strip()


def pattern_from_path(abs_path: str) -> str:
    parts = Path(abs_path).parts
    if len(parts) >= 3:
        return parts[-3]
    return "?"


def abbreviate_pattern(name: str, max_len: int = 48) -> str:
    if len(name) <= max_len:
        return name
    return name[: max_len - 1] + "…"


def load_entries(jsonl_path: Path):
    with open(jsonl_path) as f:
        return [json.loads(line) for line in f]


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--jsonl", default="data/continuity/train.jsonl")
    p.add_argument("--images-root", required=True,
                   help="dir containing <pattern>/<positive|negative>/*.png")
    p.add_argument("--num", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--balance", action="store_true",
                   help="force equal positive/negative counts")
    p.add_argument("--rule-chars", type=int, default=220,
                   help="truncate rule text to this many characters")
    p.add_argument("--cell-size", type=float, default=3.0,
                   help="width (inches) per grid cell")
    p.add_argument("--out", default="data/training_sample.png")
    args = p.parse_args()

    jsonl = Path(args.jsonl)
    images_root = Path(args.images_root).expanduser()
    if not jsonl.is_file():
        p.error(f"not a file: {jsonl}")
    if not images_root.is_dir():
        p.error(f"not a dir: {images_root}")

    entries = load_entries(jsonl)
    print(f"Loaded {len(entries)} training entries from {jsonl}")

    rng = random.Random(args.seed)
    if args.balance:
        pos = [e for e in entries if e["messages"][-1]["content"].strip().lower() == "positive"]
        neg = [e for e in entries if e["messages"][-1]["content"].strip().lower() == "negative"]
        half = args.num // 2
        sample = (rng.sample(pos, min(half, len(pos)))
                  + rng.sample(neg, min(args.num - half, len(neg))))
        rng.shuffle(sample)
    else:
        sample = rng.sample(entries, min(args.num, len(entries)))

    n = len(sample)
    cols = 5 if n >= 20 else max(1, int(n ** 0.5))
    rows = (n + cols - 1) // cols
    unique_patterns = {pattern_from_path(e["images"][0]) for e in sample}
    print(f"Sampled {n} examples ({len(unique_patterns)} distinct patterns), "
          f"grid={rows}x{cols}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    cell_w = args.cell_size
    cell_h = args.cell_size * 1.45
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cell_w * cols, cell_h * rows),
                             squeeze=False)

    missing = 0
    for i, entry in enumerate(sample):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        abs_img = entry["images"][0]
        local_img = rewrite_image_path(abs_img, images_root)
        answer = entry["messages"][-1]["content"].strip().lower()
        user_msg = entry["messages"][0]["content"]
        rule = extract_rule(user_msg)
        pattern = pattern_from_path(abs_img)

        if local_img.is_file():
            try:
                img = Image.open(local_img)
                ax.imshow(img)
            except Exception as ex:
                missing += 1
                ax.text(0.5, 0.5, f"(load error:\n{ex})",
                        ha="center", va="center", fontsize=7,
                        transform=ax.transAxes, color="red")
        else:
            missing += 1
            ax.text(0.5, 0.5, f"(image missing)\n{local_img.name}",
                    ha="center", va="center", fontsize=7,
                    transform=ax.transAxes, color="red")

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#2ca02c" if answer == "positive" else "#d62728")
            spine.set_linewidth(2)

        rule_short = rule[: args.rule_chars]
        if len(rule) > args.rule_chars:
            rule_short += "…"
        rule_wrapped = "\n".join(
            textwrap.wrap(rule_short,
                          width=int(cell_w * 14),
                          break_long_words=False)
        )
        badge = "POS" if answer == "positive" else "NEG"
        badge_color = "#2ca02c" if answer == "positive" else "#d62728"
        caption = f"{badge}  |  {abbreviate_pattern(pattern, 38)}\n{rule_wrapped}"
        ax.set_title(caption, fontsize=6, loc="left",
                     color=badge_color, pad=4)

    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].axis("off")

    fig.suptitle("Continuity training sample — "
                 "Question: 'Classify this image as Positive or Negative. "
                 "Only answer with positive or negative.'",
                 fontsize=10, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.995])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"\nWrote {out_path}")
    if missing:
        print(f"[warn] {missing} image(s) missing/unreadable "
              f"— check --images-root (jsonl prefix: "
              f"'{FIR_PREFIX_PATTERNS[0]}')")


if __name__ == "__main__":
    main()

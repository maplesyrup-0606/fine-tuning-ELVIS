#!/usr/bin/env python3
"""
Slide-ready PNG cards: the 3+3 example images of a pattern and its programmatic rubric.

One card per principle (or --per-principle N), for randomly sampled patterns. The rubric is
the ground-truth rule from data/<principle>/rubrics.json (generate_programmatic_rules.py
--rubrics); the images are the first 3 positive / 3 negative of <split>/<pattern>
(train_v3 = the Stage 1 examples ELVIS shows). No model involved.

Usage:
  python scripts/rule_cards.py                                   # one card per principle
  python scripts/rule_cards.py --per-principle 3 --seed 1
  python scripts/rule_cards.py --principles closure --pattern 120_
Writes results/rule_cards/<principle>_<pattern id>.png
"""

import argparse
import json
import os
import random
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw

from show_episode import font

ALL_PRINCIPLES = ["continuity", "proximity", "similarity", "closure", "symmetry"]
TILE, PAD, GAP = 224, 16, 28          # GAP separates the positive and negative groups
POS, NEG, INK, BG = "#1a7f4b", "#c0392b", "#0b0b0b", "white"
WRAP = 105


def card(principle, examples, rubric):
    """examples: [(path, is_positive)] x6 -> PIL image."""
    f_title, f_head, f_text = font(30), font(22), font(22)
    width = 2 * PAD + 6 * TILE + 4 * PAD + GAP
    lines = [("Rule (programmatic, from the generator's parameters):", True)]
    for para in rubric.strip().split("\n"):
        lines += [(l, False) for l in (textwrap.wrap(para, WRAP) or [""])]
    height = PAD + 44 + 34 + TILE + PAD + 12 + len(lines) * 30 + PAD
    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)

    d.text((PAD, PAD), principle, fill=INK, font=f_title)
    y_head, y_img = PAD + 44, PAD + 44 + 34
    x = PAD
    for k, (path, is_pos) in enumerate(examples):
        if k == 3:
            x += GAP
        if k in (0, 3):
            d.text((x, y_head), "Positive examples" if is_pos else "Negative examples",
                   fill=POS if is_pos else NEG, font=f_head)
        img.paste(Image.open(path).convert("RGB").resize((TILE, TILE)), (x, y_img))
        d.rectangle([x - 2, y_img - 2, x + TILE + 1, y_img + TILE + 1], outline=POS if is_pos else NEG, width=4)
        x += TILE + PAD
    y = y_img + TILE + PAD + 12
    for text, bold in lines:
        d.text((PAD, y), text, fill=INK, font=f_head if bold else f_text,
               stroke_width=1 if bold else 0, stroke_fill=INK)
        y += 30
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--split", default="train_v3", help="where the example images come from")
    parser.add_argument("--principles", nargs="+", default=ALL_PRINCIPLES)
    parser.add_argument("--per-principle", type=int, default=1)
    parser.add_argument("--pattern", default=None, help="substring of the pattern name (default: random)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/rule_cards")
    args = parser.parse_args()

    data = os.environ.get("ELVIS_DATA")
    if not data:
        sys.exit("ERROR: env var ELVIS_DATA is not set")
    data = Path(data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for principle in args.principles:
        rubrics = json.load(open(f"data/{principle}/rubrics.json"))["rubrics"]
        split_dir = data / principle / args.split
        patterns = sorted(p.name for p in split_dir.iterdir() if p.is_dir() and p.name in rubrics)
        if args.pattern:
            patterns = [p for p in patterns if args.pattern in p]
        if not patterns:
            print(f"[skip] {principle}: no matching pattern with a rubric in {split_dir}")
            continue
        rng = random.Random(f"{args.seed}-{principle}")
        for pattern in rng.sample(patterns, min(args.per_principle, len(patterns))):
            pick = lambda side: sorted((split_dir / pattern / side).glob("*.png"))[:3]
            examples = [(p, True) for p in pick("positive")] + [(p, False) for p in pick("negative")]
            path = out / f"{principle}_{pattern[:3]}.png"
            card(principle, examples, rubrics[pattern]).save(path)
            print(f"{principle:11s} {pattern}  -> {path}")


if __name__ == "__main__":
    main()

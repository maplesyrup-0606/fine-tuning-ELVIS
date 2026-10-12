#!/usr/bin/env python3
"""
Render episodes as contact sheets for checking ground-truth rules by eye.

Each sheet shows one (pattern, episode) from rules_<split>.json:
  row 1: the 6 Stage 1 examples, numbered Image 1-6 in the order ELVIS feeds them
  row 2: the 6 Stage 2 queries with their labels
  below: the rule text, whose `Image 4-6` lines refer to row 1

Usage:
  python scripts/show_episode.py --rules data/continuity/rules_ft_val.json --pattern 066_ --out results/episodes
  python scripts/show_episode.py --rules data/continuity/rules_ft_val.json --random 10 --out results/episodes
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TILE = 224
PAD = 12
LABEL_H = 28
TEXT_LINE_H = 22


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def render(ep, data_root):
    examples = ([("Image %d (pos)" % (i + 1), p) for i, p in enumerate(ep["examples"]["positive"])]
                + [("Image %d (neg)" % (i + 4), p) for i, p in enumerate(ep["examples"]["negative"])])
    queries = ([("query pos", p) for p in ep["queries"]["positive"]]
               + [("query neg", p) for p in ep["queries"]["negative"]])
    rule_lines = [f"{ep['pattern']}  (episode {ep['episode']})", ""] + ep["rule"].split("\n")

    width = PAD + 6 * (TILE + PAD)
    row_h = LABEL_H + TILE + PAD
    height = PAD + 2 * row_h + PAD + len(rule_lines) * TEXT_LINE_H + PAD
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    f_label, f_text = font(16), font(15)

    for row, tiles in enumerate((examples, queries)):
        y = PAD + row * row_h
        for col, (label, rel) in enumerate(tiles):
            x = PAD + col * (TILE + PAD)
            color = "darkgreen" if "pos" in label else "firebrick"
            draw.text((x, y + 4), label, fill=color, font=f_label)
            img = Image.open(data_root / rel).convert("RGB").resize((TILE, TILE))
            sheet.paste(img, (x, y + LABEL_H))
            draw.rectangle([x - 1, y + LABEL_H - 1, x + TILE, y + LABEL_H + TILE], outline=color, width=2)
        if row == 0:  # separate the Stage 1 examples from the Stage 2 queries
            draw.line([PAD, y + row_h - PAD // 2, width - PAD, y + row_h - PAD // 2], fill="gray", width=1)

    y = PAD + 2 * row_h + PAD
    for line in rule_lines:
        draw.text((PAD, y), line, fill="black", font=f_text)
        y += TEXT_LINE_H
    return sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--rules", required=True, help="rules_<split>.json from generate_programmatic_rules.py")
    parser.add_argument("--pattern", default=None, help="substring of the pattern name, e.g. 066_")
    parser.add_argument("--episode", type=int, default=None, help="only this episode index")
    parser.add_argument("--random", type=int, default=0, metavar="N", help="render N random episodes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/episodes")
    args = parser.parse_args()

    if not os.environ.get("ELVIS_DATA"):
        sys.exit("ERROR: env var ELVIS_DATA is not set")
    data_root = Path(os.environ["ELVIS_DATA"])

    with open(args.rules) as f:
        episodes = json.load(f)["episodes"]
    if args.pattern:
        episodes = [e for e in episodes if args.pattern in e["pattern"]]
    if args.episode is not None:
        episodes = [e for e in episodes if e["episode"] == args.episode]
    if args.random:
        episodes = random.Random(args.seed).sample(episodes, min(args.random, len(episodes)))
    if not episodes:
        sys.exit("ERROR: no episodes match")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for ep in episodes:
        path = out / f"episode_{ep['pattern'][:3]}_e{ep['episode']}.png"
        render(ep, data_root).save(path)
        print(path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Render ELVIS eval episodes from a sweep run: images, the model's rule, its answers.

For one eval run dir (results/eval_v2/<variant>/<version>/<checkpoint>, or the
base run dir) and one principle, each sheet shows:
  row 1: the 6 Stage 1 examples ELVIS fed (train/<pattern>/{pos,neg}/00000-00002)
  row 2: the 6 test images (test/<pattern>/...) with the model's answer (OK / WRONG)
  below: the rule the model wrote, and — for continuity — the ground-truth rule
         for the same examples (generate_programmatic_rules.py)
Both rules are also printed as text.

Usage:
  python scripts/show_eval_episode.py --run-dir results/eval_v2/both/v0-.../checkpoint-378 --pattern 066_
  python scripts/show_eval_episode.py --run-dir results/eval_v2/base/<ts> --random 5
"""

import argparse
import json
import os
import random
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw

from rubric_specs import SPECS
from show_episode import LABEL_H, PAD, TEXT_LINE_H, TILE, font

WRAP = 150


def gt_rule(data_root, principle, pattern, color_names):
    if principle not in SPECS:
        return None
    spec = SPECS[principle]
    slots = spec.parse(pattern)
    negs = sorted((data_root / principle / "train" / pattern / "negative").glob("*.json"))[:3]
    return spec.render(slots, [spec.violations(json.load(open(p))["img_data"], slots, color_names)
                               for p in negs])


def sheet(data_root, principle, pattern, entry, model_rule, truth):
    train = data_root / principle / "train" / pattern
    test = data_root / principle / "test" / pattern
    pick = lambda d: sorted(d.glob("*.png"))[:3]
    examples = ([(f"Image {i + 1} (pos)", p, None) for i, p in enumerate(pick(train / "positive"))]
                + [(f"Image {i + 4} (neg)", p, None) for i, p in enumerate(pick(train / "negative"))])
    preds = {s["id"]: s for s in entry["samples"]}
    queries = []
    for side, label in (("positive", 1), ("negative", 0)):
        for i, p in enumerate(pick(test / side)):
            s = preds.get(f"{side}_{i}")
            ok = s is not None and s["predicted"] == label
            queries.append((f"test {side[:3]}: {'OK' if ok else 'WRONG'}", p, ok))

    text = [f"{pattern}   accuracy {entry['accuracy']:.1f}%", "", "MODEL RULE:"]
    text += [l for para in model_rule.split("\n") for l in (textwrap.wrap(para, WRAP) or [""])]
    if truth:
        text += ["", "GROUND-TRUTH RULE (same examples):"] + truth.split("\n")

    width = PAD + 6 * (TILE + PAD)
    row_h = LABEL_H + TILE + PAD
    height = PAD + 2 * row_h + PAD + len(text) * TEXT_LINE_H + PAD
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    f_label, f_text = font(16), font(15)
    for row, tiles in enumerate((examples, queries)):
        y = PAD + row * row_h
        for col, (label, path, ok) in enumerate(tiles):
            x = PAD + col * (TILE + PAD)
            color = ("darkgreen" if "pos" in label else "firebrick") if ok is None else \
                    ("darkgreen" if ok else "firebrick")
            draw.text((x, y + 4), label, fill=color, font=f_label)
            img.paste(Image.open(path).convert("RGB").resize((TILE, TILE)), (x, y + LABEL_H))
            draw.rectangle([x - 1, y + LABEL_H - 1, x + TILE, y + LABEL_H + TILE], outline=color,
                           width=2 if ok is None else 4)
        if row == 0:
            draw.line([PAD, y + row_h - PAD // 2, width - PAD, y + row_h - PAD // 2], fill="gray", width=1)
    y = PAD + 2 * row_h + PAD
    for line in text:
        draw.text((PAD, y), line, fill="black", font=f_text)
        y += TEXT_LINE_H
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--run-dir", required=True, help="dir containing <principle>/ result folders")
    parser.add_argument("--principle", default="continuity")
    parser.add_argument("--pattern", default=None, help="substring of the pattern name, e.g. 066_")
    parser.add_argument("--random", type=int, default=0, metavar="N")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="default: <run-dir>/samples")
    args = parser.parse_args()

    for env in ("ELVIS_ROOT", "ELVIS_DATA"):
        if not os.environ.get(env):
            sys.exit(f"ERROR: env var {env} is not set")
    sys.path.insert(0, os.environ["ELVIS_ROOT"])
    from scripts import config
    data_root = Path(os.environ["ELVIS_DATA"])

    run_dir = Path(args.run_dir)
    files = sorted((run_dir / args.principle).glob("*.json"))
    files = [f for f in files if not f.name.endswith(".tmp.json")]
    if not files:
        sys.exit(f"ERROR: no result JSON in {run_dir / args.principle}")
    results = json.load(open(files[-1]))

    patterns = sorted(results)
    if args.pattern:
        patterns = [p for p in patterns if args.pattern in p]
    if args.random:
        patterns = random.Random(args.seed).sample(patterns, min(args.random, len(patterns)))
    if not patterns:
        sys.exit("ERROR: no patterns match")

    out = Path(args.out or run_dir / "samples")
    out.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        entry = results[pattern]
        model_rule = entry.get("logic_rules") or ""
        truth = gt_rule(data_root, args.principle, pattern, config.color_large)
        path = out / f"eval_{args.principle}_{pattern[:3]}.png"
        sheet(data_root, args.principle, pattern, entry, model_rule, truth).save(path)
        answers = " ".join(f"{s['id']}={'OK' if s['predicted'] == s['label'] else 'WRONG'}"
                           for s in entry["samples"])
        print(f"\n### {pattern}  (accuracy {entry['accuracy']:.1f}%)  -> {path}")
        print(f"answers: {answers}")
        print("--- model rule ---\n" + model_rule)
        if truth:
            print("--- ground-truth rule ---\n" + truth)


if __name__ == "__main__":
    main()

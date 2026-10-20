#!/usr/bin/env python3
"""
Check that evaluation images never appear in the fine-tuning splits.

Hashes the decoded pixels of every image in $ELVIS_DATA/<principle>/<split>/<pattern>/
{positive,negative}/*.png and reports, for each (eval split, train split) pair, how many
eval images are pixel-identical to some training image (any pattern, any label).
Pixel hashes rather than file bytes, so the same image saved twice still matches.
All images in a pattern folder are hashed, including any the dataset builder did not use,
so the check is a superset of what training saw.

CPU only, no model. Exit code 1 if any eval image matches a training image.

Usage:
  python scripts/check_split_overlap.py                         # all principles
  python scripts/check_split_overlap.py --principles continuity --show 10
"""

import argparse
import hashlib
import os
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

ALL_PRINCIPLES = ["continuity", "proximity", "similarity", "closure", "symmetry"]
# test images and Stage 1 example images (standard mode): new set, then ELVIS's original set
EVAL_SPLITS = ["test_v3", "train_v3", "test", "train"]
TRAIN_SPLITS = ["ft_train", "ft_val"]
# (eval, other): overlap with a training split is a failure; test vs its examples is informational.
PAIRS = [(e, t) for e in EVAL_SPLITS for t in TRAIN_SPLITS] + [("test_v3", "train_v3"), ("test", "train")]


def hash_split(split_dir):
    """{pixel_hash: [(pattern, label, file), ...]} for one split."""
    out = defaultdict(list)
    for png in sorted(split_dir.glob("*/*/*.png")):
        pattern, label = png.parent.parent.name, png.parent.name
        if label not in ("positive", "negative"):
            continue
        with Image.open(png) as im:
            h = hashlib.sha1(im.convert("RGB").tobytes()).hexdigest()
        out[h].append((pattern, label, png.name))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--principles", nargs="+", default=ALL_PRINCIPLES)
    parser.add_argument("--show", type=int, default=5, help="example matches to print per pair")
    args = parser.parse_args()

    data = os.environ.get("ELVIS_DATA")
    if not data:
        sys.exit("ERROR: env var ELVIS_DATA is not set")

    failed = False
    for principle in args.principles:
        print(f"\n=================== {principle} ===================")
        hashes = {}
        for split in EVAL_SPLITS + TRAIN_SPLITS:
            d = Path(data, principle, split)
            if not d.is_dir():
                print(f"  [skip] {d} not found")
                continue
            hashes[split] = hash_split(d)
            n = sum(len(v) for v in hashes[split].values())
            repeats = n - len(hashes[split])
            print(f"  {split:9s}: {n:6d} images, {len(hashes[split]):6d} distinct"
                  + (f"  ({repeats} repeats within the split)" if repeats else ""))

        for ev, tr in PAIRS:
            if ev not in hashes or tr not in hashes:
                continue
            shared = set(hashes[ev]) & set(hashes[tr])
            n_eval = sum(len(hashes[ev][h]) for h in shared)
            same_pattern = sum(1 for h in shared for p, _, _ in hashes[ev][h]
                               if p in {q for q, _, _ in hashes[tr][h]})
            print(f"  {ev:9s} vs {tr:9s}: " + ("OK  0 images match" if not shared else
                  f"OVERLAP  {n_eval} images match ({same_pattern} in the same pattern)"))
            for h in sorted(shared)[:args.show]:
                e, t = hashes[ev][h][0], hashes[tr][h][0]
                print(f"      {ev}/{e[0]}/{e[1]}/{e[2]}  ==  {tr}/{t[0]}/{t[1]}/{t[2]}")
            if shared and tr in TRAIN_SPLITS:
                failed = True

    print("\nRESULT:", "FAIL — eval images found in training splits" if failed else "PASS — no overlap")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

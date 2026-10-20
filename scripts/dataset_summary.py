#!/usr/bin/env python3
"""
Size of the original ELVIS evaluation set (train/test) vs the corrected one (train_v3/test_v3).

Counts patterns, images (positive + negative PNGs) and disk size per principle and split
under $ELVIS_DATA/<principle>/<split>/<pattern>/{positive,negative}/*.png.
train = Stage 1 example images, test = images the model classifies.

Usage:
  python scripts/dataset_summary.py
  python scripts/dataset_summary.py --splits train test train_v3 test_v3 ft_train ft_val
"""

import argparse
import os
import sys
from pathlib import Path

ALL_PRINCIPLES = ["continuity", "proximity", "similarity", "closure", "symmetry"]
SETS = {"ELVIS": ("train", "test"), "ELVIS-corrected": ("train_v3", "test_v3")}


def count(split_dir):
    """(patterns, images, bytes) for one split; None if it does not exist."""
    if not split_dir.is_dir():
        return None
    pngs = [p for p in split_dir.glob("*/*/*.png") if p.parent.name in ("positive", "negative")]
    patterns = sum(1 for p in split_dir.iterdir() if p.is_dir())
    return patterns, len(pngs), sum(p.stat().st_size for p in pngs)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--principles", nargs="+", default=ALL_PRINCIPLES)
    parser.add_argument("--splits", nargs="+", default=[s for pair in SETS.values() for s in pair])
    args = parser.parse_args()

    data = os.environ.get("ELVIS_DATA")
    if not data:
        sys.exit("ERROR: env var ELVIS_DATA is not set")

    print(f"{'principle':12s} {'split':9s} {'patterns':>9s} {'images':>9s} {'img/pattern':>12s} {'MB':>8s}")
    totals = {s: [0, 0, 0] for s in args.splits}
    for principle in args.principles:
        for split in args.splits:
            c = count(Path(data, principle, split))
            if c is None:
                print(f"{principle:12s} {split:9s} {'(missing)':>9s}")
                continue
            pat, img, size = c
            print(f"{principle:12s} {split:9s} {pat:9d} {img:9d} {img / max(pat, 1):12.1f} {size / 1e6:8.1f}")
            totals[split] = [a + b for a, b in zip(totals[split], c)]
        print()

    print("TOTAL")
    for split in args.splits:
        pat, img, size = totals[split]
        print(f"{'':12s} {split:9s} {pat:9d} {img:9d} {img / max(pat, 1):12.1f} {size / 1e6:8.1f}")
    for name, (tr, te) in SETS.items():
        if tr in totals and te in totals:
            print(f"{name:16s}: {totals[tr][1] + totals[te][1]:7d} images "
                  f"({totals[tr][1]} examples + {totals[te][1]} test), "
                  f"{(totals[tr][2] + totals[te][2]) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

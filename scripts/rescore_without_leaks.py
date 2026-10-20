#!/usr/bin/env python3
"""
Re-score saved ELVIS results with overlapping test images removed.

A test image is excluded when its pixels are identical to any image in the fine-tuning
splits (default ft_train and ft_val: first --per-side images per side, as build_v3_dataset.py
uses them), any pattern, any label. The same exclusion is applied to every model (base
included), so all models are scored on the same cleaned test set.
Known source: similarity palette positives have fixed positions, so they repeat across
seeds and patterns (see check_split_overlap.py).

Accuracy/F1 are recomputed as ELVIS does (mean over patterns of per-pattern values),
dropping excluded samples; a pattern with no samples left is dropped.
Writes summary_clean.json next to each summary.json (plot with
plot_principle_curves.py --summary summary_clean.json) and prints orig vs clean.

Usage:
  python scripts/rescore_without_leaks.py --root results/eval_v3_rubric_on_test_v3
  python scripts/rescore_without_leaks.py --root results/eval_v3_on_test_v3 --root results/eval_v2_on_test_v3
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from PIL import Image

ALL_PRINCIPLES = ["continuity", "proximity", "similarity", "closure", "symmetry"]


def pixel_hash(png):
    with Image.open(png) as im:
        return hashlib.sha1(im.convert("RGB").tobytes()).hexdigest()


def first_images(split_dir, per_side):
    """{(pattern, 'positive_0'): path} for the first per_side images per side (ELVIS/builder order)."""
    out = {}
    for pattern in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for label in ("positive", "negative"):
            for i, png in enumerate(sorted((pattern / label).glob("*.png"))[:per_side]):
                out[(pattern.name, f"{label}_{i}")] = png
    return out


def f1(samples):
    tp = sum(s["predicted"] == 1 and s["label"] == 1 for s in samples)
    fp = sum(s["predicted"] == 1 and s["label"] == 0 for s in samples)
    fn = sum(s["predicted"] == 0 and s["label"] == 1 for s in samples)
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def rescore(result_json, excluded):
    """{accuracy, f1, orig_accuracy, excluded_samples, dropped_patterns} for one principle."""
    results = json.load(open(result_json))
    orig, acc, f1s, n_excl, dropped = [], [], [], 0, 0
    for pattern, r in results.items():
        orig.append(r["accuracy"])
        kept = [s for s in r["samples"] if (pattern, s["id"]) not in excluded]
        n_excl += len(r["samples"]) - len(kept)
        if not kept:
            dropped += 1
            continue
        acc.append(100 * sum(s["predicted"] == s["label"] for s in kept) / len(kept))
        f1s.append(f1(kept))
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    return {"accuracy": mean(acc), "f1": mean(f1s), "orig_accuracy": mean(orig),
            "excluded_samples": n_excl, "dropped_patterns": dropped}


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--root", action="append", required=True, help="eval results folder (repeatable)")
    parser.add_argument("--test-split", default="test_v3")
    parser.add_argument("--train-splits", nargs="+", default=["ft_train", "ft_val"])
    parser.add_argument("--img-num", type=int, default=3, help="test images per side ELVIS evaluates")
    parser.add_argument("--per-side", type=int, default=6, help="fine-tuning images per side (build_v3_dataset)")
    parser.add_argument("--principles", nargs="+", default=ALL_PRINCIPLES)
    args = parser.parse_args()

    data = os.environ.get("ELVIS_DATA")
    if not data:
        sys.exit("ERROR: env var ELVIS_DATA is not set")

    excluded, record = {}, {}
    for p in args.principles:
        seen = {}
        for split in args.train_splits:
            for (pat, sid), png in first_images(Path(data, p, split), args.per_side).items():
                seen.setdefault(pixel_hash(png), []).append(f"{split}/{pat}/{png.parent.name}/{png.name}")
        record[p] = []
        for (pat, sid), png in first_images(Path(data, p, args.test_split), args.img_num).items():
            matches = seen.get(pixel_hash(png))
            if matches:
                record[p].append({"pattern": pat, "id": sid, "file": f"{png.parent.name}/{png.name}",
                                  "matches": matches})
        excluded[p] = {(e["pattern"], e["id"]) for e in record[p]}
        print(f"{p:11s}: excluding {len(excluded[p]):4d} {args.test_split} images "
              f"({len({pat for pat, _ in excluded[p]})} patterns) identical to {'/'.join(args.train_splits)}")
    # The exact exclusion list, so the cleaned numbers can be audited and reproduced.
    out = Path(f"data/exclusions_{args.test_split}.json")
    out.parent.mkdir(exist_ok=True)
    json.dump({"test_split": args.test_split, "matched_against": args.train_splits,
               "img_num": args.img_num, "per_side": args.per_side, "excluded": record},
              open(out, "w"), indent=2)
    print(f"Exclusion list: {out}")

    print(f"\n{'run':55s} {'principle':11s} {'orig':>7s} {'clean':>7s} {'diff':>6s} {'excl':>5s} {'dropped':>8s}")
    for root in map(Path, args.root):
        for summary_path in sorted(root.glob("**/summary.json")):
            run = str(summary_path.parent.relative_to(root))
            summary = json.load(open(summary_path))
            clean = {}
            for p in args.principles:
                files = sorted(summary_path.parent.glob(f"{p}/*.json"))
                if not files:
                    continue
                r = rescore(files[-1], excluded[p])
                clean[p] = {"accuracy": r["accuracy"], "f1": r["f1"],
                            "excluded_samples": r["excluded_samples"], "dropped_patterns": r["dropped_patterns"]}
                print(f"{run:55s} {p:11s} {r['orig_accuracy']:7.2f} {r['accuracy']:7.2f} "
                      f"{r['accuracy'] - r['orig_accuracy']:+6.2f} {r['excluded_samples']:5d} {r['dropped_patterns']:8d}")
            summary.update(principles=clean, excluded_from=args.train_splits,
                           excluded_images={p: len(v) for p, v in excluded.items()})
            json.dump(summary, open(summary_path.with_name("summary_clean.json"), "w"), indent=2)


if __name__ == "__main__":
    main()

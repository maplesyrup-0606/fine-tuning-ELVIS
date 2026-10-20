#!/usr/bin/env python3
"""
Sanity-check generated splits: every label must agree with its metadata.

For each pattern in $ELVIS_DATA/<principle>/<split>/, reads the per-image JSON
written by ELVIS main.py and checks it against the principle's generator
semantics, as encoded in rubric_specs.py:

  positive  - grouping intact (spec.grouping_broken is False)
            - every `rel` prop within its logic values
            - every `irrel` prop constant across objects
  negative  - every `irrel` prop constant across objects
            - grouping and/or at least one rel prop broken; a negative that
              breaks nothing is an "accidental positive" (label noise)

The pattern name is the source of truth for rel/irrel; the JSON's prop_* and
group_num fields are NOT used (main.py derives them by substring match on the
name, so they are wrong whenever a prop is irrel).

CPU only, no model. Exit code 1 if any hard check fails.
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from rubric_specs import get_spec


def check_image(spec, json_path, is_positive, slots, color_names):
    """Return (hard_errors, accidental_positive, broken) for one image."""
    with open(json_path) as f:
        objs = json.load(f)["img_data"]
    if not objs:
        return ["no objects"], False, []
    errors = spec.irrelevant_errors(objs, slots, color_names)
    broken = [c for c, _ in spec.violations(objs, slots, color_names)]

    if is_positive:
        errors += [f"positive breaks {b}" for b in broken]
        return errors, False, broken
    return errors, not broken, broken


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--split", required=True, help="e.g. ft_train, ft_val")
    parser.add_argument("--num-samples", type=int, required=True,
                        help="expected images per side per pattern")
    parser.add_argument("--principle", default="continuity")
    parser.add_argument("--expected-patterns", type=int, default=None,
                        help="default: the principle spec's pattern count")
    args = parser.parse_args()

    for env in ("ELVIS_ROOT", "ELVIS_DATA"):
        if not os.environ.get(env):
            sys.exit(f"ERROR: env var {env} is not set")
    sys.path.insert(0, os.environ["ELVIS_ROOT"])
    from scripts import config
    color_names = config.color_large
    spec = get_spec(args.principle)
    expected = args.expected_patterns or spec.expected_patterns

    split_root = Path(os.environ["ELVIS_DATA"]) / args.principle / args.split
    if not split_root.is_dir():
        sys.exit(f"ERROR: split dir not found at {split_root}")
    patterns = sorted(p.name for p in split_root.iterdir() if p.is_dir())

    hard = []            # (pattern, side, file, message)
    accidental = []      # (pattern, file)
    unverified = Counter()   # generator -> negatives breaking nothing checkable
    neg_breaks = Counter()
    n_images = 0

    for pattern in patterns:
        parsed = spec.parse(pattern)
        grouping_checked = spec.generators[parsed["generator"]].grouping_checked
        for side in ("positive", "negative"):
            side_dir = split_root / pattern / side
            pngs = sorted(side_dir.glob("*.png"))
            jsons = sorted(side_dir.glob("*.json"))
            if len(pngs) != args.num_samples or len(jsons) != args.num_samples:
                hard.append((pattern, side, "-", f"{len(pngs)} png / {len(jsons)} json, "
                                                  f"expected {args.num_samples}"))
            for jp in jsons:
                n_images += 1
                errors, acc, broken = check_image(spec, jp, side == "positive", parsed, color_names)
                hard += [(pattern, side, jp.name, e) for e in errors]
                if acc and grouping_checked:
                    accidental.append((pattern, jp.name))
                elif acc:
                    unverified[parsed["generator"]] += 1
                if side == "negative":
                    neg_breaks.update(broken or (["nothing"] if grouping_checked else ["grouping?"]))

    n_neg = n_images // 2
    print(f"Split:      {split_root}")
    print(f"Patterns:   {len(patterns)} (expected {expected})")
    print(f"Images:     {n_images} (expected {expected * 2 * args.num_samples})")
    print(f"Hard errors: {len(hard)}")
    for pattern, side, name, msg in hard[:20]:
        print(f"  {pattern}/{side}/{name}: {msg}")
    if len(hard) > 20:
        print(f"  ... {len(hard) - 20} more")
    print(f"Accidental positives (negatives breaking nothing): {len(accidental)} / {n_neg}")
    trained = [(p, n) for p, n in accidental if not spec.exclude_from_training(spec.parse(p))]
    if accidental:
        print(f"  in patterns excluded from training: {len(accidental) - len(trained)}")
        print(f"  in patterns used for training:      {len(trained)}  (must be 0)")
    for pattern, name in (trained or accidental)[:10]:
        print(f"  {pattern}/negative/{name}")
    if unverified:
        print("Negatives breaking no checkable rule, in generators whose grouping is not visible in "
              "metadata (expected: these broke the grouping):")
        for gen, count in unverified.most_common():
            print(f"  {gen}: {count}")
    print("What negatives break (an image can break several):")
    for what, count in neg_breaks.most_common():
        print(f"  {what:8s} {count}")

    ok = not hard and not trained and len(patterns) == expected
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Sanity-check generated continuity splits: every label must agree with its metadata.

For each pattern in $ELVIS_DATA/continuity/<split>/, reads the per-image JSON
written by ELVIS main.py and checks it against the generator's semantics
(ELVIS scripts/continuity/util_*.py + scripts/utils/data_utils.assign_property):

  positive  - every object on the layout (group_id >= 0)
            - every `rel` prop within its logic values
            - every `irrel` prop constant across objects
  negative  - every `irrel` prop constant across objects
            - at least one of {layout, each rel prop} broken; a negative that
              breaks nothing is an "accidental positive" (label noise)

The pattern name is the source of truth for rel/irrel; the JSON's prop_* and
group_num fields are NOT used (main.py derives them by substring match on the
name, so they are wrong whenever a prop is irrel).

CPU only, no model. Exit code 1 if any hard check fails.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

PATTERN_RE = re.compile(
    r"^\d+_\d+_(?P<generator>.+?)_rel_(?P<rel>.+)_(?P<groups>\d+)_(?P<quantity>[a-z0-9]+)_irrel_(?P<irrel>.+)$"
)

# Logic values shared by all four continuity generators (`logic = {...}` in util_*.py).
ALLOWED_SHAPES = {"square", "circle"}
ALLOWED_COLORS = {"green", "yellow"}


def parse_pattern(name):
    m = PATTERN_RE.match(name)
    if not m:
        raise ValueError(f"unrecognized pattern name: {name}")
    return m["rel"].split("_"), m["irrel"].split("_")


def obj_props(objs, color_names):
    return {
        "shape": [o["shape"] for o in objs],
        "color": [color_names[o["color_name"]] for o in objs],
        "size": [round(o["size"], 9) for o in objs],
    }


def rel_ok(prop, values):
    if prop == "shape":
        return set(values) <= ALLOWED_SHAPES
    if prop == "color":
        return set(values) <= ALLOWED_COLORS
    return len(set(values)) == 1  # size: one fixed size


def check_image(json_path, is_positive, rel, irrel, color_names):
    """Return (hard_errors, accidental_positive, broken) for one image."""
    with open(json_path) as f:
        objs = json.load(f)["img_data"]
    if not objs:
        return ["no objects"], False, []
    props = obj_props(objs, color_names)
    errors = []

    for p in irrel:
        if len(set(props[p])) != 1:
            errors.append(f"irrel {p} not constant")

    on_layout = all(o["group_id"] >= 0 for o in objs)
    broken = [] if on_layout else ["layout"]
    broken += [p for p in rel if not rel_ok(p, props[p])]

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
    parser.add_argument("--expected-patterns", type=int, default=432)
    args = parser.parse_args()

    for env in ("ELVIS_ROOT", "ELVIS_DATA"):
        if not os.environ.get(env):
            sys.exit(f"ERROR: env var {env} is not set")
    sys.path.insert(0, os.environ["ELVIS_ROOT"])
    from scripts import config
    color_names = config.color_large

    split_root = Path(os.environ["ELVIS_DATA"]) / args.principle / args.split
    if not split_root.is_dir():
        sys.exit(f"ERROR: split dir not found at {split_root}")
    patterns = sorted(p.name for p in split_root.iterdir() if p.is_dir())

    hard = []            # (pattern, side, file, message)
    accidental = []      # (pattern, file)
    neg_breaks = Counter()
    n_images = 0

    for pattern in patterns:
        rel, irrel = parse_pattern(pattern)
        for side in ("positive", "negative"):
            side_dir = split_root / pattern / side
            pngs = sorted(side_dir.glob("*.png"))
            jsons = sorted(side_dir.glob("*.json"))
            if len(pngs) != args.num_samples or len(jsons) != args.num_samples:
                hard.append((pattern, side, "-", f"{len(pngs)} png / {len(jsons)} json, "
                                                  f"expected {args.num_samples}"))
            for jp in jsons:
                n_images += 1
                errors, acc, broken = check_image(jp, side == "positive", rel, irrel, color_names)
                hard += [(pattern, side, jp.name, e) for e in errors]
                if acc:
                    accidental.append((pattern, jp.name))
                if side == "negative":
                    neg_breaks.update(broken or ["nothing"])

    n_neg = n_images // 2
    print(f"Split:      {split_root}")
    print(f"Patterns:   {len(patterns)} (expected {args.expected_patterns})")
    print(f"Images:     {n_images} (expected {args.expected_patterns * 2 * args.num_samples})")
    print(f"Hard errors: {len(hard)}")
    for pattern, side, name, msg in hard[:20]:
        print(f"  {pattern}/{side}/{name}: {msg}")
    if len(hard) > 20:
        print(f"  ... {len(hard) - 20} more")
    print(f"Accidental positives (negatives breaking nothing): {len(accidental)} / {n_neg}")
    for pattern, name in accidental[:10]:
        print(f"  {pattern}/negative/{name}")
    print("What negatives break (an image can break several):")
    for what, count in neg_breaks.most_common():
        print(f"  {what:8s} {count}")

    ok = not hard and len(patterns) == args.expected_patterns
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

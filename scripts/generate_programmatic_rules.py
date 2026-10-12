#!/usr/bin/env python3
"""
Build ground-truth Stage 1 rules for every (pattern, episode) of a continuity split.

v2 replacement for generate_rules.py (base Qwen3-VL-4B guessing rules from 3+3
images). Each rule is derived from the generator metadata, so it is correct by
construction. Schema: EXPERIMENT.md "Rule schema".

Episode layout (per pattern, per side): episode k uses images
  6k..6k+2  as Stage 1 examples  (ELVIS feeds positives first, then negatives)
  6k+3..6k+5 as Stage 2 queries
so a split generated with --num_samples 12 has 2 episodes per pattern.

Output (data/continuity/rules_<split>.json by default): one entry per episode
with its example/query image paths (relative to $ELVIS_DATA), the structured
answer key (`slots`, used by the rule grader) and the rendered `rule` text.

Semantics (ELVIS scripts/continuity/util_*.py + data_utils.assign_property):
  - positive: on the layout, every rel prop within its logic values
  - irrel props: one value shared by every object in an image
  - props in neither list: random
  - negative: breaks the layout and/or at least one rel prop

CPU only, no model. Exits 1 if any example image contradicts its label.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from check_ft_data import ALLOWED_COLORS, ALLOWED_SHAPES, PATTERN_RE

PROPS = ("shape", "color", "size")
EPISODE_SIZE = 3  # examples per side = queries per side = ELVIS img_num
NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four"}

GROUPING = {
    "non_overlap_one_split_n": lambda g: (
        f"objects are evenly spaced along one straight main line that splits into "
        f"{NUMBER_WORDS[g]} straight branches"),
    "with_intersected_n_splines": lambda g: (
        f"objects are evenly spaced along {NUMBER_WORDS[g]} smooth curves that cross at one shared point"),
    "non_intersected_n_splines": lambda g: (
        "objects are evenly spaced along one smooth curve" if g == 1 else
        f"objects are evenly spaced along {NUMBER_WORDS[g]} separate smooth curves that do not cross"),
    "feature_continuity_overlap_splines": lambda g: (
        f"objects are evenly spaced along {NUMBER_WORDS[g]} smooth curves; the first curve has a "
        f"second row of objects running closely parallel to it"),
}

RELEVANT = {
    "shape": "shape (only square and circle)",
    "color": "color (only green and yellow)",
    "size": "size (all objects the same size)",
}


def parse_pattern(name):
    m = PATTERN_RE.match(name)
    if not m or m["generator"] not in GROUPING:
        raise ValueError(f"unrecognized pattern name: {name}")
    rel, irrel = m["rel"].split("_"), m["irrel"].split("_")
    return {
        "generator": m["generator"],
        "groups": int(m["groups"]),
        "relevant": rel,
        "irrelevant": irrel,
        "free": [p for p in PROPS if p not in rel and p not in irrel],
    }


def violations(objs, relevant, color_names):
    """Ordered list of (category, evidence) the image breaks."""
    out = []
    if any(o["group_id"] < 0 for o in objs):
        out.append(("scattered", "objects do not follow the lines"))
    for prop in relevant:
        if prop == "shape":
            off = sorted({o["shape"] for o in objs} - ALLOWED_SHAPES)
            if off:
                out.append(("shape", f"has {', '.join(off[:3])}"))
        elif prop == "color":
            if {color_names[o["color_name"]] for o in objs} - ALLOWED_COLORS:
                out.append(("color", "has colors other than green and yellow"))
        elif len({round(o["size"], 9) for o in objs}) > 1:
            out.append(("size", "objects differ in size"))
    return out


def render(spec, neg_violations):
    irrelevant = ([f"{p} (same for every object in an image)" for p in spec["irrelevant"]]
                  + [f"{p} (varies freely)" for p in spec["free"]])
    lines = [
        "Principle: continuity",
        f"Grouping: {GROUPING[spec['generator']](spec['groups'])}",
        f"Relevant: {'; '.join(RELEVANT[p] for p in spec['relevant'])}",
        f"Irrelevant: {'; '.join(irrelevant)}",
    ]
    for i, vs in enumerate(neg_violations):
        lines.append(f"Image {EPISODE_SIZE + 1 + i}: {'; '.join(f'{c} ({e})' for c, e in vs)}")
    return "\n".join(lines)


def load_objs(png_path):
    with open(png_path.with_suffix(".json")) as f:
        return json.load(f)["img_data"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--split", required=True, help="e.g. ft_train, ft_val")
    parser.add_argument("--principle", default="continuity")
    parser.add_argument("--output", default=None,
                        help="default: data/<principle>/rules_<split>.json")
    parser.add_argument("--patterns", default=None,
                        help="comma-separated substrings; process only matching patterns (debug)")
    parser.add_argument("--print", type=int, default=0, metavar="N",
                        help="print the first N rules")
    args = parser.parse_args()

    for env in ("ELVIS_ROOT", "ELVIS_DATA"):
        if not os.environ.get(env):
            sys.exit(f"ERROR: env var {env} is not set")
    sys.path.insert(0, os.environ["ELVIS_ROOT"])
    from scripts import config
    color_names = config.color_large

    data_root = Path(os.environ["ELVIS_DATA"])
    split_root = data_root / args.principle / args.split
    if not split_root.is_dir():
        sys.exit(f"ERROR: split dir not found at {split_root}")
    patterns = sorted(p.name for p in split_root.iterdir() if p.is_dir())
    if args.patterns:
        wanted = tuple(args.patterns.split(","))
        patterns = [p for p in patterns if any(w in p for w in wanted)]

    episodes, errors = [], []
    for pattern in patterns:
        spec = parse_pattern(pattern)
        sides = {s: sorted((split_root / pattern / s).glob("*.png")) for s in ("positive", "negative")}
        n = min(len(v) for v in sides.values())
        if n % (2 * EPISODE_SIZE):
            errors.append(f"{pattern}: {n} images per side is not a multiple of {2 * EPISODE_SIZE}")
        for k in range(n // (2 * EPISODE_SIZE)):
            start = 2 * EPISODE_SIZE * k
            examples = {s: sides[s][start:start + EPISODE_SIZE] for s in sides}
            queries = {s: sides[s][start + EPISODE_SIZE:start + 2 * EPISODE_SIZE] for s in sides}

            for p in examples["positive"]:
                if violations(load_objs(p), spec["relevant"], color_names):
                    errors.append(f"{p}: positive example breaks the rule")
            neg_v = [violations(load_objs(p), spec["relevant"], color_names) for p in examples["negative"]]
            for p, vs in zip(examples["negative"], neg_v):
                if not vs:
                    errors.append(f"{p}: negative example breaks nothing")

            def rel_paths(d):
                return {s: [str(p.relative_to(data_root)) for p in d[s]] for s in d}

            episodes.append({
                "pattern": pattern,
                "episode": k,
                "examples": rel_paths(examples),
                "queries": rel_paths(queries),
                "slots": {**spec, "image_violations": [[c for c, _ in vs] for vs in neg_v]},
                "rule": render(spec, neg_v),
            })

    if errors:
        print(f"ERROR: {len(errors)} problems. First 10:")
        for e in errors[:10]:
            print(f"  {e}")
        return 1

    for ep in episodes[:args.print]:
        print(f"### {ep['pattern']} (episode {ep['episode']})\n{ep['rule']}\n")

    output = Path(args.output or f"data/{args.principle}/rules_{args.split}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "generator": "scripts/generate_programmatic_rules.py",
            "principle": args.principle,
            "split": args.split,
            "split_root": str(split_root),
            "episode_size": EPISODE_SIZE,
            "num_patterns": len(patterns),
            "num_episodes": len(episodes),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "episodes": episodes,
    }
    with open(output, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"Wrote {len(episodes)} episode rules ({len(patterns)} patterns) to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

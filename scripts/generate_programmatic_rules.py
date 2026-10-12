#!/usr/bin/env python3
"""
Generate ground-truth Stage 1 rules for continuity patterns from their task names.

Replaces the distilled rules from generate_rules.py (base Qwen3-VL-4B guessing
the rule from 3+3 images) with rules derived directly from how ELVIS generates
each pattern. Every pattern folder name fully determines its rule:

    {idx:03d}_{task_counter}_{generator}_rel_{props}_{groups}_{quantity}_irrel_{props}

Semantics (ELVIS scripts/continuity/util_*.py + scripts/utils/data_utils.py):
  - Positive: objects lie on the generator's layout, and every `rel` property
    is restricted to its logic values (shape in {square, circle}, color in
    {green, yellow}, size fixed).
  - `irrel` properties: one shared value for every object, in both positive
    and negative images, so they carry no signal.
  - Properties in neither list: random in both positive and negative images.
  - Negative: `cf_params` is a proper sublist of rel + ["continuity"], so at
    least one of {layout, each rel property} is broken (randomized).

Output has the same format as generate_rules.py, so build_dataset.py consumes
it unchanged. No GPU, no model; runs in seconds.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import yaml


PATTERN_RE = re.compile(
    r"^\d+_\d+_(?P<generator>.+?)_rel_(?P<rel>.+)_(?P<groups>\d+)_(?P<quantity>[a-z0-9]+)_irrel_(?P<irrel>.+)$"
)

PROPS = ("shape", "color", "size")

NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}

# Layout of positive images, keyed by ELVIS generator function name.
LAYOUTS = {
    "non_overlap_one_split_n": lambda g: (
        f"the objects form one straight main line that splits at a single point into "
        f"{NUMBER_WORDS[g]} straight branches, with evenly spaced objects along each"
    ),
    "with_intersected_n_splines": lambda g: (
        f"the objects form {NUMBER_WORDS[g]} smooth curves that cross each other at one shared point "
        f"near the center of the image, with evenly spaced objects along each curve"
    ),
    "non_intersected_n_splines": lambda g: (
        "the objects form one smooth curve with evenly spaced objects along it" if g == 1 else
        f"the objects form {NUMBER_WORDS[g]} separate smooth curves that do not cross, "
        f"with evenly spaced objects along each curve"
    ),
    "feature_continuity_overlap_splines": lambda g: (
        f"the objects form {NUMBER_WORDS[g]} smooth curves with evenly spaced objects along each; "
        f"the first curve is doubled by a second row of objects running closely parallel to it"
    ),
}

REL_CONDITIONS = {
    "shape": "every object is a square or a circle",
    "color": "every object is green or yellow",
    "size": "all objects are the same size",
}

REL_VIOLATIONS = {
    "shape": "some object has a shape other than square or circle",
    "color": "some object has a color other than green or yellow",
    "size": "the objects have different sizes",
}


def parse_pattern(name):
    m = PATTERN_RE.match(name)
    if not m:
        raise ValueError(f"unrecognized pattern name: {name}")
    rel = m["rel"].split("_")
    irrel = m["irrel"].split("_")
    unknown = [p for p in rel + irrel if p not in PROPS]
    if unknown or m["generator"] not in LAYOUTS:
        raise ValueError(f"unrecognized generator or property in pattern name: {name}")
    return {
        "generator": m["generator"],
        "rel": rel,
        "irrel": irrel,
        "free": [p for p in PROPS if p not in rel and p not in irrel],
        "groups": int(m["groups"]),
    }


def join_list(items):
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def build_rule(name):
    spec = parse_pattern(name)
    layout = LAYOUTS[spec["generator"]](spec["groups"])
    conditions = [REL_CONDITIONS[p] for p in spec["rel"]]
    violations = ["the objects are scattered randomly instead of lying along these lines"]
    violations += [REL_VIOLATIONS[p] for p in spec["rel"]]

    ignored = [f"object {join_list(spec['irrel'])} (the same for every object in both "
               f"positive and negative images)"]
    if spec["free"]:
        ignored.append(f"object {join_list(spec['free'])} (random in both)")
    return (
        f"An image is positive only if all of these hold: {layout}; "
        f"{'; '.join(conditions)}. "
        f"An image is negative if any of them fails: {'; or '.join(violations)}. "
        f"Ignore {' and '.join(ignored)}"
    )


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default=None,
                        help="override rules.cache_path from config")
    parser.add_argument("--patterns", default=None,
                        help="comma-separated substrings; process only matching patterns (debug)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print rules instead of writing them")
    args = parser.parse_args()

    cfg = load_config(args.config)
    principle = cfg["data"]["principle"]
    output_path = args.output or cfg["rules"]["cache_path"]

    elvis_data = os.environ.get(cfg["paths"]["elvis_data_env"])
    if not elvis_data:
        sys.exit(f"ERROR: env var {cfg['paths']['elvis_data_env']} is not set")
    train_root = Path(elvis_data) / principle / "train"
    if not train_root.is_dir():
        sys.exit(f"ERROR: train dir not found at {train_root}")

    all_patterns = sorted(p.name for p in train_root.iterdir() if p.is_dir())
    if args.patterns:
        wanted = tuple(args.patterns.split(","))
        patterns = [p for p in all_patterns if any(w in p for w in wanted)]
    else:
        patterns = all_patterns
    print(f"Discovered {len(all_patterns)} patterns in {train_root}; processing {len(patterns)}.")

    rules = {p: build_rule(p) for p in patterns}

    if args.dry_run:
        for p, r in rules.items():
            print(f"{p}\n  {r}\n")
        return

    payload = {
        "meta": {
            "source_model": None,
            "generator": "programmatic",
            "principle": principle,
            "num_patterns": len(rules),
            "elvis_data_path": str(train_root),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "rules": rules,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {len(rules)} rules to {output_path}")


if __name__ == "__main__":
    main()

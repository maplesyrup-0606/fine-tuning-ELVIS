#!/usr/bin/env python3
"""
Build ground-truth rules from ELVIS generator metadata (no model in the loop).

Two outputs:

  --rubrics (v3)   one rubric per pattern: Principle / Grouping / Relevant / Irrelevant.
                   Depends only on the pattern name, so it is identical across splits.
                   Default output: data/<principle>/rubrics.json
  default   (v2)   one rule per (pattern, episode) of a split: the rubric plus
                   `Image 4-6` lines describing the episode's example negatives.
                   Default output: data/<principle>/rules_<split>.json

All principle-specific knowledge (name format, grouping sentences, allowed values,
how a broken grouping shows in metadata) lives in rubric_specs.py. Schema:
EXPERIMENT.md "Rule schema".

Episode layout (v2; per pattern, per side): episode k uses images
  6k..6k+2  as Stage 1 examples  (ELVIS feeds positives first, then negatives)
  6k+3..6k+5 as Stage 2 queries
so a split generated with --num_samples 12 has 2 episodes per pattern. Each
entry stores example/query paths (relative to $ELVIS_DATA), the structured answer
key (`slots`, used by the rule grader) and the rendered `rule`.

CPU only, no model. Exits 1 if any example image contradicts its label.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from rubric_specs import get_spec

EPISODE_SIZE = 3  # examples per side = queries per side = ELVIS img_num


def load_objs(png_path):
    with open(png_path.with_suffix(".json")) as f:
        return json.load(f)["img_data"]


def build_rubrics(spec, patterns):
    return {p: spec.render(spec.parse(p)) for p in patterns}


def build_episodes(spec, split_root, data_root, patterns, color_names):
    episodes, errors = [], []
    for pattern in patterns:
        slots = spec.parse(pattern)
        sides = {s: sorted((split_root / pattern / s).glob("*.png")) for s in ("positive", "negative")}
        n = min(len(v) for v in sides.values())
        if n % (2 * EPISODE_SIZE):
            errors.append(f"{pattern}: {n} images per side is not a multiple of {2 * EPISODE_SIZE}")
        for k in range(n // (2 * EPISODE_SIZE)):
            start = 2 * EPISODE_SIZE * k
            examples = {s: sides[s][start:start + EPISODE_SIZE] for s in sides}
            queries = {s: sides[s][start + EPISODE_SIZE:start + 2 * EPISODE_SIZE] for s in sides}

            for p in examples["positive"]:
                if spec.violations(load_objs(p), slots, color_names):
                    errors.append(f"{p}: positive example breaks the rule")
            neg_v = [spec.violations(load_objs(p), slots, color_names) for p in examples["negative"]]
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
                "slots": {**slots, "image_violations": [[c for c, _ in vs] for vs in neg_v]},
                "rule": spec.render(slots, neg_v, first_neg_index=EPISODE_SIZE + 1),
            })
    return episodes, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--split", required=True,
                        help="split whose pattern folders are read, e.g. ft_train, ft_val, test")
    parser.add_argument("--principle", default="continuity")
    parser.add_argument("--rubrics", action="store_true",
                        help="v3: write one rubric per pattern instead of per-episode rules")
    parser.add_argument("--output", default=None,
                        help="default: data/<principle>/rubrics.json or rules_<split>.json")
    parser.add_argument("--patterns", default=None,
                        help="comma-separated substrings; process only matching patterns (debug)")
    parser.add_argument("--print", type=int, default=0, metavar="N",
                        help="print the first N rules/rubrics")
    args = parser.parse_args()

    for env in ("ELVIS_ROOT", "ELVIS_DATA"):
        if not os.environ.get(env):
            sys.exit(f"ERROR: env var {env} is not set")
    sys.path.insert(0, os.environ["ELVIS_ROOT"])
    from scripts import config
    spec = get_spec(args.principle)

    data_root = Path(os.environ["ELVIS_DATA"])
    split_root = data_root / args.principle / args.split
    if not split_root.is_dir():
        sys.exit(f"ERROR: split dir not found at {split_root}")
    patterns = sorted(p.name for p in split_root.iterdir() if p.is_dir())
    if args.patterns:
        wanted = tuple(args.patterns.split(","))
        patterns = [p for p in patterns if any(w in p for w in wanted)]

    meta = {
        "generator": "scripts/generate_programmatic_rules.py",
        "principle": args.principle,
        "split": args.split,
        "split_root": str(split_root),
        "num_patterns": len(patterns),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    if args.rubrics:
        rubrics = build_rubrics(spec, patterns)
        for p in list(rubrics)[:args.print]:
            print(f"### {p}\n{rubrics[p]}\n")
        output = Path(args.output or f"data/{args.principle}/rubrics.json")
        payload = {"meta": meta, "rubrics": rubrics}
        summary = f"{len(rubrics)} rubrics"
    else:
        episodes, errors = build_episodes(spec, split_root, data_root, patterns, config.color_large)
        if errors:
            print(f"ERROR: {len(errors)} problems. First 10:")
            for e in errors[:10]:
                print(f"  {e}")
            return 1
        for ep in episodes[:args.print]:
            print(f"### {ep['pattern']} (episode {ep['episode']})\n{ep['rule']}\n")
        output = Path(args.output or f"data/{args.principle}/rules_{args.split}.json")
        meta.update({"episode_size": EPISODE_SIZE, "num_episodes": len(episodes)})
        payload = {"meta": meta, "episodes": episodes}
        summary = f"{len(episodes)} episode rules ({len(patterns)} patterns)"

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"Wrote {summary} to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

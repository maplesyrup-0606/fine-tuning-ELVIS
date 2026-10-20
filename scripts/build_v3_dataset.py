#!/usr/bin/env python3
"""
Build ms-swift JSONL for v3 fine-tuning: classification only, with ground-truth rubrics.

For every pattern of every requested principle (minus the spec's training exclusions),
takes the first --per-side positive and negative images of the split and emits one row
per image:
  <image> + Stage 2 prompt containing the pattern's rubric  ->  "positive" / "negative"
The rubric is the pattern-level one (Principle / Grouping / Relevant / Irrelevant) from
data/<principle>/rubrics.json (generate_programmatic_rules.py --rubrics). No Stage 1
rows, so no loss_scale balancing.

Wording (scripts/prompts.py): ft_train rows sample one of TRAIN_VARIANTS per row
(seeded); other splits use ELVIS Mode 1 verbatim.

Usage:
  python scripts/build_v3_dataset.py --split ft_train --principles continuity --name continuity
  python scripts/build_v3_dataset.py --split ft_val   --principles all        --name all

Writes data/v3/<split>_<name>.jsonl and <...>.meta.json. CPU only.
"""

import argparse
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml

from prompts import TRAIN_VARIANTS, stage2_prompt
from rubric_specs import SPECS, get_spec

ALL_PRINCIPLES = ["continuity", "proximity", "similarity", "closure", "symmetry"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", required=True, help="ft_train or ft_val")
    parser.add_argument("--principles", nargs="+", required=True, help="principle names, or 'all'")
    parser.add_argument("--name", required=True, help="output tag, e.g. continuity or all")
    parser.add_argument("--per-side", type=int, default=6, help="images per side per pattern")
    parser.add_argument("--out-dir", default="data/v3")
    args = parser.parse_args()

    principles = ALL_PRINCIPLES if args.principles == ["all"] else args.principles
    for p in principles:
        get_spec(p)   # exits if a principle has no spec
    with open(args.config) as f:
        seed = yaml.safe_load(f)["experiment"]["seed"]
    if not os.environ.get("ELVIS_DATA"):
        sys.exit("ERROR: env var ELVIS_DATA is not set")
    data_root = Path(os.environ["ELVIS_DATA"])
    rng = random.Random(seed)
    sample_variant = args.split == "ft_train"

    rows, problems, excluded = [], [], []
    per_principle = {}
    for principle in principles:
        spec = SPECS[principle]
        rubrics_path = Path(f"data/{principle}/rubrics.json")
        if not rubrics_path.exists():
            sys.exit(f"ERROR: {rubrics_path} missing — run generate_programmatic_rules.py "
                     f"--principle {principle} --split ft_train --rubrics")
        rubrics = json.load(open(rubrics_path))["rubrics"]
        split_root = data_root / principle / args.split
        patterns = sorted(p.name for p in split_root.iterdir() if p.is_dir())
        missing = sorted(set(patterns) - set(rubrics))
        if missing:
            sys.exit(f"ERROR: {len(missing)} {principle} patterns have no rubric, e.g. {missing[0]}")

        n_rows = n_patterns = n_excluded = 0
        for pattern in patterns:
            if spec.exclude_from_training(spec.parse(pattern)):
                excluded.append(pattern)
                n_excluded += 1
                continue
            n_patterns += 1
            for label in ("positive", "negative"):
                images = sorted((split_root / pattern / label).glob("*.png"))[:args.per_side]
                if len(images) < args.per_side:
                    problems.append(f"{pattern}/{label}: {len(images)} images, need {args.per_side}")
                for img in images:
                    variant = rng.choice(TRAIN_VARIANTS) if sample_variant else "elvis"
                    rows.append({
                        "messages": [
                            {"role": "user", "content": "<image>" + stage2_prompt(variant, rubrics[pattern])},
                            {"role": "assistant", "content": label},
                        ],
                        "images": [str(img.resolve())],
                        "_variant": variant,
                    })
                    n_rows += 1
        per_principle[principle] = {"patterns": n_patterns, "rows": n_rows, "excluded_patterns": n_excluded}

    if problems:
        print(f"ERROR: {len(problems)} problems. First 10:")
        for p in problems[:10]:
            print(f"  {p}")
        return 1

    rng.shuffle(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.split}_{args.name}.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps({"messages": r["messages"], "images": r["images"]}) + "\n")

    meta = {
        "split": args.split,
        "principles": per_principle,
        "rows": len(rows),
        "labels": dict(Counter(r["messages"][1]["content"] for r in rows)),
        "variants": dict(sorted(Counter(r["_variant"] for r in rows).items())),
        "per_side": args.per_side,
        "excluded": excluded,
        "seed": seed,
        "elvis_data": str(data_root),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(f"{out}.meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {len(rows)} rows to {out}")
    for p, s in per_principle.items():
        print(f"  {p:11s} {s['patterns']:4d} patterns  {s['rows']:6d} rows  ({s['excluded_patterns']} excluded)")
    print(f"  labels: {meta['labels']}   wording: {meta['variants']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

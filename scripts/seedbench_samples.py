#!/usr/bin/env python3
"""
Export SEED-Bench-2 example questions per skill: the image(s)/frames plus a text card with
the question, options, correct answer and each model's answer.

Writes results/seedbench_samples/skill_<id>/<index>/{frame_00.png, ..., info.txt}.
Images come from the dataset tsv ($LMUData/SEEDBench2.tsv, base64 in 'image', a list for
multi-image/video items); answers from results/seedbench_<key>/*_enriched.xlsx.

Usage:
  python scripts/seedbench_samples.py --skills 19 20 21 22 --per-skill 3
  python scripts/seedbench_samples.py --skills 22 --disagree      # only questions where models differ
"""

import argparse
import ast
import base64
import glob
import io
import os
import random
from pathlib import Path

import pandas as pd
from PIL import Image

PREFIX = "Qwen3-VL-4B-Instruct"
MODELS = {"base": "t0p", "continuity-llm": "v3-continuity-llm-e3-t0p",
          "continuity-both": "v3-continuity-both-e3-t0p", "all-llm": "v3-all-llm-e3-t0p",
          "all-both": "v3-all-both-e3-t0p"}


def images_of(cell):
    """Base64 string, or a string holding a list of them -> list of PIL images."""
    s = str(cell)
    items = ast.literal_eval(s) if s.startswith("[") else [s]
    return [Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB") for b in items]


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--skills", nargs="+", type=int, required=True)
    parser.add_argument("--per-skill", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disagree", action="store_true", help="only questions where the models' hits differ")
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/seedbench_samples")
    args = parser.parse_args()

    answers = {}
    for name, suffix in MODELS.items():
        f = glob.glob(f"{args.results}/seedbench_{PREFIX}-{suffix}/*_enriched.xlsx")
        if not f:
            raise SystemExit(f"no *_enriched.xlsx for {PREFIX}-{suffix}")
        answers[name] = pd.read_excel(f[0]).set_index("index")
    ref = answers["base"]
    tsv = Path(os.environ.get("LMUData", Path.home() / "LMUData")) / "SEEDBench2.tsv"
    data = pd.read_csv(tsv, sep="\t").set_index("index")

    rng = random.Random(args.seed)
    for skill in args.skills:
        idx = list(ref.index[ref["question_type_id"] == skill])
        if args.disagree:
            idx = [i for i in idx if len({int(answers[m].at[i, "hit"]) for m in answers}) > 1]
        picked = rng.sample(idx, min(args.per_skill, len(idx)))
        name = ref.loc[idx[0], "question_type"] if idx else "?"
        print(f"skill {skill} ({name}): {len(idx)} candidate questions, exporting {len(picked)}")
        for i in picked:
            d = Path(args.out) / f"skill_{skill}" / str(i)
            d.mkdir(parents=True, exist_ok=True)
            for k, im in enumerate(images_of(data.at[i, "image"])):
                im.save(d / f"frame_{k:02d}.png")
            row = ref.loc[i]
            opts = "\n".join(f"  {c}. {row[c]}" for c in "ABCD" if c in row and pd.notna(row[c]))
            lines = [f"skill {skill}: {name}", f"index {i}", "", f"Q: {row['question']}", opts,
                     f"correct: {row['answer']}", "", "model answers:"]
            for m, df in answers.items():
                pred = str(df.at[i, "prediction"]).replace("\n", " ")
                lines.append(f"  {m:16s} {'RIGHT' if df.at[i, 'hit'] else 'wrong':5s}  {pred[:300]}")
            (d / "info.txt").write_text("\n".join(lines) + "\n")
    print(f"written to {args.out}/")


if __name__ == "__main__":
    main()

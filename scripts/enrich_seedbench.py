#!/usr/bin/env python3
"""
Enrich a VLMEvalKit SEED-Bench-2 result into the full-taxonomy *_enriched.xlsx
shape that `plot_seedbench.py` consumes.

Two input modes:

  xlsx mode (default) — judged final results
    --input <*_gpt-4o-mini_result.xlsx>
    Joins against the HF `SEED-Bench_v2_level1_2_3.json` on question_id
    (recovered from image_path via regex) to add fine-grained taxonomy.
    `hit` already comes from VLMEvalKit's gpt-4o-mini judge.

  pkl mode — in-flight / un-judged chain results
    --pkl <T*/*_SEEDBench2.pkl>  --base-enriched <base's enriched xlsx>
    Uses the base's already-enriched xlsx as the index->taxonomy map
    (so no HF download needed). Regex-extracts an answer letter from
    each prediction and computes an approximate `hit`. Suitable for
    mid-chain monitoring; final GPT-4o-mini accuracy will differ.

Join key for xlsx mode: `(\\d+)_(\\d+)` on image_path -> question_id;
verified 100% match across 24371 base rows.
Join key for pkl mode: VLMEvalKit's lossy `index` column, which has
~3099 collisions (two questions sharing one `index`). We assign each
pkl entry to the first unused base row at that index.

Examples:
    # Final judged base
    python scripts/enrich_seedbench.py \\
        --input results/seedbench_base/Qwen3-VL-4B-Instruct_SEEDBench2_gpt-4o-mini_result.xlsx

    # Mid-chain variant (approximation)
    python scripts/enrich_seedbench.py \\
        --pkl $SCRATCH/VLMEvalKit/outputs/Qwen3-VL-4B-Instruct-both-e2/T20261001-112509/01_SEEDBench2.pkl \\
        --base-enriched results/seedbench_base/Qwen3-VL-4B-Instruct_SEEDBench2_gpt-4o-mini_result_enriched.xlsx \\
        --output results/seedbench_both_e2_live/both_e2_approx_SEEDBench2_gpt-4o-mini_result_enriched.xlsx
"""

import argparse
import json
import pickle
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

import pandas as pd


HF_JSON_URL = ("https://huggingface.co/datasets/AILab-CVC/SEED-Bench-2/"
               "resolve/main/SEED-Bench_v2_level1_2_3.json")

QUESTION_TYPE_NAMES = {
    1:  "Scene Understanding",
    2:  "Instance Identity",
    3:  "Instance Attributes",
    4:  "Instance Location",
    5:  "Instances Counting",
    6:  "Spatial Relation",
    7:  "Instance Interaction",
    8:  "Visual Reasoning",
    9:  "Text Understanding",
    10: "Celebrity Recognition",
    11: "Landmark Recognition",
    12: "Chart Understanding",
    13: "Visual Referring Expression",
    14: "Science Knowledge",
    15: "Emotion Recognition",
    16: "Visual Mathematics",
    17: "Difference Spotting",
    18: "Meme Comprehension",
    19: "Global Video Understanding",
    20: "Action Recognition",
    21: "Action Prediction",
    22: "Procedure Understanding",
    23: "In-Context Captioning",
    24: "Interleaved Image-Text Analysis",
    25: "Text-to-Image Generation",
    26: "Next Image Prediction",
    27: "Text-Image Creation",
}

LETTER_PATS = [
    re.compile(r"(?i)final answer[^A-Da-d]*([A-Da-d])"),
    re.compile(r"(?i)correct answer[^A-Da-d]*([A-Da-d])"),
    re.compile(r"\*\*([A-Da-d])[\.\*]"),
    re.compile(r"^\s*([A-Da-d])[\.\s]"),
    re.compile(r"\b([A-Da-d])\b"),
]


def ensure_hf_json(cache_path: Path) -> Path:
    if cache_path.is_file() and cache_path.stat().st_size > 1_000_000:
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {HF_JSON_URL}\n          → {cache_path} ...")
    urllib.request.urlretrieve(HF_JSON_URL, cache_path)
    print(f"Cached {cache_path.stat().st_size // 1_000_000} MB")
    return cache_path


def extract_letter(pred) -> str | None:
    s = str(pred)
    for pat in LETTER_PATS:
        m = pat.findall(s)
        if m:
            return m[-1].upper()
    return None


def enrich_from_xlsx(input_xlsx: Path, hf_json: Path, out_xlsx: Path):
    with open(hf_json) as f:
        hf = json.load(f)
    hf_df = pd.DataFrame(hf["questions"])
    hf_slim = hf_df[["question_id", "question_type_id", "data_type", "data_source"]]

    df = pd.read_excel(input_xlsx)
    m = df["image_path"].astype(str).str.extract(r"(\d+)_(\d+)")
    df["question_id"] = m[0] + "_" + m[1]

    merged = df.merge(hf_slim, on="question_id", how="left", indicator=True)
    matched = (merged["_merge"] == "both").sum()
    if matched < len(df):
        unmatched = merged[merged["_merge"] != "both"]
        raise RuntimeError(
            f"Join incomplete: {matched}/{len(df)} matched. "
            f"First unmatched image_path: {unmatched.iloc[0]['image_path']!r}"
        )
    merged = merged.drop(columns=["_merge"])
    merged["question_type"] = merged["question_type_id"].map(QUESTION_TYPE_NAMES)

    print(f"Join: {matched}/{len(df)} rows matched (100%)")
    print(f"Writing {out_xlsx}")
    merged.to_excel(out_xlsx, index=False)
    return merged


def enrich_from_pkl(pkl_path: Path, base_enriched: Path, out_xlsx: Path):
    with open(pkl_path, "rb") as f:
        live = pickle.load(f)
    print(f"Loaded pkl: {pkl_path.name}  ({len(live)} predictions)")

    base = pd.read_excel(base_enriched)
    base["index"] = base["index"].astype(int)
    print(f"Base enriched: {len(base)} rows "
          f"({base['index'].nunique()} unique indices)")

    idx_to_positions = defaultdict(list)
    for pos, idx in enumerate(base["index"].tolist()):
        idx_to_positions[idx].append(pos)
    consumed = defaultdict(int)

    rows, unmatched, oversat, no_letter, collision_hits = [], 0, 0, 0, 0
    for key, pred in live.items():
        k = int(key)
        positions = idx_to_positions.get(k, [])
        if not positions:
            unmatched += 1
            continue
        taken = consumed[k]
        if taken >= len(positions):
            oversat += 1
            continue
        pos = positions[taken]
        consumed[k] += 1
        if len(positions) > 1 and taken > 0:
            collision_hits += 1
        base_row = base.iloc[pos].to_dict()
        letter = extract_letter(pred)
        if letter is None:
            no_letter += 1
        base_row["prediction"] = pred
        base_row["pred_letter"] = letter
        base_row["hit"] = int(
            letter is not None and letter == str(base_row["answer"]).upper())
        rows.append(base_row)

    df = pd.DataFrame(rows)
    pct = len(df) / len(base) * 100
    print(f"\nProgress: {len(df)}/{len(base)} ({pct:.1f}%)")
    print(f"Unmatched pkl keys: {unmatched}")
    print(f"Pkl entries beyond base saturation: {oversat}")
    print(f"Collision-ambiguous rows (2nd occurrence of a shared index): {collision_hits}")
    print(f"Predictions where regex couldn't resolve a letter: {no_letter}")
    print(f"Overall approx accuracy: {df['hit'].mean()*100:.2f}%\n")

    print("Per-subpart approx:")
    print(df.groupby("subpart")
            .agg(n=("hit", "size"), acc=("hit", "mean"))
            .to_string())

    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_xlsx, index=False)
    print(f"\nWriting {out_xlsx}")
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--input",
                   help="xlsx mode: path to *_gpt-4o-mini_result.xlsx (judged)")
    g.add_argument("--pkl",
                   help="pkl mode: path to a VLMEvalKit chain pkl (approximate)")

    p.add_argument("--base-enriched",
                   help="pkl mode only: path to base's *_enriched.xlsx")
    p.add_argument("--output", default=None,
                   help="xlsx mode: default=<input>_enriched.xlsx; "
                        "pkl mode: required (name it *_enriched.xlsx so "
                        "plot_seedbench.py picks it up)")
    p.add_argument("--hf-json", default="data/SEED-Bench_v2_level1_2_3.json",
                   help="xlsx mode: local cache path for the HF JSON")
    args = p.parse_args()

    if args.input:
        in_path = Path(args.input)
        if not in_path.is_file():
            p.error(f"not a file: {in_path}")
        out_path = Path(args.output) if args.output else in_path.with_name(
            in_path.stem + "_enriched" + in_path.suffix)
        hf_json = ensure_hf_json(Path(args.hf_json))
        enrich_from_xlsx(in_path, hf_json, out_path)
    else:
        if not args.base_enriched:
            p.error("--pkl mode requires --base-enriched")
        if not args.output:
            p.error("--pkl mode requires --output (ending in _enriched.xlsx)")
        pkl_path = Path(args.pkl)
        base_path = Path(args.base_enriched)
        for pp in (pkl_path, base_path):
            if not pp.is_file():
                p.error(f"not a file: {pp}")
        enrich_from_pkl(pkl_path, base_path, Path(args.output))


if __name__ == "__main__":
    main()

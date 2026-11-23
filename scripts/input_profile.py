#!/usr/bin/env python3
"""
How each benchmark is presented to the model: rebuild every question's input the way VLMEvalKit + Qwen3-VL
(-cp keys, use_custom_prompt=True) build it, and measure it. Nothing is run on a GPU.

Per question:
  images        images in the input (lists / video frames count individually; vLLM keeps at most 24)
  vis_tokens    visual tokens: each image resized like qwen_vl_utils (sides rounded to 32 px, 4..16384 tokens
                per image; 1 token per 32x32 block), summed over the question's images
  img_w, img_h  median image size after resizing
  text_tokens   tokens of the text prompt (Qwen tokenizer if available, else ~chars/4)
  n_options     answer options; img_options = options that are image files (SEED-Bench generation parts)
  hint          the prompt has a Hint: line
Prompt text = qwen3_vl/prompt.py: "Hint: ..\nQuestion: ..\nOptions:\nA. ..\n" + "Answer with the option letter only."
(MMMU: "Please select the correct answer from the options above.").

Large files (SEED-Bench-2, ~30 GB) are read in chunks and sampled (reservoir sampling, --sample rows); in them an
image field that references another row is not resolved (counted as an image of unknown size).

Usage (on Fir, from fine-tuning-ELVIS; use a CPU job for SEED-Bench):
  python scripts/input_profile.py
  python scripts/input_profile.py --datasets MMStar SEEDBench2 --sample 3000 --by category
Writes results/input_profile.csv (one row per question) and prints per-benchmark / per-category summaries.
"""

import argparse
import ast
import base64
import io
import math
import os
import random
import re
from pathlib import Path

import pandas as pd

DATASETS = ["SEEDBench2", "MMMU_DEV_VAL", "MMStar", "MMBench_DEV_EN_V11", "AI2D_TEST"]
FACTOR, MIN_TOKENS, MAX_TOKENS, MAX_IMAGES = 32, 4, 16384, 24    # qwen_vl_utils / VLMEvalKit Qwen3-VL defaults
CAT_COLS = ("category", "l2-category", "subpart")
BIG = 2e9   # tsv size above which the file is streamed and sampled


def smart_resize(h, w):
    """qwen_vl_utils.smart_resize: round to multiples of FACTOR, keep pixels within [MIN, MAX] tokens."""
    min_px, max_px = MIN_TOKENS * FACTOR * FACTOR, MAX_TOKENS * FACTOR * FACTOR
    hb, wb = max(FACTOR, round(h / FACTOR) * FACTOR), max(FACTOR, round(w / FACTOR) * FACTOR)
    if hb * wb > max_px:
        beta = math.sqrt(h * w / max_px)
        hb, wb = math.floor(h / beta / FACTOR) * FACTOR, math.floor(w / beta / FACTOR) * FACTOR
    elif hb * wb < min_px:
        beta = math.sqrt(min_px / (h * w))
        hb, wb = math.ceil(h * beta / FACTOR) * FACTOR, math.ceil(w * beta / FACTOR) * FACTOR
    return hb, wb


def image_list(cell, image_map):
    """VLMEvalKit image field -> list of base64 strings (a string, a list, or another row's index)."""
    s = "" if cell is None else str(cell)
    if 0 < len(s) <= 64:
        s = image_map.get(s, s)
    if s.startswith("["):
        try:
            return list(ast.literal_eval(s))
        except (ValueError, SyntaxError):
            pass
    return [s] if s else []


def image_size(b64):
    from PIL import Image
    try:
        with Image.open(io.BytesIO(base64.b64decode(b64))) as im:
            return im.size     # (w, h); only the header is parsed
    except Exception:
        return None


def build_prompt(row, dataset):
    """Text part of the Qwen3-VL custom MCQ prompt (vlmeval/vlm/qwen3_vl/prompt.py)."""
    opts = [(c, row[c]) for c in "ABCDEFGHIJ" if c in row and str(row[c]).strip() not in ("", "nan")]
    hint = str(row.get("hint", "")).strip()
    p = f"Hint: {hint}\n" if hint and hint != "nan" else ""
    p += f"Question: {row['question']}\n"
    if opts:
        p += "Options:\n" + "".join(f"{c}. {v}\n" for c, v in opts)
        p += ("Please select the correct answer from the options above." if dataset.startswith("MMMU")
              else "Answer with the option letter only.")
    return p.rstrip(), opts


def profile_row(row, dataset, image_map, tok):
    text, opts = build_prompt(row, dataset)
    imgs = image_list(row.get("image"), image_map)
    sizes = [s for s in (image_size(b) for b in imgs[:MAX_IMAGES]) if s]
    resized = [smart_resize(h, w) for w, h in sizes]
    toks = [hb * wb // (FACTOR * FACTOR) for hb, wb in resized]
    r = dict(dataset=dataset, index=row["index"], images=len(imgs), images_over_limit=max(0, len(imgs) - MAX_IMAGES),
             vis_tokens=sum(toks), vis_tokens_per_image=(sum(toks) / len(toks)) if toks else None,
             img_w=pd.Series([w for _, w in resized]).median() if resized else None,
             img_h=pd.Series([h for h, _ in resized]).median() if resized else None,
             text_tokens=len(tok(text)) if tok else round(len(text) / 4), n_options=len(opts),
             img_options=sum(bool(re.search(r"\.(jpe?g|png|gif|bmp)\b", str(v), re.I)) for _, v in opts),
             hint=text.startswith("Hint:"))
    for c in CAT_COLS:
        if c in row and str(row[c]) not in ("", "nan"):
            r[c] = str(row[c])
    return r


def read_rows(tsv, sample, seed):
    """Rows of a dataset tsv (all, or `sample` of them). Small files: read fully, image references resolved.
    Large files: one chunked pass with a uniform reservoir sample."""
    rng = random.Random(seed)
    if tsv.stat().st_size < BIG:
        df = pd.read_csv(tsv, sep="\t", dtype=str, keep_default_na=False)
        rows = df.to_dict("records")
        image_map = dict(zip(df["index"], df["image"])) if "image" in df else {}
        if sample and len(rows) > sample:
            rows = rng.sample(rows, sample)
        return rows, image_map, len(df)
    kept, seen = [], 0
    for chunk in pd.read_csv(tsv, sep="\t", dtype=str, keep_default_na=False, chunksize=500):
        for row in chunk.to_dict("records"):
            seen += 1
            if len(kept) < sample:
                kept.append(row)
            elif (j := rng.randrange(seen)) < sample:
                kept[j] = row
    return kept, {}, seen


def summarize(df, by):
    g = df.groupby(by)
    pct = lambda col, cond: g[col].apply(lambda s: 100 * cond(s).mean()).round(1)
    return pd.DataFrame({
        "questions": g.size(),
        "images_mean": g["images"].mean().round(2),
        "multi_image_%": pct("images", lambda s: s > 1),
        "over_24_%": pct("images_over_limit", lambda s: s > 0),
        "vis_tok/img_med": g["vis_tokens_per_image"].median().round(0),
        "vis_tok/q_med": g["vis_tokens"].median().round(0),
        "vis_tok/q_p90": g["vis_tokens"].quantile(0.9).round(0),
        "img_w_med": g["img_w"].median().round(0),
        "img_h_med": g["img_h"].median().round(0),
        "text_tok_med": g["text_tokens"].median().round(0),
        "text_tok_p90": g["text_tokens"].quantile(0.9).round(0),
        "options_mode": g["n_options"].agg(lambda s: s.mode().iat[0] if len(s) else None),
        "img_options_%": pct("img_options", lambda s: s > 0),
        "hint_%": pct("hint", lambda s: s),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--sample", type=int, default=3000, help="questions per benchmark (0 = all small files)")
    parser.add_argument("--by", default=None, choices=list(CAT_COLS), help="also summarize per category column")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/input_profile.csv")
    args = parser.parse_args()

    tok = None
    try:
        from transformers import AutoTokenizer
        t = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
        tok = lambda s: t(s, add_special_tokens=False)["input_ids"]
        print("text tokens: Qwen3-VL tokenizer")
    except Exception as e:
        print(f"text tokens: ~chars/4 (tokenizer unavailable: {type(e).__name__})")

    lmu = Path(os.environ.get("LMUData", Path.home() / "LMUData"))
    rows = []
    for ds in args.datasets:
        tsv = lmu / f"{ds}.tsv"
        if not tsv.is_file():
            print(f"[skip] {ds}: no {tsv}")
            continue
        sample = args.sample or (3000 if tsv.stat().st_size >= BIG else 0)
        kept, image_map, seen = read_rows(tsv, sample, args.seed)
        if ds.startswith("MMMU"):   # the reported split only
            kept = [r for r in kept if str(r.get("split", "")) == "validation"] or kept
        rows += [profile_row(r, ds, image_map, tok) for r in kept]
        print(f"  {ds}: profiled {len(kept)} of {seen} questions")
    if not rows:
        raise SystemExit("nothing profiled")

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("\nPer benchmark (medians unless noted):")
        print(summarize(df, "dataset").to_string())
        if args.by and args.by in df:
            sub = df[df[args.by].notna()]
            print(f"\nPer {args.by}:")
            print(summarize(sub, ["dataset", args.by]).to_string())
    print(f"\nper-question rows: {args.out}")


if __name__ == "__main__":
    main()

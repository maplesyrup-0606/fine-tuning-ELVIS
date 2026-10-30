#!/usr/bin/env python3
"""
Does the model think (emit <think>...</think>), and does temperature change that?

Takes one SEED-Bench-2 L1 single-image question, shows the exact prompt the chat template
builds (any <think> pre-fill would be visible there), then answers it --repeats times at
temperature 0 (greedy) and at temperature 0.7 (Qwen's recommended sampling), printing the
raw output with special tokens kept, so a thinking block cannot be hidden.

Usage (GPU):
  python scripts/thinking_check.py
  python scripts/thinking_check.py --model results/merged/v3-all-both-e3 --repeats 5 --skill 3
"""

import argparse
import ast
import base64
import glob
import io
import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def pick_question(results, skill, row):
    f = glob.glob(f"{results}/seedbench_Qwen3-VL-4B-Instruct-t0p/*_enriched.xlsx") or \
        glob.glob(f"{results}/seedbench_base/*_enriched.xlsx")
    if not f:
        raise SystemExit("no enriched SEED-Bench result to pick a question from")
    df = pd.read_excel(f[0])
    q = df[df["question_type_id"] == skill].iloc[row]
    tsv = Path(os.environ.get("LMUData", Path.home() / "LMUData")) / "SEEDBench2.tsv"
    want = str(int(q["index"]))
    for chunk in pd.read_csv(tsv, sep="\t", usecols=["index", "image"], dtype={"index": str}, chunksize=200):
        hit = chunk[chunk["index"].str.strip() == want]
        if len(hit):
            img = str(hit.iloc[0]["image"])
            if img.startswith("["):
                img = ast.literal_eval(img)[0]
            return q, Image.open(io.BytesIO(base64.b64decode(img))).convert("RGB")
    raise SystemExit(f"index {want} not found in {tsv}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--skill", type=int, default=1, help="L1 single-image skill id (1-16)")
    parser.add_argument("--row", type=int, default=0, help="which question of that skill")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--results", default="results")
    args = parser.parse_args()

    q, image = pick_question(args.results, args.skill, args.row)
    opts = "\n".join(f"{c}. {q[c]}" for c in "ABCD" if c in q and pd.notna(q[c]))
    text = f"Question: {q['question']}\nOptions:\n{opts}\nPlease select the correct answer from the options above."
    print(f"model: {args.model}\nskill {args.skill} ({q.get('question_type', '')}), index {q['index']}, "
          f"correct answer: {q['answer']}\n")

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("=== prompt as the chat template builds it (look for <think>) ===")
    print(prompt)
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt").to(model.device)

    configs = {"temperature 0 (greedy)": dict(do_sample=False),
               "temperature 0.7, top_p 0.8, top_k 20": dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20)}
    for name, gen in configs.items():
        print(f"\n=== {name} ===")
        for r in range(args.repeats):
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, **gen)
            raw = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=False)[0]
            print(f"--- run {r + 1}: {len(raw)} chars, '<think>' present: {'<think>' in raw}")
            print(raw)


if __name__ == "__main__":
    main()

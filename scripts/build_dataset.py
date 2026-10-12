#!/usr/bin/env python3
"""
Build ms-swift chat-format JSONL for v2 multi-task fine-tuning.

Reads a rules_<split>.json from generate_programmatic_rules.py and, for every
(pattern, episode), emits:
  1 Stage 1 row: 6 example images (3 pos, then 3 neg) + Stage 1 prompt -> full GT rule
  6 Stage 2 rows: 1 query image + Stage 2 prompt containing the full GT rule -> label
Stage 2 always gets the GT rule (teacher forcing); at eval it gets the model's own.

Prompt wording (scripts/prompts.py):
  --variants train   one of TRAIN_VARIANTS sampled per row (seeded)  [train split]
  --variants elvis   ELVIS Mode 1 verbatim for every row             [val split]

Loss balance: each assistant message carries a `loss_scale` so that the two
objectives contribute `--classify-share` / (1 - share) of the weighted answer-text
tokens, with mean weight 1 over those tokens. Token counts are exact (Qwen
tokenizer). Requires `--is_binary_loss_scale false` in `swift sft`, otherwise
ms-swift silently drops the weights. The template suffix (<|im_end|> + newline,
2 tokens/row) is trained at a fixed weight of 1 and is excluded from the balance.

Writes <output> plus <output>.meta.json (counts, weights, wording mix, seed).
CPU only; needs transformers for the tokenizer.
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

from prompts import HELDOUT_VARIANT, STAGE1, TRAIN_VARIANTS, stage1_prompt, stage2_prompt


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_rows(episodes, data_root, principle, variants, rng):
    """Rows without loss_scale yet; returns (rows, kinds) with kinds[i] in {stage1, stage2}."""
    rows, kinds = [], []
    pick = (lambda: rng.choice(TRAIN_VARIANTS)) if variants == "train" else (lambda: variants)
    for ep in episodes:
        ex_pos, ex_neg = ep["examples"]["positive"], ep["examples"]["negative"]
        v = pick()
        rows.append({
            "messages": [
                {"role": "user", "content": "<image>" * (len(ex_pos) + len(ex_neg))
                 + stage1_prompt(v, principle, len(ex_pos), len(ex_neg))},
                {"role": "assistant", "content": ep["rule"]},
            ],
            "images": [str(data_root / p) for p in ex_pos + ex_neg],
            "_variant": v,
        })
        kinds.append("stage1")
        for label in ("positive", "negative"):
            for q in ep["queries"][label]:
                v = pick()
                rows.append({
                    "messages": [
                        {"role": "user", "content": "<image>" + stage2_prompt(v, ep["rule"])},
                        {"role": "assistant", "content": label},
                    ],
                    "images": [str(data_root / q)],
                    "_variant": v,
                })
                kinds.append("stage2")
    return rows, kinds


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", choices=["train", "val"], default="train",
                        help="selects rules/output paths and wording defaults from config")
    parser.add_argument("--rules", default=None, help="override the rules file")
    parser.add_argument("--output", default=None, help="override the output JSONL")
    parser.add_argument("--variants", default=None,
                        help="'train' (sample per row) or one wording name; default: train for "
                             "--split train, elvis for --split val")
    parser.add_argument("--classify-share", type=float, default=None,
                        help="override training.objective_balance.classify_share")
    args = parser.parse_args()

    cfg = load_config(args.config)
    principle = cfg["data"]["principle"]
    seed = cfg["experiment"]["seed"]
    rules_path = args.rules or cfg["rules"][f"{args.split}_path"]
    output_path = Path(args.output or cfg["training"][
        "dataset_path" if args.split == "train" else "val_dataset_path"])
    variants = args.variants or ("train" if args.split == "train" else "elvis")
    share = args.classify_share if args.classify_share is not None \
        else cfg["training"]["objective_balance"]["classify_share"]
    if variants not in ("train", *STAGE1):
        sys.exit(f"ERROR: unknown --variants {variants!r}")
    if variants == HELDOUT_VARIANT:
        sys.exit("ERROR: the held-out wording must never appear in a training/val file")

    elvis_data = os.environ.get(cfg["paths"]["elvis_data_env"])
    if not elvis_data:
        sys.exit(f"ERROR: env var {cfg['paths']['elvis_data_env']} is not set")
    data_root = Path(elvis_data)

    with open(rules_path) as f:
        episodes = json.load(f)["episodes"]
    print(f"Loaded {len(episodes)} episodes from {rules_path}")

    rng = random.Random(seed)
    rows, kinds = build_rows(episodes, data_root, principle, variants, rng)

    missing = sorted({p for r in rows for p in r["images"] if not Path(p).exists()})
    if missing:
        print(f"ERROR: {len(missing)} images not found. First 5:")
        for p in missing[:5]:
            print(f"  {p}")
        return 1

    # Exact answer-text token counts -> loss_scale per objective.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_id"])
    n_tok = [len(tokenizer(r["messages"][1]["content"], add_special_tokens=False)["input_ids"]) for r in rows]
    stage1_tok = [n for n, k in zip(n_tok, kinds) if k == "stage1"]
    t1 = sum(stage1_tok)
    t2 = sum(n for n, k in zip(n_tok, kinds) if k == "stage2")
    weights = {"stage1": (1 - share) * (t1 + t2) / t1, "stage2": share * (t1 + t2) / t2}

    order = list(range(len(rows)))
    rng.shuffle(order)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as out:
        for i in order:
            row, kind = rows[i], kinds[i]
            row["messages"][1]["loss_scale"] = round(weights[kind], 6)
            out.write(json.dumps({"messages": row["messages"], "images": row["images"]}) + "\n")

    counts = Counter(kinds)
    variant_mix = Counter(f"{k}/{r['_variant']}" for r, k in zip(rows, kinds))
    meta = {
        "rules": str(rules_path),
        "rows": {"total": len(rows), "stage1": counts["stage1"], "stage2": counts["stage2"]},
        "answer_text_tokens": {"stage1": t1, "stage2": t2,
                               "stage1_per_row": {"min": min(stage1_tok), "max": max(stage1_tok)}},
        "unweighted_share": {"stage1": t1 / (t1 + t2), "stage2": t2 / (t1 + t2)},
        "classify_share": share,
        "loss_scale": weights,
        "variants": dict(sorted(variant_mix.items())),
        "seed": seed,
        "tokenizer": cfg["model"]["base_id"],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(str(output_path) + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {len(rows)} rows ({counts['stage1']} Stage 1, {counts['stage2']} Stage 2) to {output_path}")
    print(f"Answer-text tokens: Stage 1 {t1}, Stage 2 {t2} "
          f"(unweighted share {t1 / (t1 + t2):.1%} / {t2 / (t1 + t2):.1%})")
    print(f"loss_scale for a {1 - share:.0%}/{share:.0%} split: "
          f"stage1 {weights['stage1']:.3f}, stage2 {weights['stage2']:.3f}")
    print(f"Meta: {output_path}.meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

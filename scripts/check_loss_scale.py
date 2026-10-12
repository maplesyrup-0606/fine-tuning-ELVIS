#!/usr/bin/env python3
"""
Verify that per-row `loss_scale` reaches the trained tokens in ms-swift 4.5.3.

Encodes one real Stage 1 row (6 images -> rule) and one Stage 2 row (1 image +
rule -> label) through the Qwen3-VL template exactly as `swift sft` would, and
prints the weight on every trained token. Run twice internally:
  1. is_binary_loss_scale=False  -> weights should be kept
  2. default (binary)            -> weights silently dropped (loss_scale None)

Also reports the weight on the end-of-answer token (<|im_end|>), which ms-swift
encodes as a separate suffix and may NOT receive the row's weight.

Needs ms-swift + the Qwen3-VL processor (no model weights, no GPU).
"""

import argparse
import json
import os
import sys
from pathlib import Path

from prompts import stage1_prompt, stage2_prompt


def build_rows(ep, data_root, w1, w2):
    ex = ep["examples"]["positive"] + ep["examples"]["negative"]
    stage1 = {
        "messages": [
            {"role": "user", "content": "<image>" * len(ex) + stage1_prompt("elvis", "continuity", 3, 3)},
            {"role": "assistant", "content": ep["rule"], "loss_scale": w1},
        ],
        "images": [str(data_root / p) for p in ex],
    }
    stage2 = {
        "messages": [
            {"role": "user", "content": "<image>" + stage2_prompt("elvis", ep["rule"])},
            {"role": "assistant", "content": "positive", "loss_scale": w2},
        ],
        "images": [str(data_root / ep["queries"]["positive"][0])],
    }
    return stage1, stage2


def summarize(name, enc, tokenizer, expected):
    labels, scale = enc["labels"], enc.get("loss_scale")
    trained = [i for i, l in enumerate(labels) if l != -100]
    print(f"\n[{name}] trained tokens: {len(trained)}")
    if scale is None:
        print("  loss_scale: None  (weights dropped -> every trained token counts as 1)")
        return False
    # The template appends a suffix ("<|im_end|>" + "\n") after the answer text; it is
    # trained, but as a separate context that does not receive the row's loss_scale.
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    first_end = next(i for i in trained if labels[i] == end_id)
    body = [scale[i] for i in trained if i < first_end]
    suffix = [scale[i] for i in trained if i >= first_end]
    print(f"  answer-text tokens: {len(body)}  weights: {sorted(set(round(w, 4) for w in body))}")
    print(f"  suffix tokens:      {len(suffix)}  weights: {sorted(set(round(w, 4) for w in suffix))}")
    print(f"  untrained tokens with nonzero weight: "
          f"{sum(1 for i, l in enumerate(labels) if l == -100 and scale[i] != 0)}")
    tail = trained[-4:]
    print("  last trained tokens:", [(tokenizer.decode([labels[i]]), round(scale[i], 4)) for i in tail])
    return all(abs(w - expected) < 1e-6 for w in body)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--rules", default="data/continuity/rules_ft_val.json")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--w-rule", type=float, default=0.561)
    parser.add_argument("--w-classify", type=float, default=4.568)
    parser.add_argument("--max-pixels", type=int, default=224 * 224)
    args = parser.parse_args()

    if not os.environ.get("ELVIS_DATA"):
        sys.exit("ERROR: env var ELVIS_DATA is not set")
    data_root = Path(os.environ["ELVIS_DATA"])
    with open(args.rules) as f:
        ep = json.load(f)["episodes"][0]

    import swift
    from swift import get_processor, get_template
    print(f"ms-swift {swift.__version__}")
    processor = get_processor(args.model)
    tokenizer = getattr(processor, "tokenizer", processor)
    rows = build_rows(ep, data_root, args.w_rule, args.w_classify)

    ok = True
    for binary in (False, None):
        template = get_template(processor, max_pixels=args.max_pixels, is_binary_loss_scale=binary)
        template.set_mode("train")
        tag = "is_binary_loss_scale=False" if binary is False else "default (binary)"
        print(f"\n===== {tag} =====")
        r1 = summarize("Stage 1 row", template.encode(rows[0]), tokenizer, args.w_rule)
        r2 = summarize("Stage 2 row", template.encode(rows[1]), tokenizer, args.w_classify)
        if binary is False:
            ok = r1 and r2

    print("\nRESULT:", "PASS — row weights reach the answer tokens" if ok
          else "FAIL — row weights do not reach the answer tokens as expected")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

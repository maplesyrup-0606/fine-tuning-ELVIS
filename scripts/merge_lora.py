#!/usr/bin/env python3
"""
Merge a LoRA adapter into base Qwen3-VL-4B and save as a standalone HF dir.

VLMEvalKit's Qwen3VLChat loads via AutoModelForImageTextToText.from_pretrained,
which expects a complete model (not an adapter). We fuse the LoRA delta into
the base weights once, write out an ~8 GB HF dir, and point VLMEvalKit at that.

Usage:
    python scripts/merge_lora.py \\
        --adapter results/checkpoints/llm/v1-.../checkpoint-324 \\
        --out     results/merged/llm-e324
"""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--base", default="Qwen/Qwen3-VL-4B-Instruct",
                   help="HF id or local path of base model")
    p.add_argument("--adapter", required=True,
                   help="path to peft adapter dir (e.g., checkpoint-324)")
    p.add_argument("--out", required=True,
                   help="output dir for merged standalone model")
    args = p.parse_args()

    adapter_dir = Path(args.adapter)
    if not (adapter_dir / "adapter_config.json").is_file():
        p.error(f"{adapter_dir} does not look like a peft adapter "
                f"(no adapter_config.json)")

    print(f"Loading base: {args.base}")
    base = AutoModelForImageTextToText.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    print(f"Loading adapter: {args.adapter}")
    peft_model = PeftModel.from_pretrained(base, args.adapter)

    print("Merging adapter into base weights...")
    merged = peft_model.merge_and_unload()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {out}")
    merged.save_pretrained(out, safe_serialization=True)

    print("Saving processor...")
    AutoProcessor.from_pretrained(args.base).save_pretrained(out)

    print(f"Done. Merged model + processor at: {out}")


if __name__ == "__main__":
    main()

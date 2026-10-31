#!/usr/bin/env python3
"""
Does the model think (emit <think>...</think>), and does temperature change that?
Runs VLMEvalKit's own code path, not a re-implementation:

  dataset  = build_dataset('SEEDBench2')                     (as run.py)
  model    = supported_VLM[<key>]()                          (our vlmeval/config.py entry: vLLM, sampling, penalties)
  struct   = same prompt rule as vlmeval/inference.py infer_data
  response = model.generate(message=struct, dataset='SEEDBench2')   (same call as infer_data)

One question is answered --repeats times per temperature. The -t0p and -t07 config entries
differ only in temperature and the vLLM wrapper reads self.temperature on every call, so
setting model.temperature reproduces both entries. Prints the message VLMEvalKit builds, the
chat-template prompt, and each raw response (what VLMEvalKit saves as `prediction`, since
SPLIT_THINK is not set).

Usage (GPU):
  python scripts/thinking_check.py
  python scripts/thinking_check.py --key Qwen3-VL-4B-Instruct-v3-all-both-e3-t0p --row 5
"""

import argparse

from vlmeval.config import supported_VLM
from vlmeval.dataset import build_dataset

DATASET = "SEEDBench2"


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--key", default="Qwen3-VL-4B-Instruct-t0p", help="model key in vlmeval/config.py")
    parser.add_argument("--subpart", default="Single-Image & Text Comprehension", help="SEED-Bench-2 part")
    parser.add_argument("--row", type=int, default=0, help="which question of that part")
    parser.add_argument("--temps", type=float, nargs="+", default=[0.0, 0.7])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    dataset = build_dataset(DATASET)
    data = dataset.data
    line = data[data["subpart"] == args.subpart].iloc[args.row]
    print(f"key: {args.key}\n{args.subpart}, row {args.row}, index {line['index']}, correct answer: {line['answer']}\n")

    model = supported_VLM[args.key]()

    # Same prompt rule as vlmeval/inference.py infer_data.
    if hasattr(dataset, "force_use_dataset_prompt") and dataset.force_use_dataset_prompt:
        struct = dataset.build_prompt(line)
    elif hasattr(model, "use_custom_prompt") and model.use_custom_prompt(DATASET):
        struct = model.build_prompt(line, dataset=DATASET)
    else:
        struct = dataset.build_prompt(line)
    print("=== message VLMEvalKit builds (struct) ===")
    for item in struct:
        print(f"  {item['type']}: {item['value']}")

    # The exact text the chat template produces for this message (look for <think>).
    try:
        msgs = [{"role": "user", "content": model._prepare_content(struct, dataset=DATASET)}]
        if getattr(model, "system_prompt", None) is not None:
            msgs.insert(0, {"role": "system", "content": model.system_prompt})
        print("\n=== chat-template prompt ===")
        print(model.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    except Exception as e:  # private helper; the check below does not depend on it
        print(f"\n(could not render chat-template prompt: {type(e).__name__}: {e})")

    print(f"\nsettings from config: top_p={model.top_p} top_k={model.top_k} "
          f"presence_penalty={model.presence_penalty} repetition_penalty={model.repetition_penalty} "
          f"max_new_tokens={model.max_new_tokens}")
    for t in args.temps:
        model.temperature = t
        print(f"\n=== temperature {t} ===")
        for r in range(args.repeats):
            response = model.generate(message=struct, dataset=DATASET)
            print(f"--- run {r + 1}: {len(response)} chars | '<think>' {'<think>' in response} | "
                  f"'</think>' {'</think>' in response}")
            print(response)


if __name__ == "__main__":
    main()

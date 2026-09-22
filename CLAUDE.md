# fine-tuning-ELVIS — Claude Code Instructions

## What This Is
LoRA fine-tuning of Qwen3-VL-4B on ELVIS continuity to test whether Gestalt-perception improvement transfers to general vision (SEED-Bench-2). Sibling project to `~/Documents/Projects/ELVIS`; depends on ELVIS at import time via `ELVIS_ROOT` env var.

## Source of Truth
Every locked experimental decision lives in `config.yaml`. When something changes, update `config.yaml`, not code. When something is ambiguous, check `config.yaml` first.

## Tech Stack
- Python 3.10+, PyTorch, HuggingFace Transformers
- **ms-swift** for LoRA fine-tuning (framework does not know about ELVIS)
- **peft** for adapter loading at eval time
- **VLMEvalKit** for SEED-Bench-2
- WandB for tracking
- Slurm on Fir (Alliance Canada)

## ELVIS Coupling
- `ELVIS_ROOT` env var must be set. Scripts do `sys.path.insert(0, os.environ["ELVIS_ROOT"])` at import.
- Imports used from ELVIS: `scripts.baseline_models.qwen.{load_qwen_model, infer_logic_rules, evaluate_llm}`, `scripts.config`, `scripts.utils.*`.
- **Do not** copy-paste ELVIS code into this repo. If a change is needed, either wrap or commit to ELVIS proper.

## Data Flow
```
ELVIS train/ images  ─┐
                      ├─→ generate_rules.py ──→ data/rules.json
Stage 1 (base model)  ─┘                        (pattern_id → rule_text)

data/rules.json ─┐
                 ├─→ build_dataset.py ──→ data/train.jsonl
Fine-tuning imgs ─┘                      (ms-swift chat format)

data/train.jsonl ──→ ms-swift SFT ──→ results/checkpoints/{epoch-N}
                            │
                            ├─→ eval_continuity.py (per epoch)  ──→ acc
                            └─→ eval_seedbench.py (final)       ──→ acc
```

## Conventions
- Per-pattern fine-tuning samples use images 4–9 per side; images 1–3 per side are reserved as Stage 1 examples (matches how eval runs).
- Rules are generated **once** with the base model. No runtime cache. No per-epoch regeneration.
- Test split (`ELVIS_DATA/continuity/test/`) is **never** touched by this repo. It's held out for eval.
- All scripts should read `config.yaml` rather than hardcoding values.

## Early-Stop Logic
- Trigger 1: continuity test accuracy ≥ 67% (7% absolute over 59.76% base).
- Trigger 2: accuracy drops between consecutive epochs (patience 1).
- Hard cap: 3 epochs.
- Final checkpoint selection: highest continuity test accuracy across all saved checkpoints.

## Do Not
- Do not modify ELVIS eval protocols (Mode 1 must stay as-is for comparability with the correlation-plot baseline of 59.76%).
- Do not regenerate the test split.
- Do not train the vision tower or merger — LLM-only LoRA.
- Do not add features not present in `config.yaml`. Update the config first if a decision changes.

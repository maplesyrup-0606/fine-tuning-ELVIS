# fine-tuning-ELVIS

LoRA fine-tuning of Qwen3-VL-4B on the ELVIS continuity Gestalt-perception task, testing whether Gestalt-principle fine-tuning transfers to general vision performance on SEED-Bench-2.

## Hypothesis

Prior work on ELVIS ([ELVIS repo](https://github.com/maplesyrup-0606/ELVIS)) plotted correlations between Gestalt-principle accuracy and general vision benchmark scores (MMMU, MMStar, MMBench, AI2D) across multiple VLM families and sizes. Several Gestalt principles showed non-trivial correlation with general vision ability. This experiment tests the causal direction: **does improving Gestalt perception (via fine-tuning on continuity) improve general vision?**

Continuity is the first principle tested; other principles may follow depending on results.

## Method

- **Base model**: Qwen3-VL-4B-Instruct
- **Training**: LoRA (rank 4, alpha 8, attention-only, LLM only) via [ms-swift](https://github.com/modelscope/ms-swift)
- **Data**: ELVIS continuity train split, regenerated at 9 pos + 9 neg per pattern (5,184 samples)
- **Evaluation**:
  - Per-epoch: ELVIS continuity test set, Mode 1 protocol (unchanged from ELVIS baseline)
  - Final: full SEED-Bench-2 via VLMEvalKit
- **Baseline reference**: Qwen3-VL-4B on ELVIS continuity Mode 1 = 59.76%. Target = 67% (~7% absolute improvement) as early-stop trigger.

The full experiment spec — every locked decision — lives in [`config.yaml`](config.yaml).

## Repository structure

```
fine-tuning-ELVIS/
├── config.yaml          # single source of truth for every decision
├── scripts/
│   ├── generate_rules.py            # pre-generate Stage 1 rules for all patterns
│   ├── build_dataset.py             # build ms-swift JSONL from rules + images
│   ├── train.py                     # ms-swift LoRA fine-tuning wrapper
│   ├── eval_continuity.py           # ELVIS Mode 1 eval with LoRA-adapted model
│   ├── eval_seedbench.py            # SEED-Bench-2 baseline + post-FT eval
│   └── regenerate_continuity_train.py  # invokes ELVIS main.py to regen train split
├── slurm/               # Fir cluster job scripts
├── data/                # rules.json, train.jsonl (gitignored)
└── results/             # checkpoints, eval outputs (gitignored)
```

## Dependency on ELVIS

This repo depends on the [ELVIS](https://github.com/maplesyrup-0606/ELVIS) codebase for:

- Raw dataset (`ELVIS_DATA/continuity/{train,test}/`)
- Stage 1 rule generation (`scripts.baseline_models.qwen.infer_logic_rules`)
- Mode 1 evaluation primitives (`infer_logic_rules`, `evaluate_llm`)
- Model loading (`load_qwen_model`) — kept consistent between fine-tuning and eval
- Pattern generation for regenerating the train split (`scripts/main.py`)

**Coupling mechanism**: set the `ELVIS_ROOT` env var to the ELVIS checkout path. Scripts insert it into `sys.path` at import time.

**Required changes to ELVIS** (out-of-repo, tracked in Task #7):
- `scripts/main.py`: add `--num_samples` and `--splits` CLI flags so we can regenerate the train split at 9+9 without touching test.

## Setup

```bash
# 1. Point at the ELVIS checkout
export ELVIS_ROOT=/path/to/ELVIS
export ELVIS_DATA=/path/to/elvis/data
export ELVIS_FINETUNING_RESULTS=/path/to/results

# 2. Install
pip install -r requirements.txt

# 3. Regenerate the continuity train split (one-time, uses ELVIS)
python scripts/regenerate_continuity_train.py

# 4. Pre-generate Stage 1 rules with the base model
python scripts/generate_rules.py

# 5. Build the ms-swift dataset
python scripts/build_dataset.py

# 6. (Prerequisite) Run SEED-Bench-2 baseline on the base model
python scripts/eval_seedbench.py --model base

# 7. Fine-tune
python scripts/train.py

# 8. Evaluate best checkpoint on SEED-Bench-2
python scripts/eval_seedbench.py --model best_checkpoint
```

Slurm scripts wrapping steps 3–8 for the Fir cluster live in [`slurm/`](slurm/).

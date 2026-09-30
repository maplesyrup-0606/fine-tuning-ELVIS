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

## Current Status (as of last session)

**Done**
- Repo scaffolded, `config.yaml` locked, git initialized, ELVIS coupling wired via `ELVIS_ROOT`.
- Task #7: continuity train split regenerated on Fir at 448×448, 9 pos + 9 neg per pattern × 432 patterns. First 3 per side → Stage 1 examples (matches eval); remaining 6 → fine-tuning samples (5,184 total).
- ELVIS-side patches (in `maplesyrup-0606/ELVIS`, both pulled on Fir):
  1. `scripts/main.py` gains `--num_samples` and `--splits` CLI flags; `save_principle_patterns` wipes only requested splits.
  2. `scripts/main.py` output path aligned with `evaluate_models.py`: when `ELVIS_DATA` is set, no `res_XXX_pin_False/` prefix. Fixed the gen-vs-eval path asymmetry.
- Data on Fir now flat: `ELVIS_DATA=/scratch/merc0606/ELVIS/data` with principle folders directly under it (moved out of the misleading `res_448_pin_False/` wrapper).
- Task #6 (partial): `slurm/eval_seedbench_baseline.sh` written — h100:1, 3h wallclock, self-chained via `--dependency=afterany` (cap 10 iters), email notifications, results copied to `$RESULTS_DIR/seedbench_base/`.

**In progress**
- Task #8: SEEDBench2 baseline eval chain submitted on Fir (see `squeue -u $USER`). On completion, outputs land in `$ELVIS_FINETUNING_RESULTS/seedbench_base/` — record the score as the "before" reference for the post-fine-tuning delta.

**Pending**
- Task #3: `scripts/generate_rules.py` (pre-generate Stage 1 rules with base Qwen3-VL-4B) + `scripts/build_dataset.py` (emit `data/train.jsonl` in ms-swift chat format).
- Task #4: `scripts/train.py` — ms-swift LoRA fine-tuning wrapper reading `config.yaml`; per-epoch continuity eval hook.
- Task #5: `scripts/eval_continuity.py` (Mode 1 with LoRA-adapted model) + `scripts/eval_seedbench.py` (harness wrapper for both baseline and post-FT runs).
- Task #6 (remaining): `slurm/train.sh` (3g.40gb MIG per config), `slurm/eval_seedbench_ft.sh` (post-FT, same chain pattern as baseline).

## Fir cluster environment (Alliance Canada)

- Login node `login2`; username `merc0606`; PI `lsigal`; Slurm account `def-lsigal` (SLURM auto-decorates GPU jobs as `def-lsigal_g`).
- `$SCRATCH` = `/scratch/merc0606` (also symlinked from `/home/merc0606/scratch`).
- Venv at `$SCRATCH/venv/elvis` (Python 3.11).
- Repos side by side: `$SCRATCH/{ELVIS, VLMEvalKit, fine-tuning-ELVIS}`.
- HF cache redirected to `$SCRATCH/hf_cache` (home quota is tiny).

**Always load these modules before activating the venv:**
```bash
module load StdEnv/2023 gcc opencv rdkit arrow
source $SCRATCH/venv/elvis/bin/activate
```
Reason: on Alliance Canada, `rdkit`, `pyarrow`, and `opencv-python` are *modules*, not pip packages — wheelhouse ships dummy wheels that intentionally fail installs with instructions to load the module first.

**VLMEvalKit install quirk**: `pip install -e .` succeeds, but `from vlmeval.config import supported_VLM` surfaces missing benchmark-specific deps one at a time on first import (hit `rouge_score` so far — install as they appear).

## Chained-job pattern (reusable for training + post-FT eval)

`eval_seedbench_baseline.sh` uses a self-chaining pattern that fits 24h+ jobs into 3h queue-friendly slots on Fir:
- Wall = 3h; wraps `python run.py ...` in `set +e` to capture nonzero exit.
- On nonzero exit AND `CHAIN_ITER < CHAIN_MAX`: `sbatch --dependency=afterany:$SLURM_JOB_ID --export=ALL,CHAIN_ITER=$NEXT_ITER ...` re-submits itself.
- On exit 0: chain terminates cleanly.
- Only one job in the chain is runnable at any moment (linked list, not fan-out).
- Relies on VLMEvalKit's per-item resume (idempotent ID-keyed output DataFrame with atomic dump).
- Same pattern reusable for training (checkpoint every epoch → resume from latest checkpoint) and post-FT eval.

## Terminology
User prefers **"fine-tuning"** over "SFT". Applies everywhere in this project.

#!/bin/bash
#SBATCH --job-name=finetune-qwen3vl4b-continuity
#SBATCH --account=def-lsigal
#SBATCH --time=06:00:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --output=logs/train_%x_%j.out
#SBATCH --error=logs/train_%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mercurymcindoe@gmail.com

set -euo pipefail

# Usage: sbatch slurm/train.sh <llm|projector|both>
# Auto-chains on wallclock kill (exit 143/137) up to CHAIN_MAX times. Override
# with `sbatch --export=ALL,CHAIN_MAX=<n> slurm/train.sh <variant>`.
VARIANT=${1:-}
if [ -z "$VARIANT" ]; then
    echo "ERROR: variant required. Usage: sbatch slurm/train.sh <llm|projector|both>"
    exit 1
fi
case "$VARIANT" in
    llm|projector|both) ;;
    *) echo "ERROR: variant must be one of: llm, projector, both (got: $VARIANT)"; exit 1 ;;
esac

CHAIN_ITER=${CHAIN_ITER:-1}
CHAIN_MAX=${CHAIN_MAX:-3}

# --- Environment (mirrors eval_seedbench_baseline.sh) ---
module load python/3.11.5 cuda/12.6 opencv/4.13.0 rdkit arrow

VENV=${VENV:-$SCRATCH/venv/elvis}
source "$VENV/bin/activate"

export HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME"

# WandB — key must be pre-configured on the login node via `wandb login` (writes
# to ~/.netrc, which compute nodes can read) OR by exporting WANDB_API_KEY in
# the submitting shell. Without it, wandb will fall back to offline mode.
if [ -z "${WANDB_API_KEY:-}" ] && [ ! -f "$HOME/.netrc" ]; then
    echo "WARNING: no WANDB_API_KEY and no ~/.netrc — wandb will run offline"
    export WANDB_MODE=offline
fi

RESULTS_DIR=${ELVIS_FINETUNING_RESULTS:-$SLURM_SUBMIT_DIR/results}
mkdir -p "$RESULTS_DIR" logs

# --- Sanity ---
echo "=== Job $SLURM_JOB_ID on $(hostname) — variant=$VARIANT chain=$CHAIN_ITER/$CHAIN_MAX ==="
nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
python -c "import swift; print('ms-swift:', swift.__version__)"
python -c "import peft; print('peft:', peft.__version__)"

# --- Run ---
# 3 epochs × 162 steps ≈ 60 min per variant. 6h wallclock is generous slack.
# Auto-resume: if a previous run of this variant crashed after saving
# checkpoint-epoch-N, resubmitting the same job picks up from there.
cd "$SLURM_SUBMIT_DIR"

LATEST_CKPT=$(ls -td "$RESULTS_DIR/checkpoints/$VARIANT"/checkpoint-* 2>/dev/null | head -1 || true)
RESUME_ARGS=""
if [ -n "$LATEST_CKPT" ] && [ -d "$LATEST_CKPT" ]; then
    echo "=== Auto-resume from $LATEST_CKPT ==="
    RESUME_ARGS="--resume-from $LATEST_CKPT"
fi

# Capture python's exit code without triggering `set -e` early exit.
set +e
python scripts/train.py \
    --variant "$VARIANT" \
    --output-dir "$RESULTS_DIR/checkpoints/$VARIANT" \
    $RESUME_ARGS
EXIT_CODE=$?
set -e

echo "=== Python exited with code $EXIT_CODE ==="
ls -la "$RESULTS_DIR/checkpoints/$VARIANT/" 2>/dev/null || true

# Auto-chain only on wallclock kill (SIGTERM=143 or SIGKILL=137). Any other
# nonzero exit is a real error (OOM, config, import) — don't loop on it.
if { [ "$EXIT_CODE" -eq 143 ] || [ "$EXIT_CODE" -eq 137 ]; } \
   && [ "$CHAIN_ITER" -lt "$CHAIN_MAX" ]; then
    NEXT=$((CHAIN_ITER + 1))
    echo "=== Wallclock kill (exit $EXIT_CODE); chaining iter $NEXT/$CHAIN_MAX for variant=$VARIANT ==="
    sbatch --dependency=afterany:$SLURM_JOB_ID \
           --export=ALL,CHAIN_ITER=$NEXT \
           "$0" "$VARIANT"
fi

exit $EXIT_CODE

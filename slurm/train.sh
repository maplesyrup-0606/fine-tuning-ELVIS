#!/bin/bash
#SBATCH --job-name=finetune-qwen3vl4b-continuity
#SBATCH --account=rrg-lsigal
#SBATCH --time=06:00:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --output=logs/train_%x_%j.out
#SBATCH --error=logs/train_%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mercurymcindoe@gmail.com

set -euo pipefail

# Usage: sbatch slurm/train.sh <llm|projector|both> [<run>]
#   <run>: a v3 run under config v3.runs (continuity, all); omit for v2.
#   e.g.  sbatch -J v3-all-both slurm/train.sh both all
# Auto-chains on wallclock kill (exit 143/137) up to CHAIN_MAX times. Override
# with `sbatch --export=ALL,CHAIN_MAX=<n> slurm/train.sh <variant> [<run>]`.
VARIANT=${1:-}
RUN=${2:-}
RUN_ARGS=${RUN:+--run $RUN}
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
# v2: 3 epochs × 189 steps ≈ 35-40 min per variant (v1 measured ~2.5 s/step;
# v2 steps carry ~1.3x the tokens) + per-epoch validation. 6h wallclock is slack.
# Auto-resume: if a previous run of this variant was killed after saving
# a checkpoint, resubmitting the same job picks up from the newest one.
# ms-swift saves under <ckpt_dir>/v<N>-<timestamp>/checkpoint-<step>.
cd "$SLURM_SUBMIT_DIR"

# v2: <results>/checkpoints_v2/<variant>;  v3: <results>/checkpoints_v3/<run>/<variant>
CKPT_DIR=$(ELVIS_FINETUNING_RESULTS="$RESULTS_DIR" python scripts/train.py --variant "$VARIANT" $RUN_ARGS --print-output-dir)
echo "=== Checkpoints: $CKPT_DIR ==="

LATEST_CKPT=$(ls -td "$CKPT_DIR"/v*/checkpoint-* 2>/dev/null | head -1 || true)
RESUME_ARGS=""
if [ -n "$LATEST_CKPT" ] && [ -d "$LATEST_CKPT" ]; then
    echo "=== Auto-resume from $LATEST_CKPT ==="
    RESUME_ARGS="--resume-from $LATEST_CKPT"
fi

# Capture python's exit code without triggering `set -e` early exit.
set +e
python scripts/train.py \
    --variant "$VARIANT" $RUN_ARGS \
    --output-dir "$CKPT_DIR" \
    $RESUME_ARGS
EXIT_CODE=$?
set -e

echo "=== Python exited with code $EXIT_CODE ==="
ls -la "$CKPT_DIR"/v*/ 2>/dev/null || true

# Auto-chain only on wallclock kill (SIGTERM=143 or SIGKILL=137). Any other
# nonzero exit is a real error (OOM, config, import) — don't loop on it.
if { [ "$EXIT_CODE" -eq 143 ] || [ "$EXIT_CODE" -eq 137 ]; } \
   && [ "$CHAIN_ITER" -lt "$CHAIN_MAX" ]; then
    NEXT=$((CHAIN_ITER + 1))
    echo "=== Wallclock kill (exit $EXIT_CODE); chaining iter $NEXT/$CHAIN_MAX for variant=$VARIANT ==="
    sbatch --dependency=afterany:$SLURM_JOB_ID \
           --export=ALL,CHAIN_ITER=$NEXT \
           "$0" "$VARIANT" "$RUN"
fi

exit $EXIT_CODE

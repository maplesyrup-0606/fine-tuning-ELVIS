#!/bin/bash
#SBATCH --job-name=finetune-qwen3vl4b-continuity
#SBATCH --account=def-lsigal
#SBATCH --time=03:00:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mercurymcindoe@gmail.com
#SBATCH --signal=B:USR1@120

set -euo pipefail

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
echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
python -c "import swift; print('ms-swift:', swift.__version__)"
python -c "import peft; print('peft:', peft.__version__)"

# --- Run ---
# 3 epochs × 162 steps ≈ 60 min training + adapter saves. Single-shot, no chain.
# SIGUSR1 handler mirrors the eval script so wallclock overruns don't wedge the
# adapter save half-written.
cd "$SLURM_SUBMIT_DIR"

handle_usr1() {
    echo "=== SIGUSR1 received ($(date)): wallclock in <120s, terminating training (PID $PYTHON_PID) ==="
    if [ -n "${PYTHON_PID:-}" ]; then
        kill -TERM $PYTHON_PID 2>/dev/null || true
        ( sleep 60 && kill -KILL $PYTHON_PID 2>/dev/null || true ) &
    fi
}
trap handle_usr1 USR1

set +e
python scripts/train.py --output-dir "$RESULTS_DIR/checkpoints" &
PYTHON_PID=$!
wait $PYTHON_PID
RC=$?
set -e

echo "=== train.py exited with $RC ==="
if [ $RC -ne 0 ]; then
    echo "Training failed. Adapters (if any) are in $RESULTS_DIR/checkpoints/"
    exit $RC
fi

echo "=== Done. Checkpoints in $RESULTS_DIR/checkpoints/ ==="
ls -la "$RESULTS_DIR/checkpoints/" 2>/dev/null || true

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
# 3 epochs × 162 steps ≈ 60 min. 8h wallclock is generous slack; if we blow past
# that, something is stuck and we want to investigate rather than silently retry.
cd "$SLURM_SUBMIT_DIR"

python scripts/train.py --output-dir "$RESULTS_DIR/checkpoints"

echo "=== Done. Checkpoints in $RESULTS_DIR/checkpoints/ ==="
ls -la "$RESULTS_DIR/checkpoints/" 2>/dev/null || true

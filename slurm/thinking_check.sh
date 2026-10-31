#!/bin/bash
#SBATCH --account=rrg-lsigal
#SBATCH --time=1:00:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --output=logs/thinking_check_%j.out
#SBATCH --error=logs/thinking_check_%j.err

set -euo pipefail

# Runs VLMEvalKit's own SEED-Bench-2 code path on one question (see scripts/thinking_check.py).
# Usage:
#   sbatch slurm/thinking_check.sh                                                   # base, -t0p entry
#   sbatch slurm/thinking_check.sh --key Qwen3-VL-4B-Instruct-v3-all-both-e3-t0p --row 5
# Output: logs/thinking_check_<jobid>.out

module load python/3.11.5 cuda/12.6 opencv/4.13.0 rdkit arrow
source "${VENV:-$SCRATCH/venv/elvis}/bin/activate"
export HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}
export TRANSFORMERS_CACHE=$HF_HOME/transformers
mkdir -p logs

cd "$SLURM_SUBMIT_DIR"
python scripts/thinking_check.py "$@"

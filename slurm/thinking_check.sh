#!/bin/bash
#SBATCH --account=rrg-lsigal
#SBATCH --time=0:30:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/thinking_check_%j.out
#SBATCH --error=logs/thinking_check_%j.err

set -euo pipefail

# Usage:
#   sbatch slurm/thinking_check.sh                                      # base model
#   sbatch slurm/thinking_check.sh --model results/merged/v3-all-both-e3 --repeats 5
# Output: logs/thinking_check_<jobid>.out

module load python/3.11.5 cuda/12.6 opencv/4.13.0 rdkit arrow
source "${VENV:-$SCRATCH/venv/elvis}/bin/activate"
export HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}
mkdir -p logs

cd "$SLURM_SUBMIT_DIR"
python scripts/thinking_check.py "$@"

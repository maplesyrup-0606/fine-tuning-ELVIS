#!/bin/bash
#SBATCH --account=def-lsigal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/rescore_%j.out
#SBATCH --error=logs/rescore_%j.err

set -euo pipefail

# Re-score saved VLMEvalKit MCQ answers with the GPT judge (scripts/rescore_vlmeval.py).
# CPU only: no model is run. def-lsigal because rrg-lsigal only has a GPU allocation.
# Needs the OpenAI key in $VLMEVALKIT_DIR/.env (OPENAI_API_KEY=...); compute nodes can reach OpenAI.
# Usage:
#   sbatch slurm/rescore_vlmeval.sh                                   # every SEEDBench2 run under results/
#   sbatch slurm/rescore_vlmeval.sh --keys Qwen3-VL-4B-Instruct-t0p   # only these
# Output: logs/rescore_<jobid>.out

module load python/3.11.5 cuda/12.6 opencv/4.13.0 rdkit arrow
source "${VENV:-$SCRATCH/venv/elvis}/bin/activate"
VLMEVALKIT_DIR=${VLMEVALKIT_DIR:-$SCRATCH/VLMEvalKit}
set -a; source "$VLMEVALKIT_DIR/.env"; set +a
mkdir -p logs

cd "$SLURM_SUBMIT_DIR"
python -u scripts/rescore_vlmeval.py "$@"

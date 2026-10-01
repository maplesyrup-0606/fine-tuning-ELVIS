#!/bin/bash
#SBATCH --account=def-lsigal
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/merge_%x_%j.out
#SBATCH --error=logs/merge_%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mercurymcindoe@gmail.com

set -euo pipefail

# Usage:
#   sbatch -J merge-<name> slurm/merge_lora.sh <adapter_dir> <out_dir>
#
# Example:
#   sbatch -J merge-llm-e2 slurm/merge_lora.sh \
#       $SCRATCH/fine-tuning-ELVIS/results/checkpoints/llm/v1-.../checkpoint-324 \
#       $SCRATCH/fine-tuning-ELVIS/results/merged/llm-e324
#
# CPU-only — merge is a forward-free matrix add, no GPU needed.
ADAPTER=${1:?usage: sbatch slurm/merge_lora.sh <adapter_dir> <out_dir>}
OUT=${2:?missing out dir}

module load python/3.11.5 cuda/12.6 opencv/4.13.0 rdkit arrow

VENV=${VENV:-$SCRATCH/venv/elvis}
source "$VENV/bin/activate"

export HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}
export TRANSFORMERS_CACHE=$HF_HOME/transformers

mkdir -p logs

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "Adapter: $ADAPTER"
echo "Out:     $OUT"
echo "HF_HOME: $HF_HOME"

cd "$SLURM_SUBMIT_DIR"
python scripts/merge_lora.py --adapter "$ADAPTER" --out "$OUT"

echo "=== Done. Output: $OUT ==="
du -sh "$OUT"

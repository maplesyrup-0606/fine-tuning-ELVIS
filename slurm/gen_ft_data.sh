#!/bin/bash
#SBATCH --account=def-lsigal
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=gen_ft_data
#SBATCH --output=logs/gen_ft_data_%j.out
#SBATCH --error=logs/gen_ft_data_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mercurymcindoe@gmail.com

set -euo pipefail

# Generate the v2 fine-tuning splits (separate from ELVIS train/test, which
# are the eval set) and sanity-check every label against its metadata.
#
# Usage:
#   sbatch slurm/gen_ft_data.sh
#
# Writes (wipes + rewrites only these two splits):
#   $ELVIS_DATA/continuity/ft_train/  12 per side (2 episodes / pattern), seed 1
#   $ELVIS_DATA/continuity/ft_val/     6 per side (1 episode / pattern),  seed 2
# CPU only. ~10 min expected (~48 img/s locally); 1h wallclock is a buffer.

PRINCIPLE=continuity
TRAIN_SAMPLES=12
VAL_SAMPLES=6
TRAIN_SEED=1
VAL_SEED=2

# --- Environment ---
module load python/3.11.5 cuda/12.6 opencv/4.13.0 rdkit arrow  # same stack as the venv was built with

VENV=${VENV:-$SCRATCH/venv/elvis}
source "$VENV/bin/activate"

export ELVIS_ROOT=${ELVIS_ROOT:-$SCRATCH/ELVIS}
export ELVIS_DATA=${ELVIS_DATA:-$ELVIS_ROOT/data}
FT_ROOT=${FT_ROOT:-$SCRATCH/fine-tuning-ELVIS}

echo "ELVIS_ROOT=$ELVIS_ROOT"
echo "ELVIS_DATA=$ELVIS_DATA"
echo "ELVIS commit: $(git -C "$ELVIS_ROOT" log --oneline -1)"

# --- Generate ---
cd "$ELVIS_ROOT"
python -m scripts.main --principle "$PRINCIPLE" --img_size 224 \
    --splits ft_train --num_samples "$TRAIN_SAMPLES" --seed "$TRAIN_SEED"
python -m scripts.main --principle "$PRINCIPLE" --img_size 224 \
    --splits ft_val --num_samples "$VAL_SAMPLES" --seed "$VAL_SEED"

# --- Check (both run even if the first fails) ---
cd "$FT_ROOT"
status=0
python scripts/check_ft_data.py --principle "$PRINCIPLE" --split ft_train --num-samples "$TRAIN_SAMPLES" || status=1
python scripts/check_ft_data.py --principle "$PRINCIPLE" --split ft_val --num-samples "$VAL_SAMPLES" || status=1
exit $status

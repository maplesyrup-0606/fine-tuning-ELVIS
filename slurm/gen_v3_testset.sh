#!/bin/bash
#SBATCH --account=rrg-lsigal
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --job-name=gen_v3_testset
#SBATCH --output=logs/gen_v3_testset_%j.out
#SBATCH --error=logs/gen_v3_testset_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mercurymcindoe@gmail.com

set -euo pipefail

# Evaluation set generated with the fixed ELVIS generators, next to ELVIS's own splits:
#   $ELVIS_DATA/<principle>/train_v3/   Stage 1 example images (3 per side, seed 3)
#   $ELVIS_DATA/<principle>/test_v3/    test images            (3 per side, seed 4)
# Same layout as ELVIS's train/test; evaluate with
#   sbatch --export=ALL,SPLIT_SUFFIX=v3[,RUBRICS=1] slurm/eval_elvis.sh <variant> <checkpoint>
# Both splits are regenerated on every run (main.py wipes a split before writing it).
#
# Usage:
#   sbatch slurm/gen_v3_testset.sh                      # all five principles
#   sbatch slurm/gen_v3_testset.sh proximity closure    # subset
# Time: ~15k images in total; minutes on a laptop CPU, 2h wallclock is a buffer.

PRINCIPLES=("$@")
if [ ${#PRINCIPLES[@]} -eq 0 ]; then
    PRINCIPLES=(continuity proximity similarity closure symmetry)
fi
SAMPLES=3   # ELVIS evaluation format: img_num=3 examples and 3 test images per side

# --- Environment ---
module load python/3.11.5 cuda/12.6 opencv/4.13.0 rdkit arrow  # same stack as the venv was built with

VENV=${VENV:-$SCRATCH/venv/elvis}
source "$VENV/bin/activate"

export ELVIS_ROOT=${ELVIS_ROOT:-$SCRATCH/ELVIS}
export ELVIS_DATA=${ELVIS_DATA:-$ELVIS_ROOT/data}
FT_ROOT=${FT_ROOT:-$SCRATCH/fine-tuning-ELVIS}

echo "ELVIS_DATA=$ELVIS_DATA"
echo "ELVIS commit: $(git -C "$ELVIS_ROOT" log --oneline -1)"
echo "fine-tuning-ELVIS commit: $(git -C "$FT_ROOT" log --oneline -1)"
echo "Principles: ${PRINCIPLES[*]}"

for P in "${PRINCIPLES[@]}"; do
    echo; echo "=================== $P ==================="
    cd "$ELVIS_ROOT"
    python -m scripts.main --principle "$P" --img_size 224 --splits train_v3 --num_samples "$SAMPLES" --seed 3
    python -m scripts.main --principle "$P" --img_size 224 --splits test_v3  --num_samples "$SAMPLES" --seed 4

    cd "$FT_ROOT"
    for SPLIT in train_v3 test_v3; do
        echo "--- label check $P/$SPLIT"
        python scripts/check_ft_data.py --principle "$P" --split "$SPLIT" --num-samples "$SAMPLES"
    done
done

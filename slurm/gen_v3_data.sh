#!/bin/bash
#SBATCH --account=def-lsigal
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --job-name=gen_v3_data
#SBATCH --output=logs/gen_v3_data_%j.out
#SBATCH --error=logs/gen_v3_data_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mercurymcindoe@gmail.com

set -euo pipefail

# v3 data: generate fine-tuning splits for each principle (fixed ELVIS generators),
# label-check them, build per-pattern rubrics, then build the v3 classification JSONLs.
#
# Usage:
#   sbatch slurm/gen_v3_data.sh                       # all five principles
#   sbatch slurm/gen_v3_data.sh proximity similarity  # subset (datasets still built for continuity + all)
#   sbatch --export=ALL,FORCE=1 slurm/gen_v3_data.sh  # regenerate splits that already exist
#
# Per principle: ft_train and ft_val, 6 per side (seed 1 / seed 2). Existing splits are
# reused (continuity's v2 ft_train has 12 per side; the builder takes the first 6).
# Outputs: $ELVIS_DATA/<principle>/{ft_train,ft_val}/, data/<principle>/rubrics.json,
#          data/v3/{ft_train,ft_val}_{continuity,all}.jsonl (+ .meta.json)
# Time: ~51k images to generate for the four new principles (~48 img/s on a laptop CPU,
# i.e. ~20 min there); 4h wallclock is a generous buffer for slower nodes.

PRINCIPLES=("$@")
if [ ${#PRINCIPLES[@]} -eq 0 ]; then
    PRINCIPLES=(continuity proximity similarity closure symmetry)
fi
FORCE=${FORCE:-0}
SAMPLES=6
TRAIN_SEED=1
VAL_SEED=2

# --- Environment ---
module load python/3.11.5 cuda/12.6 opencv/4.13.0 rdkit arrow  # same stack as the venv was built with

VENV=${VENV:-$SCRATCH/venv/elvis}
source "$VENV/bin/activate"

export ELVIS_ROOT=${ELVIS_ROOT:-$SCRATCH/ELVIS}
export ELVIS_DATA=${ELVIS_DATA:-$ELVIS_ROOT/data}
export HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}
FT_ROOT=${FT_ROOT:-$SCRATCH/fine-tuning-ELVIS}

echo "ELVIS_ROOT=$ELVIS_ROOT"
echo "ELVIS_DATA=$ELVIS_DATA"
echo "ELVIS commit: $(git -C "$ELVIS_ROOT" log --oneline -1)"
echo "fine-tuning-ELVIS commit: $(git -C "$FT_ROOT" log --oneline -1)"
echo "Principles: ${PRINCIPLES[*]}   FORCE=$FORCE"

# Images per side in a split (counted in its first pattern), for the label check.
per_side() {
    local dirs pngs
    dirs=("$ELVIS_DATA/$1/$2"/*/)
    pngs=("${dirs[0]}positive"/*.png)
    echo "${#pngs[@]}"
}

for P in "${PRINCIPLES[@]}"; do
    echo; echo "=================== $P ==================="

    # --- 1. Generate splits (skip if present, unless FORCE=1) ---
    cd "$ELVIS_ROOT"
    for SPLIT_SEED in "ft_train:$TRAIN_SEED" "ft_val:$VAL_SEED"; do
        SPLIT=${SPLIT_SEED%%:*}; SEED=${SPLIT_SEED##*:}
        if [ -d "$ELVIS_DATA/$P/$SPLIT" ] && [ "$FORCE" != "1" ]; then
            echo "--- $P/$SPLIT exists, reusing (FORCE=1 to regenerate)"
        else
            echo "--- generating $P/$SPLIT ($SAMPLES per side, seed $SEED)"
            python -m scripts.main --principle "$P" --img_size 224 \
                --splits "$SPLIT" --num_samples "$SAMPLES" --seed "$SEED"
        fi
    done

    # --- 2. Label check (stops the job on FAIL) ---
    cd "$FT_ROOT"
    for SPLIT in ft_train ft_val; do
        echo "--- label check $P/$SPLIT"
        python scripts/check_ft_data.py --principle "$P" --split "$SPLIT" \
            --num-samples "$(per_side "$P" "$SPLIT")"
    done

    # --- 3. Rubrics (pattern names are the same in every split) ---
    python scripts/generate_programmatic_rules.py --principle "$P" --split ft_train --rubrics
done

# --- 4. Build the v3 datasets ---
cd "$FT_ROOT"
echo; echo "=================== datasets ==================="
for SPLIT in ft_train ft_val; do
    python scripts/build_v3_dataset.py --split "$SPLIT" --principles continuity --name continuity
    python scripts/build_v3_dataset.py --split "$SPLIT" --principles all --name all
done
ls -la data/v3/

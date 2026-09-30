#!/bin/bash
#SBATCH --account=def-lsigal
#SBATCH --time=03:00:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --output=logs/eval_elvis_%x_%j.out
#SBATCH --error=logs/eval_elvis_%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mercurymcindoe@gmail.com

set -euo pipefail

# Usage:
#   sbatch -J <name> slurm/eval_elvis.sh <variant> <checkpoint_path> [<principle> ...]
#   sbatch -J base slurm/eval_elvis.sh base "" continuity proximity similarity closure symmetry
#
# <variant> ∈ {llm, projector, both, base}
# For variant=base, pass "" as checkpoint (evaluates un-fine-tuned model).
# Default principles: continuity proximity similarity closure symmetry.
VARIANT=${1:-}
CHECKPOINT=${2:-}
shift 2 || true
PRINCIPLES=("$@")
if [ ${#PRINCIPLES[@]} -eq 0 ]; then
    PRINCIPLES=(continuity proximity similarity closure symmetry)
fi

if [ -z "$VARIANT" ]; then
    echo "ERROR: variant required. Usage: sbatch -J <name> slurm/eval_elvis.sh <variant> <checkpoint_path> [<principles...>]"
    exit 1
fi
case "$VARIANT" in
    llm|projector|both) [ -n "$CHECKPOINT" ] || { echo "ERROR: checkpoint required for variant=$VARIANT"; exit 1; } ;;
    base) ;;  # no checkpoint needed
    *) echo "ERROR: variant must be one of: llm, projector, both, base (got: $VARIANT)"; exit 1 ;;
esac

# --- Environment ---
module load python/3.11.5 cuda/12.6 opencv/4.13.0 rdkit arrow

VENV=${VENV:-$SCRATCH/venv/elvis}
source "$VENV/bin/activate"

export ELVIS_ROOT=${ELVIS_ROOT:-$SCRATCH/ELVIS}
export ELVIS_DATA=${ELVIS_DATA:-$ELVIS_ROOT/data}
export HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}
export TRANSFORMERS_CACHE=$HF_HOME/transformers

RESULTS_DIR=${ELVIS_FINETUNING_RESULTS:-$SLURM_SUBMIT_DIR/results}
mkdir -p "$RESULTS_DIR/eval" logs

# --- Sanity ---
echo "=== Job $SLURM_JOB_ID on $(hostname) — variant=$VARIANT ==="
echo "Checkpoint: ${CHECKPOINT:-<base, no adapter>}"
echo "Principles: ${PRINCIPLES[*]}"
echo "ELVIS_ROOT=$ELVIS_ROOT"
echo "ELVIS_DATA=$ELVIS_DATA"
nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available())"
python -c "import peft, transformers; print('peft:', peft.__version__, '| transformers:', transformers.__version__)"

if [ -n "$CHECKPOINT" ] && [ ! -d "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint $CHECKPOINT is not a directory"
    exit 1
fi

# --- Run ---
cd "$SLURM_SUBMIT_DIR"

CKPT_ARG=""
if [ -n "$CHECKPOINT" ]; then
    CKPT_ARG="--checkpoint $CHECKPOINT"
fi

python scripts/eval_elvis.py \
    --variant "$VARIANT" \
    $CKPT_ARG \
    --principles "${PRINCIPLES[@]}" \
    --output-dir "$RESULTS_DIR/eval"

echo "=== Done. Summary: $RESULTS_DIR/eval/$VARIANT/summary.json ==="
cat "$RESULTS_DIR/eval/$VARIANT/summary.json" 2>/dev/null || true

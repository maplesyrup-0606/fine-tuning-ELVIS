#!/bin/bash
#SBATCH --job-name=seedbench2-baseline-qwen3vl4b
#SBATCH --account=def-lsigal
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/seedbench_base_%j.out
#SBATCH --error=logs/seedbench_base_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mercurymcindoe@gmail.com

set -euo pipefail

# --- Environment ---
module load StdEnv/2023 gcc opencv rdkit arrow

VENV=${VENV:-$SCRATCH/venv/elvis}
source "$VENV/bin/activate"

VLMEVALKIT_DIR=${VLMEVALKIT_DIR:-$SCRATCH/VLMEvalKit}

# Redirect HF cache to scratch (home quota is small on Alliance Canada)
export HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME"

RESULTS_DIR=${ELVIS_FINETUNING_RESULTS:-$SCRATCH/fine-tuning-ELVIS-results}
mkdir -p "$RESULTS_DIR"
mkdir -p logs

# --- Sanity ---
echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
python -c "from vlmeval.config import supported_VLM; assert 'Qwen3-VL-4B-Instruct' in supported_VLM, 'model key missing'; print('vlmeval OK')"

# --- Run ---
# Job is capped at 3h; VLMEvalKit resumes from per-item partial output on the
# next chained submission. If python exits nonzero (SIGTERM at time limit),
# we chain another iteration below.
cd "$VLMEVALKIT_DIR"
set +e
python run.py --data SEEDBench2 --model Qwen3-VL-4B-Instruct --verbose
RC=$?
set -e

# --- Publish (partial or complete) results ---
OUT_SRC="$VLMEVALKIT_DIR/outputs/Qwen3-VL-4B-Instruct"
DEST="$RESULTS_DIR/seedbench_base"
mkdir -p "$DEST"
if [ -d "$OUT_SRC" ]; then
    cp -u "$OUT_SRC"/*SEEDBench2* "$DEST/" 2>/dev/null || true
    echo "Copied any available SEEDBench2 outputs to $DEST"
    ls -la "$DEST" | grep -i seedbench || true
fi

# --- Chain to next iteration if not done ---
CHAIN_ITER=${CHAIN_ITER:-1}
CHAIN_MAX=${CHAIN_MAX:-10}

if [ $RC -eq 0 ]; then
    echo "=== Eval completed (iter $CHAIN_ITER). Chain done. ==="
    exit 0
fi

if [ $CHAIN_ITER -ge $CHAIN_MAX ]; then
    echo "=== Chain cap $CHAIN_MAX reached without completion (exit $RC). Investigate. ==="
    exit 1
fi

NEXT_ITER=$((CHAIN_ITER + 1))
echo "=== Iter $CHAIN_ITER exited $RC — submitting iter $NEXT_ITER (of max $CHAIN_MAX) ==="
cd "$SLURM_SUBMIT_DIR"
sbatch --dependency=afterany:$SLURM_JOB_ID \
       --export=ALL,CHAIN_ITER=$NEXT_ITER,CHAIN_MAX=$CHAIN_MAX \
       slurm/eval_seedbench_baseline.sh

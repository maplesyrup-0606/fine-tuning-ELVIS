#!/bin/bash
#SBATCH --job-name=seedbench2-baseline-qwen3vl4b
#SBATCH --account=def-lsigal
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/seedbench_base_%j.out
#SBATCH --error=logs/seedbench_base_%j.err

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
cd "$VLMEVALKIT_DIR"
python run.py --data SEEDBench2 --model Qwen3-VL-4B-Instruct --verbose

# --- Publish results to our results dir ---
OUT_SRC="$VLMEVALKIT_DIR/outputs/Qwen3-VL-4B-Instruct"
if [ -d "$OUT_SRC" ]; then
    cp "$OUT_SRC"/*SEEDBench2* "$RESULTS_DIR/" 2>/dev/null || true
    echo "Copied SEEDBench2 outputs to $RESULTS_DIR"
    ls -la "$RESULTS_DIR" | grep -i seedbench || true
else
    echo "WARNING: expected outputs dir not found at $OUT_SRC"
    exit 1
fi

echo "=== Done ==="

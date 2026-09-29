#!/bin/bash
#SBATCH --job-name=generate-rules-qwen3vl4b
#SBATCH --account=def-lsigal
#SBATCH --time=01:00:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --output=logs/generate_rules_%j.out
#SBATCH --error=logs/generate_rules_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mercurymcindoe@gmail.com

set -euo pipefail

# --- Environment (mirrors eval_seedbench_baseline.sh) ---
module load python/3.11.5 cuda/12.6 opencv/4.13.0 rdkit arrow

VENV=${VENV:-$SCRATCH/venv/elvis}
source "$VENV/bin/activate"

# ELVIS coupling — script imports scripts.baseline_models.qwen via sys.path
export ELVIS_ROOT=${ELVIS_ROOT:-$SCRATCH/ELVIS}
# ELVIS_DATA is inherited from the shell that submitted the job

mkdir -p logs data

# --- Sanity ---
echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi
echo "ELVIS_ROOT=$ELVIS_ROOT"
echo "ELVIS_DATA=$ELVIS_DATA"
python -c "import torch; print('cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"

# --- Run ---
# SMOKE=1 → generate rules for just 3 patterns (validates pipeline in <1 min).
# unset   → full 432-pattern run (~15-30 min).
cd "$SLURM_SUBMIT_DIR"

if [ "${SMOKE:-0}" = "1" ]; then
    echo "=== SMOKE mode: 3 patterns → data/rules_smoke.json ==="
    python scripts/generate_rules.py --patterns 000,001,002 --output data/rules_smoke.json
    echo "--- smoke output ---"
    cat data/rules_smoke.json
else
    echo "=== Full run: all patterns → data/rules.json ==="
    python scripts/generate_rules.py
    echo "--- output summary ---"
    python -c "import json; d=json.load(open('data/rules.json')); print(d['meta']); print('sample rule:', list(d['rules'].values())[0][:200])"
fi

echo "=== Done ==="

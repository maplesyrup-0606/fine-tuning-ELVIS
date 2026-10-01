#!/bin/bash
#SBATCH --account=def-lsigal
#SBATCH --time=03:00:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --output=logs/seedbench_%x_%j.out
#SBATCH --error=logs/seedbench_%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mercurymcindoe@gmail.com
#SBATCH --signal=B:USR1@120

set -euo pipefail

# Usage:
#   sbatch -J seedbench-llm-e2  slurm/eval_seedbench_variant.sh Qwen3-VL-4B-Instruct-llm-e2
#   sbatch -J seedbench-both-e2 slurm/eval_seedbench_variant.sh Qwen3-VL-4B-Instruct-both-e2
#
# <MODEL_NAME> must already be registered in $VLMEVALKIT_DIR/vlmeval/config.py
# (partial(vlm.Qwen3VLChat, model_path=...)).
MODEL_NAME=${1:?usage: sbatch -J <jobname> slurm/eval_seedbench_variant.sh <MODEL_NAME>}

# --- Environment (identical to baseline) ---
module load python/3.11.5 cuda/12.6 opencv/4.13.0 rdkit arrow

VENV=${VENV:-$SCRATCH/venv/elvis}
source "$VENV/bin/activate"

VLMEVALKIT_DIR=${VLMEVALKIT_DIR:-$SCRATCH/VLMEvalKit}

export HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME"

RESULTS_DIR=${ELVIS_FINETUNING_RESULTS:-$SLURM_SUBMIT_DIR/results}
mkdir -p "$RESULTS_DIR"
mkdir -p logs

# --- Sanity ---
echo "=== Job $SLURM_JOB_ID on $(hostname) — model=$MODEL_NAME ==="
nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
python -c "from vlmeval.config import supported_VLM; assert '$MODEL_NAME' in supported_VLM, 'model key $MODEL_NAME missing from supported_VLM'; print('vlmeval OK')"

# --- Run ---
# Same SIGUSR1 shutdown + --reuse chain pattern as baseline (apples-to-apples).
cd "$VLMEVALKIT_DIR"

handle_usr1() {
    echo "=== SIGUSR1 received ($(date)): wallclock in <120s, terminating python (PID $PYTHON_PID) ==="
    if [ -n "${PYTHON_PID:-}" ]; then
        kill -TERM $PYTHON_PID 2>/dev/null || true
        ( sleep 60 && kill -KILL $PYTHON_PID 2>/dev/null || true ) &
    fi
}
trap handle_usr1 USR1

set +e
python run.py --data SEEDBench2 --model "$MODEL_NAME" --reuse --verbose &
PYTHON_PID=$!
wait $PYTHON_PID
RC=$?
set -e

# --- Publish (partial or complete) results ---
OUT_SRC="$VLMEVALKIT_DIR/outputs/$MODEL_NAME"
DEST="$RESULTS_DIR/seedbench_$MODEL_NAME"
mkdir -p "$DEST"
if [ -d "$OUT_SRC" ]; then
    cp -u "$OUT_SRC"/*SEEDBench2* "$DEST/" 2>/dev/null || true
    echo "Copied any available SEEDBench2 outputs to $DEST"
    ls -la "$DEST" | grep -i seedbench || true
fi

# --- Chain to next iteration if not done ---
CHAIN_ITER=${CHAIN_ITER:-1}
CHAIN_MAX=${CHAIN_MAX:-15}

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
sbatch --job-name="$SLURM_JOB_NAME" \
       --dependency=afterany:$SLURM_JOB_ID \
       --export=ALL,CHAIN_ITER=$NEXT_ITER,CHAIN_MAX=$CHAIN_MAX \
       slurm/eval_seedbench_variant.sh "$MODEL_NAME"

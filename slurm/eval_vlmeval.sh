#!/bin/bash
#SBATCH --account=rrg-lsigal
#SBATCH --time=03:00:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --output=logs/eval_%x_%j.out
#SBATCH --error=logs/eval_%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mercurymcindoe@gmail.com
#SBATCH --signal=B:USR1@120

set -euo pipefail

# One VLMEvalKit model x dataset run, resubmitting itself until the score file exists.
# Usage:
#   sbatch -J sb2-t0p-cp   slurm/eval_vlmeval.sh Qwen3-VL-4B-Instruct-t0p-cp
#   sbatch -J mmstar-t0p-cp slurm/eval_vlmeval.sh Qwen3-VL-4B-Instruct-t0p-cp MMStar
#
# <MODEL_NAME> must already be registered in $VLMEVALKIT_DIR/vlmeval/config.py
# (partial(vlm.Qwen3VLChat, model_path=...)).
#   sbatch -J s3-t07-cp-s3  slurm/eval_vlmeval.sh Qwen3-VL-4B-Instruct-t07-cp-s3 "MMMU_DEV_VAL MMStar MMBench_DEV_EN_V11 AI2D_TEST"
#
# <DATASET> defaults to SEEDBench2; any VLMEvalKit dataset key works, or several (quoted, space-separated):
# one job then loads the model once and runs them all; it is done when every dataset has a score file.
# (MMMU_DEV_VAL, MMStar, MMBench_DEV_EN_V11, AI2D_TEST, VisFactor, OCRBench, ...).
# MCQ datasets use the GPT judge: the OpenAI key is read from $VLMEVALKIT_DIR/.env.
MODEL_NAME=${1:?usage: sbatch -J <jobname> slurm/eval_vlmeval.sh <MODEL_NAME> [DATASET]}
DATASET=${2:-SEEDBench2}

# --- Environment (identical to baseline) ---
module load python/3.11.5 cuda/12.6 opencv/4.13.0 rdkit arrow

VENV=${VENV:-$SCRATCH/venv/elvis}
source "$VENV/bin/activate"

VLMEVALKIT_DIR=${VLMEVALKIT_DIR:-$SCRATCH/VLMEvalKit}
# OpenAI key for the MCQ judge; without it VLMEvalKit silently falls back to exact matching
# ("Failed in Prefetch" -> unparsed answers counted wrong).
if [ -f "$VLMEVALKIT_DIR/.env" ]; then set -a; source "$VLMEVALKIT_DIR/.env"; set +a; fi
[ -n "${OPENAI_API_KEY:-}" ] || echo "!! WARNING: OPENAI_API_KEY not set - MCQ answers will be scored without the GPT judge"

export HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME"

RESULTS_DIR=${ELVIS_FINETUNING_RESULTS:-$SLURM_SUBMIT_DIR/results}
mkdir -p "$RESULTS_DIR"
mkdir -p logs

# --- Sanity ---
echo "=== Job $SLURM_JOB_ID on $(hostname) — model=$MODEL_NAME  dataset=$DATASET ==="
nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
python -c "from vlmeval.config import supported_VLM; assert '$MODEL_NAME' in supported_VLM, 'model key $MODEL_NAME missing from supported_VLM'; print('vlmeval model OK')"

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
# shellcheck disable=SC2086  # $DATASET may hold several dataset names
python run.py --data $DATASET --model "$MODEL_NAME" --reuse --verbose &
PYTHON_PID=$!
wait $PYTHON_PID
RC=$?
set -e

# --- Publish (partial or complete) results ---
OUT_SRC="$VLMEVALKIT_DIR/outputs/$MODEL_NAME"
MISSING=""
for D in $DATASET; do
    # Preserve legacy seedbench_<model> dir layout; other datasets land at <dataset>_<model>
    if [ "$D" = "SEEDBench2" ]; then
        DEST="$RESULTS_DIR/seedbench_$MODEL_NAME"
    else
        DEST="$RESULTS_DIR/${D}_$MODEL_NAME"
    fi
    mkdir -p "$DEST"
    if [ -d "$OUT_SRC" ]; then
        cp -u "$OUT_SRC"/*_${D}[._]* "$DEST/" 2>/dev/null || true
        echo "Copied any available $D outputs to $DEST"
        ls -la "$DEST" | grep -iE "${D}|.pkl|.xlsx|.csv" || true
    fi
    # VLMEvalKit exits 0 even when a dataset errors out (e.g. "Engine core initialization
    # failed"), so exit code alone is not "done": require each dataset's score file as well.
    [ -n "$(find -L "$OUT_SRC" -name "*_${D}_*acc*" 2>/dev/null | head -1)" ] || MISSING="$MISSING $D"
done

# --- Chain to next iteration if not done ---
CHAIN_ITER=${CHAIN_ITER:-1}
CHAIN_MAX=${CHAIN_MAX:-15}

if [ $RC -eq 0 ] && [ -z "$MISSING" ]; then
    echo "=== Eval completed (iter $CHAIN_ITER): score files for all of: $DATASET. Chain done. ==="
    exit 0
fi
if [ $RC -eq 0 ]; then
    echo "=== Exit 0 but no score file under $OUT_SRC for:$MISSING; run did not finish; chaining ==="
    RC=1
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
       slurm/eval_vlmeval.sh "$MODEL_NAME" "$DATASET"

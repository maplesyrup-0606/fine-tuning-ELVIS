#!/bin/bash
# Submit ELVIS evaluations of base + every v3 checkpoint on test_v3 (fixed generators).
# Run on the login node (it only calls sbatch):
#
#   bash slurm/launch_v3_evals.sh                  # rubric mode, all checkpoints  (13 jobs)
#   CKPTS=final bash slurm/launch_v3_evals.sh      # rubric mode, last checkpoint  (5 jobs)
#   MODE=standard bash slurm/launch_v3_evals.sh    # standard ELVIS: model writes its own rule
#   DRY=1 bash slurm/launch_v3_evals.sh            # print the sbatch commands, submit nothing
#
# rubric   -> results/eval_v3_rubric_on_test_v3/
# standard -> results/eval_v3_on_test_v3/ (v3), results/eval_v2_on_test_v3/ (base)

set -euo pipefail
cd "$(dirname "$0")/.."

MODE=${MODE:-rubric}
CKPTS=${CKPTS:-all}
DRY=${DRY:-0}
case "$MODE" in
    rubric)   RUB=1; TAG=rub ;;
    standard) RUB=0; TAG=std ;;
    *) echo "MODE must be rubric or standard"; exit 1 ;;
esac

n=0
submit() {   # $1 job name, $2 variant, $3 checkpoint ("" for base)
    local cmd=(sbatch --time=10:00:00 --export="ALL,RUBRICS=$RUB,SPLIT_SUFFIX=v3" -J "$1" slurm/eval_elvis.sh "$2" "$3")
    if [ "$DRY" = "1" ]; then echo "${cmd[*]}"; else "${cmd[@]}"; fi
    n=$((n + 1))
}

for RUN in continuity all; do
    for V in llm both; do
        ckpts=(results/checkpoints_v3/$RUN/$V/v*/checkpoint-*)
        if [ ! -d "${ckpts[0]}" ]; then echo "WARNING: no checkpoints for $RUN/$V"; continue; fi
        if [ "$CKPTS" = "final" ]; then
            last=${ckpts[0]}
            for c in "${ckpts[@]}"; do [ "${c##*-}" -gt "${last##*-}" ] && last=$c; done
            ckpts=("$last")
        fi
        for CKPT in "${ckpts[@]}"; do
            submit "$TAG-v3-$RUN-$V-${CKPT##*-}" "$V" "$CKPT"
        done
    done
done
submit "$TAG-v3-base" base ""

if [ "$DRY" = "1" ]; then echo "Would submit $n jobs"; else echo "Submitted $n jobs"; fi

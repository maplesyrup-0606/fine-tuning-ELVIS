# Fir Cluster Bootstrap

You develop this repo locally (Mac) but every model call runs on Fir. This file is the "what to run on Fir after `git clone`" checklist.

The experiment is three headline steps:

1. **Run SEED-Bench-2 on base Qwen3-VL-4B** — gives us the "before" number.
2. **Fine-tune Qwen3-VL-4B on ELVIS continuity** — LoRA rank 4, ms-swift.
3. **Run SEED-Bench-2 on the fine-tuned checkpoint** — gives us the "after" number. Compare.

Steps 0.a–0.c below are one-time setup + data prep that must happen before step 1.

---

## 0.a Prerequisites on Fir

Assumed already in place:
- ELVIS repo cloned at some path — call it `$ELVIS_ROOT`
- ELVIS dataset materialized at `$ELVIS_DATA` (same convention as ELVIS)
- Python 3.10+ env matching ELVIS's `requirements.txt`, CUDA 12.1
- `wandb login` already run

## 0.b Clone + env

```bash
cd /home/$USER
git clone <your-github-url>/fine-tuning-ELVIS.git
cd fine-tuning-ELVIS

# Add to ~/.bashrc (Slurm reads from environment)
export ELVIS_ROOT=/home/$USER/ELVIS                             # adjust
export ELVIS_DATA=/scratch/$USER/ELVIS_DATA                     # adjust
export ELVIS_FINETUNING_RESULTS=/scratch/$USER/fine-tuning-ELVIS-results
mkdir -p "$ELVIS_FINETUNING_RESULTS"

pip install -r requirements.txt
```

Sanity check the ELVIS coupling:
```bash
python -c "import sys, os; sys.path.insert(0, os.environ['ELVIS_ROOT']); from scripts.baseline_models.qwen import infer_logic_rules; print('OK')"
```

## 0.c One-time data prep (before step 1)

```bash
# Apply the small CLI change to ELVIS main.py (--num_samples, --splits). Task #7.
# See ELVIS-side commit instructions (TBD when Task #7 runs).

sbatch slurm/regenerate_data.sh    # regenerate ELVIS_DATA/continuity/train at 9+9
sbatch slurm/generate_rules.sh     # pre-generate Stage 1 rules with base model
python scripts/build_dataset.py    # cheap; build train.jsonl on login node
```

---

## Step 1 — SEED-Bench-2 baseline on base model

```bash
sbatch slurm/eval_seedbench_baseline.sh
```

Writes `results/seedbench_base.json`. This is the "before" number that step 3 compares against.

## Step 2 — Fine-tune

```bash
sbatch slurm/train.sh
```

Runs ms-swift SFT with the config in `config.yaml`. Per-epoch ELVIS-continuity eval runs inside the job. Early-stops when continuity accuracy hits 67% or regresses. Writes LoRA checkpoints to `results/checkpoints/` and picks the best one.

## Step 3 — SEED-Bench-2 on fine-tuned checkpoint

```bash
sbatch slurm/eval_seedbench_ft.sh
```

Writes `results/seedbench_ft.json`. Compare against `seedbench_base.json` — that delta is the answer to whether Gestalt fine-tuning transferred.

---

## Current status

- [x] Repo scaffolded
- [ ] ELVIS `main.py` CLI changes (Task #7)
- [ ] Data prep scripts (Task #3)
- [ ] Training script (Task #4)
- [ ] Evaluation scripts (Task #5)
- [ ] Slurm wrappers (Task #6)
- [ ] Baseline SEED-Bench-2 (Task #8, executes as step 1 above)

Scripts and Slurm wrappers referenced above are not yet written. This file will be updated with exact wallclock and GPU tier estimates once they exist.

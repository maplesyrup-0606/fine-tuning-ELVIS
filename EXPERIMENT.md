# Experiment Setup

Concrete spec of the fine-tuning run on Qwen3-VL-4B / ELVIS continuity.
Authoritative source is [`config.yaml`](config.yaml); this doc is the human
companion. For project-level context see [`README.md`](README.md).

## Goal

Fine-tune Qwen3-VL-4B on ELVIS continuity, then measure:

1. **In-distribution gain** on the continuity test split.
2. **Negative-transfer** on the four sibling principles
   (proximity, similarity, closure, symmetry) via ELVIS Mode 1.
3. **General-vision transfer** on full SEED-Bench-2 via VLMEvalKit.

## Model

| | Value |
|---|---|
| Base checkpoint | `Qwen/Qwen3-VL-4B-Instruct` |
| Precision | bfloat16 |
| Framework | ms-swift 4.5.3 |

Note: ms-swift must be ≥ 4.5.3. Earlier versions lack
`get_multimodal_target_regex`, which silently no-ops aligner LoRA when
`target_modules=['all-linear']` is combined with `freeze_aligner=false`.

## Data

### Source

- ELVIS continuity, train split, regenerated at the ratios below.
- Test split used only for evaluation — never seen during training.

### Regeneration ratios

| | Value |
|---|---|
| Patterns (train split) | 432 |
| Positive examples per pattern | 9 |
| Negative examples per pattern | 9 |
| Image size | 224 × 224 |

### Split used by the fine-tuning pipeline

For each pattern, the 9 positive and 9 negative examples are partitioned:

- **First 3 per side** → held out as Stage 1 few-shot examples (match the
  ELVIS Mode 1 eval protocol with `img_num=3`).
- **Remaining 6 per side** → fine-tuning classification samples.

Total fine-tuning samples:
```
432 patterns × 2 sides × 6 samples = 5,184
```

### Rule generation

Stage 1 rules are pre-generated **once** with the base (un-fine-tuned)
Qwen3-VL-4B. The rules are baked into `data/continuity/train.jsonl`. No
runtime rule cache; no per-epoch regeneration. Keeps rules identical to
what ELVIS Mode 1 eval would feed the base model, so the only moving
part during training is classification.

Rule cache: `data/continuity/rules.json`.

### Dataset format

`data/continuity/train.jsonl` — one JSON per line, prompt + classification
label, prompt template matches the eval verbatim.

## LoRA

### Shared hyperparameters (all three variants)

| | Value |
|---|---|
| Rank | 4 |
| Alpha | 8 |
| Dropout | 0.0 |
| Target modules | `all-linear` |

### Three-variant ablation

Each variant is a separate `swift sft` run with its own output dir and
WandB run. Selected via `python scripts/train.py --variant {llm|projector|both}`.

| Variant | `freeze_llm` | `freeze_vit` | `freeze_aligner` | Trainable params |
|---|---|---|---|---|
| `llm` | false | **true** | **true** | 8.26 M |
| `projector` | **true** | **true** | false | 0.24 M |
| `both` | false | **true** | false | 8.49 M |

Vision tower (`vit`) is always frozen — the hypothesis is about mid-level
Gestalt perception, not low-level visual features.

Arithmetic identity: `llm + projector ≈ both` (to the 4th decimal),
confirming ms-swift's `all-linear + freeze_*` routing correctly selects
the intended component via `get_multimodal_target_regex`.

Mechanism: ms-swift's `all-linear` target expands to every `nn.Linear` in
the model; the `freeze_*` flags then exclude components so LoRA is only
applied to non-frozen linears. See
[ms-swift PR #10153](https://github.com/modelscope/ms-swift/pull/10153)
(merged 2026-09-16) for the aligner-routing fix included in 4.5.3.

## Training loop

| | Value |
|---|---|
| Per-device batch size | 4 |
| Gradient accumulation | 8 |
| Effective batch size | 32 |
| Epochs | 3 |
| Steps per epoch | 162 |
| Total steps | 486 |
| Learning rate | 5.0 × 10⁻⁵ |
| LR scheduler | cosine |
| Warmup ratio | 0.03 (3%) |
| Seed | 42 |
| Gradient checkpointing | on (40 GB MIG OOM safety) |
| Checkpoint cadence | per epoch |
| `save_total_limit` | 3 |

### Early stopping

| | Value |
|---|---|
| Metric | continuity test accuracy |
| Absolute target | 0.67 (~7% absolute over 0.5976 base) |
| Regression patience | 1 (stop if an epoch's accuracy < previous) |

### Experiment tracking

| | Value |
|---|---|
| Logger | WandB |
| Project | `Qwen3-VL-4B-fine-tuning-continuity` |
| Run names | `continuity-{llm,projector,both}` |

## Evaluation protocol

### In-distribution (per epoch)

- ELVIS continuity test split
- Protocol: `scripts.baseline_models.qwen._run_qwen_baseline` (Mode 1), unchanged
- Few-shot: `img_num=3` — identical to published ELVIS baselines
- Metric: classification accuracy (`continuity_test_accuracy`)

### Negative-transfer (per best checkpoint per variant)

- Same `_run_qwen_baseline` protocol
- Principles: `[proximity, similarity, closure, symmetry]` (continuity is
  the in-distribution target)
- Scope: best checkpoint per variant, selected by continuity accuracy
- Runs via `scripts/eval_elvis.py` with `--principles all` — one call
  covers all 5 principles per checkpoint

### General-vision transfer (final, on winning variant)

- Benchmark: SEED-Bench-2 (full, all 27 dimensions)
- Harness: VLMEvalKit
- vLLM backend (same as the base-model baseline → apples-to-apples)
- Judge: GPT-4o-mini for answer extraction (requires `OPENAI_API_KEY`)
- LoRA merged into base weights (`scripts/merge_lora.py`) and registered
  in `vlmeval/config.py`; VLMEvalKit loads the merged dir as any other
  HF model

### Baseline reference values

| | Value | Source |
|---|---|---|
| Continuity Mode 1 (base) | 0.5976 | `ELVIS/results/continuity/baseline/20260630` |
| Sibling Mode 1 (base) | fresh run at `results/eval/base/<ts>/` | this experiment |
| SEED-Bench-2 (base) | pending OpenAI key for scoring | this experiment |

## Checkpoints

Per-variant checkpoint path layout:

```
$ELVIS_FINETUNING_RESULTS/
└── checkpoints/
    └── <variant>/
        └── v<N>-<YYYYMMDD-HHMMSS>/
            ├── checkpoint-162/   # epoch 1
            ├── checkpoint-324/   # epoch 2
            └── checkpoint-486/   # epoch 3
```

Each `checkpoint-N/` is a peft adapter dir (`adapter_config.json` +
`adapter_model.safetensors`, ~16 MB), not a standalone model.

## Infrastructure

| | Value |
|---|---|
| Cluster | Fir (Compute Canada / Alliance) |
| Training GPU | MIG 3g.40gb slice (H100) |
| SEED-Bench-2 GPU | MIG 3g.40gb slice (H100) |
| Chaining | SIGUSR1 shutdown + `--reuse` for SEED-Bench-2; auto-resume from latest `checkpoint-*` for training |
| Max chain iterations | 15 for SEED-Bench-2; 3 for training |

---

# v2 Plan: Ground-Truth Rules + Multi-Task Fine-Tuning (2026-10-12)

## Motivation

v1 trained only Stage 2 (classify) using Stage 1 rules distilled from base
Qwen3-VL-4B. Those rules are unreliable: 396/432 are truncated mid-sentence by
`infer_logic_rules`' `max_new_tokens=128`, only 66/432 reach a description of
the negatives, and none state the actual criteria (allowed shapes/colors,
irrelevant properties). At ELVIS eval, Stage 2 only sees the rule text — the
6 example images never reach it — so whatever Stage 1 omits, Stage 2 cannot
recover. v2 trains both stages against ground-truth rules derived from the
generator metadata.

## Decided

1. **Rules from metadata, not a model.** Built per episode from each image's
   saved JSON (`shape`, `color_name`, `size`, `group_id`; `group_id = -1` means
   scattered) plus the task name (generator, `rel`/`irrel` props, group count).
   Do NOT trust the JSON's `prop_*` / `group_num` fields: `main.py` derives them
   by substring match on the pattern name (`prop_shape` is true even when shape
   is irrel; `group_num` is the last name token).
2. **Fixed, principle-neutral rule schema** (our fine-tuning target; ELVIS
   never enforces or parses it): `Principle / Grouping / Relevant / Irrelevant /
   Image 4–6`.
3. **Multi-task fine-tuning, one `swift sft` run per variant.** One JSONL mixes
   - Stage 1 rows: 6 example images + ELVIS `infer_logic_rules` prompt → GT rule
   - Stage 2 rows: 1 image + ELVIS `evaluate_llm` prompt with GT rule → label
   Trained with teacher forcing (Stage 2 always gets the GT rule, never a
   model-generated one). Chained only at validation / eval.
4. **Separate fine-tuning data folders.** ELVIS `train`/`test` are the eval set
   (Mode 1 uses `train/<p>/{pos,neg}/00000–00002` as Stage 1 examples and
   `test/<p>/...` as queries) and stay untouched — avoids leakage and keeps all
   prior numbers comparable.
5. **Episode layout.** Episode = 3+3 examples + 3+3 queries (6 per side,
   12 images) → 1 Stage 1 row + 6 Stage 2 rows.
   - Fine-tuning train: 12 per side (2 episodes / pattern) → 864 Stage 1 rows,
     5,184 Stage 2 rows (same Stage 2 count as v1)
   - Fine-tuning val: 6 per side (1 episode / pattern), same shape as one eval
6. **Validation, no LLM judge.**
   - Classification accuracy with the model's own rule (chained, = ELVIS metric)
   - Classification accuracy with the GT rule (Stage 2 in isolation)
   - The gap between the two = rule quality
   - Heading-by-heading rule grading against the answer key (fine-tuned models only)
7. **Prompt paraphrases** in training (ELVIS verbatim prompt is one of them),
   plus one held-out paraphrase for a robustness eval.
8. **Raise the Stage 1 token cap at eval** for every model, base included
   (base re-run gives the new reference number).

## ms-swift 4.5.3 loss weighting (verified from source)

- Per-row weight: `"loss_scale": <float>` on the assistant message
  (`swift/loss_scale/base.py`, `_inner_call`).
- **Silently ignored by default**: the default loss-scale class is binary, so
  `_encode_context_list` (`swift/template/base.py:1166-1187`) drops the weights
  to a 0/1 mask. Requires `--is_binary_loss_scale false`
  (`swift/arguments/base_args/template_args.py:142`).
- Loss = Σ(token_loss × weight) / (unweighted count of label tokens)
  (`swift/trainers/seq2seq_trainer.py:178-214`). Weights act per token, and
  weights < 1 also shrink the effective LR, so normalize to mean weight 1 over
  answer tokens.
- For an equal split: w₁·T₁ = w₂·T₂ with mean token weight 1, where Tᵢ = total
  answer tokens per objective, counted with the tokenizer at build time.

## Open decisions

- **A.** Objective balance: 50/50 (recommended) vs classification-primary
- **B.** Number of prompt paraphrases; new Stage 1 token cap value
- **C.** Extra example-set reshuffling within the train pool (default: skip)

## Task list

**Phase 0 — housekeeping**
- [ ] 0.1 Commit v1 results (EXPERIMENT.md, summary_matrix.csv, plots, scripts)
- [ ] 0.2 (optional) Held-out-paraphrase eval on v1 checkpoints — is the +23 pts prompt-specific?

**Phase 1 — data generation**
- [ ] 1.1 ELVIS `main.py`: allow fine-tuning split names, add `--seed`
- [ ] 1.2 Generate fine-tuning train (12/side) + val (6/side) on Fir
- [ ] 1.3 Sanity check: every label consistent with its metadata

**Phase 2 — rules**
- [ ] 2.1 Finalize rule schema (2–3 filled examples for review)
- [ ] 2.2 Rule builder: per episode, from metadata → rules JSON
- [ ] 2.3 Write prompt paraphrases (one held out)

**Phase 3 — training data**
- [ ] 3.1 `build_dataset.py`: Stage 1 + Stage 2 rows, `loss_scale` weights, train + val files
- [ ] 3.2 Smoke-test `loss_scale` on Fir (encode rows, one-step dry run)
- [ ] 3.3 `train.py`: `--is_binary_loss_scale false`

**Phase 4 — eval tooling** (parallel with 2–3)
- [ ] 4.1 Validation script: chained runs on val → own-rule acc, GT-rule acc, rule grading
- [ ] 4.2 Rule-grading parser (heading-based)
- [ ] 4.3 ELVIS `qwen.py`: Stage 1 token cap as a parameter
- [ ] 4.4 Re-run base with the new cap

**Phase 5 — run**
- [ ] 5.1 Train 3 variants
- [ ] 5.2 Validate each epoch checkpoint → best per variant
- [ ] 5.3 ELVIS 5-principle eval + held-out-paraphrase eval on best checkpoints
- [ ] 5.4 Compare against v1 (`summary_matrix.csv`)

Dependencies: 1 → 2 → 3 → 5; 4 parallel with 2–3. A blocks 3.1; B blocks 2.3, 4.4.

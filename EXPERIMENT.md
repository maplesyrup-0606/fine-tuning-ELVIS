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

## Rule schema (decided 2026-10-12)

One rule per (pattern, episode). Example (`351_..._non_intersected_n_splines_rel_color_size_1_s_irrel_shape`):

```
Principle: continuity
Grouping: objects are evenly spaced along one smooth curve
Relevant: color (only green and yellow); size (all objects the same size)
Irrelevant: shape (same for every object in an image)
Image 4: size (objects differ in size)
Image 5: scattered (objects do not follow the lines); color (has colors other than green and yellow)
Image 6: size (objects differ in size)
```

- `Image 4–6` = the episode's 3 example negatives, in the order ELVIS feeds them
  (positives first, then negatives, sorted by filename).
- Violation categories: `scattered`, `shape`, `color`, `size`. Shape evidence names
  the off shapes (up to 3); color evidence is generic — matplotlib color names are
  not reliably nameable and grading never uses evidence.
- Irrelevant lists `irrel` props ("same for every object in an image") and props in
  neither list ("varies freely").
- No constant `Images 1-3` line.
- **Stage 2 rows use the full rule** (including `Image 4–6`): at eval, ELVIS pastes
  the model's entire Stage 1 output into the Stage 2 prompt.
- Example negatives are NOT curated for diversity (single-rel patterns can have
  all 3 scattered): eval episodes have the natural mix, so training matches it.
- Checked: color-only negatives are not perceptually near-green/yellow — 0/388 in
  ft_val have all off-colors within RGB distance 150 of green/yellow; median
  100% of objects recolored.

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

- **A.** ~~Objective balance~~ **decided: 50/50** (2026-10-12). Without weights,
  classification gets 5.8% of the signal (83,486 rule-text tokens vs 5,184
  classification tokens in ft_train) — far below v1's 100%, which would confound
  any v2-vs-v1 difference. Weights (exact, ft_train, mean weight 1 over
  answer-text tokens): `w_rule = 0.531`, `w_classify = 8.552`. Balance holds
  per epoch, not per step (rule rows per 32-row step ~ 4.6 ± 2).
- **B.** ~~Number of prompt paraphrases~~ (decided: 5 train incl. ELVIS verbatim + 1 held out); ~~Stage 1 token cap~~ (decided: 1024, all models incl. base)
- **C.** Extra example-set reshuffling within the train pool (default: skip)

## Task list

**Phase 0 — housekeeping**
- [x] 0.1 Commit v1 results (EXPERIMENT.md, summary_matrix.csv, plots, scripts)
- [ ] 0.2 (optional) Held-out-paraphrase eval on v1 checkpoints — is the +23 pts prompt-specific?

**Phase 1 — data generation**
- [x] 1.1 ELVIS `main.py`: allow fine-tuning split names, add `--seed` (ELVIS 37e6cc4)
- [x] 1.2 Generate fine-tuning train (12/side, seed 1) + val (6/side, seed 2) on Fir — `slurm/gen_ft_data.sh`
- [x] 1.3 Sanity check: every label consistent with its metadata — `scripts/check_ft_data.py`
  Both splits PASS: 0 hard errors, 0 accidental positives. Negatives break: layout ~51%,
  size/shape/color ~20% each. → ~half the negatives are separable by layout alone; report
  val/eval accuracy split by layout-broken vs property-broken negatives (add to 4.1).

**Phase 2 — rules**
- [x] 2.1 Finalize rule schema — see "Rule schema" above
- [ ] 2.2 Rule builder: per episode, from metadata → rules JSON
- [x] 2.3 Write prompt paraphrases (one held out) — `scripts/prompts.py`: `elvis` (verbatim, checked against qwen.py) + p1–p4 for training, `heldout` for robustness eval; one variant sampled per row

**Phase 3 — training data**
- [x] 3.1 `build_dataset.py`: Stage 1 + Stage 2 rows, `loss_scale` weights, train + val files
  (tested locally on seed-identical data): train_v2 6,048 rows (864 S1 / 5,184 S2),
  val_v2 3,024 rows (432 / 2,592, ELVIS wording only); weights 0.531 / 8.552;
  wording mix ~20% per variant; 60 sampled rows encoded through the ms-swift
  Qwen3-VL template: tag/image counts match, all answer tokens carry their
  loss_scale, sequence length S1 437–470, S2 168–209 tokens.
- [x] 3.2 Smoke-test `loss_scale` — `scripts/check_loss_scale.py` PASS locally (ms-swift 4.5.3):
  answer-text tokens carry the row weight; the template suffix (`<|im_end|>` + `\n`,
  2 tokens/row) is trained at fixed weight 1; without `--is_binary_loss_scale false`
  weights are dropped (`loss_scale: None`). Gradient clipping: ms-swift does not override `max_grad_norm` (no mention in `swift/arguments/`), so the transformers default 1.0 applies.
- [x] 3.3 `train.py`: `--is_binary_loss_scale false` + `--val_dataset` (val loss per epoch).
  Verified with ms-swift's own parser (`parse_args(SftArguments, ...)`): no unrecognized
  args; is_binary_loss_scale=False, eval_strategy=epoch, split_dataset_ratio=0.0,
  max_grad_norm=1.0. Steps: 6,048 rows / 32 = 189 per epoch, 567 total.

**Phase 4 — eval tooling** (parallel with 2–3)
- [ ] 4.1 Validation script: chained runs on val → own-rule acc, GT-rule acc, rule grading
- [ ] 4.2 Rule-grading parser (heading-based)
- [x] 4.3 ELVIS `qwen.py`: Stage 1 token cap as a parameter — `infer_logic_rules(max_new_tokens=128)`,
  `_run_qwen_baseline(rule_max_new_tokens=128)`; ELVIS defaults unchanged. v2 eval passes
  `evaluation.rule_max_new_tokens: 1024` (config). v2 eval results go to
  `results/eval_v2/` (`evaluation.results_subdir`; eval_elvis.py, eval_elvis.sh,
  aggregate_results.py all read it) so v1 and v2 rows never mix.
- [ ] 4.4 Re-run base with the new cap

**Phase 5 — run**
- [ ] 5.1 Train 3 variants
- [ ] 5.2 Validate each epoch checkpoint → best per variant
- [ ] 5.3 ELVIS 5-principle eval + held-out-paraphrase eval on best checkpoints
- [ ] 5.4 Compare against v1 (`summary_matrix.csv`)

## v2 ELVIS results (2026-10-13)

Base re-run with `rule_max_new_tokens=1024`: continuity 60.92, proximity 55.19,
similarity 57.04, closure 69.60, symmetry 61.27. Untruncated rules alone lift the base
model a lot on closure (+14.3 vs v1's 128-cap base) and similarity (+4.1), so compare
each version against its own base. (v1 fine-tuned checkpoints were also scored at 128.)

| v2 `both` | continuity | proximity | similarity | closure | symmetry | mean |
|---|---|---|---|---|---|---|
| e1 (189) | 88.27 | 60.63 | 59.00 | 82.02 | 62.42 | 70.47 |
| e2 (378) | 90.39 | 60.88 | 60.90 | 78.61 | 62.42 | 70.64 |
| e3 (567) | **93.21** | 62.98 | 61.27 | 79.47 | 61.34 | **71.65** |

- Continuity: v2 `both` e3 93.21 vs v1 best 82.83; still rising at e3 although
  eval_loss bottomed at e2.
- Siblings: v2 gains over its own base are smaller than v1's (similarity +4.2 vs
  +10.4, closure +9.9 vs +30.0). Likely cause: the rule schema is continuity-specific,
  so the model writes continuity-style rules for other principles (to verify with
  `show_eval_episode.py --principle closure`).
- `projector` never learned the rule format (eval_token_acc ~0.59) → no v2 benefit.
- Closure for `both` peaks at e1 (82.0) then ~79: specialization, not overfitting
  (continuity held-out accuracy keeps rising).

**Checkpoint selection criterion (decided):** best mean over the 5 principles, with no
principle below its own base. Winner: `both` e3 (`checkpoint-567`) → SEED-Bench-2.

Optional analysis (not blocking): grade all eval rules against GT per slot; from 5
samples, single-relevant-property rules were right, the one 2-property pattern dropped
`size` and listed one violation per image.

Dependencies: 1 → 2 → 3 → 5; 4 parallel with 2–3. A blocks 3.1; B blocks 2.3, 4.4.

---

# v3 Plan: Classification-only with Ground-Truth Rubrics (2026-10-19)

## Motivation
v2's multi-task setup coupled two different skills (rule writing vs. rule-following
classification) in one LoRA. v3 drops rule writing and trains only Stage 2 classification
with programmatic ground-truth rubrics.

## Decided
1. **Train only binary classification:** 1 query image + GT rubric -> positive/negative
   (same row format as v2's Stage 2 rows; no Stage 1 rows, no loss_scale balancing needed).
2. **Rubric = pattern-level rule only:** `Principle / Grouping / Relevant / Irrelevant`.
   No `Image 4-6` lines (they describe example negatives the classifier never sees).
   One rubric per pattern, identical at train and eval.
3. **Two runs:** (a) continuity only — comparable to v1/v2, transfer to the other 4
   principles still measurable; (b) all 5 principles.
4. **Variants:** `llm`, `both` (`projector` dropped — it could not learn in v2).

## Notes / TODO
- **Eval:** customize ELVIS eval to use the pre-generated rubric instead of a
  model-written rule (Stage 1 skipped). Also keep standard Mode 1 for comparability. (Deferred.)
- **Other principles:** analyze proximity / similarity / closure / symmetry generators
  and define a rubric schema per principle before generating their data.

## v3 generator survey decisions (2026-10-19)

Guiding rule: a label must be visible in the image — the model reads pixels, not metadata.

- **Rubric code restructured:** `scripts/rubric_specs.py` (per-principle specs); continuity
  output verified byte-identical to v2's. `generate_programmatic_rules.py --rubrics` writes
  one rubric per pattern (`data/<principle>/rubrics.json`).
- **ELVIS generator fixes (always on, ELVIS repo, uncommitted):**
  - proximity `red_triangle`: size was forced constant, so size-relevant negatives with
    intact clusters were identical to positives (116/216 in a sample -> 0 after fix).
  - similarity `palette`: "count" was never applied; count-broken negatives now have
    uneven wedge counts (0/252 -> 252/252 negatives uneven; positives stay even).
  - Data already generated on Fir (incl. ELVIS train/test) is unaffected; regenerating
    from this repo now yields fixed data.
- **No fix, rubric adapts:** proximity `circle_features` never applies irrelevant props
  -> rubric says "varies freely"; `fixed_props` holds them per cluster.
- **closure:** no fix. Exclude the 20 shape-relevant `feature_circle` patterns from
  training (the illusory-circle notch makes same-shape pentagons look mixed). The
  blue/red coin-flip cue in 8 color-free `feature_circle` patterns is left as ELVIS has it.
- **Keep ELVIS's value lists** (e.g. closure color = only blue/red); the rubric states them.
- **Eval (later):** report full ELVIS scores plus a "valid subset" excluding patterns
  with known-bad labels; ELVIS `test` stays untouched.
- **symmetry:** bilateral axis-broken negatives now never reuse the positives' axis
  (was 12/72 identical-looking negatives -> 0). Exclude the 50 axis-relevant
  `rotational_symmetry_pattern` patterns from training ("axis" is never applied and has
  no defined meaning for rotational symmetry).
- **closure `feature_*`:** size-broken negatives now differ by >= 1.5x (new
  `data_utils.get_distinct_random_sizes`; was 3/48 near-equal -> 0); `feature_square`
  negatives can now break the illusory square (scattered pieces), as `feature_triangle`
  already did.
- **ELVIS files changed (uncommitted):** proximity/util_red_triangle.py,
  similarity/util_palette.py, symmetry/util_symmetry_bilateral.py,
  symmetry/util_solar_system.py, utils/data_utils.py, closure/util_feature_{square,
  triangle,circle}.py. Before/after images: fine-tuning-ELVIS/results/survey/.
- **Training exclusions:** closure `feature_circle` shape-relevant (20), symmetry
  rotational axis-relevant (50).

## v3 rubric specs (2026-10-19) — all five principles

`scripts/rubric_specs.py`: one PrincipleSpec per principle, one GeneratorSpec per ELVIS
generator (grouping/relevant wording, irrelevant scope, violation checks, optional object
filter and training exclusion). Checked on seed-2 ft_val samples (3/side) from the fixed
ELVIS generators; earlier principles re-verified unchanged after each addition.

| Principle | Patterns | Distinct rubrics | Label check | Train exclusions |
|---|---|---|---|---|
| continuity | 432 | 132 | PASS, 0 accidental positives | 0 |
| proximity | 588 | 147 | PASS; grouping-only negatives verified scattered geometrically | 0 |
| similarity | 720 | 162 | PASS, 0 accidental positives | 0 |
| closure | 596 | 212 | PASS; grouping-only negatives verified geometrically | 20 (feature_circle shape) |
| symmetry | 216 | 70 | PASS; 54 accidental positives, all in excluded patterns | 50 (rotational axis) |

Wording decisions: red_triangle `exist` = "at least one cluster" (45% of exist positives
have the feature in every cluster); similarity groups worded as "one X, different from the
other groups" (no value names); palette count omitted unless relevant; separate_big_*
"evenly spaced along the outline(s)"; symmetry describes positions only ("pairs at mirror
positions", "objects at the same distances along each arm") since shapes are not mirrored.
Samples with rubrics: results/survey/rubric_*.png.

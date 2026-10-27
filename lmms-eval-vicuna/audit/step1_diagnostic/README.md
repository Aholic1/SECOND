# Step 1 diagnostic cohort

This directory contains 100 fixed-seed, stratified POPE Popular cases for
qualitative visualization and later causal intervention. It is a diagnostic
cohort, not a statistically representative sample.

Use the complete 3000-sample cohort in
`audit/step1_full/pope_pop_3000_baseline/` for metrics and population-level
claims. Retain `audit/manifests/pope_pop_seed42_80.json` only for fast smoke and
regression checks.

## Composition

| Visual stratum | Count | Role |
| --- | ---: | --- |
| `TP_simple` | 20 | Correct positive, simple existence case |
| `TP_relation` | 10 | Correct positive, relation heuristic matched |
| `TN_correct` | 20 | Correct rejection of an absent object |
| `FN_simple` | 20 | Missed positive, simple existence case |
| `FN_relation` | 15 | Missed positive, relation heuristic matched |
| `FP_hallucination` | 15 | Incorrect claim that an absent object exists |

The total is 100. Selection uses seed 42 and includes only readable images.
`diagnostic_manifest.json` stores the selected records and
`diagnostic_summary.json` stores the aggregate checks.

The `has_relation` feature is a keyword heuristic for balancing inspection
cases; it is not a semantic ground-truth annotation and must not be reported as
one.

## Analysis policy

Compare outcomes within each ground-truth class:

- `TP` versus `FN` asks what SECOND retained or missed when the object exists.
- `TN` versus `FP` asks what SECOND used when the object is absent and the model
  either rejects it correctly or hallucinates it.

Do not use a pooled correct-versus-incorrect comparison as the primary analysis,
because it mixes positive and negative questions.

The 100 cases are appropriate for:

- original-image overlays of selected and discarded token regions;
- attention and selection-score heatmaps;
- case pairing within the two comparisons above;
- sufficiency, comprehensiveness, deletion, and insertion interventions.

They are not appropriate for estimating accuracy, error prevalence, IoU means,
or confidence intervals for the full POPE population.

## Rebuild

From `lmms-eval-vicuna`:

```bash
python3 tools/second_audit.py select_diagnostic \
  --features audit/step1_full/pope_pop_3000_baseline/features.jsonl \
  --strata-config audit/step1_diagnostic/strata_config.json \
  --output audit/step1_diagnostic/diagnostic_manifest.json \
  --seed 42
```

Run `python3 audit/validate_step1.py` after rebuilding. The validator checks the
sample count, exact stratum counts, unique question IDs, and image readability.

Files ending in `_enhanced` belong to Step 1.5 and are not part of the frozen
Step 1 diagnostic cohort.

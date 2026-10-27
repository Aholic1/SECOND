# SECOND visual-token audit: Step 1

## Status and scope

Step 1 freezes the POPE Popular baseline used to analyze SECOND's visual-token
selection. The quantitative cohort is the complete set of 3000 samples. No
population-level result may be estimated from a small sample.

Two smaller collections have narrower roles:

- `audit/step1_diagnostic/diagnostic_manifest.json` contains 100 fixed-seed,
  stratified cases for attention/token-mask visualization and causal tests.
- `audit/manifests/pope_pop_seed42_80.json` is retained for smoke and regression
  checks only. It is not deprecated, but it must not be used for quantitative
  conclusions.

Correct and incorrect cases must be compared within the same ground-truth class:
`TP` versus `FN` for positive questions, and `TN` versus `FP` for negative
questions. Pooling all correct cases against all errors confounds the answer
class with correctness.

## Frozen inputs

- Prediction log:
  `logs/liuhaotian__llava-v1.6-vicuna-7b/20260810_001849_samples_pope_pop.jsonl`
- Evaluation results:
  `logs/liuhaotian__llava-v1.6-vicuna-7b/20260810_001849_results.json`
- Prediction log SHA256:
  `0c8a71ef7bda1048dbc4649e0207eeb241a42492458a7ce10080ef5a0d5a5cc3`
- Results SHA256:
  `0e9f8866ec5efae81dfb34b02717ca96db0d95d619c59e45fe09b71c36c8ad77`
- Historical prediction commit:
  `4ad65872d9c03ea7b60ea68c2b663d22a373ec33`
- Audit-artifact base commit:
  `a28e0a5ef17868f3c3058f382d486a10b3c47688`

These commits describe different events. The prediction commit is reported by
the original results file. The audit commit is the repository snapshot used to
derive the Step 1 artifacts. The later commit that adds the frozen artifacts is
recorded by Git history rather than embedded into itself.

## Dataset and run configuration

- Model: `liuhaotian/llava-v1.6-vicuna-7b`
- Task: `pope_pop`
- Dataset: `lmms-lab/POPE`, configuration `Full`, split `popular`
- Task hash: `b5aa871bc6960e8d3d209d20d5308b27d69ae7ffd5da5d7db6804a6c1569e3a6`
- Effective samples: 3000
- Batch size: 1 on two processes / GPUs 0 and 1
- SECOND mode: `recursion`, grid `2x2`, stages `-2 -1 0 1`
- Selection: `attn_topk`, threshold `1.0`
- Positional interpolation: `bilinear_interpolation`
- Contrastive alphas: `0.8 0.8 0.8`

The complete machine-readable record is
`audit/step1_full/pope_pop_3000_baseline/baseline_provenance.json`.
`environment.txt` records the environment used to build and validate the audit
artifacts. It is not evidence of the historical prediction environment.

## Baseline results

| Measure | Value |
| --- | ---: |
| Samples | 3000 |
| TP | 1270 |
| TN | 1420 |
| FN | 230 |
| FP | 80 |
| Accuracy | 0.8966666667 |
| Precision | 0.9407407407 |
| Recall | 0.8466666667 |
| F1 | 0.8912280702 |

The primary failure comparisons for later steps are therefore `TP` versus `FN`
and `TN` versus `FP`.

## Implementation

`tools/second_audit.py` provides five Step 1 commands:

- `index`: normalize every prediction and assign `TP/TN/FN/FP`.
- `summarize`: calculate full-cohort counts and metrics.
- `map_images`: map questions to COCO files and record dimensions/checksums.
- `extract_features`: derive lightweight fields used only for stratification.
- `select_diagnostic`: select the 100 fixed-seed diagnostic cases.

`audit/step1_run.sh` rebuilds these artifacts and writes their provenance. It
does not capture the dirty worktree as a patch, because unrelated Step 2 changes
must not become part of the Step 1 record.

## Outputs

The full-cohort directory contains:

- `baseline_predictions.jsonl`: immutable copy of all 3000 predictions.
- `full_index.jsonl`: normalized predictions and outcome strata.
- `image_mapping.jsonl`: COCO paths, dimensions, readability, and checksums.
- `features.jsonl`: lightweight diagnostic-selection features.
- `summary.json`: full-cohort metrics.
- `baseline_sha256.txt`: SHA256 values for the prediction and results inputs.
- `baseline_provenance.json`: structured model, task, input, commit, and run data.
- `commit.txt`: audit-artifact base commit.
- `environment.txt`: audit/processing environment snapshot.

The diagnostic directory contains the stratification configuration, the
100-sample manifest, its summary, and documentation. Files with `enhanced` in
their name belong to Step 1.5 and are outside this Step 1 freeze.

## Rebuild and validation

From `lmms-eval-vicuna`:

```bash
bash audit/step1_run.sh
python3 audit/validate_step1.py
python3 audit/test_step1.py
```

The enhanced validator checks row counts, unique and aligned question IDs,
outcome counts, correctness labels, summary consistency, all 3000 image files
and dimensions, input hashes, diagnostic composition, and provenance.

## Completion checklist

- [x] `baseline_predictions.jsonl` contains 3000 rows.
- [x] `full_index.jsonl` contains 3000 unique question IDs.
- [x] Index, image mapping, and feature ID sets are identical.
- [x] Counts are TP=1270, TN=1420, FN=230, FP=80.
- [x] All 3000 mapped images exist, are readable, and have valid dimensions.
- [x] `features.jsonl` contains 3000 rows.
- [x] The diagnostic manifest contains 100 unique samples in all six strata.
- [x] Prediction and results SHA256 values are recorded.
- [x] Prediction and audit commits have distinct, documented meanings.
- [x] The audit environment is labeled separately from the prediction run.
- [x] Machine-readable provenance is present and validated.

Step 2 may begin only from this frozen baseline. Step 2 instrumentation,
attention dumps, causal matrices, and experiment logs are intentionally excluded
from the Step 1 commit.

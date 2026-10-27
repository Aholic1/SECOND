# SECOND Token Selection Audit v1 Design

## Objective

Determine whether SECOND's visual-token selection causes prediction changes on
POPE Popular under the current contrastive-decoding configuration. The first
phase is a mechanism-screening experiment, not a population-level benchmark or
a new selection method.

The audit must preserve the original Full SECOND computation while exposing the
exact tensors and decisions used by the selector.

## Scope

The first phase covers 80 POPE Popular samples selected from the existing Full
SECOND run:

- 30 false negatives (FN)
- 10 false positives (FP)
- 20 true positives (TP)
- 20 true negatives (TN)

Samples are drawn randomly within each Full SECOND outcome stratum with a fixed
seed. Attributes such as small target, multiple objects, occlusion, and edge
location are annotated only after sampling. Curated qualitative examples are
kept separate from the inferential audit set.

COCO bounding-box projection, full-dataset effect estimates, a new token
selector, and the complete Selection x Contrastive-Decoding factorial experiment
are out of scope for v1.

## Experimental Conditions

Every condition uses the same model weights, prompt, sample manifest, image
preprocessing where applicable, greedy decoding, and generation parameters.

| ID | Condition | Recursive multiscale | Selection | Contrastive decoding |
| --- | --- | --- | --- | --- |
| A | Standard LLaVA | No | No | No |
| B | Recursive-AllToken-Raw | Yes | No | No |
| C | Recursive-AllToken-CD | Yes | No | Yes |
| D | Full SECOND | Yes | Yes | Yes |

The comparisons have deliberately limited interpretations:

- A to B measures the combined effect of entering SECOND's recursive multiscale
  pathway. It must not be described as a pure contrastive-decoding effect.
- B to C measures contrastive decoding when all visual tokens are retained.
- C to D measures token selection under the current contrastive-decoding
  configuration.

`C correct, D wrong` is Selection Harm under the current CD configuration.
`C wrong, D correct` is Selection Benefit. `B correct, C wrong` is CD Harm in
the AllToken condition, not Selection Harm.

The design does not claim a complete causal decomposition. If v1 finds a stable
C-to-D effect, phase 2 adds condition E, Recursive-Selection-Raw, to form a 2x2
Selection x CD experiment, with Standard LLaVA retained as an external reference.

## Intervention Semantics

### Raw decoding

Raw decoding takes the final recursive stage's logits immediately before
SECOND's contrastive combination and applies greedy argmax directly. It must
explicitly bypass both the stage-logit difference combination and candidate
truncation. Setting `contrastive_alphas` to zero is not an acceptable
implementation of Raw decoding.

### AllToken

AllToken leaves the diagnostic computation intact:

1. Compute the exact `ret_attn` returned by `get_heatmap`.
2. Compute SECOND's existing entropy and requested keep ratio.
3. Compute the attention score cutoff and the selector's proposed mask.
4. Log all proposed-selection state.
5. Replace only the mask applied to the next stage with an all-one mask.

This makes C and D differ only in whether the proposed token-selection mask is
applied. The audit stores both `proposed_mask` and `applied_mask` so an AllToken
run cannot be mistaken for a run in which the selector was skipped.

## Sampling and Reproducibility

The manifest generator reads the existing D sample log, reconstructs TP/TN/FP/FN
from normalized predictions and ground truth, samples within each stratum using
seed 42, and writes an immutable manifest containing at least:

- `doc_id`
- `question_id`
- dataset image identifier
- question
- ground truth
- original D prediction
- original D outcome stratum

All four runs consume the same manifest. Audit output is joined by stable sample
identifier rather than output order, because distributed evaluation may reorder
records.

The current run contains only exact `Yes` and `No` responses, so its reported
confusion matrix is not affected by invalid-response parsing. The new evaluator
must nevertheless report invalid or unparseable responses explicitly.

## Audit Data Model

### Sample-level record

Each condition records:

- audit schema version
- condition ID
- sample and question identifiers
- model and generation configuration
- question, ground truth, raw response, normalized prediction, and correctness
- final selected token and decoded response
- source hashes or run identifier

Condition A has no recursive-stage selection records.

### Stage-transition record

For each non-final source stage and audited generation token, record:

- `token_idx`
- `source_stage` and `target_stage`
- raw selector input `ret_attn`
- SECOND row-mean entropy
- normalized SECOND entropy
- diagnostic flattened-2D entropy
- requested keep ratio
- actual attention-score cutoff (`threshold_value`)
- proposed and applied selected-token counts
- total-token count
- proposed and applied keep ratios
- mask before selection
- proposed mask
- applied mask
- per-stage next-token logits needed for the Yes/No margin

Raw tensors are stored in a lossless numeric format; PNG heatmaps and overlays
are derived artifacts and are never used for metric calculation.

The existing variable named `calculated_threshold` is logged as
`requested_keep_ratio`. It is not the attention-score threshold. The actual keep
ratio is measured from the mask because tied attention values may select more
tokens than requested.

### Yes/No margin

The margin is `L(Yes) - L(No)` for the next answer token at each recursive stage
and after final decoding. Candidate token IDs are resolved using the exact prompt
context and tokenizer. The implementation verifies that each candidate is a
single continuation token for the current Vicuna tokenizer; otherwise it records
the sequence log-probability difference or marks the scalar margin unavailable
with an explicit reason.

Selection computed at source stage `s` affects target stage `s+1`. Analyses must
therefore compare the next-stage logits or compare C and D at matched stages. A
source-stage attention map and its same-stage logits must not be described as a
before/after selection pair.

For POPE v1, tensor dumping defaults to the first answer token (`token_idx=0`),
which determines the Yes/No response. The schema retains `token_idx` so later
audits can include all generated tokens.

## POPE Metric Correction

The current `pope_yes_ratio` aggregates ground-truth labels. The audit introduces
unambiguous metrics:

- `pope_yes_ratio`: corrected compatibility name for the fraction of normalized
  model predictions equal to Yes
- `pope_pred_yes_ratio`: fraction of normalized model predictions equal to Yes
- `pope_gt_yes_ratio`: fraction of ground-truth labels equal to Yes
- `pope_invalid_ratio`: fraction that cannot be normalized to Yes or No

Existing result files remain unchanged for historical traceability. In new runs,
the former ground-truth behavior is available only under the explicit
`pope_gt_yes_ratio` name; `pope_yes_ratio` and `pope_pred_yes_ratio` both report
the prediction ratio. Metric output includes evaluation schema version 2 so old
and new result files cannot be silently mixed. Audit reports use the explicit
`pope_pred_yes_ratio` and `pope_gt_yes_ratio` names.

## Analysis

Because the 80-sample manifest intentionally oversamples Full SECOND errors, it
does not estimate population Accuracy, F1, or overall Selection Harm Rate.

The v1 report contains:

- C-to-D paired flips within each original D stratum
- Selection Harm count: `C correct, D wrong`
- Selection Benefit count: `C wrong, D correct`
- FN Rescue Rate: `count(D=FN and C=TP) / count(D=FN)`
- B-to-C CD harm and benefit counts
- A-to-B changes, described only as recursive-pathway changes
- counts and 95% Wilson confidence intervals for conditional proportions

Any displayed net effect is labeled as a diagnostic-set quantity. A
population-level rate requires either evaluation on all 3000 samples or a
representative design with appropriate sampling weights.

## Equivalence and Validation

Adding audit logging must not change Full SECOND behavior.

Validation has two levels:

1. Prediction parity: rerun condition D on the manifest and require exact
   per-sample equality with the existing Full SECOND predictions. If parity
   fails, stop before interpreting any audit result.
2. Selector parity: unit tests compare the original selector's returned mask and
   keep-ratio behavior with the instrumented selector on deterministic synthetic
   attention maps, including tied scores and boundary keep ratios.

The old run did not save entropy, cutoffs, or masks, so those internal values
cannot be retrospectively compared with it. Internal equivalence is established
by selector parity tests and by keeping D's applied mask on the original code
path, not by claiming unavailable historical comparisons.

Additional tests cover:

- Raw decoding bypasses CD and candidate truncation.
- AllToken logs the proposed mask but applies an all-one mask.
- D applies the proposed mask unchanged.
- requested and actual keep ratios are distinguished under ties.
- POPE predicted/ground-truth Yes ratios and invalid ratio.
- deterministic manifest sampling and exact stratum counts.
- audit records from multiple processes are collision-free and joinable by
  sample identifier.

## Failure Handling

- A missing or duplicate manifest ID aborts the run.
- A NaN/Inf attention, entropy, cutoff, or logit is recorded and fails validation
  rather than being silently plotted.
- Audit artifacts are written per process/sample and merged after evaluation to
  avoid concurrent writes to one JSONL file.
- Existing logs are never overwritten; each run uses a unique output directory.
- If AllToken exceeds the model context length or GPU memory, the run reports the
  exact stage and token count. It does not silently reduce the mask.

## Phase-2 Gate

Proceed to the full 3000-sample audit, condition E, COCO bbox coverage, and
target-scale/occlusion analysis only if all validation checks pass and condition
C rescues at least 5 of the 30 sampled D false negatives. This threshold is a
predeclared resource-allocation gate, not a statistical-significance claim. A
repeat run must reproduce all first-token predictions for the identified flips.
The phase-1 report must show the paired prediction flips and their internal
selection traces before proposing a replacement token selector.

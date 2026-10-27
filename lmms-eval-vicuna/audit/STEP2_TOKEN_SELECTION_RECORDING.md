# Step 2: record where SECOND actually used visual tokens

## Goal

Step 2 records the spatial decision made at every SECOND transition for the
first generated token in POPE. Token 0 is the Yes/No decision token; the full
generated token sequence is still stored in `sample.json`.

The capture distinguishes four objects:

- `mask_before`: target-stage mask before selection.
- `proposed_mask`: the selector's decision on the common 48x48 canvas.
- `applied_mask`: the mask written into SECOND after the A/B/C/D condition.
- `consumed_mask`: the resized patch-grid mask at the exact point where the
  multimodal model removes zeroed visual features.

Only `consumed_mask` and its indices describe the visual tokens actually used by
the next stage. The three target grids are 12x12, 24x24, and 48x48 for
transitions `-2 -> -1`, `-1 -> 0`, and `0 -> 1`.

The recorder keeps only GPU tensor references while SECOND is deciding. It
computes diagnostic summaries, copies arrays to CPU, and writes artifacts only
after the generated sequence is complete. The pre-decision mask update follows
the same in-place selector path as an uninstrumented condition-D run.

## Artifact modes

The diagnostic run uses `visual` mode for the fixed 100-sample Step 1 diagnostic
manifest. It saves the original image, padded model-input canvas, attention
heatmap, proposed mask, applied mask, and consumed mask for each transition.

Before that capture, an uninstrumented reference run executes the same 100
samples at the same Git commit. The diagnostic capture must reproduce all 100
reference predictions exactly. This isolates recorder effects from prediction
drift relative to the historical Step 1 run, which was produced at an older Git
commit.

The full 3000-sample run uses `compact` mode. It saves JSON metadata and
compressed lossless NPZ arrays without duplicating PNG images.

Each NPZ contains:

- `ret_attn`
- `mask_before`
- `proposed_mask`
- `applied_mask`
- `consumed_mask`
- `consumed_keep_indices` and `consumed_drop_indices`
- `consumed_keep_coords_rc` and `consumed_drop_coords_rc`

`sample.json` records the question, ground truth, prediction, generated token
IDs, original size, square model-input size, and exact padding. Grid cells map
proportionally to the padded square canvas; subtracting the recorded padding
maps them back to the original image.

## Required environment

Run from the repository with the requested environment active:

```bash
conda activate SECOND_clean
cd /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna
```

The run script refuses to start in any other conda environment.

## Preflight tests

```bash
python -m unittest \
  tests.test_second_audit_core \
  tests.test_second_audit_recorder \
  tests.test_recursion_audit \
  tests.test_llava_audit_plumbing \
  tests.test_second_audit_manifest \
  tests.test_step2_capture_validator \
  tests.test_step2_run_script \
  tests.test_step2_reference
```

## Capture commands

First generate the uninstrumented, current-code reference for the diagnostic
cohort:

```bash
bash audit/run_step2_reference.sh
```

Then run the visual diagnostic capture:

```bash
bash audit/run_step2_capture.sh diagnostic
```

The capture script finds the latest reference, verifies its Git commit and
source-file fingerprint against the current working tree, checks that it passed
validation, and requires zero prediction mismatches against it. Inspect the
capture's `validation_report.json`. Continue only when its status is `PASS`,
then run the complete quantitative cohort:

```bash
bash audit/run_step2_capture.sh full
```

Both commands use condition D, two processes, GPUs 0 and 1, and the exact SECOND
parameters frozen in Step 1. Override GPU/process choices only when required:

```bash
CUDA_VISIBLE_DEVICES=2,3 NUM_PROCESSES=2 MAIN_PROCESS_PORT=29902 \
  bash audit/run_step2_capture.sh full
```

Outputs are written under timestamped directories in
`audit/runs/step2_token_selection_v2/`. Set `SECOND_STEP2_RUN_ROOT` to use another
output root.

## Automatic validation

After inference, the script automatically verifies:

- exactly 100 or 3000 unique question IDs;
- all three transitions for every sample;
- schema v2 and expected artifact mode;
- binary masks and finite attention arrays;
- target grids of 12x12, 24x24, and 48x48;
- `consumed_mask` equals the model's nearest-neighbor resize of `applied_mask`;
- keep/drop indices and coordinates reconstruct `consumed_mask` exactly;
- image/padding transforms and diagnostic PNG dimensions;
- the generated first token agrees with the recorded Yes/No token IDs and logit
  margin;
- diagnostic predictions exactly match the same-commit uninstrumented reference;
- ground-truth labels match the frozen Step 1 data.

Historical Step 1 prediction differences are reported as warnings because that
run came from commit `4ad6587`, while Step 2 runs use the current commit. They do
not establish whether the recorder changed inference. A mismatch against the
same-commit diagnostic reference does fail validation. Any missing sample,
duplicate ID, malformed array, mask mismatch, decoding-trace mismatch, or
same-commit prediction drift also fails with a nonzero exit status.

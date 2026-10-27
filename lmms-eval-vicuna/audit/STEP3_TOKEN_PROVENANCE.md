# Step 3: visual-token provenance

Step 3 records where the visual embeddings that SECOND actually consumed came
from. It is an audit of the existing forward path; it does not change the
selection mask or the generated answer.

The runtime trace is created during the audited token's four recursive forward
passes. The trace keeps GPU tensors and Python layout metadata only. It does not
copy a mask to CPU, calculate statistics, render images, or write files during
the forward pass. After the final pass and token decision, the recorder moves
the trace to CPU and writes the artifacts.

For every candidate patch the final provenance JSONL records:

* source stage and source type (`low_res_base`, `base`, or `anyres_tile`);
* AnyRes tile row/column and local patch row/column;
* row-major candidate index, stage sequence index, visual sequence index, and
  decoder sequence index;
* preprocessing, padded-canvas, and original-image rectangles;
* the fraction of the candidate footprint outside the original image;
* actual selected/rejected, mask-selected, CLS/newline, and truncation flags.

The mapping uses the actual runtime grid, tile dimensions, model-input canvas,
and the recorded center-padding transform. AnyRes coordinates are inverted
through the same resize-and-center-pad geometry used by `process_anyres_image`.
The original-image rectangle is then clipped to the original image bounds.

`runtime_offline_match` compares the candidate indices that were appended by
the actual feature-construction loops with the indices reconstructed from the
consumed mask. A PASS therefore means that the spatial records describe the
sequence that entered the model, rather than a guessed regular grid.

## What this step does not establish

`ret_attn` remains SECOND's rank-transformed selection signal. It is not raw
multi-head causal attention and must not be interpreted as causal importance.
This step also does not test whether a selected region is question-relevant,
perform a counterfactual token ablation, or claim that a selected patch caused
the answer. Those are later attention and causal validation steps.

## Run order on root-31

Run from `/home/yxq/Downloads/SECOND-main/lmms-eval-vicuna` after activating
`SECOND_clean`:

```bash
bash audit/run_step3_provenance.sh reference
bash audit/run_step3_provenance.sh diagnostic
```

The reference command is an uninstrumented 100-sample run for the same commit.
The diagnostic command then runs the 100 samples with condition D, checks the
existing Step 2 compatibility contract, and runs the Step 3 validator. Do not
start a 3000-sample provenance run until the diagnostic `validation_report.json`
has `"status": "PASS"` and `"errors": []`.

Expected diagnostic artifacts are under:

```text
audit/runs/step3_provenance_v1/diagnostic_<UTC>/artifacts/D/rank_*/<sample>/
```

`provenance_summary.json` contains one summary for each forward stage. The
`*_runtime_trace.npz` files preserve the consumed masks and actual append-order
indices used for independent validation. The `*_provenance.jsonl` files are the
candidate-level reconstruction. In visual mode,
`*_geometry.png` shows the coordinate transform and
`*_source_stage_*_selection.png` shows green selected, red rejected, and gray
fully padded footprints. These colors describe the mask state, not importance.

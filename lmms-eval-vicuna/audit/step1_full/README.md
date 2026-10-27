# Step 1 full POPE baseline

`pope_pop_3000_baseline/` is the quantitative cohort for the SECOND visual-token
audit. It contains all 3000 POPE Popular predictions from the frozen run. No
sampling is applied.

Use this cohort for counts, model metrics, selected-token statistics, overlap
metrics, and confidence intervals. Analyze positive questions as `TP` versus
`FN` and negative questions as `TN` versus `FP`.

The 100-case diagnostic manifest is intended for detailed visualization and
causal experiments. The legacy 80-case manifest is only a smoke/regression set;
neither small collection replaces the full cohort for quantitative claims.

## Integrity

The directory records:

- the immutable prediction copy and normalized index;
- aligned image mappings and stratification features;
- aggregate baseline metrics;
- SHA256 values for the raw prediction and evaluation-result inputs;
- model, task, dataset, run parameters, and commit provenance;
- the environment used to build and validate these audit artifacts.

The environment snapshot is not asserted to be the historical environment that
generated the original predictions. See `baseline_provenance.json` for the exact
scope of every provenance field.

Validate from the `lmms-eval-vicuna` directory:

```bash
python3 audit/validate_step1.py
```

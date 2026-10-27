# Step 6A: Does token selection cause the observed wrong answers?

This version reuses and verifies the completed baseline in `step6_interventions_v1`,
keeps its frozen 48-question cohort and run ordering, and creates a new immutable
study at `step6_causal_v2`. It does not alter the old study. QID 1025 remains in
all 48-item runs and in D1 replay, but is excluded from primary mechanism results
because its D1/D2 masks differed. Primary cases are 2253, 2559, 311, 645, 1043.

The existing opt-in model hook must already be installed from the original Step 6
package. The v2 driver verifies its source snapshot against the completed v1
plan. If source/runtime changed, stop. It symlinks prior baseline folders into
the new study (read-only by procedure), verifies completion/output hashes, and
never overwrites them.

Run commands one at a time on the server, and stop on any FAIL:

```bash
cd /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna
conda activate SECOND_clean
export CUDA_VISIBLE_DEVICES=0,1
export NUM_PROCESSES=2
unset SECOND_INTERVENTION_PLAN
python -B audit/step6_interventions.py --help
python -B audit/step6_causal_v2.py prepare
python -B audit/step6_causal_v2.py freeze
python -B audit/step6_causal_v2.py replay
cat audit/runs/step6_causal_v2/replay_gate.json
# Continue only when status is PASS.
python -B audit/step6_causal_v2.py rescue
python -B audit/step6_causal_v2.py controls
python -B audit/step6_causal_v2.py context
python -B audit/step6_causal_v2.py report
cat audit/runs/step6_causal_v2/intervention_report.json
```

`freeze` creates the geometry and mask plans without inference. `replay` performs
one full 48-item run and verifies the exact D1 mask was consumed for every item.
The replay gate requires exact input metadata, masks and primary prediction/token
records for the five primary cases. It reports signal/margin changes instead of
mistaking every floating-point difference for a failed mask replay. QID 1025's
instability remains visible in the report.

Further runs: 4 target-rescue plans (three individual stages and all stages),
12 equal-count non-target controls (same 4 plans x seeds 17, 29, 43), and 4
stage-restoration plans for target-complete cases. All keep the 48-item run
context; only the five named primary cases are modified. The intervention affects
first-answer-token masks; later tokens use ordinary adaptive selection. This is
not a population estimate and does not isolate contrastive decoding.

The non-target pool consists of dropped image-content cells outside annotated
target boxes. It is not guaranteed background and is not matched on location,
scale or selector score. A target-rescue advantage is supportive evidence for a
target-location effect in these cases, but a strong claim should follow with
spatial/score-matched controls. The annotations are boxes, not segmentation masks.

`status: COMPLETE` means all planned runs and artifact checks completed. It does
not mean the hypothesis is confirmed. Compare each target rescue with all
equal-count controls, report prediction and margin changes from D replay, and
inspect per-stage output. Full-stage restoration in 311/645/1043 localizes a
stage dependency but cannot alone prove semantic context loss or a CD-specific
mechanism.

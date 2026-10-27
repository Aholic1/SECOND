# Step 6: fixed first-token mask interventions

Scope: exploratory causal tests on 2253/2559 (target rescue) and
311/645/1025/1043 (target-complete degradation). Keep the original 48-question
manifest, ordering, GPUs 0,1, two processes and generation seeds. Never infer
population performance from these selected cases. Review original target labels
and boxes before interpreting a rescue as restoration of semantic evidence.

Install only the files in step6_interventions_v1.zip under lmms-eval-vicuna.
The installer patches one exact source anchor and saves a byte-exact backup.
No experiment or historical report is overwritten. Hook disabled unless the
driver explicitly supplies SECOND_INTERVENTION_PLAN.

```bash
cd /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna
conda activate SECOND_clean
export CUDA_VISIBLE_DEVICES=0,1
export NUM_PROCESSES=2
unset SECOND_INTERVENTION_PLAN
python -B audit/test_step6_interventions.py
python audit/install_step6_hook.py
python audit/step6_interventions.py prepare
python audit/step6_interventions.py baseline
python audit/step6_interventions.py freeze
python audit/step6_interventions.py replay
cat audit/runs/step6_interventions_v1/replay_gate.json
# Continue only if replay gate PASS.
python audit/step6_interventions.py rescue
python audit/step6_interventions.py controls
python audit/step6_interventions.py context
python audit/step6_interventions.py report
```

Run each line separately; stop on any FAIL. No threshold bypass. Baseline
requires exact repeated capture records for the six focal cases and the expected
C=yes/D=no outcome. Replay requires exact capture records for all 48. Failures
indicate a reproducibility prerequisite was not met, not that a hypothesis was
disproved. Preserve partial run directories on failure. Completed runs are
verified before skipping; do not edit frozen scripts/plans during this study.

Design: 4 fresh baseline runs, 1 replay, 4 target-rescue variants (three single
stages plus all), 12 random-control variants (same four stage choices, seeds
17/29/43), 4 full-stage restoration variants. Total 25 runs x 48 questions =
1200 question-condition evaluations; actual time depends on the machine.
Controls sample dropped content cells outside every annotated target bbox, not
guaranteed semantic background. Token budget is matched at each consumed grid,
not at the 48x48 proposal grid. No replacement sampling. Three seeds are an
exploratory control, not a precise statistical estimate. This first pass does
not spatially match random controls; target-specific positive findings require
additional position/scale/score-matched controls before a strong claim.

All first-token stage masks are replayed for all 48 questions. Intervention
edits only named cases/stages. Later generated tokens use ordinary adaptive
selection. Thus this experiment tests the direct effect conditional on fixed
first-token masks; it does not test fully adaptive downstream mask propagation.
Even all-stage restoration is not asserted equivalent to condition C, since
later generated tokens are unmodified. Stage margin records are saved in the
existing per-transition JSON artifacts. This study does not isolate CD itself.

Interpretation:
- Target rescue beats equal-count controls consistently: evidence for targeted
  local-information loss in these cases, conditional on bbox validity.
- Both target and random rescue work: added context/token budget or decision
  boundary sensitivity remains an alternative explanation.
- Neither works: first-token bbox rescue is insufficient; does not disprove
  all target-information-loss mechanisms.
- Full-stage restoration repairs target-complete cases: localizes stage/token
  dependency, not necessarily semantic context loss or a CD-specific defect.
- If only combined restoration repairs: investigate cross-stage interaction;
  inspect original stage logits and final CD margin before attributing cause.

Share intervention_report.json, replay_gate.json, plans/geometry.json, and
selected token_000_stage_* JSON metadata for follow-up. COMPLETE in the report
means all planned artifacts exist; it does not mean the hypothesis is true.

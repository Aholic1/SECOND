# Step 6 causal v3: feasible equal-budget interventions

The previous freeze stopped because the non-target dropped pool was too small
for at least one target-rescue dose. This is a valid design limitation. Do not
sample with replacement or silently reduce a control. v3 preserves the original
48-question execution context and changes the intervention to a same-budget
reallocation: restore selected dropped target cells and remove the same number
of retained non-target content cells. Feasible-dose controls restore the same
number of dropped non-target cells and remove the same retained non-target
cells. Token count therefore stays fixed in both arms.

An additive full-target-rescue arm is also frozen as a descriptive restoration;
it is explicitly not a causal comparison because token count increases. A
same-budget full-target reallocation arm is frozen when enough retained
non-target cells exist. It is compared with baseline and is explicitly marked as
lacking a same-dose control if the non-target dropped pool is too small. Matched-dose
target/control arms are the primary comparison. Dose zero is reported as not
estimable, never converted into a pass.

QID 1025 remains in all 48-row runs but is excluded from the primary causal
interpretation because its D1/D2 masks differ. Primary cases: 2253, 2559,
311, 645, 1043. Target reallocation is only for 2253/2559; full-stage restore
is only for 311/645/1043.

Upload this v3 file and run one command at a time:

```bash
cd /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna
conda activate SECOND_clean
unzip /home/yxq/Downloads/step6_causal_v3.zip -d .
export CUDA_VISIBLE_DEVICES=0,1
export NUM_PROCESSES=2
unset SECOND_INTERVENTION_PLAN
python -B audit/test_step6_causal_v3.py
python -B audit/step6_causal_v3.py prepare
python -B audit/step6_causal_v3.py freeze
python -B audit/step6_causal_v3.py replay
cat audit/runs/step6_causal_v3/replay_gate.json
# Continue only when the gate says PASS.
python -B audit/step6_causal_v3.py rescue
python -B audit/step6_causal_v3.py controls
python -B audit/step6_causal_v3.py context
python -B audit/step6_causal_v3.py report
cat audit/runs/step6_causal_v3/intervention_report.json
```

The rescue command runs both full-dose target reallocation and feasible matched-
dose target arms. The control command runs matched-dose non-target arms for
three fixed seeds. The report records the actual dose, prediction, margin, and
whether the run was estimable. A target arm is supportive only when it improves
over baseline and its same-dose non-target controls do not show the same effect
across seeds. These selected cases do not estimate population rates. The
intervention fixes first-token masks; later generated tokens remain adaptive.

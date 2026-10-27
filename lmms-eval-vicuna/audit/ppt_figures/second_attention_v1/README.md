# SECOND attention-distribution figures

These figures provide attention-distribution evidence; outcomes and audit roles are recorded here, not inside the PNGs.

## 结构标签对照 / Structural-label glossary

- 输入图像 / Input
- 阶段 -2→-1 / Stage -2→-1
- 阶段 -1→0 / Stage -1→0
- 阶段 0→1 / Stage 0→1
- 最终删除区域 / Removed Tokens
- 最终注意力 / Final Attention
- 保留比例 / Keep Ratio
- 相对均匀注意力 / Relative-to-uniform attention

## QID 887 — `q887_attention_distribution.png`

- Audit role: C-to-D token-selection benefit
- Question: Is there a car in the image?
- Ground truth: yes
- Predictions (A/B/C/D): no / no / no / yes
- Keep ratios (-2→-1, -1→0, 0→1): 1.0000, 0.9861, 0.9601
- Source artifacts: /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_1/887/sample.json, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_1/887/input_image.png, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_1/887/token_000_stage_-2_to_-1.json, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_1/887/token_000_stage_-1_to_0.json, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_1/887/token_000_stage_0_to_1.json, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_1/887/token_000_stage_-2_to_-1.npz, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_1/887/token_000_stage_-1_to_0.npz, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_1/887/token_000_stage_0_to_1.npz
- Source logs (A/B/C/D): /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/A/results/liuhaotian__llava-v1.6-vicuna-7b/20260812_023541_samples_pope_pop_audit.jsonl, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/B/results/liuhaotian__llava-v1.6-vicuna-7b/20260812_023725_samples_pope_pop_audit.jsonl, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/C/results/liuhaotian__llava-v1.6-vicuna-7b/20260812_024244_samples_pope_pop_audit.jsonl, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/results/liuhaotian__llava-v1.6-vicuna-7b/20260812_024804_samples_pope_pop_audit.jsonl
- Suggested speaker note: Compare the shared-scale attention distribution and final removed cells with the paired A/B/C/D outcomes; this visual is descriptive evidence, not a causal conclusion.

## QID 2254 — `q2254_attention_distribution.png`

- Audit role: B-to-C contrastive-decoding harm
- Question: Is there a dining table in the image?
- Ground truth: no
- Predictions (A/B/C/D): no / no / yes / yes
- Keep ratios (-2→-1, -1→0, 0→1): 0.9444, 0.9722, 0.9497
- Source artifacts: /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_0/2254/sample.json, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_0/2254/input_image.png, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_0/2254/token_000_stage_-2_to_-1.json, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_0/2254/token_000_stage_-1_to_0.json, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_0/2254/token_000_stage_0_to_1.json, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_0/2254/token_000_stage_-2_to_-1.npz, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_0/2254/token_000_stage_-1_to_0.npz, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_0/2254/token_000_stage_0_to_1.npz
- Source logs (A/B/C/D): /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/A/results/liuhaotian__llava-v1.6-vicuna-7b/20260812_023541_samples_pope_pop_audit.jsonl, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/B/results/liuhaotian__llava-v1.6-vicuna-7b/20260812_023725_samples_pope_pop_audit.jsonl, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/C/results/liuhaotian__llava-v1.6-vicuna-7b/20260812_024244_samples_pope_pop_audit.jsonl, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/results/liuhaotian__llava-v1.6-vicuna-7b/20260812_024804_samples_pope_pop_audit.jsonl
- Suggested speaker note: Compare the shared-scale attention distribution and final removed cells with the paired A/B/C/D outcomes; this visual is descriptive evidence, not a causal conclusion.

## QID 2284 — `q2284_attention_distribution.png`

- Audit role: B-to-C contrastive-decoding harm
- Question: Is there a car in the image?
- Ground truth: no
- Predictions (A/B/C/D): no / no / yes / yes
- Keep ratios (-2→-1, -1→0, 0→1): 0.9444, 0.9306, 0.9149
- Source artifacts: /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_0/2284/sample.json, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_0/2284/input_image.png, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_0/2284/token_000_stage_-2_to_-1.json, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_0/2284/token_000_stage_-1_to_0.json, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_0/2284/token_000_stage_0_to_1.json, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_0/2284/token_000_stage_-2_to_-1.npz, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_0/2284/token_000_stage_-1_to_0.npz, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_0/2284/token_000_stage_0_to_1.npz
- Source logs (A/B/C/D): /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/A/results/liuhaotian__llava-v1.6-vicuna-7b/20260812_023541_samples_pope_pop_audit.jsonl, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/B/results/liuhaotian__llava-v1.6-vicuna-7b/20260812_023725_samples_pope_pop_audit.jsonl, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/C/results/liuhaotian__llava-v1.6-vicuna-7b/20260812_024244_samples_pope_pop_audit.jsonl, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/results/liuhaotian__llava-v1.6-vicuna-7b/20260812_024804_samples_pope_pop_audit.jsonl
- Suggested speaker note: Compare the shared-scale attention distribution and final removed cells with the paired A/B/C/D outcomes; this visual is descriptive evidence, not a causal conclusion.

## QID 2920 — `q2920_attention_distribution.png`

- Audit role: B-to-C contrastive-decoding harm
- Question: Is there a dining table in the image?
- Ground truth: no
- Predictions (A/B/C/D): no / no / yes / yes
- Keep ratios (-2→-1, -1→0, 0→1): 1.0000, 0.9931, 0.9462
- Source artifacts: /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_1/2920/sample.json, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_1/2920/input_image.png, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_1/2920/token_000_stage_-2_to_-1.json, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_1/2920/token_000_stage_-1_to_0.json, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_1/2920/token_000_stage_0_to_1.json, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_1/2920/token_000_stage_-2_to_-1.npz, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_1/2920/token_000_stage_-1_to_0.npz, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_1/2920/token_000_stage_0_to_1.npz
- Source logs (A/B/C/D): /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/A/results/liuhaotian__llava-v1.6-vicuna-7b/20260812_023541_samples_pope_pop_audit.jsonl, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/B/results/liuhaotian__llava-v1.6-vicuna-7b/20260812_023725_samples_pope_pop_audit.jsonl, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/C/results/liuhaotian__llava-v1.6-vicuna-7b/20260812_024244_samples_pope_pop_audit.jsonl, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/results/liuhaotian__llava-v1.6-vicuna-7b/20260812_024804_samples_pope_pop_audit.jsonl
- Suggested speaker note: Compare the shared-scale attention distribution and final removed cells with the paired A/B/C/D outcomes; this visual is descriptive evidence, not a causal conclusion.

## QID 2679 — `q2679_attention_distribution.png`

- Audit role: B-to-C contrastive-decoding benefit
- Question: Is there a toilet in the image?
- Ground truth: yes
- Predictions (A/B/C/D): yes / no / yes / yes
- Keep ratios (-2→-1, -1→0, 0→1): 0.9444, 0.9167, 0.9236
- Source artifacts: /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_0/2679/sample.json, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_0/2679/input_image.png, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_0/2679/token_000_stage_-2_to_-1.json, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_0/2679/token_000_stage_-1_to_0.json, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_0/2679/token_000_stage_0_to_1.json, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_0/2679/token_000_stage_-2_to_-1.npz, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_0/2679/token_000_stage_-1_to_0.npz, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_0/2679/token_000_stage_0_to_1.npz
- Source logs (A/B/C/D): /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/A/results/liuhaotian__llava-v1.6-vicuna-7b/20260812_023541_samples_pope_pop_audit.jsonl, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/B/results/liuhaotian__llava-v1.6-vicuna-7b/20260812_023725_samples_pope_pop_audit.jsonl, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/C/results/liuhaotian__llava-v1.6-vicuna-7b/20260812_024244_samples_pope_pop_audit.jsonl, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/results/liuhaotian__llava-v1.6-vicuna-7b/20260812_024804_samples_pope_pop_audit.jsonl
- Suggested speaker note: Compare the shared-scale attention distribution and final removed cells with the paired A/B/C/D outcomes; this visual is descriptive evidence, not a causal conclusion.

## QID 1859 — `q1859_attention_distribution.png`

- Audit role: A-to-B recursive-pathway benefit
- Question: Is there a laptop in the image?
- Ground truth: yes
- Predictions (A/B/C/D): no / yes / yes / yes
- Keep ratios (-2→-1, -1→0, 0→1): 0.9444, 0.9167, 0.9236
- Source artifacts: /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_0/1859/sample.json, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_0/1859/input_image.png, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_0/1859/token_000_stage_-2_to_-1.json, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_0/1859/token_000_stage_-1_to_0.json, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_0/1859/token_000_stage_0_to_1.json, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_0/1859/token_000_stage_-2_to_-1.npz, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_0/1859/token_000_stage_-1_to_0.npz, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/artifacts/D/rank_0/1859/token_000_stage_0_to_1.npz
- Source logs (A/B/C/D): /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/A/results/liuhaotian__llava-v1.6-vicuna-7b/20260812_023541_samples_pope_pop_audit.jsonl, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/B/results/liuhaotian__llava-v1.6-vicuna-7b/20260812_023725_samples_pope_pop_audit.jsonl, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/C/results/liuhaotian__llava-v1.6-vicuna-7b/20260812_024244_samples_pope_pop_audit.jsonl, /home/yxq/Downloads/SECOND-main/lmms-eval-vicuna/audit/runs/pope_pop_audit_v1/D/results/liuhaotian__llava-v1.6-vicuna-7b/20260812_024804_samples_pope_pop_audit.jsonl
- Suggested speaker note: Compare the shared-scale attention distribution and final removed cells with the paired A/B/C/D outcomes; this visual is descriptive evidence, not a causal conclusion.

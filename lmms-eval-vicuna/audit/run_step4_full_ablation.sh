#!/usr/bin/env bash
set -euo pipefail

COND="${1:-}"

if [[ "$COND" != "B" && "$COND" != "C" ]]; then
    echo "Usage: bash audit/run_step4_full_ablation.sh {B|C}" >&2
    exit 1
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "SECOND_clean" ]]; then
    echo "Activate SECOND_clean first." >&2
    exit 1
fi

EVAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$EVAL_ROOT"

RUN_ROOT="$EVAL_ROOT/audit/runs/step4_ablation_v1"
RUN_DIR="$RUN_ROOT/full_${COND}_$(date -u +%Y%m%dT%H%M%SZ)"
ARTIFACT_ROOT="$RUN_DIR/artifacts"
RESULTS_DIR="$RUN_DIR/results"

if [[ "$COND" == "B" ]]; then
    MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29920}"
else
    MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29921}"
fi

mkdir -p "$ARTIFACT_ROOT" "$RESULTS_DIR"

unset SECOND_AUDIT_MANIFEST || true
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NUM_PROCESSES="${NUM_PROCESSES:-2}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

python3 - "$RUN_DIR/run_manifest.json" "$COND" "$MAIN_PROCESS_PORT" <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

path, condition, port = sys.argv[1:]

definitions = {
    "B": "recursive all-token without contrastive decoding",
    "C": "recursive all-token with contrastive decoding",
}

payload = {
    "schema_version": 1,
    "purpose": "Step 4 full 3000-sample ablation",
    "mode": "full",
    "task": "pope_pop",
    "expected_count": 3000,
    "condition": condition,
    "condition_definition": definitions[condition],
    "artifact_mode": "compact",
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "runtime": {
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "num_processes": int(os.environ["NUM_PROCESSES"]),
        "main_process_port": int(port),
    },
    "second_parameters": {
        "generation_type": "recursion",
        "fix_grid": "2x2",
        "attention_thresholding_type": "attn_topk",
        "attention_threshold": 1.0,
        "positional_embedding_type": "bilinear_interpolation",
        "stages": [-2, -1, 0, 1],
        "contrastive_alphas": [0.8, 0.8, 0.8],
    },
}

Path(path).write_text(json.dumps(payload, indent=2) + "\n")
PY

MODEL_ARGS="pretrained=liuhaotian/llava-v1.6-vicuna-7b,audit_condition=${COND},audit_output_dir=${ARTIFACT_ROOT},audit_token_idx=0,audit_artifact_mode=compact"

python -m accelerate.commands.launch \
    --num_processes="$NUM_PROCESSES" \
    --main_process_port "$MAIN_PROCESS_PORT" \
    -m lmms_eval \
    --model llava \
    --model_args "$MODEL_ARGS" \
    --tasks pope_pop \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "step4_full_${COND}" \
    --output_path "$RESULTS_DIR" \
    --generation_type recursion \
    --fix_grid 2x2 \
    --attention_thresholding_type attn_topk \
    --attention_threshold 1.0 \
    --positional_embedding_type bilinear_interpolation \
    --stages -2 -1 0 1 \
    --verbosity DEBUG \
    --contrastive_alphas 0.8 0.8 0.8 \
    2>&1 | tee "$RUN_DIR/console.log"

python3 audit/validate_step2_capture.py \
    --artifact-root "$ARTIFACT_ROOT" \
    --baseline-predictions \
      audit/step1_full/pope_pop_3000_baseline/baseline_predictions.jsonl \
    --expected-count 3000 \
    --condition "$COND" \
    --artifact-mode compact \
    --report "$RUN_DIR/validation_report.json"

echo "Condition ${COND} completed: $RUN_DIR"

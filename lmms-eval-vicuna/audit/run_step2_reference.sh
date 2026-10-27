#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "SECOND_clean" ]]; then
    echo "Error: activate the SECOND_clean conda environment before running this script." >&2
    exit 1
fi

if [[ $# -ne 0 ]]; then
    echo "Usage: bash audit/run_step2_reference.sh" >&2
    exit 1
fi

EVAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HISTORICAL_BASELINE="$EVAL_ROOT/audit/step1_full/pope_pop_3000_baseline/baseline_predictions.jsonl"
DIAGNOSTIC_MANIFEST="$EVAL_ROOT/audit/step1_diagnostic/diagnostic_manifest.json"
RUN_ROOT="${SECOND_STEP2_REFERENCE_ROOT:-$EVAL_ROOT/audit/runs/step2_reference_v2}"
RUN_DIR="$RUN_ROOT/diagnostic_$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_DIR="$RUN_DIR/results"
REFERENCE_PREDICTIONS="$RUN_DIR/reference_predictions.jsonl"
SOURCE_FINGERPRINT="$(
    cd "$EVAL_ROOT"
    sha256sum \
        lmms_eval/models/llava.py \
        lmms_eval/recursion_utils.py \
        lmms_eval/second_audit.py \
        lmms_eval/tasks/pope/audit_utils.py \
        lmms_eval/tasks/pope/pope_pop_audit.yaml \
        LLaVA-NeXT/llava/model/llava_arch.py \
        LLaVA-NeXT/llava/model/language_model/llava_llama.py \
        LLaVA-NeXT/llava/model/language_model/llava_recursion.py \
        | sha256sum | cut -d' ' -f1
)"

if [[ -e "$RUN_DIR" ]]; then
    echo "Error: run directory already exists: $RUN_DIR" >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR"
export SECOND_AUDIT_MANIFEST="$DIAGNOSTIC_MANIFEST"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NUM_PROCESSES="${NUM_PROCESSES:-2}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29890}"
export EVAL_ROOT RUN_DIR RESULTS_DIR REFERENCE_PREDICTIONS
export HISTORICAL_BASELINE DIAGNOSTIC_MANIFEST SOURCE_FINGERPRINT

python - <<'PY'
import json
import os
import subprocess
from pathlib import Path


eval_root = Path(os.environ["EVAL_ROOT"])
payload = {
    "schema_version": 1,
    "purpose": "current-code uninstrumented diagnostic reference",
    "task": "pope_pop_audit",
    "expected_count": 100,
    "git_commit": subprocess.check_output(
        ["git", "-C", str(eval_root), "rev-parse", "HEAD"], text=True
    ).strip(),
    "source_fingerprint_sha256": os.environ["SOURCE_FINGERPRINT"],
    "diagnostic_manifest": str(
        Path(os.environ["DIAGNOSTIC_MANIFEST"]).relative_to(eval_root)
    ),
    "runtime": {
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "python": subprocess.check_output(["python", "--version"], text=True).strip(),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "num_processes": int(os.environ["NUM_PROCESSES"]),
        "main_process_port": int(os.environ["MAIN_PROCESS_PORT"]),
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
Path(os.environ["RUN_DIR"], "run_manifest.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
PY

MODEL_ARGS="pretrained=liuhaotian/llava-v1.6-vicuna-7b"

python -m accelerate.commands.launch \
    --num_processes="$NUM_PROCESSES" \
    --main_process_port "$MAIN_PROCESS_PORT" \
    -m lmms_eval \
    --model llava \
    --model_args "$MODEL_ARGS" \
    --tasks pope_pop_audit \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "step2_reference_diagnostic" \
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

mapfile -t SAMPLE_LOGS < <(
    find "$RESULTS_DIR" -type f -name '*_samples_pope_pop_audit.jsonl' -print
)
if [[ ${#SAMPLE_LOGS[@]} -ne 1 ]]; then
    echo "Error: expected one POPE sample log, found ${#SAMPLE_LOGS[@]}" >&2
    exit 1
fi
cp "${SAMPLE_LOGS[0]}" "$REFERENCE_PREDICTIONS"

python "$EVAL_ROOT/audit/validate_step2_reference.py" \
    --reference-predictions "$REFERENCE_PREDICTIONS" \
    --historical-baseline "$HISTORICAL_BASELINE" \
    --expected-manifest "$DIAGNOSTIC_MANIFEST" \
    --expected-count 100 \
    --report "$RUN_DIR/reference_report.json"

printf 'Current-code diagnostic reference passed validation: %s\n' "$RUN_DIR"

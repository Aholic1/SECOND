#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "SECOND_clean" ]]; then
    echo "Error: activate the SECOND_clean conda environment before running this script." >&2
    exit 1
fi

if [[ $# -ne 1 || ( "$1" != "reference" && "$1" != "diagnostic" ) ]]; then
    echo "Usage: bash audit/run_step3_provenance.sh {reference|diagnostic}" >&2
    exit 1
fi

MODE="$1"
EVAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HISTORICAL_BASELINE="$EVAL_ROOT/audit/step1_full/pope_pop_3000_baseline/baseline_predictions.jsonl"
DIAGNOSTIC_MANIFEST="$EVAL_ROOT/audit/step1_diagnostic/diagnostic_manifest.json"
REFERENCE_ROOT="${SECOND_STEP3_REFERENCE_ROOT:-$EVAL_ROOT/audit/runs/step3_reference_v1}"
CAPTURE_ROOT="${SECOND_STEP3_RUN_ROOT:-$EVAL_ROOT/audit/runs/step3_provenance_v1}"
SOURCE_FINGERPRINT="$(
    cd "$EVAL_ROOT"
    sha256sum \
        lmms_eval/models/llava.py \
        lmms_eval/recursion_utils.py \
        lmms_eval/second_audit.py \
        lmms_eval/second_provenance.py \
        lmms_eval/tasks/pope/audit_utils.py \
        lmms_eval/tasks/pope/pope_pop_audit.yaml \
        LLaVA-NeXT/llava/model/llava_arch.py \
        LLaVA-NeXT/llava/model/language_model/llava_llama.py \
        LLaVA-NeXT/llava/model/language_model/llava_recursion.py \
        | sha256sum | cut -d' ' -f1
)"
CURRENT_COMMIT="$(git -C "$EVAL_ROOT" rev-parse HEAD)"

export SECOND_AUDIT_MANIFEST="$DIAGNOSTIC_MANIFEST"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NUM_PROCESSES="${NUM_PROCESSES:-2}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

if [[ "$MODE" == "reference" ]]; then
    RUN_DIR="$REFERENCE_ROOT/diagnostic_$(date -u +%Y%m%dT%H%M%SZ)"
    RESULTS_DIR="$RUN_DIR/results"
    REFERENCE_PREDICTIONS="$RUN_DIR/reference_predictions.jsonl"
    MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29900}"
    mkdir -p "$RESULTS_DIR"

    export EVAL_ROOT RUN_DIR RESULTS_DIR REFERENCE_PREDICTIONS
    export HISTORICAL_BASELINE DIAGNOSTIC_MANIFEST SOURCE_FINGERPRINT
    export CURRENT_COMMIT MAIN_PROCESS_PORT
    python - <<'PY'
import json
import os
from pathlib import Path


payload = {
    "schema_version": 1,
    "purpose": "Step 3 current-code uninstrumented diagnostic reference",
    "task": "pope_pop_audit",
    "expected_count": 100,
    "git_commit": os.environ["CURRENT_COMMIT"],
    "source_fingerprint_sha256": os.environ["SOURCE_FINGERPRINT"],
    "diagnostic_manifest": str(
        Path(os.environ["DIAGNOSTIC_MANIFEST"]).relative_to(os.environ["EVAL_ROOT"])
    ),
    "runtime": {
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
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
        --log_samples_suffix "step3_reference_diagnostic" \
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
    printf 'Step 3 current-code reference passed: %s\n' "$RUN_DIR"
    exit 0
fi

if [[ -n "${SECOND_STEP3_REFERENCE:-}" ]]; then
    REFERENCE_PREDICTIONS="$SECOND_STEP3_REFERENCE"
    REFERENCE_DIR="$(dirname "$REFERENCE_PREDICTIONS")"
else
    REFERENCE_DIR="$(
        find "$REFERENCE_ROOT" -mindepth 1 -maxdepth 1 -type d \
            -name 'diagnostic_*' -printf '%T@ %p\n' \
            | sort -nr | head -n 1 | cut -d' ' -f2-
    )"
    REFERENCE_PREDICTIONS="$REFERENCE_DIR/reference_predictions.jsonl"
fi
if [[ -z "$REFERENCE_DIR" || ! -f "$REFERENCE_PREDICTIONS" ]]; then
    echo "Error: no Step 3 current-code reference found." >&2
    echo "Run: bash audit/run_step3_provenance.sh reference" >&2
    exit 1
fi
python - "$REFERENCE_DIR/run_manifest.json" "$REFERENCE_DIR/reference_report.json" "$CURRENT_COMMIT" "$SOURCE_FINGERPRINT" <<'PY'
import json
import sys
from pathlib import Path


manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
report = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if manifest.get("git_commit") != sys.argv[3]:
    raise SystemExit("reference run commit does not match the current commit")
if manifest.get("source_fingerprint_sha256") != sys.argv[4]:
    raise SystemExit("reference run source fingerprint does not match current files")
if report.get("status") != "PASS":
    raise SystemExit("reference run did not pass validation")
PY

RUN_DIR="$CAPTURE_ROOT/diagnostic_$(date -u +%Y%m%dT%H%M%SZ)"
ARTIFACT_ROOT="$RUN_DIR/artifacts"
RESULTS_DIR="$RUN_DIR/results"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29901}"
mkdir -p "$ARTIFACT_ROOT" "$RESULTS_DIR"
export EVAL_ROOT RUN_DIR ARTIFACT_ROOT RESULTS_DIR REFERENCE_PREDICTIONS
export HISTORICAL_BASELINE DIAGNOSTIC_MANIFEST SOURCE_FINGERPRINT
export CURRENT_COMMIT MAIN_PROCESS_PORT
python - <<'PY'
import json
import os
from pathlib import Path


payload = {
    "schema_version": 1,
    "capture_schema_version": 2,
    "provenance_schema_version": 1,
    "mode": "diagnostic",
    "task": "pope_pop_audit",
    "expected_count": 100,
    "condition": "D",
    "artifact_mode": "visual",
    "audit_token_idx": 0,
    "git_commit": os.environ["CURRENT_COMMIT"],
    "source_fingerprint_sha256": os.environ["SOURCE_FINGERPRINT"],
    "current_code_reference": os.environ["REFERENCE_PREDICTIONS"],
    "diagnostic_manifest": str(
        Path(os.environ["DIAGNOSTIC_MANIFEST"]).relative_to(os.environ["EVAL_ROOT"])
    ),
    "runtime": {
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
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

MODEL_ARGS="pretrained=liuhaotian/llava-v1.6-vicuna-7b,audit_condition=D,audit_output_dir=$ARTIFACT_ROOT,audit_token_idx=0,audit_artifact_mode=visual"
python -m accelerate.commands.launch \
    --num_processes="$NUM_PROCESSES" \
    --main_process_port "$MAIN_PROCESS_PORT" \
    -m lmms_eval \
    --model llava \
    --model_args "$MODEL_ARGS" \
    --tasks pope_pop_audit \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "step3_diagnostic_D" \
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

python "$EVAL_ROOT/audit/validate_step2_capture.py" \
    --artifact-root "$ARTIFACT_ROOT" \
    --baseline-predictions "$HISTORICAL_BASELINE" \
    --reference-predictions "$REFERENCE_PREDICTIONS" \
    --expected-manifest "$DIAGNOSTIC_MANIFEST" \
    --expected-count 100 \
    --condition D \
    --artifact-mode visual \
    --report "$RUN_DIR/step2_compatibility_report.json"

python "$EVAL_ROOT/audit/validate_step3_provenance.py" \
    --artifact-root "$ARTIFACT_ROOT" \
    --baseline-predictions "$HISTORICAL_BASELINE" \
    --reference-predictions "$REFERENCE_PREDICTIONS" \
    --expected-manifest "$DIAGNOSTIC_MANIFEST" \
    --expected-count 100 \
    --condition D \
    --artifact-mode visual \
    --report "$RUN_DIR/validation_report.json"

printf 'Step 3 diagnostic provenance passed: %s\n' "$RUN_DIR"

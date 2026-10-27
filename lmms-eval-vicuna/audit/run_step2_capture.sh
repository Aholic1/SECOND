#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "SECOND_clean" ]]; then
    echo "Error: activate the SECOND_clean conda environment before running this script." >&2
    exit 1
fi

if [[ $# -ne 1 || ( "$1" != "diagnostic" && "$1" != "full" ) ]]; then
    echo "Usage: bash audit/run_step2_capture.sh {diagnostic|full}" >&2
    exit 1
fi

MODE="$1"
EVAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HISTORICAL_BASELINE="$EVAL_ROOT/audit/step1_full/pope_pop_3000_baseline/baseline_predictions.jsonl"
DIAGNOSTIC_MANIFEST="$EVAL_ROOT/audit/step1_diagnostic/diagnostic_manifest.json"
REFERENCE_ROOT="${SECOND_STEP2_REFERENCE_ROOT:-$EVAL_ROOT/audit/runs/step2_reference_v2}"
RUN_ROOT="${SECOND_STEP2_RUN_ROOT:-$EVAL_ROOT/audit/runs/step2_token_selection_v2}"
RUN_DIR="$RUN_ROOT/${MODE}_$(date -u +%Y%m%dT%H%M%SZ)"
ARTIFACT_ROOT="$RUN_DIR/artifacts"
RESULTS_DIR="$RUN_DIR/results"

if [[ "$MODE" == "diagnostic" ]]; then
    TASK="pope_pop_audit"
    EXPECTED_COUNT=100
    ARTIFACT_MODE="visual"
    MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29891}"
    export SECOND_AUDIT_MANIFEST="$DIAGNOSTIC_MANIFEST"

    if [[ -n "${SECOND_STEP2_REFERENCE:-}" ]]; then
        REFERENCE_PREDICTIONS="$SECOND_STEP2_REFERENCE"
        REFERENCE_DIR="$(dirname "$REFERENCE_PREDICTIONS")"
    else
        if [[ ! -d "$REFERENCE_ROOT" ]]; then
            echo "Error: no current-code diagnostic reference found." >&2
            echo "Run: bash audit/run_step2_reference.sh" >&2
            exit 1
        fi
        REFERENCE_DIR="$(
            find "$REFERENCE_ROOT" -mindepth 1 -maxdepth 1 -type d \
                -name 'diagnostic_*' -printf '%T@ %p\n' \
                | sort -nr | head -n 1 | cut -d' ' -f2-
        )"
        REFERENCE_PREDICTIONS="$REFERENCE_DIR/reference_predictions.jsonl"
    fi
    if [[ -z "$REFERENCE_DIR" || ! -f "$REFERENCE_PREDICTIONS" ]]; then
        echo "Error: no current-code diagnostic reference found." >&2
        echo "Run: bash audit/run_step2_reference.sh" >&2
        exit 1
    fi

    CURRENT_COMMIT="$(git -C "$EVAL_ROOT" rev-parse HEAD)"
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
else
    TASK="pope_pop"
    EXPECTED_COUNT=3000
    ARTIFACT_MODE="compact"
    MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29892}"
    unset SECOND_AUDIT_MANIFEST || true
fi

if [[ -e "$RUN_DIR" ]]; then
    echo "Error: run directory already exists: $RUN_DIR" >&2
    exit 1
fi

mkdir -p "$ARTIFACT_ROOT" "$RESULTS_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NUM_PROCESSES="${NUM_PROCESSES:-2}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export MODE TASK EXPECTED_COUNT ARTIFACT_MODE MAIN_PROCESS_PORT
export EVAL_ROOT RUN_DIR ARTIFACT_ROOT RESULTS_DIR HISTORICAL_BASELINE
export DIAGNOSTIC_MANIFEST REFERENCE_PREDICTIONS

python - <<'PY'
import json
import os
import subprocess
from pathlib import Path


eval_root = Path(os.environ["EVAL_ROOT"])
payload = {
    "schema_version": 1,
    "capture_schema_version": 2,
    "mode": os.environ["MODE"],
    "task": os.environ["TASK"],
    "expected_count": int(os.environ["EXPECTED_COUNT"]),
    "condition": "D",
    "artifact_mode": os.environ["ARTIFACT_MODE"],
    "audit_token_idx": 0,
    "git_commit": subprocess.check_output(
        ["git", "-C", str(eval_root), "rev-parse", "HEAD"], text=True
    ).strip(),
    "step1_historical_baseline": str(
        Path(os.environ["HISTORICAL_BASELINE"]).relative_to(eval_root)
    ),
    "current_code_reference": (
        os.environ.get("REFERENCE_PREDICTIONS")
        if os.environ["MODE"] == "diagnostic"
        else None
    ),
    "diagnostic_manifest": (
        str(Path(os.environ["DIAGNOSTIC_MANIFEST"]).relative_to(eval_root))
        if os.environ["MODE"] == "diagnostic"
        else None
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

MODEL_ARGS="pretrained=liuhaotian/llava-v1.6-vicuna-7b,audit_condition=D,audit_output_dir=$ARTIFACT_ROOT,audit_token_idx=0,audit_artifact_mode=$ARTIFACT_MODE"

python -m accelerate.commands.launch \
    --num_processes="$NUM_PROCESSES" \
    --main_process_port "$MAIN_PROCESS_PORT" \
    -m lmms_eval \
    --model llava \
    --model_args "$MODEL_ARGS" \
    --tasks "$TASK" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "step2_${MODE}_D" \
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

VALIDATION_ARGS=(
    --artifact-root "$ARTIFACT_ROOT"
    --baseline-predictions "$HISTORICAL_BASELINE"
    --expected-count "$EXPECTED_COUNT"
    --condition D
    --artifact-mode "$ARTIFACT_MODE"
    --report "$RUN_DIR/validation_report.json"
)
if [[ "$MODE" == "diagnostic" ]]; then
    VALIDATION_ARGS+=(
        --expected-manifest "$DIAGNOSTIC_MANIFEST"
        --reference-predictions "$REFERENCE_PREDICTIONS"
    )
fi

python "$EVAL_ROOT/audit/validate_step2_capture.py" "${VALIDATION_ARGS[@]}"
printf 'Step 2 %s capture passed validation: %s\n' "$MODE" "$RUN_DIR"

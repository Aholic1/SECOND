#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: bash scripts/run_second_audit.sh SOURCE_D_SAMPLE_LOG [OUTPUT_ROOT]" >&2
    exit 2
fi

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source_log=$(readlink -f "$1")
python_bin=${PYTHON_BIN:-python3}
gpu_ids=${CUDA_VISIBLE_DEVICES:-0,1}
main_port=${SECOND_AUDIT_PORT:-29881}
timestamp=$(date -u +%Y%m%d_%H%M%S)
output_root=${2:-"${repo_dir}/audit/runs/${timestamp}"}
output_root=$(readlink -m "$output_root")

if [[ ! -f "$source_log" ]]; then
    echo "Source D sample log does not exist: $source_log" >&2
    exit 2
fi
if [[ "$output_root" == *","* ]]; then
    echo "OUTPUT_ROOT cannot contain a comma because it is passed through --model_args" >&2
    exit 2
fi
if [[ -d "$output_root" ]] && [[ -n "$(find "$output_root" -mindepth 1 -print -quit)" ]]; then
    echo "Refusing to reuse non-empty OUTPUT_ROOT: $output_root" >&2
    exit 2
fi

mkdir -p "$output_root"
manifest="$output_root/pope_pop_seed42_80.json"

cd "$repo_dir"
export PYTHONPATH="${repo_dir}/LLaVA-NeXT:${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"
"$python_bin" tools/second_audit.py sample \
    --samples-jsonl "$source_log" \
    --output "$manifest" \
    --seed 42 \
    --fn 30 \
    --fp 10 \
    --tp 20 \
    --tn 20

export SECOND_AUDIT_MANIFEST="$manifest"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

declare -A sample_logs

run_condition() {
    local condition=$1
    local fix_grid=$2
    local condition_root="$output_root/$condition"
    mkdir -p "$condition_root/results" "$condition_root/artifacts"

    echo "Running SECOND audit condition $condition"
    CUDA_VISIBLE_DEVICES="$gpu_ids" "$python_bin" -m accelerate.commands.launch \
        --num_processes=2 \
        --main_process_port "$main_port" \
        -m lmms_eval \
        --model llava \
        --model_args "pretrained=liuhaotian/llava-v1.6-vicuna-7b,audit_condition=${condition},audit_output_dir=${condition_root}/artifacts,audit_token_idx=0" \
        --tasks pope_pop_audit \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix "second_audit_${condition}" \
        --output_path "$condition_root/results" \
        --generation_type recursion \
        --fix_grid "$fix_grid" \
        --attention_thresholding_type attn_topk \
        --attention_threshold 1.0 \
        --positional_embedding_type bilinear_interpolation \
        --stages "-2" "-1" "0" "1" \
        --contrastive_alphas "0.8" "0.8" "0.8" \
        --seed 42 \
        --verbosity INFO

    mapfile -t found_logs < <(
        find "$condition_root/results" -type f \
            -name '*_samples_pope_pop_audit.jsonl' -print | sort
    )
    if [[ ${#found_logs[@]} -ne 1 ]]; then
        echo "Expected exactly one sample log for condition $condition, found ${#found_logs[@]}" >&2
        exit 1
    fi
    sample_logs[$condition]=${found_logs[0]}
}

run_condition "A" "none"
run_condition "B" "2x2"
run_condition "C" "2x2"
run_condition "D" "2x2"

"$python_bin" tools/second_audit.py parity \
    --manifest "$manifest" \
    --original "$source_log" \
    --new "${sample_logs[D]}" \
    --output "$output_root/d_parity.json"

"$python_bin" tools/second_audit.py analyze \
    --manifest "$manifest" \
    --a "${sample_logs[A]}" \
    --b "${sample_logs[B]}" \
    --c "${sample_logs[C]}" \
    --d "${sample_logs[D]}" \
    --output "$output_root/report"

echo "SECOND audit completed"
echo "Report: $output_root/report/report.md"
echo "Visual artifacts: $output_root/{B,C,D}/artifacts"

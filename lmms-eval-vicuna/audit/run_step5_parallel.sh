#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 || ( "$1" != "1" && "$1" != "2" ) ]]; then
    echo "Usage: bash audit/run_step5_parallel.sh {1|2} STUDY_DIRECTORY [--pair-a A,B] [--pair-b C,D]" >&2
    exit 1
fi
EVAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$EVAL_ROOT"
ROUND="$1"
STUDY="$2"
shift 2
python audit/step5_parallel_repeatability.py round --round "$ROUND" --study "$STUDY" "$@"

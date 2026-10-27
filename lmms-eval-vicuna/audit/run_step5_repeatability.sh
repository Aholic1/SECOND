#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
    echo "Usage: bash audit/run_step5_repeatability.sh STUDY_DIRECTORY" >&2
    exit 1
fi
EVAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$EVAL_ROOT"
python audit/step5_repeatability.py run --study "$1"

#!/bin/bash
set -euo pipefail

# Step 1: Build complete baseline data infrastructure for SECOND visual token analysis
# This script processes the full 3000-sample POPE dataset and creates a diagnostic subset

REPO_ROOT="/home/yxq/Downloads/SECOND-main"
EVAL_ROOT="$REPO_ROOT/lmms-eval-vicuna"
TOOLS="$EVAL_ROOT/tools/second_audit.py"

# Input: original baseline predictions
BASELINE_LOG="$EVAL_ROOT/logs/liuhaotian__llava-v1.6-vicuna-7b/20260810_001849_samples_pope_pop.jsonl"
RESULTS_LOG="$EVAL_ROOT/logs/liuhaotian__llava-v1.6-vicuna-7b/20260810_001849_results.json"

# Output directories
FULL_OUT="$EVAL_ROOT/audit/step1_full/pope_pop_3000_baseline"
DIAG_OUT="$EVAL_ROOT/audit/step1_diagnostic"

# COCO images directory
COCO_DIR="/home/yxq/CAAC64/coco/val2014"

echo "=========================================="
echo "SECOND Step 1: Baseline Data Infrastructure"
echo "=========================================="
echo ""

# Check prerequisites
if [ ! -f "$BASELINE_LOG" ]; then
    echo "Error: baseline log not found: $BASELINE_LOG"
    exit 1
fi

if [ ! -f "$RESULTS_LOG" ]; then
    echo "Error: results file not found: $RESULTS_LOG"
    exit 1
fi

if [ ! -d "$COCO_DIR" ]; then
    echo "Error: COCO directory not found: $COCO_DIR"
    exit 1
fi

echo "Creating output directories..."
mkdir -p "$FULL_OUT"
mkdir -p "$DIAG_OUT"

# Record environment snapshot
echo "Recording environment snapshot..."
cd "$REPO_ROOT"
git rev-parse HEAD > "$FULL_OUT/commit.txt"

python3 --version > "$FULL_OUT/environment.txt"
python3 -c "import torch; print('PyTorch:', torch.__version__)" >> "$FULL_OUT/environment.txt"
python3 -c "import transformers; print('Transformers:', transformers.__version__)" >> "$FULL_OUT/environment.txt"
python3 -c "import accelerate; print('Accelerate:', accelerate.__version__)" >> "$FULL_OUT/environment.txt"
nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 >> "$FULL_OUT/environment.txt"

# Compute immutable input SHA256 values
echo "Computing baseline input SHA256 values..."
sha256sum "$BASELINE_LOG" "$RESULTS_LOG" > "$FULL_OUT/baseline_sha256.txt"

# Copy baseline predictions
echo "Copying baseline predictions..."
cp "$BASELINE_LOG" "$FULL_OUT/baseline_predictions.jsonl"

echo ""
echo "=========================================="
echo "Processing Full Dataset (3000 samples)"
echo "=========================================="
echo ""

# Step 1: Build full index
echo "Step 1/5: Building full index..."
python3 "$TOOLS" index \
    --samples-jsonl "$FULL_OUT/baseline_predictions.jsonl" \
    --output "$FULL_OUT/full_index.jsonl"

# Step 2: Compute summary statistics
echo "Step 2/5: Computing summary statistics..."
python3 "$TOOLS" summarize \
    --index "$FULL_OUT/full_index.jsonl" \
    --output "$FULL_OUT/summary.json"

# Freeze provenance after the normalized baseline and summary exist. The
# environment snapshot describes this audit run, not necessarily the historical
# environment that produced the prediction log.
echo "Writing baseline provenance..."
export REPO_ROOT EVAL_ROOT BASELINE_LOG RESULTS_LOG FULL_OUT
python3 - <<'PY'
import hashlib
import json
import os
import subprocess
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


repo_root = Path(os.environ["REPO_ROOT"])
eval_root = Path(os.environ["EVAL_ROOT"])
baseline_log = Path(os.environ["BASELINE_LOG"])
results_log = Path(os.environ["RESULTS_LOG"])
full_out = Path(os.environ["FULL_OUT"])
results = json.loads(results_log.read_text(encoding="utf-8"))
summary = json.loads((full_out / "summary.json").read_text(encoding="utf-8"))
prediction_commit = subprocess.check_output(
    ["git", "-C", str(repo_root), "rev-parse", results["git_hash"]], text=True
).strip()
audit_commit = subprocess.check_output(
    ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
).strip()
task_config = results["configs"]["pope_pop"]

provenance = {
    "schema_version": 1,
    "prediction_source": str(baseline_log.relative_to(eval_root)),
    "results_source": str(results_log.relative_to(eval_root)),
    "prediction_source_sha256": sha256(baseline_log),
    "results_source_sha256": sha256(results_log),
    "prediction_run_commit": prediction_commit,
    "audit_commit": audit_commit,
    "commit_semantics": {
        "prediction_run_commit": "Commit reported by the historical evaluation results.",
        "audit_commit": "Repository HEAD used while deriving the Step 1 audit artifacts."
    },
    "model": {
        "backend": results["config"]["model"],
        "pretrained": results["model_name"],
        "model_args": results["config"]["model_args"]
    },
    "task": {
        "name": "pope_pop",
        "hash": results["task_hashes"]["pope_pop"],
        "num_fewshot": task_config["num_fewshot"]
    },
    "dataset": {
        "path": task_config["dataset_path"],
        "config": task_config["dataset_name"],
        "split": task_config["test_split"],
        "num_samples": results["n-samples"]["pope_pop"]["effective"]
    },
    "cohort_policy": {
        "quantitative": "All 3000 POPE Popular samples; no sampling.",
        "diagnostic": "100 samples selected by fixed-seed error/complexity strata for visualization and causal analysis.",
        "legacy_smoke_manifest": "audit/manifests/pope_pop_seed42_80.json",
        "legacy_smoke_role": "Smoke/regression checks only; never used for population-level conclusions."
    },
    "run_parameters": {
        "launcher": "accelerate.commands.launch",
        "cuda_visible_devices": "0,1",
        "num_processes": 2,
        "main_process_port": 29881,
        "batch_size": 1,
        "generation_type": "recursion",
        "fix_grid": "2x2",
        "attention_thresholding_type": "attn_topk",
        "attention_threshold": 1.0,
        "positional_embedding_type": "bilinear_interpolation",
        "stages": [-2, -1, 0, 1],
        "contrastive_alphas": [0.8, 0.8, 0.8],
        "generation_kwargs": task_config["generation_kwargs"],
        "seeds": {
            "random": results["config"]["random_seed"],
            "numpy": results["config"]["numpy_seed"],
            "torch": results["config"]["torch_seed"],
            "fewshot": results["config"]["fewshot_seed"]
        }
    },
    "metrics": summary,
    "reported_results": results["results"]["pope_pop"],
    "environment_snapshot": {
        "path": "audit/step1_full/pope_pop_3000_baseline/environment.txt",
        "scope": "Audit/processing environment at artifact generation time; not asserted to be the historical prediction environment."
    }
}
(full_out / "baseline_provenance.json").write_text(
    json.dumps(provenance, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
)
PY

# Step 3: Map images
echo "Step 3/5: Mapping images..."
python3 "$TOOLS" map_images \
    --index "$FULL_OUT/full_index.jsonl" \
    --coco-dir "$COCO_DIR" \
    --output "$FULL_OUT/image_mapping.jsonl"

# Step 4: Extract features
echo "Step 4/5: Extracting features..."
python3 "$TOOLS" extract_features \
    --index "$FULL_OUT/full_index.jsonl" \
    --image-mapping "$FULL_OUT/image_mapping.jsonl" \
    --output "$FULL_OUT/features.jsonl"

echo ""
echo "=========================================="
echo "Creating Diagnostic Subset"
echo "=========================================="
echo ""

# Step 5: Select diagnostic samples
echo "Step 5/5: Selecting diagnostic samples..."
python3 "$TOOLS" select_diagnostic \
    --features "$FULL_OUT/features.jsonl" \
    --strata-config "$DIAG_OUT/strata_config.json" \
    --output "$DIAG_OUT/diagnostic_manifest.json" \
    --seed 42

# Generate diagnostic summary
echo ""
echo "Generating diagnostic summary..."
python3 -c "
import json
from pathlib import Path

manifest = json.loads(Path('$DIAG_OUT/diagnostic_manifest.json').read_text())
samples = manifest['samples']

summary = {
    'total': len(samples),
    'by_stratum': manifest['actual_counts'],
    'unique_images': len(set(s['image_id'] for s in samples)),
    'readable_images': sum(1 for s in samples if s.get('image_readable', False)),
}

Path('$DIAG_OUT/diagnostic_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(f\"Diagnostic samples: {summary['total']}\")
print(f\"Unique images: {summary['unique_images']}\")
print(f\"Readable images: {summary['readable_images']}\")
"

echo ""
echo "=========================================="
echo "Step 1 Complete"
echo "=========================================="
echo ""
echo "Output locations:"
echo "  Full dataset (3000): $FULL_OUT"
echo "  Diagnostic subset:   $DIAG_OUT"
echo ""
echo "Next steps:"
echo "  - Review summary.json for full dataset statistics"
echo "  - Review diagnostic_manifest.json for selected samples"
echo "  - Proceed to Step 2: Record SECOND token selection"
echo ""

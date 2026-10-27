#!/usr/bin/env python3
"""Validate the frozen Step 1 baseline and its provenance."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path


AUDIT = Path(__file__).resolve().parent
EVAL_ROOT = AUDIT.parent
REPO_ROOT = EVAL_ROOT.parent
FULL = AUDIT / "step1_full" / "pope_pop_3000_baseline"
DIAG = AUDIT / "step1_diagnostic"
EXPECTED_COUNTS = {"TP": 1270, "TN": 1420, "FN": 230, "FP": 80}
EXPECTED_PREDICTION_SHA256 = "0c8a71ef7bda1048dbc4649e0207eeb241a42492458a7ce10080ef5a0d5a5cc3"
EXPECTED_RESULTS_SHA256 = "0e9f8866ec5efae81dfb34b02717ca96db0d95d619c59e45fe09b71c36c8ad77"
EXPECTED_PREDICTION_COMMIT = "4ad65872d9c03ea7b60ea68c2b663d22a373ec33"
EXPECTED_AUDIT_COMMIT = "a28e0a5ef17868f3c3058f382d486a10b3c47688"
EXPECTED_TASK_HASH = "b5aa871bc6960e8d3d209d20d5308b27d69ae7ffd5da5d7db6804a6c1569e3a6"


def report(ok: bool, message: str) -> bool:
    print(("PASS" if ok else "FAIL") + " " + message)
    return ok


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_code_structure() -> bool:
    checks = []
    tool = EVAL_ROOT / "tools" / "second_audit.py"
    try:
        compile(tool.read_text(encoding="utf-8"), str(tool), "exec")
        checks.append(report(True, "second_audit.py syntax"))
    except Exception as exc:
        checks.append(report(False, f"second_audit.py syntax: {exc}"))

    for module in ("PIL", "hashlib", "json", "argparse", "pathlib"):
        try:
            __import__("PIL.Image" if module == "PIL" else module)
            checks.append(report(True, f"import {module}"))
        except ImportError as exc:
            checks.append(report(False, f"import {module}: {exc}"))

    text = tool.read_text(encoding="utf-8")
    for command, function in (
        ("index", "build_full_index"),
        ("summarize", "summarize_index"),
        ("map_images", "map_images"),
        ("extract_features", "extract_features"),
        ("select_diagnostic", "select_diagnostic_samples"),
    ):
        checks.append(report(f"def {function}(" in text, f"command/function {command}"))

    for path in (
        AUDIT / "STEP1_IMPLEMENTATION.md",
        AUDIT / "step1_run.sh",
        AUDIT / "test_step1.py",
        DIAG / "strata_config.json",
        AUDIT / "step1_full" / "README.md",
        DIAG / "README.md",
        FULL / "baseline_provenance.json",
    ):
        checks.append(report(path.is_file(), f"required file {path.relative_to(REPO_ROOT)}"))
    return all(checks)


def check_data_integrity() -> bool:
    checks = []
    source = EVAL_ROOT / "logs" / "liuhaotian__llava-v1.6-vicuna-7b" / "20260810_001849_samples_pope_pop.jsonl"
    copied = FULL / "baseline_predictions.jsonl"
    index = read_jsonl(FULL / "full_index.jsonl")
    mapping = read_jsonl(FULL / "image_mapping.jsonl")
    features = read_jsonl(FULL / "features.jsonl")
    baseline = read_jsonl(copied)

    checks.append(report(len(baseline) == 3000, "baseline_predictions has 3000 rows"))
    checks.append(report(len(index) == 3000, "full_index has 3000 rows"))
    checks.append(report(len(mapping) == 3000, "image_mapping has 3000 rows"))
    checks.append(report(len(features) == 3000, "features has 3000 rows"))

    def ids(rows):
        return [str(row["question_id"]) for row in rows]

    baseline_ids = [str(row["pope_accuracy"]["question_id"]) for row in baseline]
    index_ids, mapping_ids, feature_ids = ids(index), ids(mapping), ids(features)
    checks.append(report(len(set(baseline_ids)) == 3000, "raw prediction question IDs are unique"))
    checks.append(report(len(set(index_ids)) == 3000, "question_id values are unique"))
    checks.append(report(
        set(baseline_ids) == set(index_ids) == set(mapping_ids) == set(feature_ids),
        "prediction/index/mapping/features IDs align",
    ))

    counts = {}
    mismatch = 0
    for row in index:
        counts[row["stratum"]] = counts.get(row["stratum"], 0) + 1
        if row.get("correct") != (row["gt"] == row["pred"]):
            mismatch += 1
    checks.append(report(mismatch == 0, "correctness fields agree with gt/pred"))
    checks.append(report(counts == EXPECTED_COUNTS, f"stratum counts are {EXPECTED_COUNTS}"))

    summary = json.loads((FULL / "summary.json").read_text(encoding="utf-8"))
    checks.append(report(summary.get("total") == 3000, "summary total is 3000"))
    checks.append(report(summary.get("counts") == EXPECTED_COUNTS, "summary counts match index"))
    expected_metrics = {
        "accuracy": 0.8966666666666666,
        "precision": 0.9407407407407408,
        "recall": 0.8466666666666667,
        "f1": 0.8912280701754387,
        "yes_ratio": 0.45,
    }
    checks.append(report(
        all(math.isclose(summary.get(key, -1), value, rel_tol=0, abs_tol=1e-12)
            for key, value in expected_metrics.items()),
        "summary metrics match the frozen baseline",
    ))

    unreadable = [row for row in mapping if row.get("readable") is not True]
    missing = [row for row in mapping if not Path(row.get("image_path", "")).is_file()]
    bad_dims = [
        row
        for row in mapping
        if not isinstance(row.get("width"), int)
        or not isinstance(row.get("height"), int)
        or row["width"] <= 0
        or row["height"] <= 0
    ]
    checks.append(report(not unreadable, "all mapped images are readable"))
    checks.append(report(not missing, "all mapped image files exist"))
    checks.append(report(not bad_dims, "all mapped image dimensions are valid"))

    results = EVAL_ROOT / "logs" / "liuhaotian__llava-v1.6-vicuna-7b" / "20260810_001849_results.json"
    recorded_hashes = {
        line.split()[0]
        for line in (FULL / "baseline_sha256.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    prediction_hash = sha256(source)
    results_hash = sha256(results)
    checks.append(report(
        prediction_hash == sha256(copied) == EXPECTED_PREDICTION_SHA256,
        "prediction SHA256 matches source and copied log",
    ))
    checks.append(report(results_hash == EXPECTED_RESULTS_SHA256, "results SHA256 matches frozen value"))
    checks.append(report(
        {prediction_hash, results_hash}.issubset(recorded_hashes),
        "baseline_sha256.txt records both immutable inputs",
    ))
    return all(checks)


def check_manifest() -> bool:
    checks = []
    config = json.loads((DIAG / "strata_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((DIAG / "diagnostic_manifest.json").read_text(encoding="utf-8"))
    samples = manifest.get("samples", [])
    checks.append(report(len(samples) == sum(config.values()) == 100, "diagnostic manifest has 100 samples"))
    actual = {}
    for sample in samples:
        key = sample.get("visual_stratum")
        actual[key] = actual.get(key, 0) + 1
    checks.append(report(actual == config, "diagnostic manifest matches strata configuration"))
    checks.append(report(len({str(sample["question_id"]) for sample in samples}) == 100, "diagnostic question IDs are unique"))
    checks.append(report(all(sample.get("image_readable") is True for sample in samples), "diagnostic images are readable"))
    smoke = json.loads((AUDIT / "manifests" / "pope_pop_seed42_80.json").read_text(encoding="utf-8"))
    checks.append(report(
        smoke.get("seed") == 42 and len(smoke.get("samples", [])) == 80,
        "legacy smoke/regression manifest retains 80 seed-42 samples",
    ))
    return all(checks)


def check_provenance() -> bool:
    provenance = json.loads((FULL / "baseline_provenance.json").read_text(encoding="utf-8"))
    required = {
        "prediction_source",
        "results_source",
        "prediction_source_sha256",
        "results_source_sha256",
        "prediction_run_commit",
        "audit_commit",
        "model",
        "task",
        "run_parameters",
        "dataset",
        "metrics",
    }
    missing = sorted(required - set(provenance))
    checks = [report(not missing, "provenance contains required fields")]
    if not missing:
        source = EVAL_ROOT / provenance["prediction_source"]
        results = EVAL_ROOT / provenance["results_source"]
        summary = json.loads((FULL / "summary.json").read_text(encoding="utf-8"))
        checks.append(report(
            provenance["prediction_source_sha256"] == sha256(source) == EXPECTED_PREDICTION_SHA256,
            "provenance prediction SHA256 is valid",
        ))
        checks.append(report(
            provenance["results_source_sha256"] == sha256(results) == EXPECTED_RESULTS_SHA256,
            "provenance results SHA256 is valid",
        ))
        checks.append(report(
            provenance["prediction_run_commit"] == EXPECTED_PREDICTION_COMMIT,
            "provenance identifies the historical prediction commit",
        ))
        checks.append(report(
            provenance["audit_commit"] == EXPECTED_AUDIT_COMMIT
            and provenance["audit_commit"] == (FULL / "commit.txt").read_text(encoding="utf-8").strip(),
            "provenance identifies the audit-artifact base commit",
        ))
        checks.append(report(
            provenance["model"].get("pretrained") == "liuhaotian/llava-v1.6-vicuna-7b",
            "provenance identifies the evaluated model",
        ))
        checks.append(report(
            provenance["task"].get("name") == "pope_pop"
            and provenance["task"].get("hash") == EXPECTED_TASK_HASH,
            "provenance identifies the task and task hash",
        ))
        checks.append(report(
            provenance["dataset"] == {
                "path": "lmms-lab/POPE",
                "config": "Full",
                "split": "popular",
                "num_samples": 3000,
            },
            "provenance records the complete POPE Popular cohort",
        ))
        checks.append(report(provenance["metrics"] == summary, "provenance metrics match summary.json"))
        policy = provenance.get("cohort_policy", {})
        checks.append(report(
            policy.get("legacy_smoke_manifest") == "audit/manifests/pope_pop_seed42_80.json"
            and "3000" in policy.get("quantitative", "")
            and "Smoke/regression" in policy.get("legacy_smoke_role", ""),
            "provenance separates the 3000-sample cohort from the 80-sample smoke set",
        ))
        environment = provenance.get("environment_snapshot", {})
        checks.append(report(
            environment.get("path") == "audit/step1_full/pope_pop_3000_baseline/environment.txt"
            and "not asserted" in environment.get("scope", ""),
            "provenance scopes the audit environment separately",
        ))
    return all(checks)


def main() -> int:
    print("SECOND Step 1 frozen-baseline validation")
    checks = [check_code_structure(), check_data_integrity(), check_manifest(), check_provenance()]
    print("RESULT " + ("PASS" if all(checks) else "FAIL"))
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

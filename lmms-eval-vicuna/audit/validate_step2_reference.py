#!/usr/bin/env python3
"""Validate a current-code, uninstrumented Step 2 diagnostic reference run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_predictions(path):
    records = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            metric = json.loads(line)["pope_accuracy"]
            question_id = str(metric["question_id"])
            if question_id in records:
                raise ValueError(f"duplicate question_id: {question_id}")
            records[question_id] = {
                "prediction": str(metric["prediction"]).lower(),
                "ground_truth": str(metric["ground_truth"]).lower(),
            }
    return records


def read_manifest_ids(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = [str(row["question_id"]) for row in payload.get("samples", [])]
    if len(ids) != len(set(ids)):
        raise ValueError("diagnostic manifest has duplicate question IDs")
    return set(ids)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-predictions", required=True, type=Path)
    parser.add_argument("--historical-baseline", required=True, type=Path)
    parser.add_argument("--expected-manifest", required=True, type=Path)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    errors = []
    warnings = []
    reference = read_predictions(args.reference_predictions)
    historical = read_predictions(args.historical_baseline)
    expected_ids = read_manifest_ids(args.expected_manifest)

    if len(expected_ids) != args.expected_count:
        errors.append(
            f"manifest contains {len(expected_ids)} IDs, expected {args.expected_count}"
        )
    reference_ids = set(reference)
    missing = sorted(expected_ids - reference_ids)
    unexpected = sorted(reference_ids - expected_ids)
    if missing:
        errors.append(f"missing reference IDs: {missing[:20]} (total {len(missing)})")
    if unexpected:
        errors.append(
            f"unexpected reference IDs: {unexpected[:20]} (total {len(unexpected)})"
        )
    if len(reference) != args.expected_count:
        errors.append(
            f"reference contains {len(reference)} samples, expected {args.expected_count}"
        )

    historical_mismatches = []
    for question_id in sorted(expected_ids & reference_ids):
        old = historical.get(question_id)
        if old is None:
            errors.append(f"historical baseline is missing question_id {question_id}")
            continue
        current = reference[question_id]
        if current["ground_truth"] != old["ground_truth"]:
            errors.append(f"ground truth changed for question_id {question_id}")
        if current["prediction"] != old["prediction"]:
            historical_mismatches.append(
                {
                    "question_id": question_id,
                    "historical": old["prediction"],
                    "current_reference": current["prediction"],
                }
            )
    if historical_mismatches:
        warnings.append(
            f"{len(historical_mismatches)} current predictions differ from the "
            "historical Step 1 baseline"
        )

    report = {
        "status": "PASS" if not errors else "FAIL",
        "expected_count": args.expected_count,
        "reference_count": len(reference),
        "historical_prediction_mismatch_count": len(historical_mismatches),
        "historical_prediction_mismatches": historical_mismatches,
        "warnings": warnings,
        "errors": errors,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

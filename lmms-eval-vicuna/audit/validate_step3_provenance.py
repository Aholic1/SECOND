#!/usr/bin/env python3
"""Validate Step 3 visual-token provenance and spatial mappings."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import zipfile

import numpy as np


EXPECTED_FORWARD_STAGES = [-2, -1, 0, 1]
EXPECTED_SOURCES = {
    -2: "low_res_base",
    -1: "low_res_base",
    0: "base",
    1: "anyres_tile",
}
REQUIRED_RECORD_FIELDS = {
    "sample_id",
    "forward_stage",
    "stage_internal",
    "stage_order",
    "source_type",
    "tile_id",
    "tile_row",
    "tile_col",
    "patch_row",
    "patch_col",
    "candidate_index",
    "selected",
    "mask_selected",
    "stage_sequence_index",
    "visual_sequence_index",
    "decoder_sequence_index",
    "preprocess_rect",
    "canvas_rect",
    "original_rect_clipped",
    "padding_fraction",
    "is_cls",
    "is_newline",
    "is_truncated",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return records


def read_predictions(path):
    records = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            metric = json.loads(line)["pope_accuracy"]
            question_id = str(metric["question_id"])
            if question_id in records:
                raise ValueError(f"duplicate prediction question_id: {question_id}")
            records[question_id] = {
                "prediction": str(metric["prediction"]).lower(),
                "ground_truth": str(metric["ground_truth"]).lower(),
            }
    return records


def manifest_ids(path):
    payload = read_json(path)
    ids = [str(row["question_id"]) for row in payload.get("samples", [])]
    if len(ids) != len(set(ids)):
        raise ValueError("diagnostic manifest contains duplicate question IDs")
    return set(ids)


def _valid_rect(rect, width, height):
    return (
        isinstance(rect, list)
        and len(rect) == 4
        and all(isinstance(value, (int, float)) and math.isfinite(value) for value in rect)
        and 0 <= rect[0] <= rect[2] <= width
        and 0 <= rect[1] <= rect[3] <= height
    )


def validate_forward(sample_id, sample_dir, transform, forward, artifact_mode, errors):
    stage = forward.get("forward_stage")
    if stage not in EXPECTED_FORWARD_STAGES:
        errors.append(f"{sample_dir}: unexpected forward stage {stage}")
        return
    provenance_path = (
        sample_dir / f"token_000_forward_stage_{stage}_provenance.jsonl"
    )
    trace_path = (
        sample_dir
        / f"token_000_forward_stage_{stage}_provenance_runtime_trace.npz"
    )
    if not provenance_path.is_file():
        errors.append(f"{sample_dir}: missing {provenance_path.name}")
        return
    if not trace_path.is_file():
        errors.append(f"{sample_dir}: missing {trace_path.name}")
        return
    try:
        records = read_jsonl(provenance_path)
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
        return
    try:
        with np.load(trace_path, allow_pickle=False) as archive:
            trace_arrays = {name: archive[name] for name in archive.files}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        errors.append(f"{trace_path}: cannot read runtime trace: {exc}")
        return

    if forward.get("runtime_offline_match") is not True:
        errors.append(f"{sample_dir}: forward stage {stage} runtime/offline mismatch")
    if forward.get("mm_patch_merge_type") != "spatial":
        errors.append(f"{sample_dir}: forward stage {stage} did not use spatial merge")
    if forward.get("image_aspect_ratio") != "anyres":
        errors.append(f"{sample_dir}: forward stage {stage} did not use AnyRes")
    if forward.get("encoded_image_count") != forward.get(
        "expected_encoded_image_count"
    ):
        errors.append(f"{sample_dir}: forward stage {stage} tile count is inconsistent")
    if forward.get("cls_token_count") != 0 or forward.get("newline_token_count") != 0:
        errors.append(f"{sample_dir}: forward stage {stage} contains special visual tokens")
    if forward.get("decoder_truncated_visual_count") != 0:
        errors.append(f"{sample_dir}: forward stage {stage} truncated visual tokens")
    if len(records) != forward.get("candidate_count"):
        errors.append(f"{provenance_path}: candidate count disagrees with summary")

    canvas = transform["model_input_size"]
    original = transform["original_size"]
    canvas_width, canvas_height = canvas["width"], canvas["height"]
    if forward.get("canvas_size") != [canvas_width, canvas_height]:
        errors.append(f"{sample_dir}: forward stage {stage} canvas size differs")
    seen = set()
    by_source = {}
    selected_records = []
    for index, record in enumerate(records):
        missing = REQUIRED_RECORD_FIELDS - set(record)
        if missing:
            errors.append(f"{provenance_path}:{index + 1}: missing {sorted(missing)}")
            continue
        if str(record["sample_id"]) != sample_id or record["forward_stage"] != stage:
            errors.append(f"{provenance_path}:{index + 1}: sample/forward identity mismatch")
        key = (record["stage_internal"], record["candidate_index"])
        if key in seen:
            errors.append(f"{provenance_path}:{index + 1}: duplicate candidate {key}")
        seen.add(key)
        by_source.setdefault(record["stage_internal"], []).append(record)

        if record["is_cls"] or record["is_newline"]:
            errors.append(f"{provenance_path}:{index + 1}: unexpected special token")
        if not _valid_rect(record["canvas_rect"], canvas_width, canvas_height):
            errors.append(f"{provenance_path}:{index + 1}: invalid canvas_rect")
        if not _valid_rect(
            record["original_rect_clipped"], original["width"], original["height"]
        ):
            errors.append(
                f"{provenance_path}:{index + 1}: invalid original_rect_clipped"
            )
        fraction = record["padding_fraction"]
        if not isinstance(fraction, (int, float)) or not 0 <= fraction <= 1:
            errors.append(f"{provenance_path}:{index + 1}: invalid padding_fraction")
        if record["selected"]:
            selected_records.append(record)
            if record["stage_sequence_index"] < 0 or record["visual_sequence_index"] < 0:
                errors.append(f"{provenance_path}:{index + 1}: selected index is negative")
        elif (
            record["stage_sequence_index"] != -1
            or record["visual_sequence_index"] != -1
            or record["decoder_sequence_index"] != -1
            or record["is_truncated"]
        ):
            errors.append(f"{provenance_path}:{index + 1}: rejected token has sequence state")

    source_summaries = forward.get("sources", [])
    if [item.get("stage_internal") for item in source_summaries] != EXPECTED_FORWARD_STAGES:
        errors.append(f"{sample_dir}: forward stage {stage} source order is incorrect")
    for source_order, source_stage in enumerate(EXPECTED_FORWARD_STAGES):
        source_records = by_source.get(source_stage, [])
        source_type = EXPECTED_SOURCES[source_stage]
        source_summary = (
            source_summaries[source_order]
            if source_order < len(source_summaries)
            else {}
        )
        stage_label = str(source_stage).replace("-", "neg")
        grid_shape = source_summary.get("grid_shape")
        if (
            not isinstance(grid_shape, list)
            or len(grid_shape) != 2
            or any(type(value) is not int or value <= 0 for value in grid_shape)
        ):
            errors.append(f"{provenance_path}: stage {source_stage} grid shape is invalid")
            continue
        mask_key = f"stage_{stage_label}_consumed_mask"
        runtime_key = f"stage_{stage_label}_runtime_keep_indices"
        if mask_key not in trace_arrays or runtime_key not in trace_arrays:
            errors.append(f"{trace_path}: missing stage {source_stage} runtime arrays")
            continue
        consumed_mask = trace_arrays[mask_key]
        runtime_indices = trace_arrays[runtime_key]
        if list(consumed_mask.shape) != grid_shape or not np.isin(
            consumed_mask, (0, 1)
        ).all():
            errors.append(f"{trace_path}: stage {source_stage} consumed mask is invalid")
            continue
        reconstructed_indices = np.flatnonzero(consumed_mask.reshape(-1)).astype(
            np.int32
        )
        if not np.array_equal(reconstructed_indices, runtime_indices):
            errors.append(
                f"{trace_path}: stage {source_stage} runtime append order differs "
                "from consumed mask"
            )
        candidate_count = grid_shape[0] * grid_shape[1]
        if source_summary.get("runtime_mask_matches_reconstruction") is not True:
            errors.append(f"{provenance_path}: stage {source_stage} runtime mask differs")
        if len(source_records) != candidate_count:
            errors.append(
                f"{provenance_path}: stage {source_stage} has {len(source_records)} "
                f"candidates, expected {candidate_count}"
            )
            continue
        if any(
            record["source_type"] != source_type
            or record["stage_order"] != source_order
            for record in source_records
        ):
            errors.append(f"{provenance_path}: stage {source_stage} source metadata differs")
        if [record["candidate_index"] for record in source_records] != list(
            range(candidate_count)
        ):
            errors.append(f"{provenance_path}: stage {source_stage} candidates are not row-major")
        selected = [record for record in source_records if record["selected"]]
        mask_selected = [
            record for record in source_records if record["mask_selected"]
        ]
        if [record["candidate_index"] for record in mask_selected] != (
            reconstructed_indices.tolist()
        ):
            errors.append(
                f"{provenance_path}: stage {source_stage} mask-selected rows differ "
                "from raw consumed mask"
            )
        if [record["candidate_index"] for record in selected] != runtime_indices.tolist():
            errors.append(
                f"{provenance_path}: stage {source_stage} selected rows differ "
                "from raw runtime trace"
            )
        if (
            source_summary.get("candidate_count") != candidate_count
            or source_summary.get("selected_count") != len(selected)
            or source_summary.get("mask_selected_count") != len(mask_selected)
            or source_summary.get("runtime_feature_count") != len(selected)
        ):
            errors.append(f"{provenance_path}: stage {source_stage} count summary differs")
        if [record["stage_sequence_index"] for record in selected] != list(
            range(len(selected))
        ):
            errors.append(f"{provenance_path}: stage {source_stage} sequence is not continuous")

    visual_length = forward.get("runtime_visual_length")
    if len(selected_records) != forward.get("selected_count") or len(
        selected_records
    ) != visual_length:
        errors.append(f"{provenance_path}: selected count differs from runtime length")
    if [record["visual_sequence_index"] for record in selected_records] != list(
        range(len(selected_records))
    ):
        errors.append(f"{provenance_path}: visual sequence indices are not continuous")
    if any(record["is_truncated"] for record in selected_records):
        errors.append(f"{provenance_path}: selected token was truncated")
    decoder_start = forward.get("decoder_visual_start")
    if [record["decoder_sequence_index"] for record in selected_records] != list(
        range(decoder_start, decoder_start + len(selected_records))
    ):
        errors.append(f"{provenance_path}: decoder sequence indices are not continuous")

    if artifact_mode == "visual":
        stem = f"token_000_forward_stage_{stage}_provenance"
        expected_images = [sample_dir / f"{stem}_geometry.png"] + [
            sample_dir / f"{stem}_source_stage_{source_stage}_selection.png"
            for source_stage in EXPECTED_FORWARD_STAGES
        ]
        for path in expected_images:
            if not path.is_file():
                errors.append(f"{sample_dir}: missing {path.name}")


def validate_sample(sample_id, sample, sample_dir, artifact_mode, errors):
    summary_path = sample_dir / "provenance_summary.json"
    if not summary_path.is_file():
        errors.append(f"{sample_dir}: missing provenance_summary.json")
        return
    try:
        summary = read_json(summary_path)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{summary_path}: cannot read summary: {exc}")
        return
    if summary.get("provenance_schema_version") != 1:
        errors.append(f"{summary_path}: unsupported provenance schema")
    if summary.get("token_idx") != 0:
        errors.append(f"{summary_path}: expected token_idx 0")
    if summary.get("all_runtime_offline_match") is not True:
        errors.append(f"{summary_path}: runtime/offline comparison failed")
    if summary.get("total_truncated_visual_tokens") != 0:
        errors.append(f"{summary_path}: visual tokens were truncated")
    forwards = summary.get("forward_stages", [])
    if [item.get("forward_stage") for item in forwards] != EXPECTED_FORWARD_STAGES:
        errors.append(f"{summary_path}: expected forward stages {EXPECTED_FORWARD_STAGES}")
        return
    transform = sample.get("image_transform", {})
    try:
        for key in ("original_size", "model_input_size", "padding"):
            if key not in transform:
                raise KeyError(key)
        for forward in forwards:
            validate_forward(
                sample_id, sample_dir, transform, forward, artifact_mode, errors
            )
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"{sample_dir}: malformed transform/provenance: {exc}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--baseline-predictions", required=True, type=Path)
    parser.add_argument("--reference-predictions", required=True, type=Path)
    parser.add_argument("--expected-manifest", required=True, type=Path)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--condition", default="D")
    parser.add_argument("--artifact-mode", choices=("visual", "compact"), required=True)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    errors = []
    warnings = []
    expected_ids = manifest_ids(args.expected_manifest)
    reference = read_predictions(args.reference_predictions)
    historical = read_predictions(args.baseline_predictions)
    if len(expected_ids) != args.expected_count:
        errors.append(
            f"manifest contains {len(expected_ids)} IDs, expected {args.expected_count}"
        )
    if set(reference) != expected_ids:
        errors.append("current-code reference IDs differ from diagnostic manifest")

    sample_paths = sorted((args.artifact_root / args.condition).glob("rank_*/*/sample.json"))
    samples = {}
    for sample_path in sample_paths:
        sample = read_json(sample_path)
        question_id = str(sample.get("question_id", sample.get("sample_id", "")))
        if question_id in samples:
            errors.append(f"duplicate captured question_id: {question_id}")
        else:
            samples[question_id] = (sample, sample_path.parent)
    if set(samples) != expected_ids:
        errors.append("captured IDs differ from diagnostic manifest")
    if len(samples) != args.expected_count:
        errors.append(f"captured {len(samples)} samples, expected {args.expected_count}")

    reference_mismatches = []
    historical_mismatches = []
    for question_id, (sample, sample_dir) in samples.items():
        validate_sample(question_id, sample, sample_dir, args.artifact_mode, errors)
        current = reference.get(question_id)
        if current is not None and sample.get("prediction") != current["prediction"]:
            reference_mismatches.append(
                {
                    "question_id": question_id,
                    "reference": current["prediction"],
                    "captured": sample.get("prediction"),
                }
            )
        old = historical.get(question_id)
        if old is not None and sample.get("prediction") != old["prediction"]:
            historical_mismatches.append(
                {
                    "question_id": question_id,
                    "baseline": old["prediction"],
                    "captured": sample.get("prediction"),
                }
            )
    if reference_mismatches:
        errors.append(
            f"{len(reference_mismatches)} predictions differ from current-code reference"
        )
    if historical_mismatches:
        warnings.append(
            f"{len(historical_mismatches)} predictions differ from historical Step 1 baseline"
        )

    report = {
        "status": "PASS" if not errors else "FAIL",
        "condition": args.condition,
        "artifact_mode": args.artifact_mode,
        "expected_count": args.expected_count,
        "captured_count": len(samples),
        "reference_prediction_mismatch_count": len(reference_mismatches),
        "reference_prediction_mismatches": reference_mismatches,
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

#!/usr/bin/env python3
"""Validate SECOND v2 token-selection captures against the frozen baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image


EXPECTED_TRANSITIONS = {
    (-2, -1): (12, 12),
    (-1, 0): (24, 24),
    (0, 1): (48, 48),
}
EXPECTED_SELECTION_SHAPE = (48, 48)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_baseline(path):
    records = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            metric = row["pope_accuracy"]
            question_id = str(metric["question_id"])
            if question_id in records:
                raise ValueError(f"duplicate baseline question_id: {question_id}")
            records[question_id] = {
                "prediction": str(metric["prediction"]).lower(),
                "ground_truth": str(metric["ground_truth"]).lower(),
            }
    return records


def manifest_ids(path):
    payload = read_json(path)
    ids = [str(row["question_id"]) for row in payload.get("samples", [])]
    if len(ids) != len(set(ids)):
        raise ValueError("expected manifest has duplicate question IDs")
    return set(ids)


def nearest_resize(mask, output_shape):
    if mask.ndim != 2:
        raise ValueError(f"source mask must be 2D, got {mask.shape}")
    if len(output_shape) != 2 or any(size <= 0 for size in output_shape):
        raise ValueError(f"output shape must contain two positive dimensions: {output_shape}")
    source_height, source_width = mask.shape
    output_height, output_width = output_shape
    rows = np.floor(
        np.arange(output_height, dtype=np.float64) * source_height / output_height
    ).astype(np.int64)
    columns = np.floor(
        np.arange(output_width, dtype=np.float64) * source_width / output_width
    ).astype(np.int64)
    return mask[rows[:, None], columns[None, :]]


def is_binary(mask):
    return mask.ndim == 2 and np.isin(mask, (0, 1)).all()


def validate_transform(sample, sample_dir, artifact_mode, errors):
    transform = sample.get("image_transform", {})
    original = transform.get("original_size", {})
    model_input = transform.get("model_input_size", {})
    padding = transform.get("padding", {})
    values = [
        original.get("width"),
        original.get("height"),
        model_input.get("width"),
        model_input.get("height"),
        padding.get("left"),
        padding.get("top"),
        padding.get("right"),
        padding.get("bottom"),
    ]
    if not all(isinstance(value, int) for value in values):
        errors.append(f"{sample_dir}: incomplete image_transform")
        return
    if original["width"] + padding["left"] + padding["right"] != model_input["width"]:
        errors.append(f"{sample_dir}: horizontal image transform is inconsistent")
    if original["height"] + padding["top"] + padding["bottom"] != model_input["height"]:
        errors.append(f"{sample_dir}: vertical image transform is inconsistent")
    if artifact_mode == "visual":
        for name, expected_size in (
            ("original_image.png", (original["width"], original["height"])),
            ("input_image.png", (model_input["width"], model_input["height"])),
        ):
            path = sample_dir / name
            if not path.is_file():
                errors.append(f"{sample_dir}: missing {name}")
                continue
            try:
                with Image.open(path) as image:
                    if image.size != expected_size:
                        errors.append(
                            f"{path}: expected image size {expected_size}, got {image.size}"
                        )
            except (OSError, ValueError) as exc:
                errors.append(f"{path}: cannot read image: {exc}")


def validate_transition(
    sample_dir, source_stage, target_stage, artifact_mode, errors, condition="D"
):
    stem = f"token_000_stage_{source_stage}_to_{target_stage}"
    metadata_path = sample_dir / f"{stem}.json"
    arrays_path = sample_dir / f"{stem}.npz"
    if not metadata_path.is_file() or not arrays_path.is_file():
        errors.append(f"{sample_dir}: missing transition {stem}")
        return

    try:
        metadata = read_json(metadata_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"{metadata_path}: cannot read metadata: {exc}")
        return
    if metadata.get("audit_schema_version") != 2:
        errors.append(f"{metadata_path}: expected audit schema 2")
    if metadata.get("source_stage") != source_stage or metadata.get("target_stage") != target_stage:
        errors.append(f"{metadata_path}: stage metadata mismatch")
    if metadata.get("condition") != condition:
        errors.append(f"{metadata_path}: condition mismatch")
    if metadata.get("token_idx") != 0:
        errors.append(f"{metadata_path}: expected token_idx 0")
    if metadata.get("consumed_matches_applied_mask") is not True:
        errors.append(f"{metadata_path}: consumed-mask verification flag is missing")

    required = {
        "ret_attn",
        "mask_before",
        "proposed_mask",
        "applied_mask",
        "consumed_mask",
        "consumed_keep_indices",
        "consumed_drop_indices",
        "consumed_keep_coords_rc",
        "consumed_drop_coords_rc",
    }
    try:
        with np.load(arrays_path, allow_pickle=False) as archive:
            missing = required - set(archive.files)
            if missing:
                errors.append(f"{arrays_path}: missing arrays {sorted(missing)}")
                return
            arrays = {name: archive[name] for name in required}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        errors.append(f"{arrays_path}: cannot read arrays: {exc}")
        return

    if not all(np.isfinite(value).all() for value in arrays.values()):
        errors.append(f"{arrays_path}: contains non-finite values")
    binary_masks = {}
    for name in ("mask_before", "proposed_mask", "applied_mask", "consumed_mask"):
        binary_masks[name] = is_binary(arrays[name])
        if not binary_masks[name]:
            errors.append(f"{arrays_path}: {name} is not a binary 2D mask")

    selection_shape = (
        arrays["proposed_mask"].shape if binary_masks["proposed_mask"] else None
    )
    if selection_shape is not None and selection_shape != EXPECTED_SELECTION_SHAPE:
        errors.append(
            f"{arrays_path}: expected selection grid {EXPECTED_SELECTION_SHAPE}, "
            f"got {selection_shape}"
        )
    if arrays["ret_attn"].ndim != 2:
        errors.append(f"{arrays_path}: ret_attn is not a 2D array")
    if selection_shape is not None and (
        arrays["ret_attn"].shape != selection_shape
        or arrays["applied_mask"].shape != selection_shape
        or arrays["mask_before"].shape != selection_shape
    ):
        errors.append(f"{arrays_path}: selection arrays do not share one grid")
    expected_consumed_shape = EXPECTED_TRANSITIONS[(source_stage, target_stage)]
    consumed = arrays["consumed_mask"]
    if consumed.shape != expected_consumed_shape:
        errors.append(
            f"{arrays_path}: expected consumed grid {expected_consumed_shape}, got {consumed.shape}"
        )
    can_reconstruct = (
        binary_masks["applied_mask"]
        and binary_masks["consumed_mask"]
        and consumed.shape == expected_consumed_shape
    )
    if can_reconstruct:
        expected_consumed = nearest_resize(arrays["applied_mask"], consumed.shape)
        if not np.array_equal(consumed, expected_consumed):
            errors.append(f"{arrays_path}: consumed mask differs from applied mask")
    else:
        return

    keep_flat = np.flatnonzero(consumed.reshape(-1)).astype(np.int32)
    drop_flat = np.flatnonzero(consumed.reshape(-1) == 0).astype(np.int32)
    keep_coords = np.argwhere(consumed == 1).astype(np.int16)
    drop_coords = np.argwhere(consumed == 0).astype(np.int16)
    expected_indices = {
        "consumed_keep_indices": keep_flat,
        "consumed_drop_indices": drop_flat,
        "consumed_keep_coords_rc": keep_coords,
        "consumed_drop_coords_rc": drop_coords,
    }
    for name, expected in expected_indices.items():
        if not np.array_equal(arrays[name], expected):
            errors.append(f"{arrays_path}: {name} does not reconstruct consumed_mask")

    metadata_checks = {
        "selection_grid_shape": list(selection_shape),
        "consumed_grid_shape": list(consumed.shape),
        "consumed_selected_token_count": int(keep_flat.size),
        "consumed_dropped_token_count": int(drop_flat.size),
        "consumed_total_token_count": int(consumed.size),
    }
    for key, expected in metadata_checks.items():
        if metadata.get(key) != expected:
            errors.append(f"{metadata_path}: {key} expected {expected}, got {metadata.get(key)}")
    expected_ratio = keep_flat.size / consumed.size
    try:
        recorded_ratio = float(metadata.get("consumed_keep_ratio", -1))
    except (TypeError, ValueError):
        recorded_ratio = -1
    if abs(recorded_ratio - expected_ratio) > 1e-12:
        errors.append(f"{metadata_path}: consumed_keep_ratio is inconsistent")

    if artifact_mode == "visual":
        for suffix in ("attention", "proposed_mask", "applied_mask", "consumed_mask"):
            if not (sample_dir / f"{stem}_{suffix}.png").is_file():
                errors.append(f"{sample_dir}: missing {stem}_{suffix}.png")

    return metadata


def validate_decoding_trace(sample, sample_dir, final_transition, errors):
    prediction = sample.get("prediction")
    if prediction not in {"yes", "no"}:
        errors.append(f"{sample_dir}: prediction is not yes/no")
        return
    token_ids = sample.get("final_token_ids")
    if not isinstance(token_ids, list) or not token_ids:
        errors.append(f"{sample_dir}: final_token_ids is empty or malformed")
        return
    if final_transition is None:
        return
    margin = final_transition.get("final_decoding_margin", {})
    if margin.get("available") is not True:
        errors.append(f"{sample_dir}: final Yes/No margin is unavailable")
        return
    yes_token_id = margin.get("yes_token_id")
    no_token_id = margin.get("no_token_id")
    if (
        type(yes_token_id) is not int
        or type(no_token_id) is not int
        or yes_token_id == no_token_id
    ):
        errors.append(f"{sample_dir}: final Yes/No token IDs are malformed")
        return
    expected_token_id = margin.get(f"{prediction}_token_id")
    if token_ids[0] != expected_token_id:
        errors.append(
            f"{sample_dir}: first generated token {token_ids[0]} does not match "
            f"the recorded {prediction} token {expected_token_id}"
        )
    try:
        yes_no_margin = float(margin["yes_no_margin"])
    except (KeyError, TypeError, ValueError):
        errors.append(f"{sample_dir}: final Yes/No margin is malformed")
        return
    if not np.isfinite(yes_no_margin):
        errors.append(f"{sample_dir}: final Yes/No margin is not finite")
    elif yes_no_margin == 0:
        # torch.argmax breaks an exact logit tie in favor of the smaller token ID.
        tied_winner = min(yes_token_id, no_token_id)
        if expected_token_id != tied_winner:
            errors.append(f"{sample_dir}: prediction disagrees with argmax tie-breaking")
    elif prediction == "yes" and yes_no_margin < 0:
        errors.append(f"{sample_dir}: Yes prediction has a non-positive margin")
    elif prediction == "no" and yes_no_margin > 0:
        errors.append(f"{sample_dir}: No prediction has a non-negative margin")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--baseline-predictions", required=True, type=Path)
    parser.add_argument("--reference-predictions", type=Path)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--expected-manifest", type=Path)
    parser.add_argument("--condition", default="D")
    parser.add_argument("--artifact-mode", choices=("visual", "compact"), required=True)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    errors = []
    historical_baseline = read_baseline(args.baseline_predictions)
    reference = (
        read_baseline(args.reference_predictions)
        if args.reference_predictions
        else None
    )
    expected_ids = (
        manifest_ids(args.expected_manifest)
        if args.expected_manifest
        else set(historical_baseline)
    )
    if len(expected_ids) != args.expected_count:
        errors.append(
            f"expected ID source contains {len(expected_ids)} rows, expected {args.expected_count}"
        )

    condition_root = args.artifact_root / args.condition
    sample_paths = sorted(condition_root.glob("rank_*/*/sample.json"))
    samples = {}
    for sample_path in sample_paths:
        sample = read_json(sample_path)
        question_id = str(sample.get("question_id", sample.get("sample_id", "")))
        if not question_id:
            errors.append(f"{sample_path}: missing question_id")
            continue
        if question_id in samples:
            errors.append(f"duplicate captured question_id: {question_id}")
            continue
        samples[question_id] = (sample, sample_path.parent)

    captured_ids = set(samples)
    missing = sorted(expected_ids - captured_ids)
    unexpected = sorted(captured_ids - expected_ids)
    if missing:
        errors.append(f"missing captured IDs: {missing[:20]} (total {len(missing)})")
    if unexpected:
        errors.append(f"unexpected captured IDs: {unexpected[:20]} (total {len(unexpected)})")
    if len(samples) != args.expected_count:
        errors.append(f"captured {len(samples)} samples, expected {args.expected_count}")

    historical_prediction_mismatches = []
    reference_prediction_mismatches = []
    if reference is not None:
        reference_ids = set(reference)
        if reference_ids != expected_ids:
            missing_reference = sorted(expected_ids - reference_ids)
            unexpected_reference = sorted(reference_ids - expected_ids)
            errors.append(
                "reference prediction IDs differ from expected IDs: "
                f"missing={missing_reference[:20]} (total {len(missing_reference)}), "
                f"unexpected={unexpected_reference[:20]} "
                f"(total {len(unexpected_reference)})"
            )
    for question_id, (sample, sample_dir) in samples.items():
        if sample.get("audit_schema_version") != 2:
            errors.append(f"{sample_dir}: expected audit schema 2")
        if sample.get("condition") != args.condition:
            errors.append(f"{sample_dir}: condition mismatch")
        if sample.get("artifact_mode") != args.artifact_mode:
            errors.append(f"{sample_dir}: artifact mode mismatch")
        validate_transform(sample, sample_dir, args.artifact_mode, errors)
        transition_metadata = {}
        for source_stage, target_stage in EXPECTED_TRANSITIONS:
            metadata = validate_transition(
                sample_dir,
                source_stage,
                target_stage,
                args.artifact_mode,
                errors,
                condition=args.condition,
            )
            transition_metadata[(source_stage, target_stage)] = metadata
        validate_decoding_trace(
            sample,
            sample_dir,
            transition_metadata.get((0, 1)),
            errors,
        )
        historical = historical_baseline.get(question_id)
        if historical is None:
            continue
        if sample.get("ground_truth") != historical["ground_truth"]:
            errors.append(f"{sample_dir}: ground truth differs from frozen baseline")
        if sample.get("prediction") != historical["prediction"]:
            historical_prediction_mismatches.append(
                {
                    "question_id": question_id,
                    "baseline": historical["prediction"],
                    "captured": sample.get("prediction"),
                }
            )
        current_reference = reference.get(question_id) if reference is not None else None
        if current_reference is not None:
            if sample.get("ground_truth") != current_reference["ground_truth"]:
                errors.append(f"{sample_dir}: ground truth differs from current reference")
            if sample.get("prediction") != current_reference["prediction"]:
                reference_prediction_mismatches.append(
                    {
                        "question_id": question_id,
                        "reference": current_reference["prediction"],
                        "captured": sample.get("prediction"),
                    }
                )
    if reference_prediction_mismatches:
        errors.append(
            f"{len(reference_prediction_mismatches)} predictions differ from the "
            "current-code reference"
        )

    warnings = []
    if historical_prediction_mismatches:
        warnings.append(
            f"{len(historical_prediction_mismatches)} predictions differ from the "
            "historical Step 1 baseline"
        )

    report = {
        "status": "PASS" if not errors else "FAIL",
        "condition": args.condition,
        "artifact_mode": args.artifact_mode,
        "expected_count": args.expected_count,
        "captured_count": len(samples),
        "reference_predictions": str(args.reference_predictions) if args.reference_predictions else None,
        "reference_prediction_mismatch_count": len(reference_prediction_mismatches),
        "reference_prediction_mismatches": reference_prediction_mismatches,
        "historical_prediction_mismatch_count": len(historical_prediction_mismatches),
        "historical_prediction_mismatches": historical_prediction_mismatches,
        "warnings": warnings,
        "errors": errors,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Render the 12 Step 5 mechanism-screen candidates on original image pixels.

Offline only: consumes compact D1/D2 artifacts, provenance JSONL, and the
frozen Step 5 manifest. No model or GPU is loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = ROOT / "audit/runs/step5_repeatability_v1"
DEFAULT_BUNDLE = ROOT / "audit/manifests/step5_changed30_matched_v1"
DEFAULT_OUTPUT = ROOT / "audit/runs/step5_mechanism_visuals_v1"
CANDIDATE_IDS = ("311", "645", "1025", "1043", "1339", "1507",
                 "2253", "2275", "2559", "2639", "2697", "2887")
TRANSITIONS = ((-2, -1), (-1, 0), (0, 1))
TARGET_FORWARD_STAGE = {-1: -1, 0: 0, 1: 1}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def nearest_resize(array, output_shape):
    """Match torch.nn.functional.interpolate(..., mode='nearest') indexing."""
    source_height, source_width = array.shape
    output_height, output_width = output_shape
    check(source_height > 0 and source_width > 0 and output_height > 0 and output_width > 0,
          f"invalid resize shapes: {array.shape} -> {output_shape}")
    rows = np.floor(np.arange(output_height) * source_height / output_height).astype(np.int64)
    cols = np.floor(np.arange(output_width) * source_width / output_width).astype(np.int64)
    return array[rows[:, None], cols[None, :]]


def load_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
    return rows


def sample_dir(study, label, qid):
    root = study / label / "artifacts" / "D"
    found = list(root.glob(f"rank_*/{qid}/sample.json"))
    check(len(found) == 1, f"{label}/{qid}: expected one compact sample artifact, found {len(found)}")
    return found[0].parent


def source_records(run_dir, qid, forward_stage, target_stage):
    path = run_dir / f"token_000_forward_stage_{forward_stage}_provenance.jsonl"
    check(path.is_file(), f"missing provenance: {path}")
    records = [r for r in load_jsonl(path)
               if str(r.get("sample_id")) == qid and int(r.get("stage_internal", -999)) == target_stage]
    check(records, f"{path}: no records for qid={qid}, target stage={target_stage}")
    indices = [int(record["candidate_index"]) for record in records]
    check(indices == list(range(len(records))), f"{path}: candidate indices are not complete row-major order")
    for record in records:
        check(record.get("selected") is record.get("mask_selected"),
              f"{path}: runtime selected flag differs from consumed mask")
        rect = record.get("original_rect_clipped")
        check(isinstance(rect, list) and len(rect) == 4 and all(math.isfinite(float(x)) for x in rect),
              f"{path}: malformed original-image rectangle")
    return records


def load_transition(run_dir, source_stage, target_stage):
    stem = f"token_000_stage_{source_stage}_to_{target_stage}"
    metadata_path = run_dir / f"{stem}.json"
    arrays_path = run_dir / f"{stem}.npz"
    check(metadata_path.is_file() and arrays_path.is_file(), f"missing transition files for {stem}")
    metadata = read(metadata_path)
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in ("ret_attn", "applied_mask", "consumed_mask")}
    for name, array in arrays.items():
        check(np.isfinite(array).all(), f"{arrays_path}: {name} has non-finite values")
    check(arrays["ret_attn"].ndim == 2 and arrays["applied_mask"].ndim == 2
          and arrays["consumed_mask"].ndim == 2, f"{arrays_path}: expected 2D arrays")
    check(tuple(arrays["ret_attn"].shape) == tuple(arrays["applied_mask"].shape),
          f"{arrays_path}: selection signal and applied mask grids differ")
    expected_consumed = nearest_resize(arrays["applied_mask"], arrays["consumed_mask"].shape)
    check(np.array_equal(expected_consumed, arrays["consumed_mask"]),
          f"{arrays_path}: consumed mask does not match the applied mask resize")
    check(metadata.get("consumed_matches_applied_mask") is True,
          f"{metadata_path}: runtime consumed-mask verification flag is missing")
    scores = nearest_resize(arrays["ret_attn"].astype(np.float32), arrays["consumed_mask"].shape)
    check(scores.size == arrays["consumed_mask"].size, f"{arrays_path}: score/mask cell count differs")
    return metadata, arrays, scores


def rect_intersects(a, b):
    return max(float(a[0]), float(b[0])) < min(float(a[2]), float(b[2])) and \
        max(float(a[1]), float(b[1])) < min(float(a[3]), float(b[3]))


def target_rectangles(row):
    rects = []
    for instance in row.get("target_bboxes", []):
        x, y, width, height = map(float, instance["bbox_xywh"])
        if width > 0 and height > 0:
            rects.append([x, y, x + width, y + height])
    return rects


def render_panel(image, records, scores, score_range, row, source_stage, target_stage):
    width, height = image.size
    minimum, maximum = score_range
    rgba = image.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    grid_shape = (int(round(math.sqrt(len(records)))),) * 2
    check(grid_shape[0] * grid_shape[1] == len(records),
          "provenance candidate count is not a square grid")
    target_boxes = target_rectangles(row)
    score_flat = scores.reshape(-1)
    for record in records:
        rect = [float(value) for value in record["original_rect_clipped"]]
        x1, y1, x2, y2 = rect
        x1, x2 = min(width, max(0.0, x1)), min(width, max(0.0, x2))
        y1, y2 = min(height, max(0.0, y1)), min(height, max(0.0, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        score = float(score_flat[int(record["candidate_index"])])
        unit = 0.5 if maximum <= minimum else min(1.0, max(0.0, (score - minimum) / (maximum - minimum)))
        alpha = int(round(35 + 170 * unit))
        color = (35, 190, 75, alpha) if record["selected"] else (225, 45, 45, alpha)
        draw.rectangle((x1, y1, x2, y2), fill=color)
    composed = Image.alpha_composite(rgba, overlay).convert("RGB")
    panel = Image.new("RGB", (width, height + 82), (24, 28, 32))
    panel.paste(composed, (0, 40))
    label_draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    gt = row["ground_truth"].upper()
    label_draw.text((8, 6),
                    f"{row['question_id']} | stage {source_stage}->{target_stage} | GT={gt} | target={row['target_object']}",
                    fill=(255, 255, 255), font=font)
    label_draw.text((8, 22), str(row["question"])[:max(1, width // 6)],
                    fill=(225, 230, 235), font=font)
    pred = f"C={row['C_prediction']}  D={row['D_prediction']}  score=SECOND ret_attn rank signal (not causal attention)"
    label_draw.text((8, height + 45), pred[:max(1, width // 6)], fill=(255, 255, 255), font=font)
    for instance, box in enumerate(target_boxes, 1):
        x1, y1, x2, y2 = box
        x1, x2 = min(width - 1, max(0.0, x1)), min(width - 1, max(0.0, x2))
        y1, y2 = min(height - 1, max(0.0, y1)), min(height - 1, max(0.0, y2))
        if x2 > x1 and y2 > y1:
            label_draw.rectangle((x1, y1 + 40, x2, y2 + 40), outline=(255, 230, 35), width=max(2, width // 400))
            label_draw.text((x1 + 2, y1 + 42), f"COCO {instance}", fill=(255, 230, 35), font=font)
    return panel


def make_contact_sheet(panels, path):
    font = ImageFont.load_default()
    width = 520
    thumbs = []
    for label, panel in panels:
        copy = panel.copy()
        copy.thumbnail((width, 650), Image.Resampling.LANCZOS)
        cell = Image.new("RGB", (width, copy.height + 24), "white")
        ImageDraw.Draw(cell).text((6, 5), label, fill="black", font=font)
        cell.paste(copy, (0, 24))
        thumbs.append(cell)
    sheet = Image.new("RGB", (width * len(thumbs), max(cell.height for cell in thumbs)), (235, 235, 235))
    x = 0
    for cell in thumbs:
        sheet.paste(cell, (x, 0))
        x += width
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY,
                        help="completed audit/runs/step5_repeatability_v1 directory")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    study, bundle, output = args.study.resolve(), args.bundle.resolve(), args.output.resolve()
    check(not output.exists(), f"output already exists: {output}; choose a new versioned directory")
    sys_path = str(ROOT / "audit")
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from step5_repeatability import capture_records, load_bundle  # noqa: PLC0415

    rows, identity = load_bundle(bundle, verify_images=True)
    parent_plan = read(study / "study_plan.json")
    check(identity == parent_plan.get("cohort"), "frozen bundle differs from the Step 5 study")
    report = read(study / "repeatability_report.json")
    candidate_ids = {str(qid) for qid in report.get("summary", {}).get(
        "mechanism_screen_candidate_ids_pending_label_review", [])}
    check(candidate_ids == set(CANDIDATE_IDS),
          f"report candidate IDs differ from the frozen 12-ID set: {sorted(candidate_ids)}")
    report_rows = {str(record["question_id"]): record for record in report.get("samples", [])}
    bundle_rows = {str(record["question_id"]): record for record in rows}
    for qid in CANDIDATE_IDS:
        check(qid in report_rows and report_rows[qid].get("mechanism_screen_candidate_pending_label_review") is True,
              f"{qid}: not marked as a mechanism-screen candidate in the repeatability report")
        check(qid in bundle_rows and bundle_rows[qid]["role"] == "changed_case",
              f"{qid}: not a frozen changed-case sample")
        check(not bundle_rows[qid].get("annotation_issues"),
              f"{qid}: annotation issue exists; do not include in this candidate visualization")

    expected = {qid: bundle_rows[qid] for qid in CANDIDATE_IDS}
    run_records = {}
    for label in ("D1", "D2"):
        run = study / label
        run_manifest = read(run / "run_manifest.json")
        check(run_manifest.get("condition") == "D", f"{label}: expected a D run")
        complete = read(run / "run_complete.json")
        check(complete.get("status") == "PASS" and complete.get("snapshot_sha256") == parent_plan["snapshot_sha256"],
              f"{label}: original completed run failed or has another snapshot")
        validation = read(run / "validation_report.json")
        check(validation.get("status") == "PASS" and not validation.get("errors"),
              f"{label}: Step 5 capture validation did not pass")
        for name, expected_hash in complete.get("output_hashes", {}).items():
            check(digest(run / name) == expected_hash, f"{label}: completed output changed: {name}")
        all_records = capture_records(run, "D", {str(row["question_id"]): row for row in rows})
        check(all_records == read(run / "records.json"), f"{label}: artifact contents differ from stored records")
        run_records[label] = all_records
    for qid in CANDIDATE_IDS:
        check(run_records["D1"][qid]["mask_hashes"] == run_records["D2"][qid]["mask_hashes"],
              f"{qid}: D1/D2 masks differ; do not treat one visualization as stable selection")

    prepared = []
    global_ranges = {}
    for qid in CANDIDATE_IDS:
        row = bundle_rows[qid]
        d1 = sample_dir(study, "D1", qid)
        d2 = sample_dir(study, "D2", qid)
        sample = read(d1 / "sample.json")
        summary_path = d1 / "provenance_summary.json"
        summary = read(summary_path)
        check(summary.get("all_runtime_offline_match") is True and summary.get("total_truncated_visual_tokens") == 0,
              f"{qid}: provenance did not verify against runtime or contains truncation")
        original = sample.get("image_transform", {}).get("original_size", {})
        check((int(original.get("width", -1)), int(original.get("height", -1))) ==
              (int(row["image_width"]), int(row["image_height"])),
              f"{qid}: provenance original size differs from frozen COCO dimensions")
        image_path = Path(row["image_path"])
        check(digest(image_path) == row["image_sha256"], f"{qid}: source image hash changed")
        image = Image.open(image_path).convert("RGB")
        check(image.size == (row["image_width"], row["image_height"]), f"{qid}: image dimensions changed")
        transitions = []
        for source_stage, target_stage in TRANSITIONS:
            forward_stage = TARGET_FORWARD_STAGE[target_stage]
            records1 = source_records(d1, qid, forward_stage, target_stage)
            records2 = source_records(d2, qid, forward_stage, target_stage)
            check([(r["candidate_index"], r["selected"], r["original_rect_clipped"]) for r in records1] ==
                  [(r["candidate_index"], r["selected"], r["original_rect_clipped"]) for r in records2],
                  f"{qid}: D1/D2 stage {target_stage} provenance differs")
            metadata1, arrays1, scores = load_transition(d1, source_stage, target_stage)
            _, arrays2, scores2 = load_transition(d2, source_stage, target_stage)
            for name in ("applied_mask", "consumed_mask"):
                check(np.array_equal(arrays1[name], arrays2[name]),
                      f"{qid}: D1/D2 {name} differs at {source_stage}->{target_stage}")
            check(arrays1["ret_attn"].shape == arrays2["ret_attn"].shape,
                  f"{qid}: D1/D2 selection signal shapes differ at {source_stage}->{target_stage}")
            attention_bitwise_equal = bool(np.array_equal(arrays1["ret_attn"], arrays2["ret_attn"]))
            attention_max_abs_difference = float(np.max(np.abs(
                arrays1["ret_attn"].astype(np.float64) - arrays2["ret_attn"].astype(np.float64)
            )))
            check(len(records1) == arrays1["consumed_mask"].size,
                  f"{qid}: provenance candidate count differs from consumed grid")
            check([int(record["selected"]) for record in records1] ==
                  arrays1["consumed_mask"].reshape(-1).astype(int).tolist(),
                  f"{qid}: selected flags differ from runtime consumed mask")
            key = f"{source_stage}_to_{target_stage}"
            values = scores.reshape(-1)
            if key not in global_ranges:
                global_ranges[key] = [float(values.min()), float(values.max())]
            else:
                global_ranges[key][0] = min(global_ranges[key][0], float(values.min()))
                global_ranges[key][1] = max(global_ranges[key][1], float(values.max()))
            transitions.append({"source_stage": source_stage, "target_stage": target_stage,
                                "forward_stage": forward_stage, "records": records1,
                                "metadata": metadata1, "arrays": arrays1, "scores": scores,
                                "scores_d2": scores2,
                                "selection_signal_d1_d2_bitwise_equal": attention_bitwise_equal,
                                "selection_signal_d1_d2_max_abs_difference": attention_max_abs_difference,
                                "d2_sample_dir": d2})
        prepared.append({"question_id": qid, "row": row, "image": image, "sample": sample,
                         "transitions": transitions})

    output.mkdir(parents=True)
    output_entries = []
    for item in prepared:
        qid, row, image = item["question_id"], item["row"], item["image"]
        sample_out = output / qid
        sample_out.mkdir()
        panels = []
        stage_entries = []
        for transition in item["transitions"]:
            source_stage, target_stage = transition["source_stage"], transition["target_stage"]
            key = f"{source_stage}_to_{target_stage}"
            panel = render_panel(image, transition["records"], transition["scores"],
                                 global_ranges[key], row, source_stage, target_stage)
            filename = f"stage_{source_stage}_to_{target_stage}.png"
            panel.save(sample_out / filename)
            panels.append((f"Stage {source_stage} -> {target_stage}", panel))
            target_boxes = target_rectangles(row)
            intersects = [record for record in transition["records"]
                          if any(rect_intersects(record["original_rect_clipped"], box) for box in target_boxes)]
            selected_intersects = sum(bool(record["selected"]) for record in intersects)
            stage_entries.append({
                "transition": [source_stage, target_stage],
                "provenance_forward_stage": transition["forward_stage"],
                "source_type": transition["records"][0]["source_type"],
                "grid_shape": list(transition["arrays"]["consumed_mask"].shape),
                "candidate_count": len(transition["records"]),
                "selected_count": int(transition["arrays"]["consumed_mask"].sum()),
                "dropped_count": int(transition["arrays"]["consumed_mask"].size - transition["arrays"]["consumed_mask"].sum()),
                "target_bbox_count": len(target_boxes),
                "candidate_cells_intersecting_any_coco_target_box": len(intersects) if target_boxes else None,
                "selected_candidate_cells_intersecting_any_coco_target_box": selected_intersects if target_boxes else None,
                "score_grid_min_max_shared_across_candidates_for_this_transition": global_ranges[key],
                "selection_signal_visualized_from_run": "D1",
                "selection_signal_D1_D2_bitwise_equal": transition["selection_signal_d1_d2_bitwise_equal"],
                "selection_signal_D1_D2_max_abs_difference": transition["selection_signal_d1_d2_max_abs_difference"],
                "image": filename,
                "note": "BBox overlap counts candidate cells, not unique image area; tile/crop overlap can count the same pixels more than once." if target_boxes else "No COCO target box is available for this question; no target-region overlap metric is reported.",
            })
        contact_name = "three_transition_contact_sheet.png"
        make_contact_sheet(panels, sample_out / contact_name)
        output_entries.append({
            "question_id": qid, "question": row["question"], "target_object": row["target_object"],
            "ground_truth": row["ground_truth"], "historical_predictions": {c: row[f"{c}_prediction"] for c in "CD"},
            "image_sha256": row["image_sha256"], "d1_artifact_dir": str(sample_dir(study, "D1", qid)),
            "d2_artifact_dir": str(sample_dir(study, "D2", qid)),
            "stages": stage_entries, "contact_sheet": contact_name,
        })
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "git_commit": parent_plan["snapshot"]["git_commit"],
        "study_plan_sha256": digest(study / "study_plan.json"),
        "repeatability_report_sha256": digest(study / "repeatability_report.json"),
        "frozen_cohort_sha256": identity["combined_sha256"],
        "candidate_ids": list(CANDIDATE_IDS),
        "selection_signal_semantics": "ret_attn is SECOND's rank-transformed selection signal. It is not raw causal attention, semantic relevance, or proof that a patch caused an answer.",
        "selection_signal_to_consumed_grid": "For stage score coloring, ret_attn is resized to the consumed feature grid with the same floor-index nearest-neighbor mapping used by torch interpolate(mode='nearest'); the masks are independently checked against saved consumed masks.",
        "geometry_semantics": "Rectangles come from runtime-validated provenance original_rect_clipped coordinates. AnyRes tile IDs and patch rows/columns are retained in the source JSONL; the plots project each patch rectangle to original-image coordinates.",
        "rendering": {"green": "runtime-consumed/retained token", "red": "runtime-dropped token", "yellow": "COCO target annotation box", "score_color_range_by_transition": global_ranges},
        "samples": output_entries,
        "limitations": [
            "These 12 cases were selected from the observed C/D changes; they are not a random sample.",
            "A selected region is not necessarily question-relevant, and ret_attn is not a causal attribution map.",
            "COCO boxes are dataset annotations, not pixel-level masks; box intersection is only a coarse localization check.",
            "For negative questions with no target box, the plots show selection but cannot define a positive target region."
        ],
    }
    (output / "visualization_report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Rendered {len(output_entries)} samples x {len(TRANSITIONS)} transitions: {output}")
    print(f"Report: {output / 'visualization_report.json'}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as exc:
        raise SystemExit(f"FAIL: {exc}")

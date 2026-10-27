#!/usr/bin/env python3
"""Freeze C/D-discordant POPE cases and deterministic matched controls; no inference."""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

VERSION = "step5-freeze-v1"
BASELINE = "audit/step1_full/pope_pop_3000_baseline"
POLICY = {
    "version": VERSION,
    "mandatory": ["same queried category", "same ground truth", "same C correctness"],
    "controls": "C equals D in the source run only; repeatability is not established",
    "exclusions": ["all case COCO images", "annotation-conflicting controls"],
    "reuse": "each control question and each control COCO image at most once",
    "objective": "maximum matched count, then minimum relaxed pairs, then minimum distance",
    "strict_calipers": {
        "largest_target_bbox_area_ratio_factor": 2.0,
        "target_center_distance_in_diagonal_units": 0.15,
        "scene_object_count_plus_one_factor": 2.0,
        "aspect_ratio_factor": 1.5,
    },
    "relaxed_calipers": {
        "largest_target_bbox_area_ratio_factor": 4.0,
        "target_center_distance_in_diagonal_units": 0.30,
        "scene_object_count_plus_one_factor": 3.0,
        "aspect_ratio_factor": 2.0,
    },
    "geometry": "all COCO target boxes saved; matching uses the largest box, not the historical single box",
    "negative_samples": "no target-size/location covariates when ground truth is no",
    "distance": "sum of caliper-normalized log/center differences; positive target geometry gets weight 2; log target instance count gets weight 0.25",
    "ties": "SHA256(seed, case question ID, control question ID), never Python hash()",
    "warning": "Controls are descriptive matched comparisons; matching alone does not identify causal effects.",
}


def fail(message):
    raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_entry(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def id_key(value):
    return int(value)


def yes_no(value, context):
    match = re.fullmatch(r"\s*(yes|no)\s*", str(value), re.I)
    if not match:
        fail(f"{context}: expected normalized yes/no, got {value!r}")
    return match.group(1).lower()


def indexed_jsonl(path):
    rows = {}
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("question_id", ""))
            if not qid.isdigit() or qid in rows:
                fail(f"{path}:{number}: missing/non-numeric/duplicate question_id {qid}")
            rows[qid] = row
    return rows


def load_samples(path, expected):
    rows = {}
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            doc, metric = raw.get("doc", {}), raw.get("pope_accuracy", {})
            qid = str(doc.get("question_id", ""))
            if not qid.isdigit() or str(metric.get("question_id")) != qid or qid in rows:
                fail(f"{path}:{number}: inconsistent/duplicate question_id")
            gt = yes_no(metric.get("ground_truth"), qid)
            if yes_no(doc.get("answer"), qid) != gt:
                fail(f"{qid}: document/metric ground truth differs")
            prediction = yes_no(metric.get("prediction"), qid)
            question = str(doc.get("question", "")).strip()
            match = re.fullmatch(r"Is there (?:an? )?(.+?) in the image\?", question, re.I)
            if not match:
                fail(f"{qid}: unsupported question template: {question}")
            if "score" in metric and float(metric["score"]) != float(prediction == gt):
                fail(f"{qid}: metric score disagrees with normalized prediction")
            rows[qid] = {
                "question_id": qid, "doc_id": raw.get("doc_id"),
                "dataset_image_id": str(doc.get("id")), "question": question,
                "target_object": match.group(1).lower(), "ground_truth": gt,
                "category": doc.get("category"), "prediction": prediction,
                "input": raw.get("input"), "prompt_hash": raw.get("prompt_hash"),
            }
    if len(rows) != expected:
        fail(f"{path}: expected {expected} samples, found {len(rows)}")
    return rows


def resolve_source(root, value):
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def load_sources(root, report_path, expected):
    report = read_json(report_path)
    if report.get("expected_count") != expected:
        fail("paired report expected_count differs")
    samples, manifests, source_paths = {}, {}, {}
    for condition in "BCD":
        run = resolve_source(root, report["inputs"][condition])
        manifest_path, validation_path = run / "run_manifest.json", run / "validation_report.json"
        manifest, validation = read_json(manifest_path), read_json(validation_path)
        if manifest.get("condition") != condition or manifest.get("expected_count") != expected:
            fail(f"{condition}: run manifest condition/count differs")
        if (validation.get("status") != "PASS" or validation.get("condition") != condition
                or validation.get("expected_count") != expected
                or validation.get("captured_count") != expected or validation.get("errors")):
            fail(f"{condition}: run validation gate did not pass")
        if report.get("validation_status", {}).get(condition) != "PASS":
            fail(f"{condition}: report validation gate did not pass")
        if manifest.get("git_commit") != report["git_commits"][condition]:
            fail(f"{condition}: recorded report/run commit differs")
        if manifest.get("second_parameters") != report["second_parameters"][condition]:
            fail(f"{condition}: recorded report/run parameters differ")
        sample_path = resolve_source(root, report["sample_files"][condition])
        if run not in sample_path.parents:
            fail(f"{condition}: sample file is outside the declared run directory")
        samples[condition] = load_samples(sample_path, expected)
        manifests[condition] = manifest
        for label, path in (("samples", sample_path), ("manifest", manifest_path), ("validation", validation_path)):
            source_paths[f"{condition}_{label}"] = path
    commits = {m.get("git_commit") for m in manifests.values()}
    if len(commits) != 1 or None in commits:
        fail("B/C/D recorded commits differ")
    if any(manifests[c].get("second_parameters") != manifests["B"].get("second_parameters") for c in "CD"):
        fail("B/C/D SECOND parameters differ")
    runtime_fields = ("conda_environment", "cuda_visible_devices", "num_processes")
    for condition in "CD":
        for field in runtime_fields:
            if manifests[condition].get("runtime", {}).get(field) != manifests["B"].get("runtime", {}).get(field):
                fail(f"B/{condition} runtime {field} differs")
    if any(set(samples[c]) != set(samples["B"]) for c in "CD"):
        fail("B/C/D question ID sets differ")
    for qid, base in samples["B"].items():
        for condition in "CD":
            for field in ("doc_id", "dataset_image_id", "question", "category", "ground_truth", "input", "prompt_hash"):
                if samples[condition][qid][field] != base[field]:
                    fail(f"{qid}: B/{condition} {field} differs")
    for condition, rows in samples.items():
        correct = sum(r["prediction"] == r["ground_truth"] for r in rows.values())
        if report["overall"][condition]["correct"] != correct:
            fail(f"{condition}: report correct count differs from raw samples")
    return report, samples, manifests, source_paths


def reconcile_cases(report, samples, expected_changed, expected_corrected, expected_degraded):
    changed = sorted((q for q in samples["C"] if samples["C"][q]["prediction"] != samples["D"][q]["prediction"]), key=id_key)
    corrected = [q for q in changed if samples["D"][q]["prediction"] == samples["D"][q]["ground_truth"]]
    degraded = [q for q in changed if q not in corrected]
    if (len(changed), len(corrected), len(degraded)) != (expected_changed, expected_corrected, expected_degraded):
        fail(f"unexpected C/D counts: changed={len(changed)}, corrected={len(corrected)}, degraded={len(degraded)}")
    effect = report["C_to_D_selection_effect"]
    if (effect["prediction_changed_count"] != len(changed)
            or effect["corrected_count"] != len(corrected) or effect["degraded_count"] != len(degraded)
            or sorted(map(str, effect["corrected_question_ids"]), key=id_key) != corrected
            or sorted(map(str, effect["degraded_question_ids"]), key=id_key) != degraded):
        fail("raw C/D changes differ from report effect lists")
    # changed_samples is the B/C/D UNION, not the 30-case C/D cohort.
    report_changed = [r for r in report["changed_samples"] if r["C_prediction"] != r["D_prediction"]]
    if len(report_changed) != len(changed) or {str(r["question_id"]) for r in report_changed} != set(changed):
        fail("raw C/D changes differ from report changed_samples")
    for row in report_changed:
        qid = str(row["question_id"])
        for condition in "BCD":
            if row[f"{condition}_prediction"] != samples[condition][qid]["prediction"]:
                fail(f"{qid}: report {condition} prediction differs")
        if row["question"] != samples["C"][qid]["question"] or row["ground_truth"] != samples["C"][qid]["ground_truth"]:
            fail(f"{qid}: report question/ground truth differs")
    return changed, corrected, degraded


def load_coco(path):
    coco = read_json(path)
    images = {str(r["id"]): r for r in coco["images"]}
    if len(images) != len(coco["images"]):
        fail("COCO image IDs are not unique")
    categories = {int(r["id"]): r["name"].lower() for r in coco["categories"]}
    objects = defaultdict(list)
    for annotation in coco["annotations"]:
        image_id = str(annotation["image_id"])
        if image_id not in images:
            fail(f"annotation references unknown COCO image {image_id}")
        if len(annotation["bbox"]) != 4 or any(not math.isfinite(float(x)) for x in annotation["bbox"]):
            fail("invalid COCO bbox")
        x, y, w, h = map(float, annotation["bbox"])
        if w <= 0 or h <= 0:
            continue
        objects[image_id].append({
            "annotation_id": annotation["id"], "category": categories[int(annotation["category_id"])],
            "bbox_xywh": [x, y, w, h], "bbox_area": w * h,
            "iscrowd": bool(annotation.get("iscrowd", 0)),
        })
    return images, categories, objects


def enrich(samples, features, mapping, coco):
    images, categories, objects = coco
    ids = set(samples["C"])
    if set(features) != ids or set(mapping) != ids:
        fail("feature/image-mapping question ID sets differ from B/C/D samples")
    rows = {}
    for qid in sorted(ids, key=id_key):
        sample = samples["C"][qid]
        feature, mapped = features[qid], mapping[qid]
        image_id = str(mapped["image_id"])
        if str(feature["image_id"]) != image_id:
            fail(f"{qid}: enhanced features and image mapping disagree on COCO ID")
        if feature["question"].strip() != sample["question"] or yes_no(feature["gt"], qid) != sample["ground_truth"]:
            fail(f"{qid}: enhanced features question/ground truth differs")
        if image_id not in images:
            fail(f"{qid}: mapped COCO image is missing from annotation file")
        image = images[image_id]
        if Path(mapped["image_path"]).name != image["file_name"] or mapped["image_filename"] != image["file_name"]:
            fail(f"{qid}: mapped filename and COCO filename disagree")
        width, height = int(image["width"]), int(image["height"])
        if width <= 0 or height <= 0:
            fail(f"{qid}: invalid image dimensions")
        for source, wkey, hkey in ((feature, "image_width", "image_height"), (mapped, "width", "height")):
            if (int(source[wkey]), int(source[hkey])) != (width, height):
                fail(f"{qid}: historical metadata and COCO dimensions differ")
        if not mapped.get("readable") or not feature.get("image_readable"):
            fail(f"{qid}: historical image readability failed")
        target = sample["target_object"]
        if target not in set(categories.values()):
            fail(f"{qid}: queried category is not a COCO category: {target}")
        targets = [r for r in objects[image_id] if r["category"] == target]
        issues = []
        if sample["ground_truth"] == "yes" and not targets:
            issues.append("GT=yes but COCO contains no queried category annotation")
        if sample["ground_truth"] == "no" and targets:
            issues.append("GT=no but COCO contains queried category annotation(s)")
        largest = max(targets, key=lambda r: (r["bbox_area"], -int(r["annotation_id"]))) if targets else None
        area_ratio, center, peripheral_distance = None, None, None
        if sample["ground_truth"] == "yes" and largest:
            x, y, w, h = largest["bbox_xywh"]
            area_ratio = (w * h) / (width * height)
            center = [(x + w / 2) / width, (y + h / 2) / height]
            peripheral_distance = math.hypot(center[0] - .5, center[1] - .5) / math.sqrt(2)
        row = {
            "question_id": qid, "doc_id": sample["doc_id"],
            "dataset_image_id": sample["dataset_image_id"],
            "image_id": image_id, "coco_image_id": image_id,
            "image_id_semantics": "image_id and coco_image_id are COCO IDs; dataset_image_id is the evaluation document ID",
            "image_path": mapped["image_path"], "image_sha256": mapped["sha256"],
            "question": sample["question"], "target_object": target,
            "ground_truth": sample["ground_truth"],
            **{f"{c}_prediction": samples[c][qid]["prediction"] for c in "BCD"},
            "C_correct": sample["prediction"] == sample["ground_truth"],
            "D_correct": samples["D"][qid]["prediction"] == sample["ground_truth"],
            "image_width": width, "image_height": height, "aspect_ratio": width / height,
            "num_objects": len(objects[image_id]), "num_noncrowd_objects": sum(not r["iscrowd"] for r in objects[image_id]),
            "target_instance_count": len(targets), "target_bboxes": targets,
            "largest_target_bbox_area_ratio": area_ratio, "largest_target_center_normalized": center,
            "largest_target_peripheral_distance": peripheral_distance,
            "annotation_issues": issues,
            "matching_geometry_available": not issues,
        }
        rows[qid] = row
    return rows


def pair_distance(case, control):
    if any(case[k] != control[k] for k in ("ground_truth", "target_object", "C_correct")):
        return None
    if not case["matching_geometry_available"] or not control["matching_geometry_available"]:
        return None
    differences = {
        "scene_log_count_difference": abs(math.log((case["num_objects"] + 1) / (control["num_objects"] + 1))),
        "log_aspect_ratio_difference": abs(math.log(case["aspect_ratio"] / control["aspect_ratio"])),
    }
    positive = case["ground_truth"] == "yes"
    if positive:
        ca, co = case["largest_target_bbox_area_ratio"], control["largest_target_bbox_area_ratio"]
        cc, oc = case["largest_target_center_normalized"], control["largest_target_center_normalized"]
        differences["target_log_area_ratio_difference"] = abs(math.log(ca / co))
        differences["target_center_distance"] = math.dist(cc, oc) / math.sqrt(2)
        differences["target_log_instance_count_difference"] = abs(math.log(case["target_instance_count"] / control["target_instance_count"]))
    for tier, factor, center, count, aspect in (("strict", 2, .15, 2, 1.5), ("relaxed", 4, .30, 3, 2)):
        if differences["scene_log_count_difference"] > math.log(count) + 1e-12:
            continue
        if differences["log_aspect_ratio_difference"] > math.log(aspect) + 1e-12:
            continue
        if positive and (differences["target_log_area_ratio_difference"] > math.log(factor) + 1e-12
                         or differences["target_center_distance"] > center + 1e-12):
            continue
        score = differences["scene_log_count_difference"] / math.log(2) + differences["log_aspect_ratio_difference"] / math.log(1.5)
        if positive:
            score += 2 * (differences["target_log_area_ratio_difference"] / math.log(2) + differences["target_center_distance"] / .15)
            score += .25 * differences["target_log_instance_count_difference"]
        return {"tier": tier, "distance": score, "covariate_differences": differences}
    return None


def edge_cost(case_id, control_id, match, seed):
    tie = int(hashlib.sha256(f"{seed}:{case_id}:{control_id}".encode()).hexdigest()[:12], 16) % 1000
    # One relaxed pair costs more than all 30 finite distance/tie costs combined.
    return (10**15 if match["tier"] == "relaxed" else 0) + round(match["distance"] * 1_000_000) * 1000 + tie


def min_cost_assignment(candidates):
    """Successive shortest paths: max cardinality, minimum total integer cost."""
    cases = sorted(candidates, key=id_key)
    images = sorted({image for values in candidates.values() for image in values}, key=id_key)
    case_nodes = {q: i + 1 for i, q in enumerate(cases)}
    image_nodes = {image: len(cases) + i + 1 for i, image in enumerate(images)}
    sink = len(cases) + len(images) + 1
    graph = [[] for _ in range(sink + 1)]

    def add_edge(a, b, cost):
        forward = [b, len(graph[b]), 1, cost]
        reverse = [a, len(graph[a]), 0, -cost]
        graph[a].append(forward)
        graph[b].append(reverse)
        return forward

    references = {}
    for q in cases:
        add_edge(0, case_nodes[q], 0)
        for image in sorted(candidates[q], key=id_key):
            references[q, image] = add_edge(case_nodes[q], image_nodes[image], candidates[q][image]["cost"])
    for image in images:
        add_edge(image_nodes[image], sink, 0)
    potentials = [0] * len(graph)
    while True:
        distance, previous = [math.inf] * len(graph), [None] * len(graph)
        distance[0] = 0
        queue = [(0, 0)]
        while queue:
            dist, node = heapq.heappop(queue)
            if dist != distance[node]:
                continue
            for i, edge in enumerate(graph[node]):
                target, _, capacity, cost = edge
                proposal = dist + cost + potentials[node] - potentials[target]
                if capacity and proposal < distance[target]:
                    distance[target], previous[target] = proposal, (node, i)
                    heapq.heappush(queue, (proposal, target))
        if previous[sink] is None:
            break
        for node, dist in enumerate(distance):
            if dist != math.inf:
                potentials[node] += dist
        node = sink
        while node:
            parent, i = previous[node]
            edge = graph[parent][i]
            edge[2] -= 1
            graph[node][edge[1]][2] += 1
            node = parent
    return {q: candidates[q][image] for (q, image), edge in references.items() if edge[2] == 0}


def match_controls(rows, changed, seed):
    case_images = {rows[q]["coco_image_id"] for q in changed}
    pool = [r for q, r in rows.items() if q not in changed
            and r["C_prediction"] == r["D_prediction"] and r["coco_image_id"] not in case_images
            and r["matching_geometry_available"]]
    pool.sort(key=lambda r: id_key(r["question_id"]))
    candidates, diagnostics = {}, {}
    for q in changed:
        case, by_image = rows[q], {}
        same_stratum = [r for r in pool if all(case[k] == r[k] for k in ("ground_truth", "target_object", "C_correct"))]
        strict_ids, relaxed_ids = [], []
        for control in same_stratum:
            distance = pair_distance(case, control)
            if distance is None:
                continue
            cq = control["question_id"]
            (strict_ids if distance["tier"] == "strict" else relaxed_ids).append(cq)
            value = {"control_question_id": cq, "control_coco_image_id": control["coco_image_id"],
                     **distance, "cost": edge_cost(q, cq, distance, seed)}
            previous = by_image.get(control["coco_image_id"])
            if previous is None or (value["cost"], id_key(cq)) < (previous["cost"], id_key(previous["control_question_id"])):
                by_image[control["coco_image_id"]] = value
        candidates[q] = by_image
        diagnostics[q] = {"same_mandatory_stratum_candidates": len(same_stratum),
                          "strict_candidates": len(strict_ids), "relaxed_candidates": len(relaxed_ids),
                          "candidate_control_images": len(by_image)}
    assignment = min_cost_assignment(candidates)
    pairs, unmatched = [], []
    for q in changed:
        if q in assignment:
            pairs.append({"pair_id": f"pair_{q}", "case_question_id": q,
                          "case_coco_image_id": rows[q]["coco_image_id"],
                          "case_change_direction": rows[q]["change_direction"],
                          **assignment[q], "candidate_counts": diagnostics[q]})
        else:
            reason = ("case annotation requires review" if not rows[q]["matching_geometry_available"]
                      else "no candidate in mandatory stratum" if not diagnostics[q]["same_mandatory_stratum_candidates"]
                      else "no candidate within declared calipers" if not candidates[q]
                      else "unique-control-image competition prevented assignment")
            unmatched.append({"question_id": q, "reason": reason, **diagnostics[q]})
    return pairs, unmatched, {"historically_equal_pool_after_image_and_annotation_exclusions": len(pool),
                              "case_unique_images": len(case_images)}


def verify_selected_images(rows):
    verified = {}
    for row in rows:
        image_id = row["coco_image_id"]
        signature = (row["image_path"], row["image_sha256"])
        if image_id in verified:
            if verified[image_id] != signature:
                fail(f"COCO image {image_id}: selected rows disagree on path/hash")
            continue
        if sha256(row["image_path"]) != row["image_sha256"]:
            fail(f"COCO image {image_id}: current image hash differs from historical mapping")
        verified[image_id] = signature
    return len(verified)


def current_worktree(root):
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True)
        if result.returncode:
            return None
        return result.stdout
    commit, status, diff = git("rev-parse", "HEAD"), git("status", "--short", "--untracked-files=normal"), git("diff", "HEAD", "--binary")
    return {
        "head": commit.decode().strip() if commit else None,
        "status": status.decode(errors="replace").splitlines() if status is not None else None,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest() if diff is not None else None,
        "scope": "Current freeze-time worktree only; historical dirty files cannot be reconstructed from a commit ID alone.",
    }


def manifest(name, rows, provenance_hash, seed):
    return {"schema_version": 2, "purpose": name, "freeze_version": VERSION, "seed": seed,
            "expected_count": len(rows), "provenance_sha256": provenance_hash, "samples": rows}


def write_bundle(output, files):
    if output.exists():
        fail(f"frozen output already exists; refusing overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".step5-freeze-", dir=output.parent))
    try:
        for name, value in files.items():
            (temporary / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        hashes = {name: sha256(temporary / name) for name in sorted(files)}
        (temporary / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8", newline="\n")
        if output.exists():
            fail(f"frozen output already exists: {output}")
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", type=Path, default=Path("audit/runs/step4_ablation_v1/bcd_full_paired_report.json"))
    parser.add_argument("--features", type=Path, default=Path(BASELINE) / "enhanced_features.jsonl")
    parser.add_argument("--image-mapping", type=Path, default=Path(BASELINE) / "image_mapping.jsonl")
    parser.add_argument("--coco", type=Path, default=Path("/home/yxq/CAAC64/annotations_trainval2014/annotations/instances_val2014.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("audit/manifests/step5_changed30_matched_v1"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-count", type=int, default=3000)
    parser.add_argument("--expected-changed", type=int, default=30)
    parser.add_argument("--expected-corrected", type=int, default=13)
    parser.add_argument("--expected-degraded", type=int, default=17)
    return parser.parse_args(argv)


def run(args):
    root = args.repo_root.resolve()
    paths = {name: resolve_source(root, getattr(args, name)) for name in ("report", "features", "image_mapping", "coco", "output_dir")}
    if paths["output_dir"].exists():
        fail(f"frozen output already exists; refusing overwrite: {paths['output_dir']}")
    report, samples, manifests, sources = load_sources(root, paths["report"], args.expected_count)
    changed, corrected, degraded = reconcile_cases(report, samples, args.expected_changed, args.expected_corrected, args.expected_degraded)
    rows = enrich(samples, indexed_jsonl(paths["features"]), indexed_jsonl(paths["image_mapping"]), load_coco(paths["coco"]))
    for q in changed:
        rows[q]["role"] = "changed_case"
        rows[q]["change_direction"] = "corrected" if q in corrected else "degraded"
    pairs, unmatched, pool_info = match_controls(rows, changed, args.seed)
    control_ids = [p["control_question_id"] for p in pairs]
    for pair in pairs:
        rows[pair["control_question_id"]]["role"] = "historically_CD_equal_control"
        rows[pair["control_question_id"]]["matched_case_question_id"] = pair["case_question_id"]
    cases = [rows[q] for q in changed]
    controls = [rows[q] for q in sorted(control_ids, key=id_key)]
    if set(changed) & set(control_ids) or len(control_ids) != len(set(control_ids)):
        fail("case/control question overlap or reused control question")
    if {r["coco_image_id"] for r in cases} & {r["coco_image_id"] for r in controls}:
        fail("case/control COCO image overlap")
    if len({r["coco_image_id"] for r in controls}) != len(controls):
        fail("control COCO image reused")
    verified_images = verify_selected_images(cases + controls)
    source_paths = {**sources, **{name: paths[name] for name in ("report", "features", "image_mapping", "coco")}, "generator": Path(__file__).resolve()}
    warnings = [
        "This outcome-selected cohort cannot estimate the overall 3000-sample accuracy effect.",
        "Historical C/D equality does not establish repeatability; repeat C and D before causal intervention.",
        "COCO bounding boxes are approximate localization; annotation absence does not by itself prove object absence.",
        "Question-to-COCO mapping uses the frozen Step 1 mapping, checked against metadata, annotations and image hashes; it is not rederived from the HF dataset here.",
        "Same recorded commit is not proof of identical historical dirty worktrees, model files or numeric execution.",
    ]
    relaxed = sum(p["tier"] == "relaxed" for p in pairs)
    if relaxed:
        warnings.append(f"{relaxed} pairs use declared relaxed calipers; inspect pairs.json before downstream analysis.")
    worktree = current_worktree(root)
    if worktree["head"] != manifests["C"]["git_commit"]:
        warnings.append("Current HEAD differs from the historical source commit (or could not be read).")
    provenance = {
        "schema_version": 1, "freeze_version": VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed, "matching_policy": POLICY,
        "source_files": {name: source_entry(path) for name, path in source_paths.items()},
        "recorded_run_manifests": manifests, "current_worktree": worktree,
        "selected_image_hashes_verified": verified_images,
        "features_usage": "Question/GT/COCO ID/dimension cross-checks only. Historical predictions, strata and single target bbox are not used for matching.",
        "warnings": warnings,
    }
    provenance_text = json.dumps(provenance, indent=2, ensure_ascii=False) + "\n"
    provenance_hash = hashlib.sha256(provenance_text.encode("utf-8")).hexdigest()
    freeze_report = {
        "status": "PASS" if not unmatched else "NEEDS_REVIEW",
        "changed_count": len(cases), "corrected_count": len(corrected), "degraded_count": len(degraded),
        "control_count": len(controls), "combined_count": len(cases) + len(controls),
        "strict_pairs": len(pairs) - relaxed, "relaxed_pairs": relaxed,
        "unmatched_count": len(unmatched), "unmatched": unmatched,
        "case_control_question_overlap": 0, "case_control_coco_image_overlap": 0,
        "control_unique_images": len(controls), "selected_images_sha256_verified": verified_images,
        "changed_groups": dict(Counter(f"{r['change_direction']}|gt_{r['ground_truth']}" for r in cases)),
        "case_annotation_issues": [{"question_id": r["question_id"], "issues": r["annotation_issues"]} for r in cases if r["annotation_issues"]],
        "all_dataset_annotation_issue_count": sum(bool(r["annotation_issues"]) for r in rows.values()),
        **pool_info, "warnings": warnings, "output_dir": str(paths["output_dir"]),
        "status_semantics": "PASS verifies freezing and declared matching rules only, not repeatability or causality.",
    }
    files = {
        "changed30.json": manifest("All original C/D-changed cases, including unmatched cases", cases, provenance_hash, args.seed),
        "controls.json": manifest("Matched controls equal under C/D in the historical source run", controls, provenance_hash, args.seed),
        "combined.json": manifest("Union for later paired repeatability experiments", sorted(cases + controls, key=lambda r: id_key(r["question_id"])), provenance_hash, args.seed),
        "pairs.json": {"schema_version": 1, "matching_policy": POLICY, "seed": args.seed, "pairs": pairs, "unmatched": unmatched},
        "freeze_provenance.json": provenance, "freeze_report.json": freeze_report,
    }
    write_bundle(paths["output_dir"], files)
    print(json.dumps(freeze_report, indent=2, ensure_ascii=False))
    return 0 if not unmatched else 2


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except (ValueError, KeyError, OSError, TypeError) as error:
        raise SystemExit(f"FAIL: {error}")

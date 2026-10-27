#!/usr/bin/env python3
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from PIL import Image
import random
import re


_BINARY_ANSWER_RE = re.compile(r"^\s*(yes|no)\b", re.IGNORECASE)
_STRATA = ("FN", "FP", "TP", "TN")


def normalize_answer(text):
    match = _BINARY_ANSWER_RE.match(str(text))
    return match.group(1).lower() if match else "invalid"


def classify_stratum(prediction, ground_truth):
    if prediction not in {"yes", "no"} or ground_truth not in {"yes", "no"}:
        raise ValueError(
            f"cannot classify non-binary pair: prediction={prediction!r}, "
            f"ground_truth={ground_truth!r}"
        )
    if prediction == "yes" and ground_truth == "yes":
        return "TP"
    if prediction == "no" and ground_truth == "no":
        return "TN"
    if prediction == "no" and ground_truth == "yes":
        return "FN"
    return "FP"


def _read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc


def _sample_record(row):
    doc = row.get("doc") or {}
    metric = row.get("pope_accuracy") or {}
    question_id = str(metric.get("question_id", doc.get("question_id", "")))
    if not question_id:
        raise ValueError("sample row is missing question_id")
    prediction = normalize_answer(
        metric.get("prediction", (row.get("filtered_resps") or [""])[0])
    )
    ground_truth = normalize_answer(metric.get("ground_truth", doc.get("answer", "")))
    return {
        "doc_id": row.get("doc_id"),
        "question_id": question_id,
        "image_id": str(doc.get("id", "")),
        "question": doc.get("question", ""),
        "ground_truth": ground_truth,
        "original_d_prediction": prediction,
        "stratum": classify_stratum(prediction, ground_truth),
    }


def build_manifest(sample_log, output, seed, counts):
    sample_log = Path(sample_log)
    output = Path(output)
    normalized_counts = {stratum: int(counts.get(stratum, 0)) for stratum in _STRATA}
    if any(value < 0 for value in normalized_counts.values()):
        raise ValueError("manifest stratum counts must be non-negative")

    by_stratum = defaultdict(list)
    seen_question_ids = set()
    for row in _read_jsonl(sample_log):
        record = _sample_record(row)
        question_id = record["question_id"]
        if question_id in seen_question_ids:
            raise ValueError(f"duplicate question_id in sample log: {question_id}")
        seen_question_ids.add(question_id)
        by_stratum[record["stratum"]].append(record)

    rng = random.Random(seed)
    selected = []
    for stratum in _STRATA:
        available = by_stratum[stratum]
        requested = normalized_counts[stratum]
        if requested > len(available):
            raise ValueError(
                f"not enough {stratum} samples: requested {requested}, "
                f"available {len(available)}"
            )
        selected.extend(rng.sample(available, requested))

    selected.sort(key=lambda row: (row["doc_id"] is None, row["doc_id"]))
    manifest = {
        "schema_version": 1,
        "source_sample_log": str(sample_log.resolve()),
        "seed": int(seed),
        "requested_counts": normalized_counts,
        "actual_counts": dict(Counter(row["stratum"] for row in selected)),
        "samples": selected,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def wilson_interval(successes, total, z=1.96):
    successes = int(successes)
    total = int(total)
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("Wilson interval requires 0 <= successes <= total")
    if total == 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def verify_prediction_parity(original, new):
    original_ids = set(original)
    new_ids = set(new)
    if original_ids != new_ids:
        raise ValueError(
            "prediction parity ID mismatch: "
            f"missing_in_new={sorted(original_ids - new_ids)}, "
            f"unexpected_in_new={sorted(new_ids - original_ids)}"
        )
    changed = [
        question_id
        for question_id in sorted(original_ids)
        if normalize_answer(original[question_id]) != normalize_answer(new[question_id])
    ]
    if changed:
        raise ValueError(
            f"prediction parity failed for {len(changed)} samples: {changed[:20]}"
        )
    return {"matched": len(original_ids), "changed": 0, "question_ids": sorted(original_ids)}


def _paired_effect(rows, left, right):
    harm_ids = [
        row["question_id"]
        for row in rows
        if bool(row[left]["correct"]) and not bool(row[right]["correct"])
    ]
    benefit_ids = [
        row["question_id"]
        for row in rows
        if not bool(row[left]["correct"]) and bool(row[right]["correct"])
    ]
    denominator = len(rows)
    harm_interval = wilson_interval(len(harm_ids), denominator)
    benefit_interval = wilson_interval(len(benefit_ids), denominator)
    return {
        "left_condition": left,
        "right_condition": right,
        "denominator": denominator,
        "harm_count": len(harm_ids),
        "harm_rate": len(harm_ids) / denominator if denominator else 0.0,
        "harm_wilson_95": list(harm_interval),
        "harm_question_ids": harm_ids,
        "benefit_count": len(benefit_ids),
        "benefit_rate": len(benefit_ids) / denominator if denominator else 0.0,
        "benefit_wilson_95": list(benefit_interval),
        "benefit_question_ids": benefit_ids,
        "flip_count": len(harm_ids) + len(benefit_ids),
    }


def analyze_pairs(rows):
    required_conditions = {"A", "B", "C", "D"}
    seen = set()
    normalized_rows = []
    for row in rows:
        question_id = str(row["question_id"])
        if question_id in seen:
            raise ValueError(f"duplicate paired question_id: {question_id}")
        seen.add(question_id)
        missing = required_conditions - set(row)
        if missing:
            raise ValueError(f"paired row {question_id} is missing conditions: {sorted(missing)}")
        normalized_rows.append({**row, "question_id": question_id, "stratum": str(row["stratum"]).upper()})

    fn_rows = [row for row in normalized_rows if row["stratum"] == "FN"]
    fn_rescue_ids = [
        row["question_id"]
        for row in fn_rows
        if bool(row["C"]["correct"]) and not bool(row["D"]["correct"])
    ]
    fn_interval = wilson_interval(len(fn_rescue_ids), len(fn_rows))

    by_stratum = {}
    for stratum in _STRATA:
        stratum_rows = [row for row in normalized_rows if row["stratum"] == stratum]
        by_stratum[stratum] = {
            "count": len(stratum_rows),
            "selection": _paired_effect(stratum_rows, "C", "D"),
            "contrastive_decoding": _paired_effect(stratum_rows, "B", "C"),
        }

    return {
        "audit_schema_version": 1,
        "diagnostic_set_size": len(normalized_rows),
        "population_rate_warning": (
            "This stratified diagnostic set does not estimate population Accuracy, "
            "F1, or overall Selection Harm Rate."
        ),
        "recursive_pathway": _paired_effect(normalized_rows, "A", "B"),
        "contrastive_decoding": _paired_effect(normalized_rows, "B", "C"),
        "selection": _paired_effect(normalized_rows, "C", "D"),
        "fn_rescue": {
            "count": len(fn_rescue_ids),
            "denominator": len(fn_rows),
            "rate": len(fn_rescue_ids) / len(fn_rows) if fn_rows else 0.0,
            "wilson_95": list(fn_interval),
            "question_ids": fn_rescue_ids,
        },
        "by_original_d_stratum": by_stratum,
    }


def _load_manifest(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported manifest schema: {payload.get('schema_version')}")
    rows = payload.get("samples", [])
    by_id = {str(row["question_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("manifest contains duplicate question_id values")
    return by_id


def load_sample_predictions(path):
    predictions = {}
    for row in _read_jsonl(path):
        record = _sample_record(row)
        question_id = record["question_id"]
        if question_id in predictions:
            raise ValueError(f"duplicate question_id in condition log: {question_id}")
        prediction = record["original_d_prediction"]
        ground_truth = record["ground_truth"]
        predictions[question_id] = {
            "prediction": prediction,
            "ground_truth": ground_truth,
            "correct": prediction == ground_truth,
        }
    return predictions


def join_condition_logs(manifest_path, condition_paths):
    manifest = _load_manifest(manifest_path)
    wanted = set(manifest)
    condition_rows = {}
    for condition in ("A", "B", "C", "D"):
        rows = load_sample_predictions(condition_paths[condition])
        actual = set(rows)
        if actual != wanted:
            raise ValueError(
                f"condition {condition} ID mismatch: "
                f"missing={sorted(wanted - actual)}, unexpected={sorted(actual - wanted)}"
            )
        condition_rows[condition] = rows

    paired = []
    for question_id, manifest_row in manifest.items():
        paired.append(
            {
                "question_id": question_id,
                "stratum": manifest_row["stratum"],
                **{
                    condition: condition_rows[condition][question_id]
                    for condition in ("A", "B", "C", "D")
                },
            }
        )
    return paired


def _report_markdown(report):
    selection = report["selection"]
    cd = report["contrastive_decoding"]
    fn = report["fn_rescue"]
    lines = [
        "# SECOND Token Selection Audit v1 Report",
        "",
        f"Diagnostic samples: {report['diagnostic_set_size']}",
        "",
        f"> {report['population_rate_warning']}",
        "",
        "| Comparison | Harm | Benefit | Denominator |",
        "| --- | ---: | ---: | ---: |",
        f"| B→C Contrastive Decoding | {cd['harm_count']} | {cd['benefit_count']} | {cd['denominator']} |",
        f"| C→D Token Selection under CD | {selection['harm_count']} | {selection['benefit_count']} | {selection['denominator']} |",
        "",
        f"FN rescue: {fn['count']}/{fn['denominator']} ({fn['rate']:.4f})",
        "",
        "Selection Harm question IDs: "
        + (", ".join(selection["harm_question_ids"]) or "none"),
        "",
        "Selection Benefit question IDs: "
        + (", ".join(selection["benefit_question_ids"]) or "none"),
        "",
    ]
    return "\n".join(lines)


def build_full_index(samples_jsonl, output):
    """Generate full index from samples without sampling."""
    samples_jsonl = Path(samples_jsonl)
    output = Path(output)

    # Load POPE dataset to get image_source mapping
    from datasets import load_dataset
    try:
        pope_dataset = load_dataset("lmms-lab/POPE", "Full", split="popular")
    except Exception as e:
        raise RuntimeError(f"Failed to load POPE dataset. Ensure internet connection or local cache exists: {e}")

    # Build mapping from question_id to COCO image_id
    question_id_to_coco_id = {}
    for item in pope_dataset:
        qid = str(item["question_id"])
        # Extract COCO image ID from image_source (e.g., "COCO_val2014_000000310196" -> "310196")
        image_source = item["image_source"]
        if image_source.startswith("COCO_val2014_"):
            coco_id = image_source.replace("COCO_val2014_", "").lstrip("0") or "0"
        else:
            coco_id = image_source
        question_id_to_coco_id[qid] = coco_id

    records = []
    seen_question_ids = set()

    for row in _read_jsonl(samples_jsonl):
        record = _sample_record(row)
        question_id = record["question_id"]

        if question_id in seen_question_ids:
            raise ValueError(f"duplicate question_id in sample log: {question_id}")
        seen_question_ids.add(question_id)

        gt = record["ground_truth"]
        pred = record["original_d_prediction"]

        # Get real COCO image_id from POPE dataset
        coco_image_id = question_id_to_coco_id.get(question_id, record["image_id"])

        records.append({
            "question_id": question_id,
            "image_id": coco_image_id,
            "question": record["question"],
            "gt": gt,
            "pred": pred,
            "correct": gt == pred,
            "stratum": record["stratum"],
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return len(records)


def summarize_index(index_path, output):
    """Compute statistics from full index."""
    index_path = Path(index_path)
    output = Path(output)

    records = list(_read_jsonl(index_path))
    total = len(records)

    counts = Counter(r["stratum"] for r in records)
    correct = sum(1 for r in records if r["correct"])

    tp = counts.get("TP", 0)
    tn = counts.get("TN", 0)
    fp = counts.get("FP", 0)
    fn = counts.get("FN", 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    yes_count = sum(1 for r in records if r["pred"] == "yes")
    yes_ratio = yes_count / total if total > 0 else 0.0

    summary = {
        "total": total,
        "counts": {
            "TP": tp,
            "TN": tn,
            "FN": fn,
            "FP": fp,
        },
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": yes_ratio,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return summary


def map_images(index_path, coco_dir, output):
    """Map question_ids to COCO image paths."""
    index_path = Path(index_path)
    coco_dir = Path(coco_dir)
    output = Path(output)

    if not coco_dir.exists():
        raise FileNotFoundError(f"COCO directory not found: {coco_dir}")

    records = list(_read_jsonl(index_path))
    mappings = []

    for record in records:
        question_id = record["question_id"]
        image_id = record["image_id"]

        # POPE uses COCO val2014
        # image_id is typically the COCO ID without leading zeros
        # COCO filename format: COCO_val2014_000000{id}.jpg
        coco_id_padded = image_id.zfill(6)
        image_filename = f"COCO_val2014_000000{coco_id_padded}.jpg"
        image_path = coco_dir / image_filename

        mapping = {
            "question_id": question_id,
            "image_id": image_id,
            "image_path": str(image_path),
            "image_filename": image_filename,
            "readable": False,
            "width": 0,
            "height": 0,
            "sha256": "",
        }

        if image_path.exists():
            try:
                with Image.open(image_path) as img:
                    mapping["width"] = img.width
                    mapping["height"] = img.height
                    mapping["readable"] = True

                # Compute SHA256
                with open(image_path, "rb") as f:
                    mapping["sha256"] = hashlib.sha256(f.read()).hexdigest()

            except Exception as e:
                mapping["error"] = str(e)

        mappings.append(mapping)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for mapping in mappings:
            f.write(json.dumps(mapping, ensure_ascii=False) + "\n")

    readable_count = sum(1 for m in mappings if m["readable"])
    return {"total": len(mappings), "readable": readable_count}


def extract_features(index_path, image_mapping_path, output):
    """Extract lightweight features for diagnostic selection."""
    index_path = Path(index_path)
    image_mapping_path = Path(image_mapping_path)
    output = Path(output)

    index_records = {r["question_id"]: r for r in _read_jsonl(index_path)}
    image_mappings = {m["question_id"]: m for m in _read_jsonl(image_mapping_path)}

    relation_keywords = ["next to", "behind", "in front of", "on", "under", "beside", "near", "above", "below"]

    features = []
    for question_id, record in index_records.items():
        mapping = image_mappings.get(question_id, {})
        question = record.get("question", "").lower()

        # Detect relation questions
        has_relation = any(keyword in question for keyword in relation_keywords)

        # Image complexity proxy: aspect ratio
        width = mapping.get("width", 0)
        height = mapping.get("height", 0)
        aspect_ratio = width / height if height > 0 else 0.0

        # Question length
        question_length = len(question.split())

        # Parse target object from question
        # POPE format: "Is there a {object} in the image?"
        target_match = re.search(r"is there an? (.+?) in the image", question)
        target_object = target_match.group(1) if target_match else ""

        feature = {
            "question_id": question_id,
            "stratum": record["stratum"],
            "image_id": record["image_id"],
            "gt": record["gt"],
            "pred": record["pred"],
            "correct": record["correct"],
            "question_length": question_length,
            "has_relation": has_relation,
            "target_object": target_object,
            "image_width": width,
            "image_height": height,
            "aspect_ratio": aspect_ratio,
            "image_readable": mapping.get("readable", False),
        }

        features.append(feature)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for feature in features:
            f.write(json.dumps(feature, ensure_ascii=False) + "\n")

    return len(features)


def select_diagnostic_samples(features_path, strata_config_path, output, seed):
    """Select diagnostic samples using stratified sampling."""
    features_path = Path(features_path)
    strata_config_path = Path(strata_config_path)
    output = Path(output)

    # Load strata configuration
    strata_config = json.loads(strata_config_path.read_text(encoding="utf-8"))

    # Load features
    features = list(_read_jsonl(features_path))

    # Group by visual stratum
    by_visual_stratum = defaultdict(list)

    for feature in features:
        stratum = feature["stratum"]
        has_relation = feature["has_relation"]
        image_readable = feature["image_readable"]

        # Skip unreadable images
        if not image_readable:
            continue

        # Define visual strata
        if stratum == "TP":
            # For TP, split by question complexity
            if has_relation:
                visual_stratum = "TP_relation"
            else:
                visual_stratum = "TP_simple"
        elif stratum == "TN":
            visual_stratum = "TN_correct"
        elif stratum == "FN":
            # For FN, split by relation vs simple
            if has_relation:
                visual_stratum = "FN_relation"
            else:
                visual_stratum = "FN_simple"
        elif stratum == "FP":
            visual_stratum = "FP_hallucination"
        else:
            continue

        feature["visual_stratum"] = visual_stratum
        by_visual_stratum[visual_stratum].append(feature)

    # Sample from each stratum
    rng = random.Random(seed)
    selected = []

    for visual_stratum, requested_count in strata_config.items():
        available = by_visual_stratum[visual_stratum]

        if requested_count > len(available):
            print(f"Warning: requested {requested_count} samples from {visual_stratum}, "
                  f"but only {len(available)} available. Taking all.")
            selected.extend(available)
        else:
            selected.extend(rng.sample(available, requested_count))

    # Create manifest
    manifest = {
        "schema_version": 2,
        "source_features": str(features_path.resolve()),
        "seed": seed,
        "strata_config": strata_config,
        "actual_counts": dict(Counter(s["visual_stratum"] for s in selected)),
        "samples": sorted(selected, key=lambda x: x["question_id"]),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return manifest


def _add_sample_parser(subparsers):
    parser = subparsers.add_parser("sample", help="build a stratified POPE audit manifest")
    parser.add_argument("--samples-jsonl", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fn", type=int, default=30)
    parser.add_argument("--fp", type=int, default=10)
    parser.add_argument("--tp", type=int, default=20)
    parser.add_argument("--tn", type=int, default=20)
    return parser


def _add_parity_parser(subparsers):
    parser = subparsers.add_parser("parity", help="verify new D predictions against the original D run")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--new", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _add_analyze_parser(subparsers):
    parser = subparsers.add_parser("analyze", help="analyze paired A/B/C/D audit logs")
    parser.add_argument("--manifest", required=True, type=Path)
    for condition in ("a", "b", "c", "d"):
        parser.add_argument(f"--{condition}", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _add_index_parser(subparsers):
    parser = subparsers.add_parser("index", help="generate full index from samples (no sampling)")
    parser.add_argument("--samples-jsonl", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _add_summarize_parser(subparsers):
    parser = subparsers.add_parser("summarize", help="compute statistics from full index")
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _add_map_images_parser(subparsers):
    parser = subparsers.add_parser("map_images", help="map question_ids to COCO image paths")
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--coco-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _add_extract_features_parser(subparsers):
    parser = subparsers.add_parser("extract_features", help="extract lightweight features for diagnostic selection")
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--image-mapping", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _add_select_diagnostic_parser(subparsers):
    parser = subparsers.add_parser("select_diagnostic", help="select diagnostic samples using stratified sampling")
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--strata-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv=None):
    parser = argparse.ArgumentParser(description="SECOND Token Selection Audit utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_sample_parser(subparsers)
    _add_parity_parser(subparsers)
    _add_analyze_parser(subparsers)
    _add_index_parser(subparsers)
    _add_summarize_parser(subparsers)
    _add_map_images_parser(subparsers)
    _add_extract_features_parser(subparsers)
    _add_select_diagnostic_parser(subparsers)
    args = parser.parse_args(argv)

    if args.command == "sample":
        manifest = build_manifest(
            args.samples_jsonl,
            args.output,
            args.seed,
            {"FN": args.fn, "FP": args.fp, "TP": args.tp, "TN": args.tn},
        )
        counts = manifest["actual_counts"]
        print(
            "manifest written to "
            f"{args.output}: FN={counts.get('FN', 0)} FP={counts.get('FP', 0)} "
            f"TP={counts.get('TP', 0)} TN={counts.get('TN', 0)}"
        )
        return 0

    if args.command == "parity":
        manifest = _load_manifest(args.manifest)
        original_rows = load_sample_predictions(args.original)
        new_rows = load_sample_predictions(args.new)
        missing_original = set(manifest) - set(original_rows)
        if missing_original:
            raise ValueError(f"original D log is missing manifest IDs: {sorted(missing_original)}")
        original = {question_id: original_rows[question_id]["prediction"] for question_id in manifest}
        new = {question_id: row["prediction"] for question_id, row in new_rows.items()}
        report = verify_prediction_parity(original, new)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"D prediction parity passed for {report['matched']} samples")
        return 0

    if args.command == "analyze":
        paired = join_condition_logs(
            args.manifest,
            {"A": args.a, "B": args.b, "C": args.c, "D": args.d},
        )
        report = analyze_pairs(paired)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (args.output / "report.md").write_text(
            _report_markdown(report), encoding="utf-8"
        )
        print(f"audit report written to {args.output}")
        return 0

    if args.command == "index":
        count = build_full_index(args.samples_jsonl, args.output)
        print(f"full index written to {args.output}: {count} samples")
        return 0

    if args.command == "summarize":
        summary = summarize_index(args.index, args.output)
        print(f"summary written to {args.output}")
        print(f"Total: {summary['total']}, Accuracy: {summary['accuracy']:.4f}, "
              f"F1: {summary['f1']:.4f}")
        print(f"Counts: TP={summary['counts']['TP']} TN={summary['counts']['TN']} "
              f"FN={summary['counts']['FN']} FP={summary['counts']['FP']}")
        return 0

    if args.command == "map_images":
        result = map_images(args.index, args.coco_dir, args.output)
        print(f"image mapping written to {args.output}")
        print(f"Total: {result['total']}, Readable: {result['readable']}")
        return 0

    if args.command == "extract_features":
        count = extract_features(args.index, args.image_mapping, args.output)
        print(f"features written to {args.output}: {count} samples")
        return 0

    if args.command == "select_diagnostic":
        manifest = select_diagnostic_samples(args.features, args.strata_config, args.output, args.seed)
        print(f"diagnostic manifest written to {args.output}")
        print(f"Selected {len(manifest['samples'])} samples")
        print(f"Counts by visual stratum: {manifest['actual_counts']}")
        return 0

    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

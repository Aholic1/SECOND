#!/usr/bin/env python3
"""Generate a validated paired B/C/D report from lmms-eval run directories."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


CONDITIONS = ("B", "C", "D")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b-run", required=True, type=Path)
    parser.add_argument("--c-run", required=True, type=Path)
    parser.add_argument("--d-run", required=True, type=Path)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing required file: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def normalize_answer(value: Any) -> str:
    text = str(value).strip().lower()
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return "invalid"


def samples_file(run_dir: Path) -> Path:
    matches = sorted((run_dir / "results").glob("*/*_samples_*.jsonl"))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one sample JSONL under {run_dir / 'results'}, "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]


def load_samples(path: Path, expected_count: int) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            metric = raw.get("pope_accuracy")
            doc = raw.get("doc")
            if not isinstance(metric, dict) or not isinstance(doc, dict):
                raise ValueError(f"missing pope_accuracy/doc in {path}:{line_number}")
            question_id = str(metric.get("question_id", doc.get("question_id", "")))
            if not question_id:
                raise ValueError(f"missing question_id in {path}:{line_number}")
            if question_id != str(doc.get("question_id")):
                raise ValueError(f"question_id disagreement in {path}:{line_number}")
            if question_id in rows:
                raise ValueError(f"duplicate question_id {question_id} in {path}")
            ground_truth = normalize_answer(metric.get("ground_truth", raw.get("target")))
            if ground_truth not in {"yes", "no"}:
                raise ValueError(
                    f"invalid ground truth for question_id {question_id} in {path}"
                )
            rows[question_id] = {
                "question_id": question_id,
                "doc_id": raw.get("doc_id"),
                "image_id": doc.get("id"),
                "question": str(doc.get("question", "")),
                "category": doc.get("category"),
                "ground_truth": ground_truth,
                "prediction": normalize_answer(metric.get("prediction")),
            }
    if len(rows) != expected_count:
        raise ValueError(f"{path}: expected {expected_count} samples, found {len(rows)}")
    return rows


def id_key(question_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(question_id))
    except ValueError:
        return (1, question_id)


def metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    tp = sum(r["ground_truth"] == "yes" and r["prediction"] == "yes" for r in rows)
    tn = sum(r["ground_truth"] == "no" and r["prediction"] == "no" for r in rows)
    fp = sum(r["ground_truth"] == "no" and r["prediction"] == "yes" for r in rows)
    fn = sum(r["ground_truth"] == "yes" and r["prediction"] == "no" for r in rows)
    invalid = sum(r["prediction"] == "invalid" for r in rows)
    correct = tp + tn
    count = len(rows)
    return {
        "count": count,
        "correct": correct,
        "accuracy": correct / count if count else None,
        "invalid": invalid,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def mcnemar_exact_p_value(corrected: int, degraded: int) -> float:
    discordant = corrected + degraded
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(corrected, degraded) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def pair_effect(
    samples: dict[str, dict[str, dict[str, Any]]],
    left: str,
    right: str,
    question_ids: Iterable[str],
) -> dict[str, Any]:
    ids = sorted(question_ids, key=id_key)
    left_rows = [samples[left][qid] for qid in ids]
    right_rows = [samples[right][qid] for qid in ids]
    changed = [qid for qid in ids if samples[left][qid]["prediction"] != samples[right][qid]["prediction"]]
    corrected = [
        qid
        for qid in ids
        if samples[left][qid]["prediction"] != samples[left][qid]["ground_truth"]
        and samples[right][qid]["prediction"] == samples[right][qid]["ground_truth"]
    ]
    degraded = [
        qid
        for qid in ids
        if samples[left][qid]["prediction"] == samples[left][qid]["ground_truth"]
        and samples[right][qid]["prediction"] != samples[right][qid]["ground_truth"]
    ]
    left_accuracy = metrics(left_rows)["accuracy"]
    right_accuracy = metrics(right_rows)["accuracy"]
    delta = right_accuracy - left_accuracy
    return {
        "left": left,
        "right": right,
        "count": len(ids),
        "left_accuracy": left_accuracy,
        "right_accuracy": right_accuracy,
        "accuracy_delta": delta,
        "accuracy_delta_percentage_points": 100.0 * delta,
        "prediction_changed_count": len(changed),
        "corrected_count": len(corrected),
        "corrected_question_ids": corrected,
        "degraded_count": len(degraded),
        "degraded_question_ids": degraded,
        "net_corrected_minus_degraded": len(corrected) - len(degraded),
        "mcnemar_exact_two_sided_p_value": mcnemar_exact_p_value(
            len(corrected), len(degraded)
        ),
    }


def target_object(question: str) -> str | None:
    match = re.fullmatch(r"Is there (?:an? )?(.+?) in the image\?", question.strip())
    return match.group(1) if match else None


def validate_runs(
    run_dirs: dict[str, Path], expected_count: int
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    manifests: dict[str, dict[str, Any]] = {}
    validations: dict[str, dict[str, Any]] = {}
    for condition, run_dir in run_dirs.items():
        manifests[condition] = read_json(run_dir / "run_manifest.json")
        validations[condition] = read_json(run_dir / "validation_report.json")
        manifest = manifests[condition]
        validation = validations[condition]
        if manifest.get("condition") != condition:
            raise ValueError(f"{run_dir}: manifest condition is not {condition}")
        if validation.get("status") != "PASS" or validation.get("condition") != condition:
            raise ValueError(f"{run_dir}: validation gate did not PASS for {condition}")
        for source, value in (("manifest", manifest), ("validation", validation)):
            if value.get("expected_count") != expected_count:
                raise ValueError(
                    f"{run_dir}: {source} expected_count is {value.get('expected_count')}, "
                    f"not {expected_count}"
                )
        if validation.get("captured_count") != expected_count:
            raise ValueError(f"{run_dir}: captured_count is not {expected_count}")
        if validation.get("errors"):
            raise ValueError(f"{run_dir}: validation errors are present")
    commits = {m.get("git_commit") for m in manifests.values()}
    if len(commits) != 1 or None in commits:
        raise ValueError(f"B/C/D git commits do not match: {commits}")
    parameters = [m.get("second_parameters") for m in manifests.values()]
    if any(value != parameters[0] for value in parameters[1:]):
        raise ValueError("B/C/D SECOND parameters do not match")
    return manifests, validations


def main() -> None:
    args = parse_args()
    run_dirs = {"B": args.b_run, "C": args.c_run, "D": args.d_run}
    manifests, validations = validate_runs(run_dirs, args.expected_count)
    sample_paths = {condition: samples_file(path) for condition, path in run_dirs.items()}
    samples = {
        condition: load_samples(sample_paths[condition], args.expected_count)
        for condition in CONDITIONS
    }

    id_sets = {condition: set(rows) for condition, rows in samples.items()}
    if any(id_sets[condition] != id_sets["B"] for condition in ("C", "D")):
        raise ValueError("B/C/D question_id sets do not match")
    question_ids = sorted(id_sets["B"], key=id_key)
    for qid in question_ids:
        reference = samples["B"][qid]
        for condition in ("C", "D"):
            candidate = samples[condition][qid]
            for field in ("doc_id", "image_id", "question", "category", "ground_truth"):
                if candidate[field] != reference[field]:
                    raise ValueError(
                        f"question_id {qid}: {field} differs between B and {condition}"
                    )

    overall = {
        condition: metrics(samples[condition][qid] for qid in question_ids)
        for condition in CONDITIONS
    }
    by_ground_truth: dict[str, Any] = {}
    for ground_truth in ("yes", "no"):
        subset = [qid for qid in question_ids if samples["B"][qid]["ground_truth"] == ground_truth]
        by_ground_truth[ground_truth] = {
            **{
                condition: metrics(samples[condition][qid] for qid in subset)
                for condition in CONDITIONS
            },
            "B_to_C": pair_effect(samples, "B", "C", subset),
            "C_to_D": pair_effect(samples, "C", "D", subset),
        }

    changed_ids = [
        qid
        for qid in question_ids
        if samples["B"][qid]["prediction"] != samples["C"][qid]["prediction"]
        or samples["C"][qid]["prediction"] != samples["D"][qid]["prediction"]
    ]
    changed_samples = []
    for qid in changed_ids:
        base = samples["B"][qid]
        changed_samples.append(
            {
                "question_id": qid,
                "doc_id": base["doc_id"],
                "image_id": base["image_id"],
                "question": base["question"],
                "target_object": target_object(base["question"]),
                "ground_truth": base["ground_truth"],
                "B_prediction": samples["B"][qid]["prediction"],
                "C_prediction": samples["C"][qid]["prediction"],
                "D_prediction": samples["D"][qid]["prediction"],
                "B_to_C_outcome": (
                    "corrected"
                    if samples["B"][qid]["prediction"] != base["ground_truth"]
                    and samples["C"][qid]["prediction"] == base["ground_truth"]
                    else "degraded"
                    if samples["B"][qid]["prediction"] == base["ground_truth"]
                    and samples["C"][qid]["prediction"] != base["ground_truth"]
                    else "unchanged_correctness"
                ),
                "C_to_D_outcome": (
                    "corrected"
                    if samples["C"][qid]["prediction"] != base["ground_truth"]
                    and samples["D"][qid]["prediction"] == base["ground_truth"]
                    else "degraded"
                    if samples["C"][qid]["prediction"] == base["ground_truth"]
                    and samples["D"][qid]["prediction"] != base["ground_truth"]
                    else "unchanged_correctness"
                ),
            }
        )

    report = {
        "schema_version": 1,
        "mode": "full",
        "expected_count": args.expected_count,
        "inputs": {condition: str(path) for condition, path in run_dirs.items()},
        "sample_files": {condition: str(path) for condition, path in sample_paths.items()},
        "git_commits": {
            condition: manifests[condition]["git_commit"] for condition in CONDITIONS
        },
        "second_parameters": {
            condition: manifests[condition]["second_parameters"] for condition in CONDITIONS
        },
        "condition_definitions": {
            "B": manifests["B"].get(
                "condition_definition", "recursive all-token without contrastive decoding"
            ),
            "C": manifests["C"].get(
                "condition_definition", "recursive all-token with contrastive decoding"
            ),
            "D": "SECOND-selected tokens with contrastive decoding",
        },
        "validation_status": {
            condition: validations[condition]["status"] for condition in CONDITIONS
        },
        "overall": overall,
        "B_to_C_cd_effect": pair_effect(samples, "B", "C", question_ids),
        "C_to_D_selection_effect": pair_effect(samples, "C", "D", question_ids),
        "by_ground_truth": by_ground_truth,
        "changed_samples": changed_samples,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"PASS: wrote paired B/C/D report to {args.output}")
    print(json.dumps({
        "overall": overall,
        "B_to_C_cd_effect": report["B_to_C_cd_effect"],
        "C_to_D_selection_effect": report["C_to_D_selection_effect"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

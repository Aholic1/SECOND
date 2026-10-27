#!/usr/bin/env python3
"""Repeat C/D inference for the two prediction-unstable Step 5 questions.

This is a focused follow-up to step5_repeatability.py. It reuses the exact
frozen questions, source snapshot, seeds, task and two-process GPU 0,1 setup;
it does not alter model code or overwrite the original study.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "audit"))
from step5_repeatability import (  # noqa: E402
    PARAMS,
    SEEDS,
    canonical,
    capture_records,
    check,
    digest,
    load_bundle,
    read,
    snapshot,
    write,
)

QUESTION_IDS = ("457", "1512")
ADDITIONAL_LABELS = ("C3", "D3", "C4", "D4", "C5", "D5")
DEFAULT_PARENT = ROOT / "audit/runs/step5_repeatability_v1"
DEFAULT_STUDY = ROOT / "audit/runs/step5_unstable_repeat_v1"
DEFAULT_BUNDLE = ROOT / "audit/manifests/step5_changed30_matched_v1"
MAX_USED_MIB = 2_048


def configure_runtime():
    check(os.environ.get("CONDA_DEFAULT_ENV") == "SECOND_clean", "activate SECOND_clean first")
    requested = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")
    check(requested == "0,1", "this follow-up is fixed to physical GPUs 0,1")
    check(os.environ.get("NUM_PROCESSES", "2") == "2", "NUM_PROCESSES must be 2")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ.setdefault("NUM_PROCESSES", "2")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")


def gpu_preflight():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True
    )
    inventory = {}
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 4:
            inventory[parts[0]] = tuple(int(part) for part in parts[1:4])
    for index in ("0", "1"):
        check(index in inventory, f"physical GPU {index} was not found")
        used, total, utilization = inventory[index]
        check(used <= MAX_USED_MIB,
              f"GPU {index}: {used}/{total} MiB in use; this study requires an idle GPU "
              f"(allowing at most {MAX_USED_MIB} MiB for display/context allocations). "
              "Wait for other jobs to finish; do not bypass this check.")
        print(f"GPU {index}: {used}/{total} MiB in use, utilization {utilization}%")


def verify_parent(parent: Path, bundle: Path):
    plan_path = parent / "study_plan.json"
    report_path = parent / "repeatability_report.json"
    check(plan_path.is_file() and report_path.is_file(),
          f"completed Step 5 report is missing under {parent}")
    parent_plan = read(plan_path)
    report = read(report_path)
    unstable = {str(qid) for qid in report.get("summary", {}).get("unstable_prediction_ids", [])}
    check(set(QUESTION_IDS) <= unstable,
          f"the parent report does not mark both {QUESTION_IDS} as prediction-unstable")
    samples = {str(row["question_id"]): row for row in report.get("samples", [])}
    for qid in QUESTION_IDS:
        check(qid in samples and samples[qid].get("outcome") == "repeat_unstable",
              f"{qid}: parent report no longer shows repeat_unstable")
    rows, identity = load_bundle(bundle, verify_images=True)
    check(identity == parent_plan["cohort"], "frozen cohort identity differs from the parent study")
    by_id = {str(row["question_id"]): row for row in rows}
    check(set(QUESTION_IDS) <= set(by_id), "one or more unstable questions are absent from the frozen manifest")
    for qid in QUESTION_IDS:
        check(by_id[qid]["role"] == "changed_case", f"{qid} is not in the frozen changed-case group")
    state = snapshot()
    check(state == parent_plan["snapshot"],
          "current commit/source/runtime/model snapshot differs from Step 5; stop and investigate")
    return parent_plan, report, samples, [by_id[qid] for qid in QUESTION_IDS], identity, state


def prepare(parent: Path, bundle: Path, study: Path):
    check(not study.exists(), f"follow-up directory already exists: {study}; choose a new versioned path")
    parent_plan, report, report_samples, rows, identity, state = verify_parent(parent, bundle)
    study.mkdir(parents=True)
    manifest_path = study / "unstable_457_1512_manifest.json"
    manifest = {
        "schema_version": 1,
        "expected_count": len(rows),
        "description": "Exact subset of the frozen Step 5 changed30_matched cohort; labels unchanged.",
        "parent_combined_sha256": identity["combined_sha256"],
        "samples": rows,
    }
    write(manifest_path, manifest)
    plan = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "question_ids": list(QUESTION_IDS),
        "additional_labels": list(ADDITIONAL_LABELS),
        "run_order": list(ADDITIONAL_LABELS),
        "parent_study": str(parent.resolve()),
        "parent_study_plan_sha256": digest(parent / "study_plan.json"),
        "parent_report_sha256": digest(parent / "repeatability_report.json"),
        "parent_snapshot_sha256": parent_plan["snapshot_sha256"],
        "bundle": identity,
        "bundle_combined_sha256": identity["combined_sha256"],
        "subset_manifest_sha256": digest(manifest_path),
        "snapshot": state,
        "snapshot_sha256": canonical(state),
        "script_sha256": digest(Path(__file__)),
        "runtime": {"conda_environment": "SECOND_clean", "cuda_visible_devices": "0,1",
                    "num_processes": 2, "seeds": SEEDS, "second_parameters": PARAMS},
        "design": {
            "existing_repeats_per_condition": 2,
            "additional_repeats_per_condition": 3,
            "total_repeats_per_condition_after_completion": 5,
            "sampling": "Same seed and same two physical GPUs; descriptive within-setup repeatability check only.",
            "limitations": [
                "The two questions were selected because the original Step 5 report found instability; results do not estimate population accuracy.",
                "Five repeats are descriptive and do not establish universal determinism or a calibrated probability.",
                "C/D repeats use the same two-GPU launch setup; no cross-GPU or cross-seed claim is made.",
                "Original POPE labels remain unchanged; this is not a relabeling or causal intervention."
            ]
        },
        "parent_unstable_question_predictions": {
            qid: report_samples[qid]["predictions"] for qid in QUESTION_IDS
        },
    }
    write(study / "followup_plan.json", plan)
    print(f"Prepared {len(rows)} frozen questions. NO MODEL RUN.")
    print(f"Plan: {study / 'followup_plan.json'}")
    print(f"Manifest: {manifest_path}")


def ensure_port_free(port: int):
    with socket.socket() as probe:
        try:
            probe.bind(("", port))
        except OSError as exc:
            raise ValueError(f"port {port} is unavailable; choose another --port-base") from exc


def sample_log_path(run: Path):
    found = list((run / "results").glob("*/*_samples_*.jsonl"))
    check(len(found) == 1, f"expected one sample log in {run / 'results'}, found {len(found)}")
    return found[0]


def verify_run(run: Path, label: str, plan: dict, expected_rows: dict):
    complete_path = run / "run_complete.json"
    check(complete_path.is_file(), f"{label}: run is incomplete; preserve its directory and inspect console.log")
    complete = read(complete_path)
    check(complete.get("status") == "PASS" and complete.get("snapshot_sha256") == plan["snapshot_sha256"],
          f"{label}: completion/snapshot check failed")
    for name, expected_hash in complete["output_hashes"].items():
        check(digest(run / name) == expected_hash, f"{label}: output changed: {name}")
    manifest = read(run / "run_manifest.json")
    check(manifest.get("repeat_label") == label and manifest.get("condition") == label[0],
          f"{label}: run manifest identity differs")
    records = capture_records(run, label[0], expected_rows)
    check(records == read(run / "records.json"), f"{label}: captured artifacts differ from frozen records")
    validation = read(run / "validation_report.json")
    check(validation.get("status") == "PASS" and not validation.get("errors"),
          f"{label}: artifact validation did not pass")
    return records


def run_one(study: Path, label: str, rows: list[dict], plan: dict, port: int):
    run = study / label
    expected = {str(row["question_id"]): row for row in rows}
    if (run / "run_complete.json").is_file():
        verify_run(run, label, plan, expected)
        print(f"Verified completed run, skipping {label}")
        return
    check(not run.exists(), f"{label}: incomplete run directory exists; preserve/rename before retrying")
    check(snapshot() == plan["snapshot"], "source/runtime changed before the next run")
    check(digest(study / "unstable_457_1512_manifest.json") == plan["subset_manifest_sha256"],
          "subset manifest changed")
    gpu_preflight()
    ensure_port_free(port)
    run.mkdir()
    (run / "artifacts").mkdir()
    (run / "results").mkdir()
    condition = label[0]
    model_args = (
        "pretrained=liuhaotian/llava-v1.6-vicuna-7b,"
        f"audit_condition={condition},audit_output_dir={run / 'artifacts'},"
        "audit_token_idx=0,audit_artifact_mode=compact"
    )
    command = [
        sys.executable, "-m", "accelerate.commands.launch", "--num_processes=2",
        "--main_process_port", str(port), "-m", "lmms_eval", "--model", "llava",
        "--model_args", model_args, "--tasks", "pope_pop_audit", "--batch_size", "1",
        "--log_samples", "--log_samples_suffix", f"step5_unstable_{label}",
        "--output_path", str(run / "results"), "--generation_type", "recursion",
        "--fix_grid", "2x2", "--attention_thresholding_type", "attn_topk",
        "--attention_threshold", "1.0", "--positional_embedding_type", "bilinear_interpolation",
        "--stages", "-2", "-1", "0", "1", "--verbosity", "DEBUG",
        "--contrastive_alphas", "0.8", "0.8", "0.8", "--seed", SEEDS,
    ]
    write(run / "run_manifest.json", {
        "condition": condition, "repeat_label": label, "expected_count": len(rows),
        "git_commit": plan["snapshot"]["git_commit"],
        "snapshot_sha256": plan["snapshot_sha256"],
        "subset_manifest_sha256": plan["subset_manifest_sha256"],
        "cuda_visible_devices": "0,1", "num_processes": 2, "seed": SEEDS,
        "parameters": PARAMS, "command": command, "main_process_port": port,
    })
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    env["NUM_PROCESSES"] = "2"
    env["SECOND_AUDIT_MANIFEST"] = str((study / "unstable_457_1512_manifest.json").resolve())
    with (run / "console.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    check(code == 0, f"{label}: inference exited {code}; inspect {run / 'console.log'}")
    check(snapshot() == plan["snapshot"], "source/runtime changed during inference")
    validation_command = [
        sys.executable, str(ROOT / "audit/validate_step2_capture.py"),
        "--artifact-root", str(run / "artifacts"),
        "--baseline-predictions", str(ROOT / "audit/step1_full/pope_pop_3000_baseline/baseline_predictions.jsonl"),
        "--expected-manifest", env["SECOND_AUDIT_MANIFEST"], "--expected-count", str(len(rows)),
        "--condition", condition, "--artifact-mode", "compact",
        "--report", str(run / "validation_report.json"),
    ]
    with (run / "validation.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(validation_command, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, text=True)
    check(result.returncode == 0, f"{label}: validation failed; inspect {run / 'validation.log'}")
    records = capture_records(run, condition, expected)
    write(run / "records.json", records)
    log_path = sample_log_path(run)
    outputs = ["records.json", "validation_report.json", "run_manifest.json", "console.log",
               "validation.log", str(log_path.relative_to(run))]
    write(run / "run_complete.json", {
        "status": "PASS", "snapshot_sha256": plan["snapshot_sha256"],
        "output_hashes": {name: digest(run / name) for name in outputs},
        "semantics": "Structure and artifact validation passed; prediction stability is reported separately.",
    })
    verify_run(run, label, plan, expected)
    print(f"Completed {label}")


def mask_signature(record):
    return record["mask_hashes"]


def summarize(study: Path):
    plan = read(study / "followup_plan.json")
    check(plan.get("script_sha256") == digest(Path(__file__)), "follow-up script changed since preparation")
    parent = Path(plan["parent_study"])
    check(digest(parent / "study_plan.json") == plan["parent_study_plan_sha256"], "parent study plan changed")
    check(digest(parent / "repeatability_report.json") == plan["parent_report_sha256"], "parent report changed")
    check(digest(study / "unstable_457_1512_manifest.json") == plan["subset_manifest_sha256"], "subset manifest changed")
    parent_plan = read(parent / "study_plan.json")
    check(parent_plan["snapshot_sha256"] == plan["parent_snapshot_sha256"], "parent snapshot reference differs")
    bundle_rows, identity = load_bundle(plan["bundle"]["folder"], verify_images=True)
    check(identity == plan["bundle"], "frozen bundle identity changed")
    full_expected = {str(row["question_id"]): row for row in bundle_rows}
    rows = read(study / "unstable_457_1512_manifest.json")["samples"]
    expected = {str(row["question_id"]): row for row in rows}
    data = {}
    for label in ("C1", "C2", "D1", "D2"):
        run = parent / label
        complete = read(run / "run_complete.json")
        check(complete.get("status") == "PASS" and complete.get("snapshot_sha256") == parent_plan["snapshot_sha256"],
              f"parent {label}: run completion/snapshot check failed")
        for name, expected_hash in complete.get("output_hashes", {}).items():
            check(digest(run / name) == expected_hash, f"parent {label}: output changed: {name}")
        records = capture_records(run, label[0], full_expected)
        check(records == read(run / "records.json"), f"parent {label}: artifacts differ from stored records")
        data[label] = {qid: records[qid] for qid in QUESTION_IDS}
    for label in ADDITIONAL_LABELS:
        data[label] = verify_run(study / label, label, plan, expected)

    samples = []
    for row in rows:
        qid = str(row["question_id"])
        observations = {label: data[label][qid] for label in ("C1", "C2", "C3", "C4", "C5", "D1", "D2", "D3", "D4", "D5")}
        for field in ("prompt", "generation_config", "image_transform"):
            check(len({json.dumps(observations[label][field], sort_keys=True, ensure_ascii=False)
                       for label in observations}) == 1,
                  f"{qid}: {field} differs across the ten repeat records")
        condition_summary = {}
        for condition in "CD":
            labels = [f"{condition}{index}" for index in range(1, 6)]
            predictions = [observations[label]["prediction"] for label in labels]
            margins = [observations[label]["yes_no_margin"] for label in labels]
            condition_summary[condition] = {
                "prediction_counts": dict(sorted(Counter(predictions).items())),
                "prediction_stable_across_five_runs": len(set(predictions)) == 1,
                "correct_count_out_of_five": sum(p == row["ground_truth"] for p in predictions),
                "final_token_sequence_distinct_count": len({tuple(observations[label]["final_token_ids"]) for label in labels}),
                "distinct_stage_mask_hash_sets": len({canonical(mask_signature(observations[label])) for label in labels}),
                "distinct_stage_selection_signal_hash_sets": len({canonical(observations[label]["attention_hashes"]) for label in labels}),
                "yes_no_margin_values": margins,
                "yes_no_margin_median": median(margins),
                "yes_no_margin_min": min(margins),
                "yes_no_margin_max": max(margins),
            }
        paired = []
        for repeat in range(1, 6):
            c = observations[f"C{repeat}"]["prediction"]
            d = observations[f"D{repeat}"]["prediction"]
            gt = row["ground_truth"]
            paired.append({"repeat": repeat, "C": c, "D": d,
                           "selection_changed_prediction": c != d,
                           "corrected_by_D": c != gt and d == gt,
                           "degraded_by_D": c == gt and d != gt})
        samples.append({
            "question_id": qid, "question": row["question"], "ground_truth": row["ground_truth"],
            "target_object": row["target_object"], "image_id": row["coco_image_id"],
            "historical_predictions": {c: row[f"{c}_prediction"] for c in "CD"},
            "condition_summary": condition_summary,
            "paired_C_D_repeats": paired,
        })
    result = {
        "schema_version": 1,
        "status": "PASS",
        "study_plan_sha256": digest(study / "followup_plan.json"),
        "parent_study": str(parent),
        "question_ids": list(QUESTION_IDS),
        "repeats_per_condition": 5,
        "samples": samples,
        "limitations": plan["design"]["limitations"],
        "interpretation": "Counts and margins describe only these outcome-selected questions under this fixed setup; they are not population estimates or causal evidence.",
    }
    write(study / "unstable_repeatability_report.json", result)
    print(json.dumps({"status": result["status"], "question_ids": result["question_ids"],
                      "summary": [{"question_id": s["question_id"], "C": s["condition_summary"]["C"],
                                   "D": s["condition_summary"]["D"]} for s in samples]},
                     indent=2, ensure_ascii=False))
    print(f"Report: {study / 'unstable_repeatability_report.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare", help="validate parent results and freeze the two-question subset; no inference")
    prep.add_argument("--parent-study", type=Path, default=DEFAULT_PARENT)
    prep.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    prep.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    run = sub.add_parser("run", help="run three additional C and three additional D repeats")
    run.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    run.add_argument("--port-base", type=int, default=29941)
    report = sub.add_parser("report", help="aggregate the original two and new three repeats per condition")
    report.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    args = parser.parse_args()
    configure_runtime()
    if args.action == "prepare":
        prepare(args.parent_study.resolve(), args.bundle.resolve(), args.study.resolve())
    elif args.action == "report":
        summarize(args.study.resolve())
    else:
        study = args.study.resolve()
        plan = read(study / "followup_plan.json")
        check(plan.get("script_sha256") == digest(Path(__file__)), "script changed since prepare")
        parent = Path(plan["parent_study"])
        check(digest(parent / "study_plan.json") == plan["parent_study_plan_sha256"],
              "parent study plan changed since prepare")
        check(digest(parent / "repeatability_report.json") == plan["parent_report_sha256"],
              "parent report changed since prepare")
        check(snapshot() == plan["snapshot"], "source/runtime changed since prepare")
        check(1024 <= args.port_base <= 65529, "port base must leave room for all six run ports")
        _, identity = load_bundle(plan["bundle"]["folder"], verify_images=True)
        check(identity == plan["bundle"], "frozen cohort identity changed")
        rows = read(study / "unstable_457_1512_manifest.json")["samples"]
        for offset, label in enumerate(ADDITIONAL_LABELS):
            run_one(study, label, rows, plan, args.port_base + offset)
        summarize(study)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"FAIL: {exc}")

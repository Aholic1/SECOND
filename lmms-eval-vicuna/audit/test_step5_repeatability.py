"""Synthetic checks: no network, production data, CUDA, or model calls."""
import copy
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import step5_repeatability as study


def case(qid="1", role="changed_case"):
    return {"question_id": qid, "coco_image_id": "100", "question": "Is there a car in the image?",
            "role": role, "ground_truth": "yes", "C_prediction": "yes",
            "D_prediction": "no" if role == "changed_case" else "yes",
            "change_direction": "degraded" if role == "changed_case" else None,
            "annotation_issues": []}


def captures(row):
    result = {}
    for label in study.ORDER:
        result[label] = {row["question_id"]: {
            "prediction": row[label[0] + "_prediction"], "prompt": "identical prompt",
            "generation_config": {"temperature": 0}, "image_transform": {"padding": 0},
            "mask_hashes": {"mask": label[0]}, "final_token_ids": [1 if label[0] == "C" else 2],
            "attention_hashes": {"attn": "same"}, "yes_no_margin": 1 if label[0] == "C" else -1,
        }}
    return result


class RepeatabilityTests(unittest.TestCase):
    def test_stable_historical_change(self):
        row = case()
        report = study.analyze_records([row], captures(row))
        self.assertEqual(report["summary"]["prediction_repeatability_status"], "PASS")
        self.assertEqual(report["samples"][0]["outcome"], "historical_change_reproduced")
        self.assertEqual(report["summary"]["mechanism_screen_candidate_ids_pending_label_review"], ["1"])

    def test_label_questionable_case_is_not_causal_candidate(self):
        row = case()
        row["annotation_issues"] = ["missing target annotation"]
        report = study.analyze_records([row], captures(row))
        self.assertEqual(report["summary"]["prediction_repeatability_status"], "PASS")
        self.assertEqual(report["summary"]["mechanism_screen_candidate_ids_pending_label_review"], [])

    def test_mask_change_despite_equal_answers_is_detected(self):
        row = case()
        records = captures(row)
        records["D2"]["1"]["mask_hashes"] = {"mask": "changed"}
        report = study.analyze_records([row], records)
        self.assertEqual(report["summary"]["prediction_repeatability_status"], "PASS")
        self.assertEqual(report["summary"]["first_answer_mask_repeatability_status"], "UNSTABLE")
        self.assertEqual(report["summary"]["mechanism_screen_candidate_ids_pending_label_review"], [])

    def test_repeat_unstable_answer(self):
        row = case()
        records = captures(row)
        records["D2"]["1"]["prediction"] = "yes"
        report = study.analyze_records([row], records)
        self.assertEqual(report["summary"]["unstable_prediction_ids"], ["1"])
        self.assertIsNone(report["samples"][0]["currently_CD_different"])

    def test_current_stable_historical_mismatch_separate(self):
        row = case()
        records = captures(row)
        for label in ("D1", "D2"):
            records[label]["1"]["prediction"] = "yes"
        report = study.analyze_records([row], records)
        self.assertEqual(report["summary"]["prediction_repeatability_status"], "PASS")
        self.assertEqual(report["summary"]["historical_prediction_reproduction_status"], "MISMATCH")
        self.assertEqual(report["samples"][0]["outcome"], "repeat_stable_but_historical_mismatch")

    def test_prompt_mismatch_rejected(self):
        row = case()
        records = captures(row)
        records["C2"]["1"]["prompt"] = "changed"
        with self.assertRaisesRegex(ValueError, "prompt differs"):
            study.analyze_records([row], records)

    def test_control_repeatability_is_reported_separately(self):
        row = case(role="historically_CD_equal_control")
        report = study.analyze_records([row], captures(row))
        self.assertEqual(report["samples"][0]["outcome"], "historical_equal_control_reproduced")
        self.assertEqual(report["summary"]["control_count"], 1)
        self.assertEqual(report["summary"]["mechanism_screen_candidate_ids_pending_label_review"], [])

    def test_attention_variation_not_confused_with_mask_variation(self):
        row = case()
        records = captures(row)
        records["D2"]["1"]["attention_hashes"] = {"attn": "changed bits"}
        report = study.analyze_records([row], records)
        self.assertEqual(report["summary"]["first_answer_mask_repeatability_status"], "PASS")
        self.assertFalse(report["samples"][0]["first_answer_attention_bitwise_equal"]["D"])

    def make_bundle(self, folder):
        image = folder / "image.jpg"
        image.write_bytes(b"synthetic image content")
        cases = []
        for i in range(30):
            row = case(str(i + 1))
            row.update({"change_direction": "corrected" if i < 13 else "degraded",
                        "image_path": str(image), "image_sha256": study.digest(image)})
            if i < 13:
                row["C_prediction"], row["D_prediction"] = "no", "yes"
            cases.append(row)
        provenance = {"fixture": "synthetic"}
        study.write(folder / "freeze_provenance.json", provenance)
        phash = study.digest(folder / "freeze_provenance.json")
        files = {"changed30.json": {"samples": cases}, "controls.json": {"samples": []},
                 "combined.json": {"samples": cases, "expected_count": 30, "provenance_sha256": phash},
                 "pairs.json": {"pairs": [], "unmatched": []},
                 "freeze_report.json": {"status": "NEEDS_REVIEW", "control_count": 0}}
        for name, value in files.items():
            study.write(folder / name, value)
        study.write(folder / "sha256.json", {name: study.digest(folder / name) for name in (*files, "freeze_provenance.json")})

    def test_partial_matching_allowed_and_blind_package_hides_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "bundle"
            folder.mkdir()
            self.make_bundle(folder)
            rows, identity = study.load_bundle(folder, verify_images=True)
            self.assertEqual(identity["historical_matching_status"], "NEEDS_REVIEW")
            out = Path(tmp) / "review"
            study.prepare_review(folder, out)
            with zipfile.ZipFile(out / "blind_review.zip") as archive:
                self.assertNotIn("review_key_private.json", archive.namelist())
                page = archive.read("index.html").decode()
                for value in ("C_prediction", "D_prediction", "ground_truth", "degraded", "corrected", "annotation_issues"):
                    self.assertNotIn(value, page)
                self.assertEqual(len(archive.namelist()), 31)
            with self.assertRaisesRegex(ValueError, "already exists"):
                study.prepare_review(folder, out)

    def test_modified_bundle_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.make_bundle(folder)
            with (folder / "combined.json").open("a") as f:
                f.write(" ")
            with self.assertRaisesRegex(ValueError, "frozen file modified"):
                study.load_bundle(folder)

    def test_runner_passes_frozen_subset_seed_and_condition_and_can_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            row = case()
            state = {"git_commit": "synthetic"}
            plan = {"snapshot": state, "snapshot_sha256": study.canonical(state),
                    "cohort": {"folder": str(folder / "bundle"), "combined_sha256": "fixture"}}
            commands = []
            def fake_stream(command, logfile, env):
                commands.append((command, env))
                logfile.write_text("synthetic test only")
                if "--report" in command:
                    study.write(Path(command[command.index("--report") + 1]), {"status": "PASS"})
            with patch.object(study, "snapshot", return_value=state), patch.object(study, "load_bundle"), \
                 patch.object(study, "stream_command", side_effect=fake_stream), \
                 patch.object(study, "capture_records", return_value={"1": {"prediction": "yes"}}):
                study.run_one(folder, "C1", [row], plan, 29931)
                self.assertEqual(len(commands), 2)
                command, env = commands[0]
                self.assertEqual(command[command.index("--tasks") + 1], "pope_pop_audit")
                self.assertEqual(command[command.index("--seed") + 1], "0,1234,1234,1234")
                self.assertIn("audit_condition=C", command[command.index("--model_args") + 1])
                self.assertEqual(env["SECOND_AUDIT_MANIFEST"], str(folder / "bundle/combined.json"))
                study.run_one(folder, "C1", [row], plan, 29931)
                self.assertEqual(len(commands), 2)

    def test_incomplete_run_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "D1").mkdir()
            with self.assertRaisesRegex(ValueError, "incomplete run exists"):
                study.run_one(folder, "D1", [case()], {}, 29931)


if __name__ == "__main__":
    unittest.main(verbosity=2)

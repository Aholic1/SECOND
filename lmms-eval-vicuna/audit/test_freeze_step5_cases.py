"""Synthetic tests only: no real POPE data, GPU, model, or production manifest writes."""
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import freeze_step5_cases as freeze


class FreezeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.report = {
            "expected_count": 4, "inputs": {}, "sample_files": {},
            "git_commits": {}, "second_parameters": {}, "validation_status": {}, "overall": {},
            "C_to_D_selection_effect": {"prediction_changed_count": 2, "corrected_count": 1,
                "degraded_count": 1, "corrected_question_ids": ["2"], "degraded_question_ids": ["1"]},
            "changed_samples": [],
        }
        predictions = {"B": ["yes", "no", "yes", "no"],
                       "C": ["yes", "no", "yes", "no"], "D": ["no", "yes", "yes", "no"]}
        for condition in "BCD":
            run = self.root / condition
            run.mkdir()
            manifest = {"condition": condition, "expected_count": 4, "git_commit": "synthetic-commit",
                        "runtime": {"conda_environment": "SECOND_clean", "num_processes": 2, "cuda_visible_devices": "0,1"},
                        "second_parameters": {"stages": [-2, -1, 0, 1]}}
            self.save(run / "run_manifest.json", manifest)
            self.save(run / "validation_report.json", {"status": "PASS", "condition": condition,
                "expected_count": 4, "captured_count": 4, "errors": []})
            samples = []
            for i, prediction in enumerate(predictions[condition], 1):
                samples.append({"doc_id": i - 1,
                    "doc": {"id": str(3000 + i), "question_id": str(i), "answer": "yes",
                            "question": "Is there a car in the image?", "category": "popular"},
                    "pope_accuracy": {"question_id": str(i), "ground_truth": "yes", "prediction": prediction,
                                      "score": int(prediction == "yes")},
                    "input": "Is there a car in the image?\nAnswer yes or no.", "prompt_hash": "fixture-prompt"})
            self.save_lines(run / "samples.jsonl", samples)
            self.report["inputs"][condition] = condition
            self.report["sample_files"][condition] = f"{condition}/samples.jsonl"
            self.report["git_commits"][condition] = manifest["git_commit"]
            self.report["second_parameters"][condition] = manifest["second_parameters"]
            self.report["validation_status"][condition] = "PASS"
            self.report["overall"][condition] = {"correct": 2}
        for i in (1, 2):
            self.report["changed_samples"].append({"question_id": str(i), "question": "Is there a car in the image?",
                "ground_truth": "yes", **{f"{c}_prediction": predictions[c][i - 1] for c in "BCD"}})
        # B/C-only entry must not become a C/D case. The C/D values agree.
        self.report["changed_samples"].append({"question_id": "3", "C_prediction": "yes", "D_prediction": "yes"})
        self.save(self.root / "report.json", self.report)
        mapping, features, images, annotations = [], [], [], []
        for i in range(1, 5):
            image_id = 100 + i
            image_path = self.root / f"COCO_val2014_{image_id:012d}.jpg"
            image_path.write_bytes(f"synthetic-image-bytes-{i}".encode())
            mapping.append({"question_id": str(i), "image_id": str(image_id), "image_path": str(image_path),
                "image_filename": image_path.name, "readable": True, "width": 640, "height": 480,
                "sha256": freeze.sha256(image_path)})
            features.append({"question_id": str(i), "image_id": str(image_id), "question": "Is there a car in the image?",
                "gt": "yes", "image_width": 640, "image_height": 480, "image_readable": True,
                "pred": "untrusted historical prediction", "target_bbox": [0, 0, 1, 1]})
            images.append({"id": image_id, "file_name": image_path.name, "width": 640, "height": 480})
            for index, bbox in enumerate(([100, 100, 30, 40], [1, 1, 2, 3])):
                annotations.append({"id": i * 10 + index, "image_id": image_id, "category_id": 1,
                                    "bbox": bbox, "iscrowd": 0})
        self.save_lines(self.root / "mapping.jsonl", mapping)
        self.save_lines(self.root / "features.jsonl", features)
        self.save(self.root / "coco.json", {"images": images, "categories": [{"id": 1, "name": "car"}], "annotations": annotations})

    def save(self, path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def save_lines(self, path, values):
        path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")

    def args(self, output="frozen"):
        return freeze.parse_args(["--repo-root", str(self.root), "--report", "report.json", "--features", "features.jsonl",
            "--image-mapping", "mapping.jsonl", "--coco", "coco.json", "--output-dir", output,
            "--expected-count", "4", "--expected-changed", "2", "--expected-corrected", "1", "--expected-degraded", "1"])

    def execute(self, output="frozen"):
        with contextlib.redirect_stdout(io.StringIO()):
            return freeze.run(self.args(output))

    def test_full_synthetic_bundle_and_immutable_output(self):
        self.assertEqual(self.execute(), 0)
        frozen = self.root / "frozen"
        report = freeze.read_json(frozen / "freeze_report.json")
        self.assertEqual((report["status"], report["control_count"], report["combined_count"]), ("PASS", 2, 4))
        cases = freeze.read_json(frozen / "changed30.json")
        self.assertEqual([r["question_id"] for r in cases["samples"]], ["1", "2"])
        self.assertEqual(cases["samples"][0]["dataset_image_id"], "3001")
        self.assertEqual(cases["samples"][0]["coco_image_id"], "101")
        self.assertEqual(len(cases["samples"][0]["target_bboxes"]), 2)
        self.assertAlmostEqual(cases["samples"][0]["largest_target_bbox_area_ratio"], 1200 / (640 * 480))
        self.assertEqual(cases["provenance_sha256"], freeze.sha256(frozen / "freeze_provenance.json"))
        for name, digest in freeze.read_json(frozen / "sha256.json").items():
            self.assertEqual(freeze.sha256(frozen / name), digest)
        pairs = freeze.read_json(frozen / "pairs.json")["pairs"]
        self.assertEqual({p["case_question_id"]: p["control_question_id"] for p in pairs}, {"1": "3", "2": "4"})
        with self.assertRaisesRegex(ValueError, "refusing overwrite"):
            self.execute()
        self.execute("second")
        self.assertEqual(freeze.read_json(frozen / "pairs.json"), freeze.read_json(self.root / "second/pairs.json"))

    def test_global_matching_avoids_greedy_loss(self):
        candidates = {"1": {"10": {"cost": 1, "control_question_id": "10"}, "20": {"cost": 5, "control_question_id": "20"}},
                      "2": {"10": {"cost": 2, "control_question_id": "10"}}}
        assigned = freeze.min_cost_assignment(candidates)
        self.assertEqual(assigned["1"]["control_question_id"], "20")
        self.assertEqual(assigned["2"]["control_question_id"], "10")

    def test_partial_match_is_not_pass_and_cases_are_preserved(self):
        coco = freeze.read_json(self.root / "coco.json")
        # Largest box scale exceeds both calipers for one control.
        coco["annotations"][4]["bbox"] = [0, 0, 600, 450]
        self.save(self.root / "coco.json", coco)
        self.assertEqual(self.execute(), 2)
        report = freeze.read_json(self.root / "frozen/freeze_report.json")
        self.assertEqual(report["status"], "NEEDS_REVIEW")
        self.assertEqual((report["changed_count"], report["control_count"]), (2, 1))

    def test_changed_prompt_is_rejected(self):
        path = self.root / "D/samples.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["input"] = "different prompt"
        self.save_lines(path, rows)
        with self.assertRaisesRegex(ValueError, "input differs"):
            self.execute()
        self.assertFalse((self.root / "frozen").exists())

    def test_image_content_change_is_rejected(self):
        path = next(self.root.glob("COCO*101.jpg"))
        path.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "image hash differs"):
            self.execute()

    def test_report_disagreement_is_rejected(self):
        self.report["C_to_D_selection_effect"]["corrected_question_ids"] = ["1"]
        self.save(self.root / "report.json", self.report)
        with self.assertRaisesRegex(ValueError, "report effect lists"):
            self.execute()

    def test_duplicate_question_is_rejected(self):
        path = self.root / "mapping.jsonl"
        with path.open("a") as handle:
            handle.write(path.read_text().splitlines()[0] + "\n")
        with self.assertRaisesRegex(ValueError, "duplicate question_id"):
            self.execute()

    def test_annotation_missing_for_case_is_reviewed_not_dropped(self):
        coco = freeze.read_json(self.root / "coco.json")
        coco["annotations"] = [a for a in coco["annotations"] if a["image_id"] != 101]
        self.save(self.root / "coco.json", coco)
        self.assertEqual(self.execute(), 2)
        report = freeze.read_json(self.root / "frozen/freeze_report.json")
        self.assertEqual(report["case_annotation_issues"][0]["question_id"], "1")
        self.assertEqual(report["changed_count"], 2)

    def test_controls_cannot_share_case_image(self):
        sources = freeze.load_sources(self.root, self.root / "report.json", 4)
        rows = freeze.enrich(sources[1], freeze.indexed_jsonl(self.root / "features.jsonl"),
            freeze.indexed_jsonl(self.root / "mapping.jsonl"), freeze.load_coco(self.root / "coco.json"))
        rows["1"]["change_direction"], rows["2"]["change_direction"] = "degraded", "corrected"
        rows["3"]["coco_image_id"] = rows["1"]["coco_image_id"]
        pairs, unmatched, _ = freeze.match_controls(rows, ["1", "2"], 42)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(unmatched[0]["question_id"], "1")

    def test_negative_geometry_is_not_fabricated(self):
        case = {"ground_truth": "no", "target_object": "car", "C_correct": True,
                "matching_geometry_available": True, "num_objects": 4, "aspect_ratio": 1.3}
        match = freeze.pair_distance(case, dict(case))
        self.assertEqual(match["tier"], "strict")
        self.assertNotIn("target_log_area_ratio_difference", match["covariate_differences"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

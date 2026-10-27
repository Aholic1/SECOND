from collections import Counter
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools.second_audit import build_manifest


REPO_ROOT = Path(__file__).parents[1]


class FakeDataset:
    def __init__(self, rows):
        self.rows = list(rows)

    def filter(self, predicate):
        return FakeDataset(row for row in self.rows if predicate(row))

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)


def load_audit_utils():
    path = REPO_ROOT / "lmms_eval" / "tasks" / "pope" / "audit_utils.py"
    spec = importlib.util.spec_from_file_location("pope_audit_utils_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_row(doc_id, prediction, ground_truth):
    return {
        "doc_id": doc_id,
        "doc": {
            "id": str(3000 + doc_id),
            "question_id": str(doc_id + 1),
            "question": f"Question {doc_id}?",
            "answer": ground_truth,
        },
        "filtered_resps": [prediction.title()],
        "pope_accuracy": {
            "question_id": str(doc_id + 1),
            "prediction": prediction,
            "ground_truth": ground_truth,
        },
    }


def write_synthetic_log(path):
    combinations = [
        ("yes", "yes"),
        ("no", "no"),
        ("no", "yes"),
        ("yes", "no"),
    ]
    rows = []
    for repeat in range(4):
        for prediction, ground_truth in combinations:
            rows.append(sample_row(len(rows), prediction, ground_truth))
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class ManifestTests(unittest.TestCase):
    def test_build_manifest_is_reproducible_and_stratified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "samples.jsonl"
            write_synthetic_log(source)
            counts = {"FN": 2, "FP": 1, "TP": 2, "TN": 2}

            one = build_manifest(source, root / "one.json", 42, counts)
            two = build_manifest(source, root / "two.json", 42, counts)

            self.assertEqual(one["samples"], two["samples"])
            self.assertEqual(
                Counter(row["stratum"] for row in one["samples"]), counts
            )
            self.assertEqual(len({row["question_id"] for row in one["samples"]}), 7)

    def test_build_manifest_rejects_duplicate_question_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "samples.jsonl"
            row = sample_row(0, "yes", "yes")
            source.write_text(
                json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "duplicate question_id"):
                build_manifest(source, root / "manifest.json", 42, {"TP": 1})

    def test_filter_rejects_missing_question_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"schema_version": 1, "samples": [{"question_id": "missing"}]}),
                encoding="utf-8",
            )
            audit_utils = load_audit_utils()

            with mock.patch.dict(
                "os.environ", {"SECOND_AUDIT_MANIFEST": str(manifest)}, clear=False
            ):
                with self.assertRaisesRegex(ValueError, "missing"):
                    audit_utils.pope_process_audit_docs(
                        FakeDataset([{"question_id": "present"}])
                    )

    def test_filter_accepts_step1_diagnostic_schema_v2(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "samples": [
                            {"question_id": "1"},
                            {"question_id": "3"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            audit_utils = load_audit_utils()

            with mock.patch.dict(
                "os.environ", {"SECOND_AUDIT_MANIFEST": str(manifest)}, clear=False
            ):
                filtered = audit_utils.pope_process_audit_docs(
                    FakeDataset(
                        [
                            {"question_id": "1"},
                            {"question_id": "2"},
                            {"question_id": "3"},
                        ]
                    )
                )

            self.assertEqual(
                [row["question_id"] for row in filtered], ["1", "3"]
            )


if __name__ == "__main__":
    unittest.main()

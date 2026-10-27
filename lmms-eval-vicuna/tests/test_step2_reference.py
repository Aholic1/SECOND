import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


REPO_ROOT = Path(__file__).parents[1]


def load_reference_validator():
    path = REPO_ROOT / "audit" / "validate_step2_reference.py"
    spec = importlib.util.spec_from_file_location("step2_reference_validator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Step2ReferenceTests(unittest.TestCase):
    def test_prediction_reader_rejects_duplicate_question_ids(self):
        validator = load_reference_validator()
        row = {
            "pope_accuracy": {
                "question_id": "1",
                "prediction": "yes",
                "ground_truth": "yes",
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text(
                json.dumps(row) + "\n" + json.dumps(row) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate question_id"):
                validator.read_predictions(path)

    def test_reference_script_is_uninstrumented_and_fixed_to_diagnostic_set(self):
        script = (REPO_ROOT / "audit" / "run_step2_reference.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('CONDA_DEFAULT_ENV:-}" != "SECOND_clean"', script)
        self.assertIn("--tasks pope_pop_audit", script)
        self.assertIn("--expected-count 100", script)
        self.assertIn("validate_step2_reference.py", script)
        self.assertIn('"source_fingerprint_sha256"', script)
        self.assertNotIn("audit_condition=", script)
        self.assertNotIn("audit_output_dir=", script)


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).parents[1]


class Step3RunScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = (REPO_ROOT / "audit" / "run_step3_provenance.sh").read_text(
            encoding="utf-8"
        )

    def test_requires_clean_environment_and_explicit_mode(self):
        self.assertIn('CONDA_DEFAULT_ENV:-}" != "SECOND_clean"', self.script)
        self.assertIn("{reference|diagnostic}", self.script)

    def test_reference_is_uninstrumented(self):
        reference_branch = self.script.split('if [[ "$MODE" == "reference" ]]', 1)[1]
        reference_branch = reference_branch.split('if [[ -n "${SECOND_STEP3_REFERENCE:-}" ]]', 1)[0]
        self.assertIn('MODEL_ARGS="pretrained=liuhaotian/llava-v1.6-vicuna-7b"', reference_branch)
        self.assertNotIn("audit_condition=", reference_branch)

    def test_capture_runs_both_noninterference_and_provenance_gates(self):
        for expected in (
            "audit_condition=D",
            "validate_step2_capture.py",
            "step2_compatibility_report.json",
            "validate_step3_provenance.py",
            "validation_report.json",
            'manifest.get("git_commit") != sys.argv[3]',
            'manifest.get("source_fingerprint_sha256") != sys.argv[4]',
        ):
            self.assertIn(expected, self.script)


if __name__ == "__main__":
    unittest.main()

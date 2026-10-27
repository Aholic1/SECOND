from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).parents[1]


class Step2RunScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = (REPO_ROOT / "audit" / "run_step2_capture.sh").read_text(
            encoding="utf-8"
        )

    def test_requires_requested_environment_and_known_mode(self):
        self.assertIn('CONDA_DEFAULT_ENV:-}" != "SECOND_clean"', self.script)
        self.assertIn('{diagnostic|full}', self.script)

    def test_diagnostic_and_full_capture_contracts_are_fixed(self):
        for expected in (
            'TASK="pope_pop_audit"',
            "EXPECTED_COUNT=100",
            'ARTIFACT_MODE="visual"',
            'TASK="pope_pop"',
            "EXPECTED_COUNT=3000",
            'ARTIFACT_MODE="compact"',
            "export SECOND_AUDIT_MANIFEST",
        ):
            self.assertIn(expected, self.script)

    def test_inference_is_followed_by_strict_baseline_validation(self):
        for expected in (
            "set -euo pipefail",
            "audit_condition=D",
            "audit_token_idx=0",
            "--generation_type recursion",
            "--stages -2 -1 0 1",
            "validate_step2_capture.py",
            '--baseline-predictions "$HISTORICAL_BASELINE"',
            '--reference-predictions "$REFERENCE_PREDICTIONS"',
            '--expected-count "$EXPECTED_COUNT"',
            '--artifact-mode "$ARTIFACT_MODE"',
            'validation_report.json',
        ):
            self.assertIn(expected, self.script)

    def test_diagnostic_requires_a_passing_reference_from_the_same_commit(self):
        for expected in (
            "run_step2_reference.sh",
            'manifest.get("git_commit") != sys.argv[3]',
            'manifest.get("source_fingerprint_sha256") != sys.argv[4]',
            'report.get("status") != "PASS"',
        ):
            self.assertIn(expected, self.script)


if __name__ == "__main__":
    unittest.main()

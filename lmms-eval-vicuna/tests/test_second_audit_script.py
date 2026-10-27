from pathlib import Path
import subprocess
import unittest


REPO_ROOT = Path(__file__).parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_second_audit.sh"


class SecondAuditScriptTests(unittest.TestCase):
    def test_driver_contains_all_conditions_and_safety_guards(self):
        text = SCRIPT.read_text(encoding="utf-8")

        for condition in ("A", "B", "C", "D"):
            self.assertIn(f'run_condition "{condition}"', text)
        self.assertIn("SECOND_AUDIT_MANIFEST", text)
        self.assertIn("NCCL_P2P_DISABLE", text)
        self.assertIn("NCCL_IB_DISABLE", text)
        self.assertIn("set -euo pipefail", text)
        self.assertIn("pope_pop_audit", text)
        self.assertIn("PYTHONPATH", text)
        self.assertIn("LLaVA-NeXT", text)

    def test_parity_runs_before_paired_analysis(self):
        text = SCRIPT.read_text(encoding="utf-8")

        self.assertLess(text.index(" parity "), text.index(" analyze "))

    def test_no_argument_invocation_prints_usage_without_mutating_files(self):
        completed = subprocess.run(
            ["bash", str(SCRIPT)],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("Usage:", completed.stderr)


if __name__ == "__main__":
    unittest.main()

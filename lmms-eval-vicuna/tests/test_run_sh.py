import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class RunScriptTest(unittest.TestCase):
    def test_disables_unsupported_nccl_transports_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            probe_output = temp_path / "nccl-env.txt"
            fake_python = temp_path / "python3"
            fake_python.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    printf '%s\\n%s\\n' "${NCCL_P2P_DISABLE-}" "${NCCL_IB_DISABLE-}" > "${PROBE_OUTPUT}"
                    """
                )
            )
            fake_python.chmod(0o755)

            env = os.environ.copy()
            env.pop("NCCL_P2P_DISABLE", None)
            env.pop("NCCL_IB_DISABLE", None)
            env["PATH"] = f"{temp_path}:{env['PATH']}"
            env["PROBE_OUTPUT"] = str(probe_output)

            completed = subprocess.run(
                ["bash", "run.sh"],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(probe_output.read_text().splitlines(), ["1", "1"])


if __name__ == "__main__":
    unittest.main()

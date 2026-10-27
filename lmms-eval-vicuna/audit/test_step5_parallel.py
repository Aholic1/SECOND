"""Synthetic scheduling tests; no GPU or model is started."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import step5_parallel_repeatability as s


class ParallelTests(unittest.TestCase):
    def test_gpu_layout_crosses_conditions(self):
        self.assertEqual(s.GPU_LAYOUT, {"C1": "0,1", "D1": "2,3", "D2": "0,1", "C2": "2,3"})
        self.assertNotEqual(s.GPU_LAYOUT["C1"], s.GPU_LAYOUT["D1"])
        self.assertEqual(s.GPU_LAYOUT["C1"], s.GPU_LAYOUT["D2"])

    def test_custom_gpu_layout_uses_selected_physical_pairs(self):
        s.set_gpu_layout("1,2", "3,4")
        self.assertEqual(s.GPU_LAYOUT, {"C1": "1,2", "D1": "3,4", "D2": "1,2", "C2": "3,4"})
        self.assertEqual(s.TARGET_GPU_INDICES, ("1", "2", "3", "4"))
        with self.assertRaisesRegex(ValueError, "overlap"):
            s.set_gpu_layout("1,2", "2,3")
        s.set_gpu_layout("0,1", "2,3")

    def test_busy_gpu_blocks(self):
        s.set_gpu_layout("0,1", "2,3")
        with patch.object(s.subprocess, "check_output", return_value="0, 39358, 46068\n1, 12, 46068\n2, 12, 46068\n3, 12, 46068"):
            with self.assertRaisesRegex(ValueError, "GPU 0"):
                s.assert_gpu_pairs_idle()

    def test_idle_gpu_passes(self):
        s.set_gpu_layout("0,1", "2,3")
        with patch.object(s.subprocess, "check_output", return_value="0, 12, 46068\n1, 12, 46068\n2, 12, 46068\n3, 12, 46068"):
            s.assert_gpu_pairs_idle()

    def test_unselected_busy_gpu_does_not_block(self):
        s.set_gpu_layout("1,2", "3,4")
        output = "0, 39359, 46068\n1, 12, 46068\n2, 12, 46068\n3, 12, 46068\n4, 12, 46068\n5, 12, 46068"
        with patch.object(s.subprocess, "check_output", return_value=output):
            s.assert_gpu_pairs_idle()
        s.set_gpu_layout("0,1", "2,3")

    def test_missing_gpu_blocks(self):
        s.set_gpu_layout("0,1", "2,3")
        with patch.object(s.subprocess, "check_output", return_value="0, 12, 46068\n1, 12, 46068"):
            with self.assertRaisesRegex(ValueError, "GPU 2 not found"):
                s.assert_gpu_pairs_idle()

    def test_independent_uuid_environments_ports_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = {"question_id": "1"}
            state = {"git_commit": "fixture", "gpu_uuid_by_index": {str(i): f"GPU-{i}" for i in range(6)},
                     "gpu_layout": s.GPU_LAYOUT}
            plan = {"snapshot": state, "snapshot_sha256": s.canonical(state),
                    "cohort": {"folder": str(root / "bundle"), "combined_sha256": "fixture"}}
            commands = []
            def fake_command(command, logfile, env):
                commands.append((command, dict(env)))
                logfile.write_text("synthetic only")
                if "--report" in command:
                    s.write(Path(command[command.index("--report") + 1]), {"status": "PASS"})
            with patch.object(s, "snapshot", return_value=state), patch.object(s, "load_bundle"), \
                 patch.object(s, "stream_command", side_effect=fake_command), \
                 patch.object(s, "capture_records", return_value={"1": {"prediction": "yes"}}):
                for label, port in (("C1", 29931), ("D1", 29932)):
                    s.run_one(root, label, [row], plan, port, s.GPU_LAYOUT[label])
                launches = [(cmd, env) for cmd, env in commands if "--model_args" in cmd]
                self.assertEqual(len(launches), 2)
                for (cmd, env), pair, port in zip(launches, ("GPU-0,GPU-1", "GPU-2,GPU-3"), ("29931", "29932")):
                    self.assertEqual(env["CUDA_VISIBLE_DEVICES"], pair)
                    self.assertEqual(env["NUM_PROCESSES"], "2")
                    self.assertIn("--num_processes=2", cmd)
                    self.assertEqual(cmd[cmd.index("--main_process_port") + 1], port)
                    self.assertEqual(cmd[cmd.index("--seed") + 1], s.SEEDS)
                count = len(commands)
                s.run_one(root, "C1", [row], plan, 29931, "0,1")
                self.assertEqual(len(commands), count)
                manifest = s.read(root / "C1/run_manifest.json")
                manifest["physical_gpu_pair"] = "2,3"
                s.write(root / "C1/run_manifest.json", manifest)
                complete = s.read(root / "C1/run_complete.json")
                complete["output_hashes"]["run_manifest.json"] = s.digest(root / "C1/run_manifest.json")
                s.write(root / "C1/run_complete.json", complete)
                with self.assertRaisesRegex(ValueError, "GPU assignment differs"):
                    s.run_one(root, "C1", [row], plan, 29931, "0,1")

    def test_failed_run_is_not_overwritten(self):
        s.set_gpu_layout("0,1", "2,3")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "D1").mkdir()
            with self.assertRaisesRegex(ValueError, "incomplete run exists"):
                s.run_one(root, "D1", [{"question_id": "1"}], {}, 29932, "2,3")


if __name__ == "__main__":
    unittest.main(verbosity=2)

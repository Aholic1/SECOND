import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from lmms_eval.second_audit import SecondAuditRecorder


class SecondAuditRecorderTests(unittest.TestCase):
    def test_recorder_writes_lossless_arrays_and_overlays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = SecondAuditRecorder(root, "D", rank=1)
            recorder.begin_sample(
                "q1",
                Image.new("RGB", (32, 32), "white"),
                {"question": "Q?"},
                original_image=Image.new("RGB", (24, 16), "gray"),
            )

            recorder.write_transition(
                "q1",
                token_idx=0,
                source_stage=-2,
                target_stage=-1,
                ret_attn=np.arange(16, dtype=np.float32).reshape(4, 4),
                proposed_mask=np.eye(4, dtype=np.float32),
                applied_mask=np.eye(4, dtype=np.float32),
                consumed_mask=np.eye(2, dtype=np.float32),
                metadata={"requested_keep_ratio": 0.5},
            )

            sample_dir = root / "D" / "rank_1" / "q1"
            arrays_path = sample_dir / "token_000_stage_-2_to_-1.npz"
            self.assertTrue(arrays_path.exists())
            self.assertTrue(
                (sample_dir / "token_000_stage_-2_to_-1_attention.png").exists()
            )
            self.assertTrue(
                (sample_dir / "token_000_stage_-2_to_-1_proposed_mask.png").exists()
            )
            self.assertTrue(
                (sample_dir / "token_000_stage_-2_to_-1_consumed_mask.png").exists()
            )
            self.assertTrue((sample_dir / "original_image.png").exists())
            with np.load(arrays_path) as arrays:
                np.testing.assert_array_equal(
                    arrays["ret_attn"], np.arange(16, dtype=np.float32).reshape(4, 4)
                )
                np.testing.assert_array_equal(arrays["consumed_mask"], np.eye(2))
                np.testing.assert_array_equal(
                    arrays["consumed_keep_indices"], np.array([0, 3])
                )

            transition = json.loads(arrays_path.with_suffix(".json").read_text())
            self.assertEqual(transition["selection_grid_shape"], [4, 4])
            self.assertEqual(transition["consumed_grid_shape"], [2, 2])
            self.assertEqual(transition["consumed_selected_token_count"], 2)

    def test_finish_sample_updates_metadata_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = SecondAuditRecorder(root, "B", rank=0)
            recorder.begin_sample(
                "q2", Image.new("RGB", (16, 16), "black"), {"question": "Q?"}
            )

            recorder.finish_sample("q2", {"raw_response": "Yes", "prediction": "yes"})

            payload = json.loads(
                (root / "B" / "rank_0" / "q2" / "sample.json").read_text()
            )
            self.assertEqual(payload["condition"], "B")
            self.assertEqual(payload["prediction"], "yes")

    def test_non_finite_transition_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = SecondAuditRecorder(Path(directory), "D", rank=0)
            recorder.begin_sample(
                "q3", Image.new("RGB", (16, 16), "black"), {"question": "Q?"}
            )

            with self.assertRaisesRegex(ValueError, "finite"):
                recorder.write_transition(
                    "q3",
                    token_idx=0,
                    source_stage=-2,
                    target_stage=-1,
                    ret_attn=np.array([[np.nan]], dtype=np.float32),
                    proposed_mask=np.ones((1, 1), dtype=np.float32),
                    applied_mask=np.ones((1, 1), dtype=np.float32),
                    consumed_mask=np.ones((1, 1), dtype=np.float32),
                    metadata={},
                )

    def test_compact_mode_skips_png_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = SecondAuditRecorder(
                root, "D", rank=0, artifact_mode="compact"
            )
            recorder.begin_sample(
                "q4", Image.new("RGB", (16, 16), "black"), {"question": "Q?"}
            )
            recorder.write_transition(
                "q4",
                token_idx=0,
                source_stage=-2,
                target_stage=-1,
                ret_attn=np.ones((4, 4), dtype=np.float32),
                proposed_mask=np.ones((4, 4), dtype=np.float32),
                applied_mask=np.ones((4, 4), dtype=np.float32),
                consumed_mask=np.ones((2, 2), dtype=np.float32),
                metadata={},
            )

            sample_dir = root / "D" / "rank_0" / "q4"
            self.assertTrue((sample_dir / "sample.json").exists())
            self.assertTrue((sample_dir / "token_000_stage_-2_to_-1.npz").exists())
            self.assertEqual(list(sample_dir.glob("*.png")), [])

    def test_consumed_mask_must_match_applied_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = SecondAuditRecorder(
                Path(directory), "D", rank=0, artifact_mode="compact"
            )
            recorder.begin_sample(
                "q5", Image.new("RGB", (16, 16), "black"), {"question": "Q?"}
            )

            with self.assertRaisesRegex(ValueError, "consumed_mask"):
                recorder.write_transition(
                    "q5",
                    token_idx=0,
                    source_stage=-2,
                    target_stage=-1,
                    ret_attn=np.ones((4, 4), dtype=np.float32),
                    proposed_mask=np.ones((4, 4), dtype=np.float32),
                    applied_mask=np.ones((4, 4), dtype=np.float32),
                    consumed_mask=np.zeros((2, 2), dtype=np.float32),
                    metadata={},
                )


if __name__ == "__main__":
    unittest.main()

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


REPO_ROOT = Path(__file__).parents[1]


def load_validator():
    path = REPO_ROOT / "audit" / "validate_step2_capture.py"
    spec = importlib.util.spec_from_file_location("step2_capture_validator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Step2CaptureValidatorTests(unittest.TestCase):
    def setUp(self):
        self.validator = load_validator()

    def write_transition(self, sample_dir, *, applied_mask=None, consumed_mask=None):
        source_stage, target_stage = -2, -1
        stem = "token_000_stage_-2_to_-1"
        selection = np.ones((48, 48), dtype=np.uint8)
        applied = selection if applied_mask is None else applied_mask
        consumed = (
            np.ones((12, 12), dtype=np.uint8)
            if consumed_mask is None
            else consumed_mask
        )
        keep = np.flatnonzero(consumed.reshape(-1)).astype(np.int32)
        drop = np.flatnonzero(consumed.reshape(-1) == 0).astype(np.int32)
        np.savez_compressed(
            sample_dir / f"{stem}.npz",
            ret_attn=np.ones((48, 48), dtype=np.float32),
            mask_before=np.zeros((48, 48), dtype=np.uint8),
            proposed_mask=selection,
            applied_mask=applied,
            consumed_mask=consumed,
            consumed_keep_indices=keep,
            consumed_drop_indices=drop,
            consumed_keep_coords_rc=np.argwhere(consumed == 1).astype(np.int16),
            consumed_drop_coords_rc=np.argwhere(consumed == 0).astype(np.int16),
        )
        metadata = {
            "audit_schema_version": 2,
            "condition": "D",
            "token_idx": 0,
            "source_stage": source_stage,
            "target_stage": target_stage,
            "selection_grid_shape": [48, 48],
            "consumed_grid_shape": list(consumed.shape),
            "consumed_selected_token_count": int(keep.size),
            "consumed_dropped_token_count": int(drop.size),
            "consumed_total_token_count": int(consumed.size),
            "consumed_keep_ratio": float(keep.size / consumed.size),
            "consumed_matches_applied_mask": True,
            "final_decoding_margin": {
                "available": True,
                "yes_token_id": 3869,
                "no_token_id": 1939,
                "yes_no_margin": 1.25,
            },
        }
        (sample_dir / f"{stem}.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )

    def test_valid_compact_transition_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            sample_dir = Path(directory)
            self.write_transition(sample_dir)
            errors = []

            self.validator.validate_transition(
                sample_dir, -2, -1, "compact", errors
            )

            self.assertEqual(errors, [])

    def test_malformed_mask_dimensions_are_reported_without_crashing(self):
        malformed_cases = {
            "applied_mask": {
                "applied_mask": np.ones(48, dtype=np.uint8),
            },
            "consumed_mask": {
                "consumed_mask": np.ones(12, dtype=np.uint8),
            },
        }
        for expected_name, overrides in malformed_cases.items():
            with self.subTest(mask=expected_name), tempfile.TemporaryDirectory() as directory:
                sample_dir = Path(directory)
                self.write_transition(sample_dir, **overrides)
                errors = []

                self.validator.validate_transition(
                    sample_dir, -2, -1, "compact", errors
                )

                self.assertTrue(any(expected_name in error for error in errors), errors)

    def test_corrupt_npz_is_reported_without_crashing(self):
        with tempfile.TemporaryDirectory() as directory:
            sample_dir = Path(directory)
            stem = "token_000_stage_-2_to_-1"
            (sample_dir / f"{stem}.json").write_text(
                json.dumps(
                    {
                        "audit_schema_version": 2,
                        "source_stage": -2,
                        "target_stage": -1,
                    }
                ),
                encoding="utf-8",
            )
            (sample_dir / f"{stem}.npz").write_bytes(b"not an npz")
            errors = []

            self.validator.validate_transition(
                sample_dir, -2, -1, "compact", errors
            )

            self.assertTrue(any("cannot read arrays" in error for error in errors))

    def test_decoding_trace_must_use_the_generated_yes_no_token(self):
        sample = {"prediction": "yes", "final_token_ids": [3869, 2]}
        transition = {
            "final_decoding_margin": {
                "available": True,
                "yes_token_id": 8241,
                "no_token_id": 3782,
                "yes_no_margin": -10.0,
            }
        }
        errors = []

        self.validator.validate_decoding_trace(
            sample, Path("sample"), transition, errors
        )

        self.assertTrue(any("first generated token" in error for error in errors))
        self.assertTrue(any("non-positive margin" in error for error in errors))

    def test_equal_yes_no_logits_follow_argmax_token_id_order(self):
        for prediction, generated_id, yes_id, no_id, should_pass in (
            ("no", 1939, 3869, 1939, True),
            ("yes", 3869, 3869, 1939, False),
            ("yes", 1939, 1939, 3869, True),
            ("no", 3869, 1939, 3869, False),
        ):
            with self.subTest(prediction=prediction, yes_id=yes_id):
                sample = {"prediction": prediction, "final_token_ids": [generated_id, 2]}
                transition = {
                    "final_decoding_margin": {
                        "available": True,
                        "yes_token_id": yes_id,
                        "no_token_id": no_id,
                        "yes_no_margin": 0.0,
                    }
                }
                errors = []
                self.validator.validate_decoding_trace(
                    sample, Path("sample"), transition, errors
                )
                if should_pass:
                    self.assertEqual(errors, [])
                else:
                    self.assertTrue(any("argmax tie-breaking" in error for error in errors))

    def test_equal_logits_with_missing_token_id_is_reported(self):
        sample = {"prediction": "no", "final_token_ids": [1939]}
        transition = {
            "final_decoding_margin": {
                "available": True,
                "no_token_id": 1939,
                "yes_no_margin": 0.0,
            }
        }
        errors = []

        self.validator.validate_decoding_trace(
            sample, Path("sample"), transition, errors
        )

        self.assertTrue(any("token IDs are malformed" in error for error in errors))


if __name__ == "__main__":
    unittest.main()

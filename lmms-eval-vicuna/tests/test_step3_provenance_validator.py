import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from lmms_eval.second_provenance import build_forward_provenance, write_jsonl


REPO_ROOT = Path(__file__).parents[1]


def load_validator():
    path = REPO_ROOT / "audit" / "validate_step3_provenance.py"
    spec = importlib.util.spec_from_file_location("step3_validator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_trace():
    specs = [
        (-2, "low_res_base", 6, 1, 1),
        (-1, "low_res_base", 12, 1, 1),
        (0, "base", 24, 1, 1),
        (1, "anyres_tile", 48, 2, 2),
    ]
    sources = []
    cursor = 0
    for stage, source_type, side, tile_rows, tile_cols in specs:
        mask = np.zeros((side, side), dtype=np.uint8)
        mask[0, 0] = 1
        source = {
            "stage_internal": stage,
            "source_type": source_type,
            "grid_height": side,
            "grid_width": side,
            "raster_height": 672 if stage == 1 else 336,
            "raster_width": 672 if stage == 1 else 336,
            "tile_rows": tile_rows,
            "tile_cols": tile_cols,
            "patch_rows_per_tile": side // tile_rows,
            "patch_cols_per_tile": side // tile_cols,
            "consumed_mask": mask,
            "runtime_keep_indices": [0],
            "runtime_feature_count": 1,
            "runtime_visual_start": cursor,
            "runtime_visual_end": cursor + 1,
        }
        cursor += 1
        sources.append(source)
    return {
        "forward_stage": 1,
        "mm_patch_merge_type": "spatial",
        "image_aspect_ratio": "anyres",
        "canvas_size": [400, 400],
        "encoded_image_count": 5,
        "sources": sources,
        "runtime_visual_length": cursor,
        "decoder_visual_start": 10,
        "decoder_visual_end": 10 + cursor,
        "decoder_retained_visual_count": cursor,
        "decoder_truncated_visual_count": 0,
        "decoder_sequence_length_before_truncation": 30,
        "decoder_sequence_length_after_truncation": 30,
        "tokenizer_model_max_length": 4096,
    }


TRANSFORM = {
    "original_size": {"width": 400, "height": 200},
    "model_input_size": {"width": 400, "height": 400},
    "padding": {"left": 0, "top": 100, "right": 0, "bottom": 100},
}


class Step3ValidatorTests(unittest.TestCase):
    def test_valid_runtime_reconstruction_passes_forward_checks(self):
        validator = load_validator()
        records, summary = build_forward_provenance("7", make_trace(), TRANSFORM)
        with tempfile.TemporaryDirectory() as directory:
            sample_dir = Path(directory)
            write_jsonl(
                sample_dir / "token_000_forward_stage_1_provenance.jsonl",
                records,
            )
            arrays = {}
            for source in make_trace()["sources"]:
                label = str(source["stage_internal"]).replace("-", "neg")
                arrays[f"stage_{label}_consumed_mask"] = source["consumed_mask"]
                arrays[f"stage_{label}_runtime_keep_indices"] = np.asarray(
                    source["runtime_keep_indices"], dtype=np.int32
                )
            np.savez_compressed(
                sample_dir
                / "token_000_forward_stage_1_provenance_runtime_trace.npz",
                **arrays,
            )
            errors = []
            validator.validate_forward(
                "7", sample_dir, TRANSFORM, summary, "compact", errors
            )
            self.assertEqual(errors, [])

    def test_validator_rejects_runtime_mismatch_and_truncation(self):
        validator = load_validator()
        records, summary = build_forward_provenance("7", make_trace(), TRANSFORM)
        summary["runtime_offline_match"] = False
        summary["decoder_truncated_visual_count"] = 1
        with tempfile.TemporaryDirectory() as directory:
            sample_dir = Path(directory)
            write_jsonl(
                sample_dir / "token_000_forward_stage_1_provenance.jsonl",
                records,
            )
            arrays = {}
            for source in make_trace()["sources"]:
                label = str(source["stage_internal"]).replace("-", "neg")
                arrays[f"stage_{label}_consumed_mask"] = source["consumed_mask"]
                arrays[f"stage_{label}_runtime_keep_indices"] = np.asarray(
                    source["runtime_keep_indices"], dtype=np.int32
                )
            np.savez_compressed(
                sample_dir
                / "token_000_forward_stage_1_provenance_runtime_trace.npz",
                **arrays,
            )
            errors = []
            validator.validate_forward(
                "7", sample_dir, TRANSFORM, summary, "compact", errors
            )
            self.assertTrue(any("runtime/offline mismatch" in error for error in errors))
            self.assertTrue(any("truncated visual tokens" in error for error in errors))


if __name__ == "__main__":
    unittest.main()

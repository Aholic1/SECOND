import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from lmms_eval.second_provenance import (
    build_forward_provenance,
    render_geometry,
    render_selection_overlay,
    source_geometry,
    write_jsonl,
)


def source(stage, mask, *, tile_rows=1, tile_cols=1):
    mask = np.asarray(mask, dtype=np.uint8)
    source_type = {
        -2: "low_res_base",
        -1: "low_res_base",
        0: "base",
        1: "anyres_tile",
    }[stage]
    patch_rows = mask.shape[0] // tile_rows
    patch_cols = mask.shape[1] // tile_cols
    count = int(mask.sum())
    return {
        "stage_internal": stage,
        "source_type": source_type,
        "grid_height": mask.shape[0],
        "grid_width": mask.shape[1],
        "raster_height": 672 if stage == 1 else mask.shape[0],
        "raster_width": 672 if stage == 1 else mask.shape[1],
        "tile_rows": tile_rows,
        "tile_cols": tile_cols,
        "patch_rows_per_tile": patch_rows,
        "patch_cols_per_tile": patch_cols,
        "consumed_mask": mask,
        "runtime_keep_indices": np.flatnonzero(mask.reshape(-1)).tolist(),
        "runtime_feature_count": count,
    }


def trace(*, retained=5):
    sources = [
        source(-2, [[1]]),
        source(-1, [[0]]),
        source(0, [[1]]),
        source(1, [[1, 0], [0, 1]], tile_rows=2, tile_cols=2),
    ]
    cursor = 0
    for item in sources:
        item["runtime_visual_start"] = cursor
        cursor += item["runtime_feature_count"]
        item["runtime_visual_end"] = cursor
    return {
        "forward_stage": 1,
        "mm_patch_merge_type": "spatial",
        "image_aspect_ratio": "anyres",
        "canvas_size": [400, 400],
        "encoded_image_count": 5,
        "sources": sources,
        "runtime_visual_length": cursor,
        "decoder_visual_start": 3,
        "decoder_visual_end": 3 + cursor,
        "decoder_retained_visual_count": retained,
        "decoder_truncated_visual_count": cursor - retained,
        "decoder_sequence_length_before_truncation": 20,
        "decoder_sequence_length_after_truncation": 20,
        "tokenizer_model_max_length": 4096,
    }


IMAGE_TRANSFORM = {
    "original_size": {"width": 400, "height": 200},
    "model_input_size": {"width": 400, "height": 400},
    "padding": {"left": 0, "top": 100, "right": 0, "bottom": 100},
}


class SecondProvenanceTests(unittest.TestCase):
    def test_reconstruction_maps_sequence_and_anyres_tile_order(self):
        records, summary = build_forward_provenance(
            "sample-1", trace(retained=4), IMAGE_TRANSFORM
        )
        self.assertTrue(summary["runtime_offline_match"])
        self.assertEqual(summary["candidate_count"], 7)
        self.assertEqual(summary["selected_count"], 4)
        self.assertEqual(summary["decoder_truncated_visual_count"], 0)

        selected = [record for record in records if record["selected"]]
        self.assertEqual(
            [record["visual_sequence_index"] for record in selected],
            [0, 1, 2, 3],
        )
        self.assertEqual(
            [record["decoder_sequence_index"] for record in selected],
            [3, 4, 5, 6],
        )
        rejected = [record for record in records if not record["selected"]]
        self.assertTrue(
            all(
                record["stage_sequence_index"] == -1
                and record["visual_sequence_index"] == -1
                and record["decoder_sequence_index"] == -1
                for record in rejected
            )
        )

        anyres = [record for record in records if record["stage_internal"] == 1]
        self.assertEqual(
            [(record["tile_id"], record["tile_row"], record["tile_col"])
             for record in anyres],
            [(0, 0, 0), (1, 0, 1), (2, 1, 0), (3, 1, 1)],
        )
        self.assertAlmostEqual(anyres[0]["padding_fraction"], 0.5)
        self.assertEqual(anyres[0]["canvas_rect"], [0.0, 0.0, 200.0, 200.0])
        self.assertEqual(
            anyres[0]["original_rect_clipped"], [0.0, 0.0, 200.0, 100.0]
        )

    def test_runtime_mask_disagreement_fails_comparison(self):
        payload = trace(retained=4)
        payload["sources"][0]["runtime_keep_indices"] = []
        _, summary = build_forward_provenance("sample-1", payload, IMAGE_TRANSFORM)
        self.assertFalse(summary["runtime_offline_match"])

    def test_truncated_selected_token_has_no_decoder_index(self):
        records, summary = build_forward_provenance(
            "sample-1", trace(retained=3), IMAGE_TRANSFORM
        )
        truncated = [record for record in records if record["is_truncated"]]
        self.assertEqual(len(truncated), 1)
        self.assertTrue(truncated[0]["selected"])
        self.assertEqual(truncated[0]["visual_sequence_index"], 3)
        self.assertEqual(truncated[0]["decoder_sequence_index"], -1)
        self.assertEqual(summary["decoder_truncated_visual_count"], 1)

    def test_anyres_geometry_reproduces_resize_and_center_padding(self):
        geometry = source_geometry(
            {
                "source_type": "anyres_tile",
                "raster_width": 672,
                "raster_height": 672,
            },
            (400, 200),
        )
        self.assertEqual((geometry.scaled_width, geometry.scaled_height), (672, 336))
        self.assertEqual((geometry.paste_x, geometry.paste_y), (0, 168))
        self.assertEqual(
            geometry.raster_rect_to_canvas([0, 168, 672, 504]),
            [0.0, 0.0, 400.0, 200.0],
        )

    def test_serialization_and_rendering_create_readable_artifacts(self):
        records, _ = build_forward_provenance(
            "sample-1", trace(retained=4), IMAGE_TRANSFORM
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jsonl_path = root / "provenance.jsonl"
            write_jsonl(jsonl_path, records)
            loaded = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
            self.assertEqual(len(loaded), len(records))

            image = Image.new("RGB", (400, 400), "black")
            geometry_path = root / "geometry.png"
            selection_path = root / "selection.png"
            render_geometry(image, records, IMAGE_TRANSFORM, geometry_path)
            render_selection_overlay(image, records, 1, selection_path)
            with Image.open(geometry_path) as rendered:
                self.assertEqual(rendered.size, image.size)
            with Image.open(selection_path) as rendered:
                self.assertEqual(rendered.size, image.size)


if __name__ == "__main__":
    unittest.main()

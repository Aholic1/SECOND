"""Reconstruct SECOND visual-token provenance from a passive runtime trace."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PROVENANCE_SCHEMA_VERSION = 1
STAGE_SOURCE_TYPES = {
    -2: "low_res_base",
    -1: "low_res_base",
    0: "base",
    1: "anyres_tile",
}


def _array(value, *, dtype=None):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=dtype)
    return array


def _rect_area(rect):
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])


def _intersect(rect, bounds):
    x1 = min(max(float(rect[0]), float(bounds[0])), float(bounds[2]))
    y1 = min(max(float(rect[1]), float(bounds[1])), float(bounds[3]))
    x2 = min(max(float(rect[2]), float(bounds[0])), float(bounds[2]))
    y2 = min(max(float(rect[3]), float(bounds[1])), float(bounds[3]))
    return [x1, y1, max(x1, x2), max(y1, y2)]


def _rounded_rect(rect):
    return [round(float(value), 6) for value in rect]


@dataclass(frozen=True)
class SourceGeometry:
    canvas_width: int
    canvas_height: int
    raster_width: int
    raster_height: int
    scaled_width: int
    scaled_height: int
    paste_x: int
    paste_y: int

    @property
    def scale_x(self):
        return self.scaled_width / self.canvas_width

    @property
    def scale_y(self):
        return self.scaled_height / self.canvas_height

    def raster_rect_to_canvas(self, rect):
        unpadded = [
            (rect[0] - self.paste_x) / self.scale_x,
            (rect[1] - self.paste_y) / self.scale_y,
            (rect[2] - self.paste_x) / self.scale_x,
            (rect[3] - self.paste_y) / self.scale_y,
        ]
        return _intersect(
            unpadded,
            [0.0, 0.0, float(self.canvas_width), float(self.canvas_height)],
        )


def source_geometry(source, canvas_size):
    canvas_width, canvas_height = map(int, canvas_size)
    if canvas_width <= 0 or canvas_height <= 0:
        raise ValueError(f"invalid canvas size: {canvas_size}")

    raster_width = int(source["raster_width"])
    raster_height = int(source["raster_height"])
    if raster_width <= 0 or raster_height <= 0:
        raise ValueError("source raster dimensions must be positive")

    if source["source_type"] == "anyres_tile":
        scale = min(raster_width / canvas_width, raster_height / canvas_height)
        scaled_width = min(math.ceil(canvas_width * scale), raster_width)
        scaled_height = min(math.ceil(canvas_height * scale), raster_height)
        paste_x = (raster_width - scaled_width) // 2
        paste_y = (raster_height - scaled_height) // 2
    else:
        scaled_width = raster_width
        scaled_height = raster_height
        paste_x = 0
        paste_y = 0

    return SourceGeometry(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        raster_width=raster_width,
        raster_height=raster_height,
        scaled_width=scaled_width,
        scaled_height=scaled_height,
        paste_x=paste_x,
        paste_y=paste_y,
    )


def _candidate_record(
    *,
    sample_id,
    forward_stage,
    source,
    source_order,
    candidate_index,
    row,
    col,
    selected,
    mask_selected,
    stage_sequence_index,
    visual_sequence_index,
    decoder_sequence_index,
    is_truncated,
    image_transform,
):
    grid_height = int(source["grid_height"])
    grid_width = int(source["grid_width"])
    canvas = image_transform["model_input_size"]
    canvas_size = (int(canvas["width"]), int(canvas["height"]))
    geometry = source_geometry(source, canvas_size)
    raster_rect = [
        col * geometry.raster_width / grid_width,
        row * geometry.raster_height / grid_height,
        (col + 1) * geometry.raster_width / grid_width,
        (row + 1) * geometry.raster_height / grid_height,
    ]
    canvas_rect = geometry.raster_rect_to_canvas(raster_rect)

    padding = image_transform["padding"]
    original = image_transform["original_size"]
    original_on_canvas = [
        float(padding["left"]),
        float(padding["top"]),
        float(padding["left"] + original["width"]),
        float(padding["top"] + original["height"]),
    ]
    original_overlap = _intersect(canvas_rect, original_on_canvas)
    original_rect = [
        original_overlap[0] - padding["left"],
        original_overlap[1] - padding["top"],
        original_overlap[2] - padding["left"],
        original_overlap[3] - padding["top"],
    ]

    raster_area = _rect_area(raster_rect)
    source_content_rect = [
        geometry.paste_x,
        geometry.paste_y,
        geometry.paste_x + geometry.scaled_width,
        geometry.paste_y + geometry.scaled_height,
    ]
    original_raster_rect = [
        geometry.paste_x + original_on_canvas[0] * geometry.scale_x,
        geometry.paste_y + original_on_canvas[1] * geometry.scale_y,
        geometry.paste_x + original_on_canvas[2] * geometry.scale_x,
        geometry.paste_y + original_on_canvas[3] * geometry.scale_y,
    ]
    original_raster_rect = _intersect(original_raster_rect, source_content_rect)
    padding_fraction = 1.0 - _rect_area(
        _intersect(raster_rect, original_raster_rect)
    ) / raster_area
    padding_fraction = min(1.0, max(0.0, padding_fraction))

    tile_rows = int(source.get("tile_rows", 1))
    tile_cols = int(source.get("tile_cols", 1))
    patch_rows = int(source.get("patch_rows_per_tile", grid_height))
    patch_cols = int(source.get("patch_cols_per_tile", grid_width))
    if source["source_type"] == "anyres_tile":
        tile_row = row // patch_rows
        tile_col = col // patch_cols
        patch_row = row % patch_rows
        patch_col = col % patch_cols
        tile_id = tile_row * tile_cols + tile_col
    else:
        tile_row = -1
        tile_col = -1
        tile_id = -1
        patch_row = row
        patch_col = col

    return {
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "sample_id": str(sample_id),
        "forward_stage": int(forward_stage),
        "stage_internal": int(source["stage_internal"]),
        "stage_order": int(source_order),
        "source_type": source["source_type"],
        "tile_id": tile_id,
        "tile_row": tile_row,
        "tile_col": tile_col,
        "tile_rows": tile_rows,
        "tile_cols": tile_cols,
        "patch_row": patch_row,
        "patch_col": patch_col,
        "candidate_index": int(candidate_index),
        "selected": bool(selected),
        "mask_selected": bool(mask_selected),
        "stage_sequence_index": int(stage_sequence_index),
        "visual_sequence_index": int(visual_sequence_index),
        "decoder_sequence_index": int(decoder_sequence_index),
        "preprocess_rect": _rounded_rect(raster_rect),
        "canvas_rect": _rounded_rect(canvas_rect),
        "original_rect_clipped": _rounded_rect(original_rect),
        "padding_fraction": round(float(padding_fraction), 8),
        "is_cls": False,
        "is_newline": False,
        "is_truncated": bool(is_truncated),
    }


def build_forward_provenance(sample_id, trace, image_transform):
    """Build candidate records and compare reconstruction with runtime truth."""
    forward_stage = int(trace["forward_stage"])
    sources = trace.get("sources", [])
    if not sources:
        raise ValueError(f"forward stage {forward_stage} has no source trace")
    if trace.get("mm_patch_merge_type") != "spatial":
        raise ValueError("Step 3 currently supports the exercised spatial merge only")

    decoder_start = int(trace["decoder_visual_start"])
    retained_visual_count = int(trace["decoder_retained_visual_count"])
    records = []
    reconstructed_runtime_match = True
    visual_cursor = 0
    source_summaries = []

    for source_order, source in enumerate(sources):
        stage = int(source["stage_internal"])
        expected_type = STAGE_SOURCE_TYPES.get(stage)
        if source["source_type"] != expected_type:
            raise ValueError(
                f"stage {stage} source type is {source['source_type']!r}, "
                f"expected {expected_type!r}"
            )
        grid_height = int(source["grid_height"])
        grid_width = int(source["grid_width"])
        if (
            grid_height
            != int(source.get("tile_rows", 1))
            * int(source.get("patch_rows_per_tile", grid_height))
            or grid_width
            != int(source.get("tile_cols", 1))
            * int(source.get("patch_cols_per_tile", grid_width))
        ):
            raise ValueError(f"stage {stage} tile and patch grids are inconsistent")
        consumed = _array(source["consumed_mask"], dtype=np.uint8)
        expected_shape = (grid_height, grid_width)
        if consumed.shape != expected_shape:
            raise ValueError(
                f"stage {stage} mask does not match runtime grid {expected_shape}"
            )
        if not np.isin(consumed, (0, 1)).all():
            raise ValueError(f"stage {stage} mask must be binary")

        mask_selected_indices = np.flatnonzero(consumed.reshape(-1))
        runtime_indices = _array(
            source["runtime_keep_indices"], dtype=np.int64
        ).reshape(-1)
        if runtime_indices.size and (
            runtime_indices.min() < 0 or runtime_indices.max() >= consumed.size
        ):
            raise ValueError(f"stage {stage} runtime keep index is out of range")
        source_match = np.array_equal(mask_selected_indices, runtime_indices)
        reconstructed_runtime_match &= source_match
        runtime_count = int(source["runtime_feature_count"])
        runtime_start = int(source["runtime_visual_start"])
        runtime_end = int(source["runtime_visual_end"])
        if (
            runtime_count != runtime_indices.size
            or runtime_start != visual_cursor
            or runtime_end != visual_cursor + runtime_count
        ):
            reconstructed_runtime_match = False

        selected_position = {int(index): pos for pos, index in enumerate(runtime_indices)}
        mask_selected = set(map(int, mask_selected_indices))
        for candidate_index in range(grid_height * grid_width):
            row, col = divmod(candidate_index, grid_width)
            selected = candidate_index in selected_position
            if selected:
                stage_index = selected_position[candidate_index]
                visual_index = visual_cursor + stage_index
                truncated = visual_index >= retained_visual_count
                decoder_index = -1 if truncated else decoder_start + visual_index
            else:
                stage_index = -1
                visual_index = -1
                decoder_index = -1
                truncated = False
            records.append(
                _candidate_record(
                    sample_id=sample_id,
                    forward_stage=forward_stage,
                    source=source,
                    source_order=source_order,
                    candidate_index=candidate_index,
                    row=row,
                    col=col,
                    selected=selected,
                    mask_selected=candidate_index in mask_selected,
                    stage_sequence_index=stage_index,
                    visual_sequence_index=visual_index,
                    decoder_sequence_index=decoder_index,
                    is_truncated=truncated,
                    image_transform=image_transform,
                )
            )
        visual_cursor += int(runtime_indices.size)
        source_summaries.append(
            {
                "stage_internal": stage,
                "source_type": source["source_type"],
                "grid_shape": [grid_height, grid_width],
                "tile_grid_shape": [
                    int(source.get("tile_rows", 1)),
                    int(source.get("tile_cols", 1)),
                ],
                "candidate_count": int(consumed.size),
                "mask_selected_count": int(mask_selected_indices.size),
                "selected_count": int(runtime_indices.size),
                "runtime_feature_count": runtime_count,
                "runtime_visual_range": [runtime_start, runtime_end],
                "runtime_mask_matches_reconstruction": bool(source_match),
            }
        )

    runtime_visual_length = int(trace["runtime_visual_length"])
    if visual_cursor != runtime_visual_length:
        reconstructed_runtime_match = False
    selected_records = [record for record in records if record["selected"]]
    visual_indices = [record["visual_sequence_index"] for record in selected_records]
    if visual_indices != list(range(runtime_visual_length)):
        reconstructed_runtime_match = False
    anyres_source = next(
        source for source in sources if int(source["stage_internal"]) == 1
    )
    expected_encoded_images = 1 + int(anyres_source["tile_rows"]) * int(
        anyres_source["tile_cols"]
    )
    if int(trace["encoded_image_count"]) != expected_encoded_images:
        reconstructed_runtime_match = False

    summary = {
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "sample_id": str(sample_id),
        "forward_stage": forward_stage,
        "mm_patch_merge_type": trace["mm_patch_merge_type"],
        "image_aspect_ratio": trace["image_aspect_ratio"],
        "canvas_size": list(map(int, trace["canvas_size"])),
        "encoded_image_count": int(trace["encoded_image_count"]),
        "expected_encoded_image_count": expected_encoded_images,
        "candidate_count": len(records),
        "selected_count": len(selected_records),
        "runtime_visual_length": runtime_visual_length,
        "decoder_visual_start": decoder_start,
        "decoder_visual_end": int(trace["decoder_visual_end"]),
        "decoder_sequence_length_before_truncation": int(
            trace["decoder_sequence_length_before_truncation"]
        ),
        "decoder_sequence_length_after_truncation": int(
            trace["decoder_sequence_length_after_truncation"]
        ),
        "tokenizer_model_max_length": trace["tokenizer_model_max_length"],
        "decoder_retained_visual_count": retained_visual_count,
        "decoder_truncated_visual_count": int(
            trace["decoder_truncated_visual_count"]
        ),
        "cls_token_count": 0,
        "newline_token_count": 0,
        "runtime_offline_match": bool(reconstructed_runtime_match),
        "sources": source_summaries,
    }
    return records, summary


def write_jsonl(path, records):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(path)


def render_geometry(image, records, image_transform, path):
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    padding = image_transform["padding"]
    original = image_transform["original_size"]
    original_box = (
        padding["left"],
        padding["top"],
        padding["left"] + original["width"] - 1,
        padding["top"] + original["height"] - 1,
    )
    draw.rectangle(original_box, outline=(255, 210, 40), width=3)
    anyres = next(
        (record for record in records if record["source_type"] == "anyres_tile"),
        None,
    )
    if anyres is not None:
        width, height = canvas.size
        for column in range(1, anyres["tile_cols"]):
            x = round(column * width / anyres["tile_cols"])
            draw.line((x, 0, x, height), fill=(40, 160, 255), width=2)
        for row in range(1, anyres["tile_rows"]):
            y = round(row * height / anyres["tile_rows"])
            draw.line((0, y, width, y), fill=(40, 160, 255), width=2)
    canvas.save(path)


def render_selection_overlay(image, records, stage_internal, path):
    base = image.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    stage_records = [
        record for record in records if record["stage_internal"] == stage_internal
    ]
    for record in stage_records:
        rect = record["canvas_rect"]
        if _rect_area(rect) == 0:
            continue
        if record["padding_fraction"] >= 1.0:
            color = (130, 130, 130, 100)
        elif record["selected"]:
            color = (32, 200, 80, 105)
        else:
            color = (235, 55, 55, 75)
        draw.rectangle(tuple(rect), fill=color)
    Image.alpha_composite(base, layer).convert("RGB").save(path)

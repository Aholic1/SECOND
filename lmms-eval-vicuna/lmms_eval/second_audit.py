from enum import Enum
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch

from lmms_eval.second_provenance import (
    PROVENANCE_SCHEMA_VERSION,
    build_forward_provenance,
    render_geometry,
    render_selection_overlay,
    write_jsonl,
)


AUDIT_SCHEMA_VERSION = 2
_BINARY_RESPONSE_RE = re.compile(r"^\s*(yes|no)\b", re.IGNORECASE)


def normalize_binary_response(text):
    match = _BINARY_RESPONSE_RE.match(str(text))
    return match.group(1).lower() if match else "invalid"


class AuditCondition(str, Enum):
    STANDARD_LLAVA = "A"
    RECURSIVE_ALLTOKEN_RAW = "B"
    RECURSIVE_ALLTOKEN_CD = "C"
    FULL_SECOND = "D"


class AuditArtifactMode(str, Enum):
    VISUAL = "visual"
    COMPACT = "compact"


def parse_audit_condition(condition):
    if isinstance(condition, AuditCondition):
        return condition
    if condition is None or str(condition).strip() == "":
        return None
    try:
        return AuditCondition(str(condition).strip().upper())
    except ValueError as exc:
        choices = ", ".join(item.value for item in AuditCondition)
        raise ValueError(f"audit_condition must be one of {choices}, got {condition!r}") from exc


def parse_audit_artifact_mode(mode):
    if isinstance(mode, AuditArtifactMode):
        return mode
    try:
        return AuditArtifactMode(str(mode).strip().lower())
    except ValueError as exc:
        choices = ", ".join(item.value for item in AuditArtifactMode)
        raise ValueError(f"audit_artifact_mode must be one of {choices}, got {mode!r}") from exc


def uses_recursion(condition):
    parsed = parse_audit_condition(condition)
    return parsed is not None and parsed is not AuditCondition.STANDARD_LLAVA


def uses_contrastive_decoding(condition):
    parsed = parse_audit_condition(condition)
    return parsed in {
        AuditCondition.RECURSIVE_ALLTOKEN_CD,
        AuditCondition.FULL_SECOND,
    }


def applies_selection(condition):
    return parse_audit_condition(condition) is AuditCondition.FULL_SECOND


def validate_audit_args(condition, output_dir, token_idx):
    parsed = parse_audit_condition(condition)
    if parsed is None:
        return None, None, int(token_idx)
    if output_dir is None or not str(output_dir).strip():
        raise ValueError("audit_output_dir is required when audit_condition is set")
    token_idx = int(token_idx)
    if token_idx < 0:
        raise ValueError("audit_token_idx must be non-negative")
    return parsed, Path(output_dir), token_idx


def applied_selection_mask(condition, proposed_mask):
    parsed = parse_audit_condition(condition)
    if parsed is AuditCondition.FULL_SECOND:
        return proposed_mask
    if parsed in {
        AuditCondition.RECURSIVE_ALLTOKEN_RAW,
        AuditCondition.RECURSIVE_ALLTOKEN_CD,
    }:
        return torch.ones_like(proposed_mask)
    raise ValueError(f"condition {condition!r} has no recursive selection transition")


def select_final_logits(stage_logits, contrastive_alphas, use_cd):
    if not stage_logits:
        raise ValueError("stage_logits must contain at least one tensor")
    final_logit = stage_logits[-1].clone()
    if not use_cd:
        return final_logit
    if len(contrastive_alphas) < len(stage_logits) - 1:
        raise ValueError("contrastive_alphas must cover every stage transition")

    top_k = min(20, final_logit.shape[-1])
    _, top_indices = stage_logits[-1].topk(top_k, dim=-1)
    final_mask = torch.zeros_like(stage_logits[-1], dtype=torch.bool)
    final_mask.scatter_(dim=-1, index=top_indices, value=True)

    for index in range(len(stage_logits) - 1):
        logit_diff = stage_logits[index + 1] - stage_logits[index]
        final_logit += contrastive_alphas[index] * logit_diff
        final_logit[~final_mask] = -1e9
    return final_logit


def _encode_without_special_tokens(tokenizer, text):
    return list(tokenizer.encode(text, add_special_tokens=False))


def resolve_binary_token_ids(tokenizer, prompt=None):
    resolved = {}
    for label in ("Yes", "No"):
        token_ids = _encode_without_special_tokens(tokenizer, label)
        if len(token_ids) != 1:
            return {
                "available": False,
                "reason": f"{label} is not one generated token: {token_ids}",
            }
        resolved[label.lower()] = int(token_ids[0])
    if resolved["yes"] == resolved["no"]:
        return {"available": False, "reason": "Yes and No resolved to the same token"}
    return {
        "available": True,
        "reason": None,
        "yes_token_id": resolved["yes"],
        "no_token_id": resolved["no"],
    }


def binary_logit_record(logits, token_ids):
    if not token_ids.get("available"):
        return {
            "available": False,
            "reason": token_ids.get("reason", "binary token IDs unavailable"),
            "yes_token_id": None,
            "no_token_id": None,
            "yes_logit": None,
            "no_logit": None,
            "yes_no_margin": None,
        }
    flat_logits = logits.detach().float().reshape(-1)
    yes_token_id = int(token_ids["yes_token_id"])
    no_token_id = int(token_ids["no_token_id"])
    yes_logit = float(flat_logits[yes_token_id].item())
    no_logit = float(flat_logits[no_token_id].item())
    if not math.isfinite(yes_logit) or not math.isfinite(no_logit):
        raise ValueError("Yes/No logits must be finite")
    return {
        "available": True,
        "reason": None,
        "yes_token_id": yes_token_id,
        "no_token_id": no_token_id,
        "yes_logit": yes_logit,
        "no_logit": no_logit,
        "yes_no_margin": yes_logit - no_logit,
    }


def _jsonable(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _atomic_write_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_jsonable) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _as_finite_array(name, value):
    if torch.is_tensor(value):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_binary_mask(name, value):
    array = _as_finite_array(name, value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D mask, got shape {array.shape}")
    if not np.isin(array, (0, 1)).all():
        raise ValueError(f"{name} must contain only binary values")
    return array.astype(np.uint8, copy=False)


def _nearest_resize_mask(mask, output_shape):
    source_height, source_width = mask.shape
    output_height, output_width = output_shape
    row_indices = np.floor(
        np.arange(output_height, dtype=np.float64) * source_height / output_height
    ).astype(np.int64)
    column_indices = np.floor(
        np.arange(output_width, dtype=np.float64) * source_width / output_width
    ).astype(np.int64)
    return mask[row_indices[:, None], column_indices[None, :]]


def _mask_index_arrays(mask):
    flat = mask.reshape(-1)
    keep_flat = np.flatnonzero(flat).astype(np.int32)
    drop_flat = np.flatnonzero(flat == 0).astype(np.int32)
    keep_coords = np.argwhere(mask == 1).astype(np.int16)
    drop_coords = np.argwhere(mask == 0).astype(np.int16)
    return keep_flat, drop_flat, keep_coords, drop_coords


def _safe_sample_id(sample_id):
    raw = str(sample_id)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
    if not safe:
        raise ValueError("sample_id must contain at least one path-safe character")
    return safe


def _attention_rgb(attention):
    attention = attention.astype(np.float32, copy=False)
    minimum = float(attention.min())
    maximum = float(attention.max())
    if maximum > minimum:
        unit = (attention - minimum) / (maximum - minimum)
    else:
        unit = np.zeros_like(attention)
    red = unit
    green = 1.0 - np.abs(2.0 * unit - 1.0)
    blue = 1.0 - unit
    return np.clip(np.stack([red, green, blue], axis=-1) * 255.0, 0, 255).astype(
        np.uint8
    )


def _blend_attention(image, attention):
    heatmap = Image.fromarray(_attention_rgb(attention), mode="RGB").resize(
        image.size, Image.Resampling.BILINEAR
    )
    return Image.blend(image.convert("RGB"), heatmap, alpha=0.55)


def _overlay_mask(image, mask, color):
    mask_image = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L").resize(
        image.size, Image.Resampling.NEAREST
    )
    color_layer = Image.new("RGB", image.size, color)
    tinted = Image.blend(image.convert("RGB"), color_layer, alpha=0.45)
    return Image.composite(tinted, image.convert("RGB"), mask_image)


class SecondAuditRecorder:
    def __init__(self, output_dir, condition, rank, token_idx=0, artifact_mode="visual"):
        parsed, output_dir, token_idx = validate_audit_args(
            condition, output_dir, token_idx
        )
        self.output_dir = output_dir
        self.condition = parsed
        self.rank = int(rank)
        self.token_idx = token_idx
        self.artifact_mode = parse_audit_artifact_mode(artifact_mode)
        self.rank_dir = self.output_dir / parsed.value / f"rank_{self.rank}"
        self.rank_dir.mkdir(parents=True, exist_ok=True)

    def _sample_dir(self, sample_id):
        return self.rank_dir / _safe_sample_id(sample_id)

    def begin_sample(self, sample_id, image, metadata, original_image=None):
        sample_dir = self._sample_dir(sample_id)
        sample_dir.mkdir(parents=False, exist_ok=False)
        if self.artifact_mode is AuditArtifactMode.VISUAL:
            image.convert("RGB").save(sample_dir / "input_image.png")
            if original_image is not None:
                original_image.convert("RGB").save(sample_dir / "original_image.png")
        payload = {
            "audit_schema_version": AUDIT_SCHEMA_VERSION,
            "condition": self.condition.value,
            "artifact_mode": self.artifact_mode.value,
            "rank": self.rank,
            "sample_id": str(sample_id),
            **dict(metadata),
        }
        _atomic_write_json(sample_dir / "sample.json", payload)
        return sample_dir

    def write_transition(
        self,
        sample_id,
        *,
        token_idx,
        source_stage,
        target_stage,
        ret_attn,
        proposed_mask,
        applied_mask,
        consumed_mask,
        metadata,
        mask_before=None,
    ):
        sample_dir = self._sample_dir(sample_id)
        if not sample_dir.is_dir():
            raise ValueError(f"sample {sample_id!r} has not been started")
        ret_attn = _as_finite_array("ret_attn", ret_attn).astype(np.float32, copy=False)
        if ret_attn.ndim != 2:
            raise ValueError(f"ret_attn must be a 2D array, got shape {ret_attn.shape}")
        proposed_mask = _as_binary_mask("proposed_mask", proposed_mask)
        applied_mask = _as_binary_mask("applied_mask", applied_mask)
        consumed_mask = _as_binary_mask("consumed_mask", consumed_mask)
        expected_consumed_mask = _nearest_resize_mask(applied_mask, consumed_mask.shape)
        if not np.array_equal(consumed_mask, expected_consumed_mask):
            raise ValueError("consumed_mask does not match the mask applied at model input")
        keep_flat, drop_flat, keep_coords, drop_coords = _mask_index_arrays(consumed_mask)
        arrays = {
            "ret_attn": ret_attn,
            "proposed_mask": proposed_mask,
            "applied_mask": applied_mask,
            "consumed_mask": consumed_mask,
            "consumed_keep_indices": keep_flat,
            "consumed_drop_indices": drop_flat,
            "consumed_keep_coords_rc": keep_coords,
            "consumed_drop_coords_rc": drop_coords,
        }
        if mask_before is not None:
            arrays["mask_before"] = _as_binary_mask("mask_before", mask_before)

        stem = f"token_{int(token_idx):03d}_stage_{source_stage}_to_{target_stage}"
        arrays_path = sample_dir / f"{stem}.npz"
        temporary = arrays_path.with_name(arrays_path.name + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(arrays_path)

        if self.artifact_mode is AuditArtifactMode.VISUAL:
            image = Image.open(sample_dir / "input_image.png").convert("RGB")
            _blend_attention(image, ret_attn).save(sample_dir / f"{stem}_attention.png")
            _overlay_mask(image, proposed_mask, (255, 48, 48)).save(
                sample_dir / f"{stem}_proposed_mask.png"
            )
            _overlay_mask(image, applied_mask, (48, 220, 96)).save(
                sample_dir / f"{stem}_applied_mask.png"
            )
            _overlay_mask(image, consumed_mask, (48, 140, 255)).save(
                sample_dir / f"{stem}_consumed_mask.png"
            )

        transition_metadata = {
            "audit_schema_version": AUDIT_SCHEMA_VERSION,
            "condition": self.condition.value,
            "sample_id": str(sample_id),
            "token_idx": int(token_idx),
            "source_stage": int(source_stage),
            "target_stage": int(target_stage),
            **dict(metadata),
            "selection_grid_shape": list(proposed_mask.shape),
            "consumed_grid_shape": list(consumed_mask.shape),
            "consumed_selected_token_count": int(keep_flat.size),
            "consumed_dropped_token_count": int(drop_flat.size),
            "consumed_total_token_count": int(consumed_mask.size),
            "consumed_keep_ratio": float(keep_flat.size / consumed_mask.size),
            "consumed_matches_applied_mask": True,
            "selection_signal_semantics": (
                "ret_attn is SECOND's rank-transformed selection signal, "
                "not raw causal attention"
            ),
        }
        _atomic_write_json(sample_dir / f"{stem}.json", transition_metadata)
        return arrays_path

    def write_provenance(self, sample_id, *, token_idx, traces):
        sample_dir = self._sample_dir(sample_id)
        sample_path = sample_dir / "sample.json"
        if not sample_path.is_file():
            raise ValueError(f"sample {sample_id!r} has not been started")
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
        image_transform = sample.get("image_transform")
        if not isinstance(image_transform, dict):
            raise ValueError("sample image_transform is required for provenance")
        if not traces:
            raise ValueError("at least one runtime provenance trace is required")

        summaries = []
        for trace in traces:
            records, summary = build_forward_provenance(
                sample_id,
                trace,
                image_transform,
            )
            forward_stage = summary["forward_stage"]
            stem = (
                f"token_{int(token_idx):03d}_forward_stage_{forward_stage}_provenance"
            )
            trace_arrays = {}
            for source in trace["sources"]:
                stage_label = str(int(source["stage_internal"])).replace("-", "neg")
                trace_arrays[f"stage_{stage_label}_consumed_mask"] = _as_binary_mask(
                    "consumed_mask", source["consumed_mask"]
                )
                trace_arrays[f"stage_{stage_label}_runtime_keep_indices"] = np.asarray(
                    source["runtime_keep_indices"], dtype=np.int32
                )
            trace_path = sample_dir / f"{stem}_runtime_trace.npz"
            temporary = trace_path.with_name(trace_path.name + ".tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **trace_arrays)
            temporary.replace(trace_path)
            write_jsonl(sample_dir / f"{stem}.jsonl", records)
            summaries.append(summary)

            if self.artifact_mode is AuditArtifactMode.VISUAL:
                image = Image.open(sample_dir / "input_image.png").convert("RGB")
                render_geometry(
                    image,
                    records,
                    image_transform,
                    sample_dir / f"{stem}_geometry.png",
                )
                for source in summary["sources"]:
                    source_stage = source["stage_internal"]
                    render_selection_overlay(
                        image,
                        records,
                        source_stage,
                        sample_dir
                        / f"{stem}_source_stage_{source_stage}_selection.png",
                    )

        payload = {
            "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
            "sample_id": str(sample_id),
            "token_idx": int(token_idx),
            "forward_stages": summaries,
            "all_runtime_offline_match": all(
                summary["runtime_offline_match"] for summary in summaries
            ),
            "total_truncated_visual_tokens": sum(
                summary["decoder_truncated_visual_count"] for summary in summaries
            ),
        }
        _atomic_write_json(sample_dir / "provenance_summary.json", payload)
        return payload

    def finish_sample(self, sample_id, metadata):
        sample_path = self._sample_dir(sample_id) / "sample.json"
        if not sample_path.is_file():
            raise ValueError(f"sample {sample_id!r} has not been started")
        payload = json.loads(sample_path.read_text(encoding="utf-8"))
        payload.update(dict(metadata))
        _atomic_write_json(sample_path, payload)

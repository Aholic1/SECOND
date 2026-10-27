"""Render PPT-ready SECOND attention audit figures and their companion index."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
import numpy as np
from PIL import Image


_TRANSITIONS = ((-2, -1), (-1, 0), (0, 1))
DEFAULT_QUESTION_IDS = ("887", "2254", "2284", "2920", "2679", "1859")

_AUDIT_ROLES = {
    "887": "C-to-D token-selection benefit",
    "2254": "B-to-C contrastive-decoding harm",
    "2284": "B-to-C contrastive-decoding harm",
    "2920": "B-to-C contrastive-decoding harm",
    "2679": "B-to-C contrastive-decoding benefit",
    "1859": "A-to-B recursive-pathway benefit",
}

ATTENTION_NORM = matplotlib.colors.LogNorm(vmin=0.25, vmax=8.0, clip=True)
ATTENTION_TICKS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
ATTENTION_CMAP = matplotlib.colormaps["turbo"].copy()
ATTENTION_COLORBAR_LABEL = "相对均匀注意力 / Relative-to-uniform attention"

_ATTENTION_TICK_LABELS = ("0.25×", "0.5×", "1×", "2×", "4×", "8×")
_KEEP_RATIO_ABSOLUTE_TOLERANCE = 1e-9

_CJK_FONT_FAMILIES = (
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
    "Microsoft YaHei",
)
_SAMPLE_LABELS = (
    "输入图像 / Input",
    "阶段 -2→-1 / Stage -2→-1",
    "阶段 -1→0 / Stage -1→0",
    "阶段 0→1 / Stage 0→1",
    "最终删除区域 / Removed Tokens",
)
_OVERVIEW_COLUMN_LABELS = (
    "输入图像 / Input",
    "最终注意力 / Final Attention",
    "最终删除区域 / Removed Tokens",
    "保留比例 / Keep Ratio",
)
_STRUCTURAL_LABEL_GLOSSARY = (
    *_SAMPLE_LABELS,
    _OVERVIEW_COLUMN_LABELS[1],
    _OVERVIEW_COLUMN_LABELS[3],
    ATTENTION_COLORBAR_LABEL,
)


@dataclass(frozen=True)
class TransitionData:
    source_stage: int
    target_stage: int
    attention_density: np.ndarray
    applied_mask: np.ndarray
    keep_ratio: float
    json_path: Path
    npz_path: Path


@dataclass(frozen=True)
class SampleData:
    question_id: str
    image: Image.Image
    metadata: dict
    transitions: tuple[TransitionData, TransitionData, TransitionData]
    sample_dir: Path


@dataclass(frozen=True)
class AuditPrediction:
    question: str
    ground_truth: str
    prediction: str
    source_path: Path


def find_cjk_font() -> Optional[FontProperties]:
    """Return the first preferred installed Chinese-capable font, if any."""
    installed = {entry.name: entry for entry in fontManager.ttflist}
    for family in _CJK_FONT_FAMILIES:
        entry = installed.get(family)
        if entry is not None:
            return FontProperties(fname=entry.fname)
    return None


def _panel_label(label: str, cjk_font: Optional[FontProperties]) -> str:
    if cjk_font is not None:
        return label
    return label.split(" / ", maxsplit=1)[-1]


def _set_panel_title(axis, label: str, cjk_font: Optional[FontProperties]) -> None:
    kwargs = {"fontsize": 14, "pad": 10}
    if cjk_font is not None:
        kwargs["fontproperties"] = cjk_font
    axis.set_title(_panel_label(label, cjk_font), **kwargs)


def _add_attention_colorbar(
    figure,
    color_axis,
    cjk_font: Optional[FontProperties],
) -> None:
    """Add the shared relative-to-uniform attention colorbar contract."""
    colorbar = figure.colorbar(
        matplotlib.cm.ScalarMappable(norm=ATTENTION_NORM, cmap=ATTENTION_CMAP),
        cax=color_axis,
        orientation="horizontal",
        ticks=ATTENTION_TICKS,
    )
    colorbar.set_ticklabels(_ATTENTION_TICK_LABELS)
    label_kwargs = {"fontsize": 11}
    if cjk_font is not None:
        label_kwargs["fontproperties"] = cjk_font
    colorbar.set_label(_panel_label(ATTENTION_COLORBAR_LABEL, cjk_font), **label_kwargs)
    colorbar.ax.tick_params(labelsize=10)


def _show_attention_overlay(axis, image: Image.Image, attention_density: np.ndarray) -> None:
    """Display an image with a bilinearly resized, fixed-alpha attention map."""
    rgb_image = image.convert("RGB")
    axis.imshow(rgb_image)
    resized_attention = np.asarray(
        Image.fromarray(np.asarray(attention_density, dtype=np.float32)).resize(
            rgb_image.size,
            Image.Resampling.BILINEAR,
        ),
        dtype=np.float32,
    )
    axis.imshow(ATTENTION_CMAP(ATTENTION_NORM(resized_attention)), alpha=0.55)


def _show_removed_tokens(axis, image: Image.Image, applied_mask: np.ndarray) -> None:
    """Display only removed mask cells as opaque red outlines over an image."""
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size
    rows, columns = applied_mask.shape
    axis.imshow(rgb_image, extent=(0, width, height, 0))
    for row, column in np.argwhere(applied_mask == 0):
        axis.add_patch(
            matplotlib.patches.Rectangle(
                (column * width / columns, row * height / rows),
                width / columns,
                height / rows,
                facecolor="red",
                edgecolor="white",
                linewidth=0.45,
            )
        )
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)


def _hide_axes(axis) -> None:
    axis.set_axis_off()


def _save_figure(figure, output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(output, dpi=240, facecolor="white")
    finally:
        plt.close(figure)


def render_sample_figure(sample: SampleData, output: Path) -> None:
    """Render one five-panel SECOND attention figure at 4K 16:9."""
    cjk_font = find_cjk_font()
    figure = plt.figure(figsize=(16, 9), dpi=240, facecolor="white")
    grid = figure.add_gridspec(
        2,
        5,
        height_ratios=(20, 1),
        hspace=0.16,
        wspace=0.08,
        left=0.035,
        right=0.985,
        top=0.90,
        bottom=0.09,
    )
    axes = [figure.add_subplot(grid[0, column]) for column in range(5)]
    figure.suptitle(f"QID {sample.question_id}", fontsize=19, y=0.965)

    axes[0].imshow(sample.image)
    _set_panel_title(axes[0], _SAMPLE_LABELS[0], cjk_font)
    _hide_axes(axes[0])

    for axis, label, transition in zip(axes[1:4], _SAMPLE_LABELS[1:4], sample.transitions):
        _show_attention_overlay(axis, sample.image, transition.attention_density)
        _set_panel_title(axis, label, cjk_font)
        _hide_axes(axis)

    _show_removed_tokens(axes[4], sample.image, sample.transitions[-1].applied_mask)
    _set_panel_title(axes[4], _SAMPLE_LABELS[4], cjk_font)
    _hide_axes(axes[4])

    color_axis = figure.add_subplot(grid[1, 1:4])
    _add_attention_colorbar(figure, color_axis, cjk_font)
    _save_figure(figure, output)


def render_overview(samples: Sequence[SampleData], output: Path) -> None:
    """Render six samples as a 4K overview of final-stage behavior."""
    if len(samples) != 6:
        raise ValueError("overview requires exactly six samples")

    cjk_font = find_cjk_font()
    figure, axes = plt.subplots(6, 4, figsize=(16, 9), dpi=240, facecolor="white")
    figure.subplots_adjust(
        left=0.03,
        right=0.985,
        top=0.94,
        bottom=0.13,
        hspace=0.44,
        wspace=0.10,
    )
    for axis, label in zip(axes[0], _OVERVIEW_COLUMN_LABELS):
        _set_panel_title(axis, label, cjk_font)

    stage_positions = np.arange(3)
    stage_labels = ("-2→-1", "-1→0", "0→1")
    keep_ratio_lower_bound = max(
        0.0,
        min(
            0.85,
            min(
                transition.keep_ratio
                for sample in samples
                for transition in sample.transitions
            ),
        ),
    )
    for row, sample in enumerate(samples):
        input_axis, attention_axis, removed_axis, ratio_axis = axes[row]
        final_transition = sample.transitions[-1]

        input_axis.imshow(sample.image)
        _hide_axes(input_axis)
        input_axis.text(
            0.03,
            0.94,
            f"QID {sample.question_id}",
            color="white",
            fontsize=8,
            transform=input_axis.transAxes,
            va="top",
        )

        _show_attention_overlay(attention_axis, sample.image, final_transition.attention_density)
        _hide_axes(attention_axis)

        _show_removed_tokens(removed_axis, sample.image, final_transition.applied_mask)
        _hide_axes(removed_axis)

        ratios = [transition.keep_ratio for transition in sample.transitions]
        ratio_axis.plot(stage_positions, ratios, color="#1f77b4", marker="o", linewidth=1.5, markersize=3)
        ratio_axis.set_xlim(-0.2, 2.2)
        ratio_axis.set_ylim(keep_ratio_lower_bound, 1.005)
        ratio_axis.set_xticks(stage_positions, stage_labels, fontsize=7)
        ratio_axis.tick_params(axis="y", labelsize=7)
        ratio_axis.grid(axis="y", linewidth=0.35, alpha=0.45)

    color_axis = figure.add_axes((0.31, 0.055, 0.38, 0.018))
    _add_attention_colorbar(figure, color_axis, cjk_font)
    _save_figure(figure, output)


def relative_attention_density(attention: np.ndarray) -> np.ndarray:
    """Normalize finite, non-negative 2D attention to a mean density of one."""
    array = np.asarray(attention, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("attention must be a finite non-negative 2D array")
    scale = float(array.max())
    if scale <= 0:
        raise ValueError("attention must have positive mass")
    scaled = array / scale
    return scaled / float(scaled.sum()) * array.size


def _validate_question_id(question_id: str) -> str:
    sample_id = str(question_id)
    if not sample_id.isascii() or not sample_id.isdigit():
        raise ValueError("question_id must contain ASCII digits only")
    return sample_id


def discover_sample(artifact_root: Path, question_id: str) -> Path:
    """Return the unique ``rank_*/<question_id>`` directory under an artifact root."""
    root = Path(artifact_root)
    sample_id = _validate_question_id(question_id)
    matches = [path for path in root.glob(f"rank_*/{sample_id}") if path.is_dir()]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one rank sample for question_id {sample_id!r}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _single_transition_path(sample_dir: Path, source_stage: int, target_stage: int, suffix: str) -> Path:
    paths = sorted(sample_dir.glob(f"token_*_stage_{source_stage}_to_{target_stage}{suffix}"))
    if not paths:
        raise ValueError(f"missing transition {source_stage}_to_{target_stage} ({suffix})")
    if len(paths) != 1:
        raise ValueError(f"expected one transition {source_stage}_to_{target_stage} ({suffix})")
    return paths[0]


def _load_transition(sample_dir: Path, source_stage: int, target_stage: int) -> TransitionData:
    json_path = _single_transition_path(sample_dir, source_stage, target_stage, ".json")
    npz_path = _single_transition_path(sample_dir, source_stage, target_stage, ".npz")
    if json_path.stem != npz_path.stem:
        raise ValueError(
            f"JSON and NPZ files must have matching transition stems for "
            f"{source_stage}_to_{target_stage}"
        )
    metadata = _load_json(json_path)
    if (
        metadata.get("source_stage") != source_stage
        or metadata.get("target_stage") != target_stage
    ):
        raise ValueError(f"transition metadata stages do not match {source_stage}_to_{target_stage}")

    try:
        with np.load(npz_path) as arrays:
            attention = np.array(arrays["ret_attn"], dtype=np.float64)
            applied_mask = np.array(arrays["applied_mask"], dtype=np.float64)
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError(f"invalid transition arrays: {npz_path}") from exc

    attention_density = relative_attention_density(attention)
    if applied_mask.ndim != 2 or applied_mask.shape != attention_density.shape:
        raise ValueError(f"applied mask shape must match attention for {source_stage}_to_{target_stage}")
    if not np.isfinite(applied_mask).all() or not np.isin(applied_mask, (0.0, 1.0)).all():
        raise ValueError(f"applied mask must be finite and binary for {source_stage}_to_{target_stage}")

    try:
        keep_ratio = float(metadata["applied_keep_ratio"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"transition keep ratio is missing or invalid for {source_stage}_to_{target_stage}") from exc
    if not np.isfinite(keep_ratio) or not 0.0 <= keep_ratio <= 1.0:
        raise ValueError(f"transition keep ratio must be in [0, 1] for {source_stage}_to_{target_stage}")
    mask_keep_ratio = float(applied_mask.mean())
    if not np.isclose(
        keep_ratio,
        mask_keep_ratio,
        rtol=0.0,
        atol=_KEEP_RATIO_ABSOLUTE_TOLERANCE,
    ):
        raise ValueError(
            f"transition keep ratio does not match applied mask mean for "
            f"{source_stage}_to_{target_stage}"
        )

    return TransitionData(
        source_stage=source_stage,
        target_stage=target_stage,
        attention_density=attention_density,
        applied_mask=applied_mask,
        keep_ratio=keep_ratio,
        json_path=json_path,
        npz_path=npz_path,
    )


def load_sample(artifact_root: Path, question_id: str) -> SampleData:
    """Load one audit sample with the three required recursive transitions."""
    sample_id = _validate_question_id(question_id)
    sample_dir = discover_sample(artifact_root, sample_id)
    sample_json_path = sample_dir / "sample.json"
    image_path = sample_dir / "input_image.png"
    if not sample_json_path.is_file():
        raise ValueError(f"missing sample.json in {sample_dir}")
    if not image_path.is_file():
        raise ValueError(f"missing input_image.png in {sample_dir}")

    metadata = _load_json(sample_json_path)
    if str(metadata.get("question_id", "")) != sample_id:
        raise ValueError(f"sample metadata question_id does not match {sample_id!r}")
    try:
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid input image: {image_path}") from exc

    transitions = tuple(
        _load_transition(sample_dir, source_stage, target_stage)
        for source_stage, target_stage in _TRANSITIONS
    )
    return SampleData(
        question_id=sample_id,
        image=image,
        metadata=metadata,
        transitions=transitions,
        sample_dir=sample_dir,
    )


def _path_is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _unique_condition_log(audit_root: Path, condition: str) -> Path:
    paths = sorted((Path(audit_root) / condition).rglob("*_samples_pope_pop_audit.jsonl"))
    if len(paths) != 1:
        raise ValueError(
            f"expected exactly one sample log for condition {condition!r}; found {len(paths)}"
        )
    return paths[0]


def _prediction_from_record(record: dict, source_path: Path) -> Optional[tuple[str, AuditPrediction]]:
    pope_accuracy = record.get("pope_accuracy")
    document = record.get("doc")
    if not isinstance(pope_accuracy, dict) or not isinstance(document, dict):
        return None
    metric_question_id = pope_accuracy.get("question_id")
    document_question_id = document.get("question_id")
    if (
        metric_question_id is not None
        and document_question_id is not None
        and str(metric_question_id) != str(document_question_id)
    ):
        raise ValueError(f"question ID disagreement in {source_path}")
    question_id = str(metric_question_id if metric_question_id is not None else document_question_id or "")
    question = document.get("question")
    ground_truth = pope_accuracy.get("ground_truth", document.get("answer", record.get("target")))
    prediction = pope_accuracy.get("prediction")
    if not question_id or not isinstance(question, str) or ground_truth is None or prediction is None:
        raise ValueError(f"incomplete POPE prediction record in {source_path}")
    return question_id, AuditPrediction(
        question=question,
        ground_truth=str(ground_truth),
        prediction=str(prediction),
        source_path=source_path,
    )


def load_audit_predictions(
    audit_root: Path, question_ids: Sequence[str]
) -> dict[str, dict[str, AuditPrediction]]:
    """Load the selected A/B/C/D predictions, rejecting ambiguous log records."""
    selected_ids = {str(question_id) for question_id in question_ids}
    predictions: dict[str, dict[str, AuditPrediction]] = {}
    for condition in "ABCD":
        source_path = _unique_condition_log(audit_root, condition)
        condition_predictions: dict[str, AuditPrediction] = {}
        try:
            lines = source_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ValueError(f"unable to read sample log: {source_path}") from exc
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL record {line_number} in {source_path}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSON object required at record {line_number} in {source_path}")
            result = _prediction_from_record(record, source_path)
            if result is None:
                continue
            question_id, prediction = result
            if question_id not in selected_ids:
                continue
            if question_id in condition_predictions:
                raise ValueError(
                    f"duplicate prediction for question_id {question_id!r} in {source_path}"
                )
            condition_predictions[question_id] = prediction
        missing_ids = selected_ids - condition_predictions.keys()
        if missing_ids:
            raise ValueError(
                f"missing selected predictions for condition {condition!r}: {sorted(missing_ids)}"
            )
        predictions[condition] = condition_predictions
    for question_id in selected_ids:
        reference = predictions["A"][question_id]
        for condition in "BCD":
            prediction = predictions[condition][question_id]
            if prediction.question != reference.question:
                raise ValueError(f"question disagreement for question_id {question_id!r}")
            if prediction.ground_truth != reference.ground_truth:
                raise ValueError(f"ground_truth disagreement for question_id {question_id!r}")
    return predictions


def _write_index(
    output: Path,
    samples: Sequence[SampleData],
    predictions: dict[str, dict[str, AuditPrediction]],
) -> None:
    lines = [
        "# SECOND attention-distribution figures",
        "",
        "These figures provide attention-distribution evidence; outcomes and audit roles are recorded here, not inside the PNGs.",
        "",
        "## 结构标签对照 / Structural-label glossary",
        "",
        *(f"- {label}" for label in _STRUCTURAL_LABEL_GLOSSARY),
        "",
    ]
    for sample in samples:
        question_id = sample.question_id
        first_prediction = predictions["A"][question_id]
        lines.extend(
            [
                f"## QID {question_id} — `q{question_id}_attention_distribution.png`",
                "",
                f"- Audit role: {_AUDIT_ROLES.get(question_id, 'Selected attention-distribution audit sample')}",
                f"- Question: {first_prediction.question}",
                f"- Ground truth: {first_prediction.ground_truth}",
                "- Predictions (A/B/C/D): "
                + " / ".join(predictions[condition][question_id].prediction for condition in "ABCD"),
                "- Keep ratios (-2→-1, -1→0, 0→1): "
                + ", ".join(f"{transition.keep_ratio:.4f}" for transition in sample.transitions),
                "- Source artifacts: "
                + ", ".join(
                    str(path)
                    for path in (
                        sample.sample_dir / "sample.json",
                        sample.sample_dir / "input_image.png",
                        *(transition.json_path for transition in sample.transitions),
                        *(transition.npz_path for transition in sample.transitions),
                    )
                ),
                "- Source logs (A/B/C/D): "
                + ", ".join(str(predictions[condition][question_id].source_path) for condition in "ABCD"),
                "- Suggested speaker note: Compare the shared-scale attention distribution and final removed cells with the paired A/B/C/D outcomes; this visual is descriptive evidence, not a causal conclusion.",
                "",
            ]
        )
    output.write_text("\n".join(lines), encoding="utf-8")


def generate_figures(
    artifact_root: Path,
    audit_root: Path,
    output: Path,
    question_ids: Sequence[str] = DEFAULT_QUESTION_IDS,
) -> tuple[Path, ...]:
    """Generate the selected figure set and Markdown companion index."""
    artifact_root = Path(artifact_root)
    output = Path(output)
    selected_ids = tuple(_validate_question_id(question_id) for question_id in question_ids)
    if _path_is_inside(output, artifact_root):
        raise ValueError("output path must not equal or be inside the artifact root")
    if len(selected_ids) != 6 or len(set(selected_ids)) != 6:
        raise ValueError("exactly six unique question IDs are required")

    samples = [load_sample(artifact_root, question_id) for question_id in selected_ids]
    predictions = load_audit_predictions(audit_root, selected_ids)
    if output.is_symlink():
        raise ValueError("output path must not be a symlink")
    output.mkdir(parents=True, exist_ok=True)

    independent_paths = [output / f"q{sample.question_id}_attention_distribution.png" for sample in samples]
    overview_path = output / "second_attention_overview.png"
    index_path = output / "README.md"
    generated = [*independent_paths, overview_path, index_path]
    symlinked_targets = [path for path in generated if path.is_symlink()]
    if symlinked_targets:
        rendered_targets = ", ".join(str(path) for path in symlinked_targets)
        raise ValueError(f"refusing to overwrite symlink output target(s): {rendered_targets}")
    resolved_output = output.resolve()
    for path in generated:
        try:
            path.resolve().relative_to(resolved_output)
        except ValueError as exc:
            raise ValueError(
                f"generated path must remain beneath resolved output directory: {path}"
            ) from exc

    for sample, path in zip(samples, independent_paths):
        render_sample_figure(sample, path)
    render_overview(samples, overview_path)
    _write_index(index_path, samples, predictions)
    return tuple(generated)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--audit-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--question-ids",
        default=",".join(DEFAULT_QUESTION_IDS),
        help="Comma-separated question IDs, in display order (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    question_ids = tuple(question_id.strip() for question_id in args.question_ids.split(",") if question_id.strip())
    try:
        generated = generate_figures(
            args.artifact_root,
            args.audit_root,
            args.output,
            question_ids,
        )
    except ValueError as exc:
        parser.error(str(exc))
    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

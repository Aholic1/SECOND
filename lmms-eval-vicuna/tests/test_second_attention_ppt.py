import json
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from PIL import Image

from tools.render_second_attention_ppt import (
    ATTENTION_CMAP,
    ATTENTION_NORM,
    ATTENTION_TICKS,
    SampleData,
    TransitionData,
    discover_sample,
    load_sample,
    relative_attention_density,
    render_overview,
    render_sample_figure,
)


TRANSITIONS = ((-2, -1), (-1, 0), (0, 1))
QUESTION_IDS = ("887", "2254", "2284", "2920", "2679", "1859")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_AUDIT_ROLES = {
    "887": "C-to-D token-selection benefit",
    "2254": "B-to-C contrastive-decoding harm",
    "2284": "B-to-C contrastive-decoding harm",
    "2920": "B-to-C contrastive-decoding harm",
    "2679": "B-to-C contrastive-decoding benefit",
    "1859": "A-to-B recursive-pathway benefit",
}


def write_synthetic_sample(root, missing_stage=None, question_id="887"):
    root = Path(root) / "rank_0"
    sample_dir = root / str(question_id)
    sample_dir.mkdir(parents=True)
    Image.new("RGB", (8, 6), "navy").save(sample_dir / "input_image.png")
    (sample_dir / "sample.json").write_text(
        json.dumps({"question_id": str(question_id), "question": "What is shown?"}),
        encoding="utf-8",
    )
    for source_stage, target_stage in TRANSITIONS:
        if f"{source_stage}_to_{target_stage}" == missing_stage:
            continue
        stem = f"token_000_stage_{source_stage}_to_{target_stage}"
        np.savez_compressed(
            sample_dir / f"{stem}.npz",
            ret_attn=np.array([[1.0, 2.0], [3.0, 4.0]]),
            applied_mask=np.array([[1.0, 1.0], [0.0, 0.0]]),
        )
        (sample_dir / f"{stem}.json").write_text(
            json.dumps(
                {
                    "source_stage": source_stage,
                    "target_stage": target_stage,
                    "applied_keep_ratio": 0.5,
                }
            ),
            encoding="utf-8",
        )
    return root.parent


def write_synthetic_audit(root):
    root = Path(root)
    artifact_root = root / "D" / "artifacts" / "D"
    for question_id in QUESTION_IDS:
        write_synthetic_sample(artifact_root, question_id=question_id)
    for condition in "ABCD":
        result_path = root / condition / "results" / "synthetic" / f"{condition}_samples_pope_pop_audit.jsonl"
        result_path.parent.mkdir(parents=True)
        records = []
        for question_id in QUESTION_IDS:
            records.append(
                {
                    "doc": {
                        "question_id": question_id,
                        "question": f"Is item {question_id} visible?",
                        "answer": "yes",
                    },
                    "pope_accuracy": {
                        "question_id": question_id,
                        "prediction": "yes" if condition in "AC" else "no",
                        "ground_truth": "yes",
                    },
                }
            )
        result_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
    return artifact_root


def synthetic_log_path(audit_root, condition):
    return Path(audit_root) / condition / "results" / "synthetic" / f"{condition}_samples_pope_pop_audit.jsonl"


def rewrite_synthetic_record(audit_root, condition, question_id, update):
    path = synthetic_log_path(audit_root, condition)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for record in records:
        if record["doc"]["question_id"] == str(question_id):
            update(record)
            break
    else:
        raise AssertionError(f"missing synthetic question ID {question_id}")
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def run_cli(audit_root, artifact_root, output, question_ids=None):
    command = [
        sys.executable,
        "tools/render_second_attention_ppt.py",
        "--artifact-root",
        str(artifact_root),
        "--audit-root",
        str(audit_root),
        "--output",
        str(output),
    ]
    if question_ids is not None:
        command.extend(("--question-ids", ",".join(question_ids)))
    return subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def run_cli_with_synthetic_audit(tmp_path, output):
    audit_root = Path(tmp_path) / "audit"
    artifact_root = write_synthetic_audit(audit_root)
    return run_cli(audit_root, artifact_root, output)


def make_sample_data(question_id="887"):
    image = Image.new("RGB", (8, 6), "navy")
    transitions = tuple(
        TransitionData(
            source_stage=source_stage,
            target_stage=target_stage,
            attention_density=np.array([[0.25, 1.0], [2.0, 8.0]]),
            applied_mask=np.array([[1.0, 1.0], [0.0, 0.0]]),
            keep_ratio=0.95 - index * 0.025,
            json_path=Path(f"transition-{index}.json"),
            npz_path=Path(f"transition-{index}.npz"),
        )
        for index, (source_stage, target_stage) in enumerate(TRANSITIONS)
    )
    return SampleData(
        question_id=str(question_id),
        image=image,
        metadata={"question": "What is shown?"},
        transitions=transitions,
        sample_dir=Path(f"sample-{question_id}"),
    )


class AttentionDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_relative_density_has_unit_mean(self):
        density = relative_attention_density(np.array([[1.0, 1.0], [2.0, 0.0]]))
        self.assertAlmostEqual(float(density.mean()), 1.0)
        self.assertTrue(np.isfinite(density).all())

    def test_relative_density_rejects_invalid_arrays(self):
        for attention in (
            np.array([[np.nan]]),
            np.array([[-1.0, 2.0]]),
            np.zeros((2, 2)),
        ):
            with self.assertRaises(ValueError):
                relative_attention_density(attention)

    def test_relative_density_normalizes_finite_values_without_overflow(self):
        density = relative_attention_density(np.array([[1e308, 1e308]]))

        self.assertTrue(np.isfinite(density).all())
        self.assertAlmostEqual(float(density.mean()), 1.0)

    def test_load_sample_requires_exact_transition_order(self):
        root = write_synthetic_sample(self.tmp_path, missing_stage="-1_to_0")
        with self.assertRaisesRegex(ValueError, "missing transition"):
            load_sample(root, "887")

    def test_load_sample_returns_validated_ordered_data(self):
        root = write_synthetic_sample(self.tmp_path)

        sample = load_sample(root, "887")

        self.assertEqual(discover_sample(root, "887"), sample.sample_dir)
        self.assertEqual(sample.question_id, "887")
        self.assertEqual(sample.image.mode, "RGB")
        self.assertEqual(sample.image.size, (8, 6))
        self.assertEqual(
            [(item.source_stage, item.target_stage) for item in sample.transitions],
            [(-2, -1), (-1, 0), (0, 1)],
        )
        self.assertAlmostEqual(sample.transitions[-1].keep_ratio, 0.5)
        self.assertTrue(np.array_equal(sample.transitions[-1].applied_mask, [[1, 1], [0, 0]]))

    def test_load_sample_rejects_nonbinary_mask(self):
        root = write_synthetic_sample(self.tmp_path)
        path = root / "rank_0" / "887" / "token_000_stage_0_to_1.npz"
        np.savez_compressed(
            path,
            ret_attn=np.ones((2, 2)),
            applied_mask=np.array([[1.0, 0.5], [0.0, 0.0]]),
        )

        with self.assertRaisesRegex(ValueError, "binary"):
            load_sample(root, "887")

    def test_load_sample_rejects_metadata_question_id_mismatch(self):
        root = write_synthetic_sample(self.tmp_path)
        sample_json = root / "rank_0" / "887" / "sample.json"
        sample_json.write_text(json.dumps({"question_id": "2254"}), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "metadata question_id"):
            load_sample(root, "887")

    def test_load_sample_rejects_applied_mask_shape_mismatch(self):
        root = write_synthetic_sample(self.tmp_path)
        npz_path = root / "rank_0" / "887" / "token_000_stage_0_to_1.npz"
        np.savez_compressed(
            npz_path,
            ret_attn=np.ones((2, 2)),
            applied_mask=np.ones((1, 2)),
        )

        with self.assertRaisesRegex(ValueError, "mask shape"):
            load_sample(root, "887")

    def test_load_sample_rejects_invalid_keep_ratios(self):
        for index, keep_ratio in enumerate((float("nan"), -0.1, 1.1)):
            with self.subTest(keep_ratio=keep_ratio):
                root = write_synthetic_sample(self.tmp_path / str(index))
                json_path = root / "rank_0" / "887" / "token_000_stage_0_to_1.json"
                metadata = json.loads(json_path.read_text(encoding="utf-8"))
                metadata["applied_keep_ratio"] = keep_ratio
                json_path.write_text(json.dumps(metadata), encoding="utf-8")

                with self.assertRaisesRegex(ValueError, r"keep ratio must be in \[0, 1\]"):
                    load_sample(root, "887")

    def test_load_sample_rejects_keep_ratio_that_disagrees_with_mask(self):
        root = write_synthetic_sample(self.tmp_path)
        json_path = root / "rank_0" / "887" / "token_000_stage_0_to_1.json"
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        metadata["applied_keep_ratio"] = 0.50000001
        json_path.write_text(json.dumps(metadata), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "does not match applied mask mean"):
            load_sample(root, "887")

    def test_load_sample_requires_matching_transition_stems(self):
        root = write_synthetic_sample(self.tmp_path)
        sample_dir = root / "rank_0" / "887"
        expected = sample_dir / "token_000_stage_-1_to_0.npz"
        mismatched = sample_dir / "token_001_stage_-1_to_0.npz"
        expected.rename(mismatched)

        with self.assertRaisesRegex(ValueError, "matching transition"):
            load_sample(root, "887")

    def test_discover_sample_rejects_duplicate_rank_matches(self):
        root = write_synthetic_sample(self.tmp_path)
        duplicate = root / "rank_1" / "887"
        duplicate.mkdir(parents=True)

        with self.assertRaisesRegex(ValueError, "exactly one"):
            discover_sample(root, "887")

    def test_discover_sample_rejects_question_id_traversal(self):
        root = write_synthetic_sample(self.tmp_path)

        with self.assertRaisesRegex(ValueError, "ASCII digits"):
            discover_sample(root, "../887")

    def test_discover_sample_rejects_question_id_glob_characters(self):
        root = write_synthetic_sample(self.tmp_path)

        for question_id in ("*", "[0-9]", "88?"):
            with self.subTest(question_id=question_id):
                with self.assertRaisesRegex(ValueError, "ASCII digits"):
                    discover_sample(root, question_id)

    def test_discover_sample_rejects_non_ascii_digits(self):
        root = write_synthetic_sample(self.tmp_path)

        with self.assertRaisesRegex(ValueError, "ASCII digits"):
            discover_sample(root, "８８７")


class FigureRenderingTests(unittest.TestCase):
    def setUp(self):
        plt.close("all")
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temporary_directory.name)

    def tearDown(self):
        plt.close("all")
        self.temporary_directory.cleanup()

    def _capture_sample_figure(self, sample):
        with patch("tools.render_second_attention_ppt._save_figure") as save_figure:
            render_sample_figure(sample, self.tmp_path / "sample.png")
        return save_figure.call_args.args[0]

    def _capture_overview(self, samples):
        with patch("tools.render_second_attention_ppt._save_figure") as save_figure:
            render_overview(samples, self.tmp_path / "overview.png")
        return save_figure.call_args.args[0]

    def test_independent_figure_is_4k_16_by_9(self):
        output = self.tmp_path / "sample.png"

        render_sample_figure(make_sample_data(), output)

        with Image.open(output) as image:
            self.assertEqual(image.size, (3840, 2160))

    def test_overview_is_4k_and_contains_six_rows(self):
        output = self.tmp_path / "overview.png"

        render_overview([make_sample_data(str(index)) for index in range(6)], output)

        with Image.open(output) as image:
            self.assertEqual(image.size, (3840, 2160))

    def test_overview_displays_each_row_question_id(self):
        figure = self._capture_overview([make_sample_data(str(index)) for index in range(6)])
        try:
            self.assertEqual(
                [[text.get_text() for text in axis.texts] for axis in figure.axes[:24:4]],
                [[f"QID {index}"] for index in range(6)],
            )
        finally:
            plt.close(figure)

    def test_sample_uses_fixed_scientific_attention_scale_and_labels(self):
        sample = make_sample_data()
        figure = self._capture_sample_figure(sample)
        try:
            self.assertEqual((ATTENTION_NORM.vmin, ATTENTION_NORM.vmax), (0.25, 8.0))
            self.assertTrue(ATTENTION_NORM.clip)
            self.assertEqual(ATTENTION_CMAP.name, "turbo")
            self.assertEqual(ATTENTION_TICKS, (0.25, 0.5, 1.0, 2.0, 4.0, 8.0))

            color_axis = figure.axes[-1]
            self.assertEqual(
                [label.get_text() for label in color_axis.get_xticklabels()],
                ["0.25×", "0.5×", "1×", "2×", "4×", "8×"],
            )

            overlay = figure.axes[1].images[-1]
            expected_density = np.asarray(
                Image.fromarray(sample.transitions[0].attention_density.astype(np.float32)).resize(
                    sample.image.size,
                    Image.Resampling.BILINEAR,
                )
            )
            expected_overlay = matplotlib.colormaps["turbo"](
                matplotlib.colors.LogNorm(vmin=0.25, vmax=8.0, clip=True)(expected_density)
            )
            self.assertEqual(overlay.get_alpha(), 0.55)
            np.testing.assert_allclose(np.asarray(overlay.get_array()), expected_overlay)
        finally:
            plt.close(figure)

    def test_both_renderers_have_the_same_labeled_attention_colorbar(self):
        with patch(
            "tools.render_second_attention_ppt.find_cjk_font",
            return_value=FontProperties(),
        ):
            sample_figure = self._capture_sample_figure(make_sample_data())
            overview_figure = self._capture_overview(
                [make_sample_data(str(index)) for index in range(6)]
            )
        try:
            for figure in (sample_figure, overview_figure):
                color_axes = [
                    axis
                    for axis in figure.axes
                    if axis.get_xlabel()
                    == "相对均匀注意力 / Relative-to-uniform attention"
                ]
                self.assertEqual(len(color_axes), 1)
                colorbar = color_axes[0]._colorbar
                self.assertIs(colorbar.norm, ATTENTION_NORM)
                np.testing.assert_allclose(colorbar.get_ticks(), ATTENTION_TICKS)
                self.assertEqual(
                    [label.get_text() for label in color_axes[0].get_xticklabels()],
                    ["0.25×", "0.5×", "1×", "2×", "4×", "8×"],
                )
        finally:
            plt.close(sample_figure)
            plt.close(overview_figure)

    def test_attention_colorbar_labels_fit_inside_the_saved_canvas(self):
        sample_figure = self._capture_sample_figure(make_sample_data())
        overview_figure = self._capture_overview(
            [make_sample_data(str(index)) for index in range(6)]
        )
        try:
            for figure in (sample_figure, overview_figure):
                figure.canvas.draw()
                color_axes = [
                    axis
                    for axis in figure.axes
                    if axis.get_xlabel() == "Relative-to-uniform attention"
                ]
                self.assertEqual(len(color_axes), 1)
                label_bounds = color_axes[0].xaxis.label.get_window_extent(
                    renderer=figure.canvas.get_renderer()
                )
                self.assertGreaterEqual(label_bounds.y0, 0.0)
        finally:
            plt.close(sample_figure)
            plt.close(overview_figure)

    def test_overview_uses_bilingual_structural_labels_when_cjk_font_exists(self):
        with patch(
            "tools.render_second_attention_ppt.find_cjk_font",
            return_value=FontProperties(),
        ):
            figure = self._capture_overview(
                [make_sample_data(str(index)) for index in range(6)]
            )
        try:
            self.assertEqual(
                [axis.get_title() for axis in figure.axes[:4]],
                [
                    "输入图像 / Input",
                    "最终注意力 / Final Attention",
                    "最终删除区域 / Removed Tokens",
                    "保留比例 / Keep Ratio",
                ],
            )
        finally:
            plt.close(figure)

    def test_removed_panels_use_only_the_final_transition_mask(self):
        sample = make_sample_data()
        transitions = tuple(
            replace(transition, applied_mask=mask)
            for transition, mask in zip(
                sample.transitions,
                (
                    np.zeros((2, 2)),
                    np.zeros((2, 2)),
                    np.array([[1.0, 1.0], [0.0, 0.0]]),
                ),
            )
        )
        final_mask_sample = replace(sample, transitions=transitions)
        figure = self._capture_sample_figure(final_mask_sample)
        overview = self._capture_overview([final_mask_sample] * 6)
        try:
            for removed_axis in (figure.axes[4], overview.axes[2]):
                self.assertEqual(len(removed_axis.images), 1)
                self.assertEqual(len(removed_axis.patches), 2)
                for patch_artist in removed_axis.patches:
                    np.testing.assert_allclose(patch_artist.get_facecolor(), (1.0, 0.0, 0.0, 1.0))
                    np.testing.assert_allclose(patch_artist.get_edgecolor(), (1.0, 1.0, 1.0, 1.0))
        finally:
            plt.close(figure)
            plt.close(overview)

    def test_overview_keep_ratio_axes_use_fixed_scale_and_stage_labels(self):
        figure = self._capture_overview([make_sample_data(str(index)) for index in range(6)])
        try:
            self.assertEqual(
                [axis.get_title() for axis in figure.axes[:4]],
                ["Input", "Final Attention", "Removed Tokens", "Keep Ratio"],
            )
            for ratio_axis in figure.axes[3::4]:
                self.assertEqual(ratio_axis.get_ylim(), (0.85, 1.005))
                self.assertEqual(
                    [label.get_text() for label in ratio_axis.get_xticklabels()],
                    ["-2→-1", "-1→0", "0→1"],
                )
        finally:
            plt.close(figure)

    def test_overview_uses_one_lower_bound_that_includes_low_keep_ratios(self):
        samples = [make_sample_data(str(index)) for index in range(6)]
        low_ratio_sample = samples[2]
        low_ratio_sample = replace(
            low_ratio_sample,
            transitions=(
                replace(low_ratio_sample.transitions[0], keep_ratio=0.42),
                *low_ratio_sample.transitions[1:],
            ),
        )
        samples[2] = low_ratio_sample

        figure = self._capture_overview(samples)
        try:
            limits = [axis.get_ylim() for axis in figure.axes[3:24:4]]
            self.assertEqual(len(set(limits)), 1)
            self.assertLessEqual(limits[0][0], 0.42)
            self.assertEqual(limits[0][1], 1.005)
        finally:
            plt.close(figure)

    def test_sample_text_is_limited_to_qid_structural_labels_and_colorbar(self):
        sample = replace(make_sample_data(), metadata={"question": "forbidden prose"})
        figure = self._capture_sample_figure(sample)
        try:
            displayed_text = {
                text
                for axis in figure.axes
                for text in (
                    [axis.get_title()]
                    + [artist.get_text() for artist in axis.texts]
                    + (
                        [label.get_text() for label in axis.get_xticklabels()]
                        + [label.get_text() for label in axis.get_yticklabels()]
                        if axis.axison
                        else []
                    )
                )
                if text
            }
            displayed_text.update(text.get_text() for text in figure.texts)
            allowed = {
                "QID 887",
                "Input",
                "Stage -2→-1",
                "Stage -1→0",
                "Stage 0→1",
                "Removed Tokens",
                "输入图像 / Input",
                "阶段 -2→-1 / Stage -2→-1",
                "阶段 -1→0 / Stage -1→0",
                "阶段 0→1 / Stage 0→1",
                "最终删除区域 / Removed Tokens",
                "Relative-to-uniform attention",
                "相对均匀注意力 / Relative-to-uniform attention",
                "0.25×",
                "0.5×",
                "1×",
                "2×",
                "4×",
                "8×",
            }
            self.assertTrue(displayed_text.issubset(allowed), displayed_text - allowed)
            self.assertNotIn("forbidden prose", displayed_text)
        finally:
            plt.close(figure)

    def test_saved_figures_have_white_background_and_leave_no_open_figures(self):
        sample_output = self.tmp_path / "sample.png"
        overview_output = self.tmp_path / "overview.png"

        render_sample_figure(make_sample_data(), sample_output)
        render_overview([make_sample_data(str(index)) for index in range(6)], overview_output)

        for output in (sample_output, overview_output):
            with Image.open(output) as image:
                self.assertEqual(image.convert("RGB").getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(plt.get_fignums(), [])


class FigureCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temporary_directory.name)
        self.output = self.tmp_path / "output"
        self.output.mkdir()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_cli_generates_exact_output_names(self):
        completed = run_cli_with_synthetic_audit(self.tmp_path, self.output)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            {path.name for path in self.output.iterdir()},
            {
                "q887_attention_distribution.png",
                "q1859_attention_distribution.png",
                "q2254_attention_distribution.png",
                "q2284_attention_distribution.png",
                "q2679_attention_distribution.png",
                "q2920_attention_distribution.png",
                "second_attention_overview.png",
                "README.md",
            },
        )

    def test_cli_preserves_unrelated_files_and_prints_each_generated_path(self):
        sentinel = self.output / "keep-me.txt"
        sentinel.write_text("unrelated output", encoding="utf-8")
        audit_root = self.tmp_path / "audit"
        artifact_root = write_synthetic_audit(audit_root)

        completed = run_cli(audit_root, artifact_root, self.output)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "unrelated output")
        expected_paths = {
            str(self.output / f"q{question_id}_attention_distribution.png")
            for question_id in QUESTION_IDS
        }
        expected_paths.update(
            {
                str(self.output / "second_attention_overview.png"),
                str(self.output / "README.md"),
            }
        )
        self.assertEqual(set(completed.stdout.splitlines()), expected_paths)

    def test_cli_rejects_output_equal_to_or_below_artifact_root(self):
        audit_root = self.tmp_path / "audit"
        artifact_root = write_synthetic_audit(audit_root)

        completed = run_cli(audit_root, artifact_root, artifact_root / "ppt-output")

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("must not equal or be inside the artifact root", completed.stderr)
        self.assertFalse((artifact_root / "ppt-output").exists())

    def test_cli_rejects_question_id_traversal_before_generating_outputs(self):
        audit_root = self.tmp_path / "audit"
        artifact_root = write_synthetic_audit(audit_root)
        question_ids = ("../887", *QUESTION_IDS[1:])

        completed = run_cli(
            audit_root,
            artifact_root,
            self.output,
            question_ids=question_ids,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("ASCII digits", completed.stderr)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_cli_rejects_question_id_glob_characters_before_generating_outputs(self):
        audit_root = self.tmp_path / "audit"
        artifact_root = write_synthetic_audit(audit_root)
        question_ids = ("*", *QUESTION_IDS[1:])

        completed = run_cli(
            audit_root,
            artifact_root,
            self.output,
            question_ids=question_ids,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("ASCII digits", completed.stderr)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_cli_refuses_symlinked_deliverables_without_modifying_their_target(self):
        audit_root = self.tmp_path / "audit"
        artifact_root = write_synthetic_audit(audit_root)
        protected = self.tmp_path / "protected-source"
        protected.write_bytes(b"original source bytes")
        for filename in ("q887_attention_distribution.png", "README.md"):
            with self.subTest(filename=filename):
                output = self.tmp_path / f"output-{filename}"
                output.mkdir()
                (output / filename).symlink_to(protected)

                completed = run_cli(audit_root, artifact_root, output)

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("symlink", completed.stderr)
                self.assertEqual(protected.read_bytes(), b"original source bytes")

    def test_cli_rejects_metric_and_document_question_id_disagreement(self):
        audit_root = self.tmp_path / "audit"
        artifact_root = write_synthetic_audit(audit_root)
        rewrite_synthetic_record(
            audit_root,
            "B",
            "887",
            lambda record: record["pope_accuracy"].update(question_id="2254"),
        )

        completed = run_cli(audit_root, artifact_root, self.output)

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("question ID disagreement", completed.stderr)

    def test_cli_rejects_cross_condition_question_and_ground_truth_disagreement(self):
        for field, changed_value in (("question", "A different question"), ("ground_truth", "no")):
            with self.subTest(field=field):
                audit_root = self.tmp_path / f"audit-{field}"
                artifact_root = write_synthetic_audit(audit_root)
                if field == "question":
                    rewrite_synthetic_record(
                        audit_root,
                        "C",
                        "887",
                        lambda record: record["doc"].update(question=changed_value),
                    )
                else:
                    rewrite_synthetic_record(
                        audit_root,
                        "C",
                        "887",
                        lambda record: record["pope_accuracy"].update(ground_truth=changed_value),
                    )

                completed = run_cli(audit_root, artifact_root, self.tmp_path / f"output-{field}")

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(f"{field} disagreement", completed.stderr)

    def test_index_contains_complete_audit_details(self):
        completed = run_cli_with_synthetic_audit(self.tmp_path, self.output)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        index_path = self.output / "README.md"
        self.assertTrue(index_path.is_file(), "CLI must create its companion index")
        index = index_path.read_text(encoding="utf-8")
        self.assertEqual(index.count("- Suggested speaker note:"), 6)
        for structural_label in (
            "输入图像 / Input",
            "阶段 -2→-1 / Stage -2→-1",
            "阶段 -1→0 / Stage -1→0",
            "阶段 0→1 / Stage 0→1",
            "最终注意力 / Final Attention",
            "最终删除区域 / Removed Tokens",
            "保留比例 / Keep Ratio",
            "相对均匀注意力 / Relative-to-uniform attention",
        ):
            self.assertIn(structural_label, index)
        for question_id in QUESTION_IDS:
            sample_dir = self.tmp_path / "audit" / "D" / "artifacts" / "D" / "rank_0" / question_id
            self.assertIn(f"## QID {question_id} — `q{question_id}_attention_distribution.png`", index)
            self.assertIn(f"- Audit role: {EXPECTED_AUDIT_ROLES[question_id]}", index)
            self.assertIn(f"- Question: Is item {question_id} visible?", index)
            self.assertIn("- Ground truth: yes", index)
            self.assertIn("- Predictions (A/B/C/D): yes / no / yes / no", index)
            self.assertIn("- Keep ratios (-2→-1, -1→0, 0→1): 0.5000, 0.5000, 0.5000", index)
            self.assertIn(str(sample_dir / "sample.json"), index)
            self.assertIn(str(sample_dir / "input_image.png"), index)
            for source_stage, target_stage in TRANSITIONS:
                stem = f"token_000_stage_{source_stage}_to_{target_stage}"
                self.assertIn(str(sample_dir / f"{stem}.json"), index)
                self.assertIn(str(sample_dir / f"{stem}.npz"), index)
            for condition in "ABCD":
                self.assertIn(str(synthetic_log_path(self.tmp_path / "audit", condition)), index)


if __name__ == "__main__":
    unittest.main()

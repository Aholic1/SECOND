# SECOND Attention PPT Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate six scientifically comparable, PPT-ready SECOND attention figures plus one overview and a companion index from the completed D-condition audit artifacts.

**Architecture:** A single focused rendering module loads and validates one sample's saved NPZ/JSON/image artifacts, converts raw attention to relative-to-uniform density, and renders independent or overview figures with a shared logarithmic color scale. A CLI selects the six approved question IDs and writes deterministic outputs to a dedicated directory without modifying source artifacts.

**Tech Stack:** Python 3.10, NumPy, Pillow, Matplotlib, standard-library `unittest`, Bash.

## Global Constraints

- Read inputs only from `audit/runs/pope_pop_audit_v1/D/artifacts/D` and the four A/B/C/D sample logs.
- Produce exactly six independent PNGs, one overview PNG, and one UTF-8 Markdown index.
- Independent PNGs are 3840×2160 pixels with a white background.
- Do not print questions, ground truths, predictions, or causal claims inside PNG files.
- Use one shared `LogNorm(vmin=0.25, vmax=8.0)` scale with ticks `0.25×, 0.5×, 1×, 2×, 4×, 8×`.
- Use bilinear interpolation for attention and nearest-neighbor interpolation for masks.
- Existing audit artifacts, logs, `run.sh`, and unrelated tests remain untouched.
- Tests must run with `python3 -m unittest` because the Codex system Python does not provide pytest.

---

### Task 1: Load and validate attention samples

**Files:**
- Create: `tools/render_second_attention_ppt.py`
- Create: `tests/test_second_attention_ppt.py`

**Interfaces:**
- Produces: `relative_attention_density(attention: np.ndarray) -> np.ndarray`.
- Produces: `discover_sample(artifact_root: Path, question_id: str) -> Path`.
- Produces: `load_sample(artifact_root: Path, question_id: str) -> SampleData`.
- `SampleData` contains the RGB input image, three ordered transitions, final applied mask, sample metadata, and source paths.

- [ ] **Step 1: Write failing normalization and validation tests**

```python
class AttentionDataTests(unittest.TestCase):
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

    def test_load_sample_requires_exact_transition_order(self):
        root = write_synthetic_sample(self.tmp_path, missing_stage="-1_to_0")
        with self.assertRaisesRegex(ValueError, "missing transition"):
            load_sample(root, "887")
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python3 -m unittest tests.test_second_attention_ppt.AttentionDataTests -v`

Expected: FAIL because `tools.render_second_attention_ppt` does not exist.

- [ ] **Step 3: Implement immutable data models and loaders**

```python
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


def relative_attention_density(attention):
    array = np.asarray(attention, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("attention must be a finite non-negative 2D array")
    total = float(array.sum())
    if total <= 0:
        raise ValueError("attention must have positive mass")
    return array / total * array.size
```

`load_sample` must locate exactly one `rank_*/<question_id>` directory, require `sample.json`, `input_image.png`, and the exact transitions `-2→-1`, `-1→0`, `0→1`. It validates matching 2D shapes, finite binary masks, finite keep ratios in `[0, 1]`, and the metadata question ID.

- [ ] **Step 4: Run the data tests and verify GREEN**

Run: `python3 -m unittest tests.test_second_attention_ppt.AttentionDataTests -v`

Expected: all `AttentionDataTests` pass.

- [ ] **Step 5: Commit the loader**

```bash
git add tools/render_second_attention_ppt.py tests/test_second_attention_ppt.py
git commit -m "feat: load SECOND attention figure data"
```

### Task 2: Render the independent and overview figures

**Files:**
- Modify: `tools/render_second_attention_ppt.py`
- Modify: `tests/test_second_attention_ppt.py`

**Interfaces:**
- Consumes: `SampleData` and `TransitionData` from Task 1.
- Produces: `render_sample_figure(sample: SampleData, output: Path) -> None`.
- Produces: `render_overview(samples: Sequence[SampleData], output: Path) -> None`.
- Produces: `find_cjk_font() -> Optional[FontProperties]`.

- [ ] **Step 1: Write failing rendering tests**

```python
class FigureRenderingTests(unittest.TestCase):
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
```

- [ ] **Step 2: Run the rendering tests and verify RED**

Run: `python3 -m unittest tests.test_second_attention_ppt.FigureRenderingTests -v`

Expected: FAIL because the rendering functions do not exist.

- [ ] **Step 3: Implement common scientific rendering helpers**

Use one module-level normalization and colormap:

```python
ATTENTION_NORM = matplotlib.colors.LogNorm(vmin=0.25, vmax=8.0, clip=True)
ATTENTION_TICKS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
ATTENTION_CMAP = matplotlib.colormaps["turbo"].copy()
```

Implement an overlay helper that displays the saved image, resizes attention with bilinear interpolation, and uses a fixed alpha of `0.55`. Implement a removed-token helper that renders `applied_mask == 0` in opaque red with a thin white cell boundary; do not tint retained cells.

`find_cjk_font` checks installed font family names in this order: `Noto Sans CJK SC`, `Source Han Sans SC`, `WenQuanYi Zen Hei`, `Microsoft YaHei`, and returns `None` if unavailable. If unavailable, panel labels fall back to the English halves only.

- [ ] **Step 4: Implement the five-panel 4K sample figure**

Use `figsize=(16, 9), dpi=240`, a five-column GridSpec, one shared horizontal colorbar, and panel labels:

```python
labels = (
    "输入图像 / Input",
    "阶段 -2→-1 / Stage -2→-1",
    "阶段 -1→0 / Stage -1→0",
    "阶段 0→1 / Stage 0→1",
    "最终删除区域 / Removed Tokens",
)
```

Only `QID <id>` and these structural labels may appear. Save with explicit white face color and without `bbox_inches="tight"` so output dimensions remain exact.

- [ ] **Step 5: Implement the six-row overview figure**

Use `figsize=(16, 9), dpi=240` with six rows and four columns: input, final attention, final removed-token overlay, and a compact line plot of the three keep ratios. The keep-ratio plot uses fixed y-limits `[0.85, 1.005]`, stage positions `-2→-1`, `-1→0`, `0→1`, and displays no causal prose.

- [ ] **Step 6: Run rendering tests and verify GREEN**

Run: `python3 -m unittest tests.test_second_attention_ppt.FigureRenderingTests -v`

Expected: all rendering tests pass and no Matplotlib figures remain open.

- [ ] **Step 7: Commit the renderer**

```bash
git add tools/render_second_attention_ppt.py tests/test_second_attention_ppt.py
git commit -m "feat: render SECOND attention PPT figures"
```

### Task 3: Add the CLI, generate the approved figure set, and verify outputs

**Files:**
- Modify: `tools/render_second_attention_ppt.py`
- Modify: `tests/test_second_attention_ppt.py`
- Generate: `audit/ppt_figures/second_attention_v1/*.png`
- Generate: `audit/ppt_figures/second_attention_v1/README.md`

**Interfaces:**
- Produces CLI:

```bash
python3 tools/render_second_attention_ppt.py \
  --artifact-root audit/runs/pope_pop_audit_v1/D/artifacts/D \
  --audit-root audit/runs/pope_pop_audit_v1 \
  --output audit/ppt_figures/second_attention_v1
```

- Defaults to question IDs `887,2254,2284,2920,2679,1859` in that order.

- [ ] **Step 1: Write failing CLI and index tests**

```python
class FigureCliTests(unittest.TestCase):
    def test_cli_generates_exact_output_names(self):
        completed = run_cli_with_synthetic_audit(self.tmp_path)
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

    def test_index_contains_predictions_but_png_labels_do_not(self):
        index = (self.output / "README.md").read_text(encoding="utf-8")
        self.assertIn("A/B/C/D", index)
        self.assertIn("887", index)
```

- [ ] **Step 2: Run CLI tests and verify RED**

Run: `python3 -m unittest tests.test_second_attention_ppt.FigureCliTests -v`

Expected: FAIL because the CLI and index writer are absent.

- [ ] **Step 3: Implement strict CLI and companion index generation**

The CLI refuses an output path equal to or inside the artifact root, creates the dedicated output directory, and overwrites only the eight exact generated filenames. It loads A/B/C/D predictions from the unique `*_samples_pope_pop_audit.jsonl` under each condition, joins by question ID, and writes the approved audit role, question, GT, predictions, keep ratios, source paths, and one suggested speaker note per sample to `README.md`.

- [ ] **Step 4: Run CLI tests and the complete new test module**

Run: `python3 -m unittest tests.test_second_attention_ppt -v`

Expected: all tests pass.

- [ ] **Step 5: Record source hashes and generate the real figures**

Run:

```bash
find audit/runs/pope_pop_audit_v1/D/artifacts/D \
  -type f \( -name '*.npz' -o -name '*.json' -o -name 'input_image.png' \) \
  -print0 | sort -z | xargs -0 sha256sum > /tmp/second_attention_sources_before.sha256

python3 tools/render_second_attention_ppt.py \
  --artifact-root audit/runs/pope_pop_audit_v1/D/artifacts/D \
  --audit-root audit/runs/pope_pop_audit_v1 \
  --output audit/ppt_figures/second_attention_v1
```

Expected: the CLI prints six independent output paths, the overview path, and the index path.

- [ ] **Step 6: Verify dimensions, file count, source immutability, and visual output**

Run:

```bash
python3 -c 'from pathlib import Path; from PIL import Image; p=Path("audit/ppt_figures/second_attention_v1"); fs=sorted(p.glob("*.png")); assert len(fs)==7; assert all(Image.open(f).size==(3840,2160) for f in fs); print(len(fs), "PNGs verified")'

find audit/runs/pope_pop_audit_v1/D/artifacts/D \
  -type f \( -name '*.npz' -o -name '*.json' -o -name 'input_image.png' \) \
  -print0 | sort -z | xargs -0 sha256sum > /tmp/second_attention_sources_after.sha256

cmp /tmp/second_attention_sources_before.sha256 /tmp/second_attention_sources_after.sha256
```

Expected: seven 3840×2160 PNGs, `README.md` exists, and `cmp` exits 0. Open the overview and at least two independent figures for visual inspection.

- [ ] **Step 7: Run the project audit regression suite**

Run: `python3 -m unittest discover -s tests -v`

Expected: all tracked audit tests plus the user's existing `test_run_sh.py` pass.

- [ ] **Step 8: Commit the tool and tests without committing experiment outputs**

```bash
git add tools/render_second_attention_ppt.py tests/test_second_attention_ppt.py
git commit -m "feat: generate SECOND attention PPT figures"
```

Generated PNGs and their README remain in `audit/ppt_figures/second_attention_v1` for direct use in the PPT; existing untracked experiment runs remain untouched.

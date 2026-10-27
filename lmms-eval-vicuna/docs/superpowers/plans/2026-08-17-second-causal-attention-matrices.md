# SECOND Causal Attention Matrices Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture Layer 0 causal self-attention for all 32 heads from the final SECOND stage for six selected POPE samples and render six comparable 4K masked matrix figures.

**Architecture:** Add a standalone, opt-in recorder that validates and area-pools one generated-step attention tensor before atomically writing rank-scoped NPZ/JSON artifacts. Thread that recorder through the existing LLaVA adapter and recursive generator only when explicit model arguments enable it. A separate renderer loads all six captures, computes one shared scale, and produces 4-by-8 head grids; a dedicated D-only runner filters POPE through a six-QID manifest.

**Tech Stack:** Python 3.10, PyTorch, NumPy, Matplotlib, Pillow, Hugging Face Accelerate, Bash, `unittest`.

## Global Constraints

- Selected QIDs, in display order: `887, 1859, 2254, 2284, 2679, 2920`.
- Run only condition D (`Full SECOND`), generated token `0`, final stage `1`, LLM layer `0`, all 32 heads.
- Render exactly six `3840 x 2160` PNGs with an explicit black causal upper triangle and red expanded-image-token rectangle.
- Black means the LLM causal mask; it never means the SECOND visual-token selection mask.
- Store at most `384 x 384` values per head as float16; never persist the full prompt matrix.
- Use one shared `99.9`-percentile upper color limit and one transform across all six samples and heads.
- Do not modify `run.sh`, existing `audit/runs`, existing `audit/ppt_figures`, or prior logs.
- New output root: `audit/causal_attention/second_causal_attention_v1`.
- Machine 31 endpoint: `yxq@10.10.65.31:10003`; project path: `/home/yxq/Downloads/SECOND-main/lmms-eval-vicuna`; default GPUs: `0,1`.

---

## File Structure

- Create `lmms_eval/causal_attention.py`: capture configuration, validation, image-token interval calculation, bounded pooling, and atomic NPZ/JSON recording.
- Modify `lmms_eval/models/llava.py`: accept opt-in capture arguments, construct the recorder, and pass it into recursive generation.
- Modify `LLaVA-NeXT/llava/model/language_model/llava_recursion.py`: request attentions only at the configured stage/token and hand the selected layer tensor to the recorder.
- Create `tools/render_second_causal_attention.py`: strict six-artifact loader, shared-scale computation, 4-by-8 rendering, and README generation.
- Create `audit/manifests/pope_pop_causal_attention_6.json`: exact six-QID task filter.
- Create `audit/manifests/pope_pop_causal_attention_smoke_887.json`: one-QID remote smoke filter.
- Create `scripts/run_second_causal_attention.sh`: D-only capture-and-render runner with output safety checks.
- Create `tests/test_causal_attention.py`: capture-core tests.
- Create `tests/test_causal_attention_plumbing.py`: adapter and recursion opt-in plumbing tests.
- Create `tests/test_render_second_causal_attention.py`: strict loader, scale, mask, output, and README tests.
- Create `tests/test_second_causal_attention_script.py`: manifests and runner contract tests.

---

### Task 1: Memory-Bounded Causal-Attention Recorder

**Files:**
- Create: `lmms_eval/causal_attention.py`
- Create: `tests/test_causal_attention.py`

**Interfaces:**
- Produces: `CaptureConfig`, `validate_capture_args(...)`, `image_token_interval(input_ids, sequence_length, image_token_index)`, `pool_attention(attention, size)`, and `CausalAttentionRecorder.write(sample_id, attention, input_ids, image_token_index)`.
- Consumes: PyTorch tensors from the recursive generator; no dependency on `SecondAuditRecorder` internals.

- [ ] **Step 1: Write failing configuration and interval tests**

```python
from pathlib import Path
import tempfile
import unittest
import torch

from lmms_eval.causal_attention import (
    CausalAttentionRecorder,
    image_token_interval,
    validate_capture_args,
)


class CausalAttentionCoreTests(unittest.TestCase):
    def test_disabled_capture_returns_none(self):
        self.assertIsNone(validate_capture_args(None, 0, 1, 0, 384))

    def test_enabled_capture_validates_non_negative_indices(self):
        with self.assertRaisesRegex(ValueError, "token_idx"):
            validate_capture_args("out", -1, 1, 0, 384)

    def test_image_token_interval_uses_expanded_sequence_length(self):
        input_ids = torch.tensor([[1, -200, 9, 10]])
        self.assertEqual(image_token_interval(input_ids, 12, -200), (1, 10))

    def test_image_token_interval_requires_one_placeholder(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            image_token_interval(torch.tensor([[1, 2, 3]]), 8, -200)
```

- [ ] **Step 2: Run tests and verify the module is missing**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_causal_attention.CausalAttentionCoreTests -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'lmms_eval.causal_attention'`.

- [ ] **Step 3: Implement configuration and interval validation**

```python
from dataclasses import dataclass
from pathlib import Path

import torch


CAPTURE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CaptureConfig:
    output_dir: Path
    token_idx: int
    stage: int
    layer: int
    pool_size: int


def validate_capture_args(output_dir, token_idx, stage, layer, pool_size):
    if output_dir is None or not str(output_dir).strip():
        return None
    values = {
        "token_idx": int(token_idx),
        "stage": int(stage),
        "layer": int(layer),
        "pool_size": int(pool_size),
    }
    for name in ("token_idx", "stage", "layer"):
        if values[name] < 0:
            raise ValueError(f"causal_attention_{name} must be non-negative")
    if values["pool_size"] <= 0:
        raise ValueError("causal_attention_pool_size must be positive")
    return CaptureConfig(Path(output_dir), **values)


def image_token_interval(input_ids, sequence_length, image_token_index):
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape [1, tokens]")
    positions = (input_ids[0] == image_token_index).nonzero(as_tuple=False).flatten()
    if positions.numel() != 1:
        raise ValueError("exactly one image token placeholder is required")
    start = int(positions.item())
    image_count = int(sequence_length) - (int(input_ids.shape[1]) - 1)
    end = start + image_count
    if not 0 <= start < end <= int(sequence_length):
        raise ValueError("expanded image token interval is invalid")
    return start, end
```

- [ ] **Step 4: Run the focused tests**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_causal_attention.CausalAttentionCoreTests -v`

Expected: PASS.

- [ ] **Step 5: Add failing pooling and atomic recorder tests**

```python
import json
import numpy as np

from lmms_eval.causal_attention import pool_attention


def test_pool_attention_bounds_shape_and_dtype(self):
    source = torch.arange(2 * 8 * 8, dtype=torch.float32).reshape(1, 2, 8, 8)
    pooled = pool_attention(source, 4)
    self.assertEqual(tuple(pooled.shape), (2, 4, 4))
    self.assertEqual(pooled.dtype, torch.float16)

def test_recorder_writes_rank_scoped_atomic_artifacts(self):
    with tempfile.TemporaryDirectory() as directory:
        recorder = CausalAttentionRecorder(
            directory, rank=1, token_idx=0, stage=1, layer=0, pool_size=4
        )
        attention = torch.tril(torch.ones((1, 2, 8, 8), dtype=torch.float32))
        path = recorder.write("887", attention, torch.tensor([[1, -200, 9]]), -200)
        self.assertEqual(path, Path(directory) / "rank_1/887/layer_00_stage_1_token_000.npz")
        with np.load(path) as arrays:
            self.assertEqual(arrays["attention"].shape, (2, 4, 4))
            self.assertEqual(arrays["attention"].dtype, np.float16)
        metadata = json.loads(path.with_suffix(".json").read_text())
        self.assertEqual(metadata["original_shape"], [2, 8, 8])
        self.assertEqual(metadata["image_token_interval"], [1, 7])
        self.assertFalse(path.with_name(path.name + ".tmp").exists())

def test_recorder_rejects_non_square_or_non_finite_attention(self):
    with tempfile.TemporaryDirectory() as directory:
        recorder = CausalAttentionRecorder(directory, 0, 0, 1, 0, 4)
        with self.assertRaisesRegex(ValueError, "square"):
            recorder.write("887", torch.ones(1, 2, 4, 3), torch.tensor([[1, -200]]), -200)
        bad = torch.ones(1, 2, 4, 4)
        bad[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            recorder.write("887", bad, torch.tensor([[1, -200]]), -200)
```

- [ ] **Step 6: Run the new tests and verify missing behavior**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_causal_attention -v`

Expected: FAIL because `pool_attention` and/or recorder writing are not implemented.

- [ ] **Step 7: Implement pooling and the recorder**

Implementation requirements:

```python
import json
import re
from pathlib import Path

import numpy as np
import torch.nn.functional as F


def pool_attention(attention, size):
    if attention.ndim != 4 or attention.shape[0] != 1:
        raise ValueError("attention must have shape [1, heads, query, key]")
    if attention.shape[-2] != attention.shape[-1]:
        raise ValueError("attention matrix must be square")
    if not bool(torch.isfinite(attention).all()):
        raise ValueError("attention must contain only finite values")
    if bool((attention < 0).any()):
        raise ValueError("attention must be non-negative")
    target = min(int(size), int(attention.shape[-1]))
    return F.adaptive_avg_pool2d(attention.float(), (target, target))[0].half().cpu()
```

`CausalAttentionRecorder` must:

- call `validate_capture_args` and expose `should_capture(token_idx, stage)`;
- create `output_dir/rank_<rank>/<ASCII-digit sample_id>` with `exist_ok=False`;
- validate exactly 32 heads at production capture time while allowing `expected_heads` injection in unit tests;
- write `attention` through a file handle to `*.npz.tmp`, then `Path.replace`;
- write JSON through `*.json.tmp`, then `Path.replace`;
- include schema, rank, QID, token, stage, layer, original and pooled shapes, half-open image interval, min/max, and pooled valid-value percentiles `50, 90, 99, 99.9`;
- reject unsafe sample IDs and pre-existing sample directories.

- [ ] **Step 8: Run capture tests and full existing tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_causal_attention -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

Expected: all tests PASS.

- [ ] **Step 9: Commit the recorder**

```bash
git add lmms_eval/causal_attention.py tests/test_causal_attention.py
git commit -m "feat: record bounded causal attention matrices"
```

---

### Task 2: Opt-In LLaVA and Recursive-Generation Plumbing

**Files:**
- Modify: `lmms_eval/models/llava.py`
- Modify: `LLaVA-NeXT/llava/model/language_model/llava_recursion.py`
- Create: `tests/test_causal_attention_plumbing.py`

**Interfaces:**
- Consumes: `CausalAttentionRecorder.should_capture(...)` and `.write(...)` from Task 1.
- Produces: new model arguments `causal_attention_output_dir`, `causal_attention_token_idx`, `causal_attention_stage`, `causal_attention_layer`, and `causal_attention_pool_size`; recursive keyword `causal_attention_recorder`.

- [ ] **Step 1: Write failing source-contract tests**

```python
import ast
from pathlib import Path
import unittest

from tests.test_llava_audit_plumbing import function_node

REPO_ROOT = Path(__file__).parents[1]


class CausalAttentionPlumbingTests(unittest.TestCase):
    def test_llava_adapter_accepts_and_passes_capture_configuration(self):
        path = REPO_ROOT / "lmms_eval/models/llava.py"
        init = function_node(path, "Llava", "__init__")
        names = {argument.arg for argument in init.args.args}
        self.assertTrue({
            "causal_attention_output_dir", "causal_attention_token_idx",
            "causal_attention_stage", "causal_attention_layer",
            "causal_attention_pool_size",
        }.issubset(names))
        source = ast.unparse(function_node(path, "Llava", "generate_until"))
        self.assertIn("causal_attention_recorder=self.causal_attention_recorder", source)

    def test_recursive_generation_requests_and_records_only_target_capture(self):
        path = REPO_ROOT / "LLaVA-NeXT/llava/model/language_model/llava_recursion.py"
        source = ast.unparse(function_node(path, "LlavaLlamaForRecursion", "generate_recursive"))
        self.assertIn("causal_attention_recorder.should_capture(token_idx, stage)", source)
        self.assertIn("output_attentions=capture_attention or not last_stage", source)
        self.assertIn("causal_attention_recorder.write", source)
```

- [ ] **Step 2: Run and verify failure**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_causal_attention_plumbing -v`

Expected: FAIL because the arguments and recorder plumbing do not exist.

- [ ] **Step 3: Add adapter arguments and construct one recorder per rank**

In `lmms_eval/models/llava.py`:

```python
from lmms_eval.causal_attention import CausalAttentionRecorder, validate_capture_args

# __init__ arguments
causal_attention_output_dir=None,
causal_attention_token_idx=0,
causal_attention_stage=1,
causal_attention_layer=0,
causal_attention_pool_size=384,

capture_config = validate_capture_args(
    causal_attention_output_dir,
    causal_attention_token_idx,
    causal_attention_stage,
    causal_attention_layer,
    causal_attention_pool_size,
)
self.causal_attention_recorder = (
    None
    if capture_config is None
    else CausalAttentionRecorder.from_config(
        capture_config, rank=self.accelerator.process_index, expected_heads=32
    )
)
```

Pass `causal_attention_recorder=self.causal_attention_recorder` into `generate_recursive` without changing the standard branch.

- [ ] **Step 4: Add target-only recursive capture**

In `generate_recursive`:

```python
causal_attention_recorder = kwargs.pop("causal_attention_recorder", None)

capture_attention = (
    causal_attention_recorder is not None
    and causal_attention_recorder.should_capture(token_idx, stage)
)

cont = self.generate(
    # existing arguments unchanged
    output_attentions=(capture_attention or not last_stage),
)

if capture_attention:
    attentions = cont.get("attentions")
    if not attentions or not attentions[0]:
        raise ValueError("generation did not return prompt attentions")
    layer = causal_attention_recorder.layer
    if layer >= len(attentions[0]):
        raise ValueError(f"causal attention layer {layer} is out of range")
    causal_attention_recorder.write(
        audit_sample_id,
        attentions[0][layer],
        input_ids,
        IMAGE_TOKEN_INDEX,
    )
```

Capture before the final-stage `break`. Ensure `audit_sample_id` is required when capture is enabled. Do not call `get_heatmap` on the final stage.

- [ ] **Step 5: Run focused and regression tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  tests.test_causal_attention_plumbing \
  tests.test_llava_audit_plumbing -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit the plumbing**

```bash
git add \
  lmms_eval/models/llava.py \
  LLaVA-NeXT/llava/model/language_model/llava_recursion.py \
  tests/test_causal_attention_plumbing.py
git commit -m "feat: capture final-stage LLaVA causal attention"
```

---

### Task 3: Strict Six-Sample 4K Renderer

**Files:**
- Create: `tools/render_second_causal_attention.py`
- Create: `tests/test_render_second_causal_attention.py`

**Interfaces:**
- Consumes: rank-scoped NPZ/JSON files written by `CausalAttentionRecorder`.
- Produces: `load_capture(raw_root, qid)`, `shared_scale(captures, percentile=99.9)`, `causal_mask(size)`, `render_capture(...)`, and `generate_figures(..., allow_subset=False)`.

- [ ] **Step 1: Write failing loader, scale, and mask tests**

```python
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.render_second_causal_attention import (
    DEFAULT_QUESTION_IDS,
    causal_mask,
    generate_figures,
    load_capture,
    shared_scale,
)


def write_capture(root, qid, rank=0, value=1.0):
    directory = Path(root) / f"rank_{rank}" / str(qid)
    directory.mkdir(parents=True)
    stem = directory / "layer_00_stage_1_token_000"
    attention = np.tril(np.full((32, 8, 8), value, dtype=np.float16))
    np.savez_compressed(stem.with_suffix(".npz"), attention=attention)
    stem.with_suffix(".json").write_text(json.dumps({
        "schema_version": 1, "question_id": str(qid), "rank": rank,
        "token_idx": 0, "stage": 1, "layer": 0,
        "original_shape": [32, 16, 16], "pooled_shape": [32, 8, 8],
        "image_token_interval": [2, 14],
    }))


class CausalAttentionRendererTests(unittest.TestCase):
    def test_causal_mask_is_exact_upper_triangle(self):
        np.testing.assert_array_equal(
            causal_mask(3),
            [[False, True, True], [False, False, True], [False, False, False]],
        )

    def test_loader_rejects_duplicate_rank_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            write_capture(directory, "887", rank=0)
            write_capture(directory, "887", rank=1)
            with self.assertRaisesRegex(ValueError, "exactly one"):
                load_capture(Path(directory), "887")

    def test_shared_scale_uses_all_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, qid in enumerate(DEFAULT_QUESTION_IDS, start=1):
                write_capture(directory, qid, value=float(index))
            captures = [load_capture(Path(directory), qid) for qid in DEFAULT_QUESTION_IDS]
            self.assertGreater(shared_scale(captures), 5.0)
```

- [ ] **Step 2: Run and verify module failure**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_render_second_causal_attention -v`

Expected: FAIL because the renderer module does not exist.

- [ ] **Step 3: Implement strict loading and one shared scale**

The loader must require:

- ASCII-digit QIDs;
- exactly one `rank_*/<qid>` match;
- matching NPZ and JSON stems;
- exactly one `attention` array with shape `[32,N,N]` and `N <= 384`;
- finite, non-negative float values;
- metadata matching QID, layer `0`, stage `1`, token `0`, pooled shape, and a valid image interval.

Implement:

```python
def causal_mask(size):
    return np.triu(np.ones((size, size), dtype=bool), k=1)


def shared_scale(captures, percentile=99.9):
    valid = []
    for capture in captures:
        keep = ~causal_mask(capture.attention.shape[-1])
        valid.append(capture.attention[:, keep].astype(np.float64, copy=False))
    vmax = float(np.percentile(np.concatenate([row.ravel() for row in valid]), percentile))
    if not np.isfinite(vmax) or vmax <= 0:
        raise ValueError("shared attention scale must be positive and finite")
    return vmax
```

- [ ] **Step 4: Add failing 4K generation and README tests**

```python
from PIL import Image

def test_generate_six_4k_figures_and_readme(self):
    with tempfile.TemporaryDirectory() as directory:
        raw = Path(directory) / "raw"
        output = Path(directory) / "output"
        for qid in DEFAULT_QUESTION_IDS:
            write_capture(raw, qid)
        generated = generate_figures(raw, output)
        pngs = [path for path in generated if path.suffix == ".png"]
        self.assertEqual(len(pngs), 6)
        for path in pngs:
            with Image.open(path) as image:
                self.assertEqual(image.size, (3840, 2160))
        readme = (output / "README.md").read_text()
        self.assertIn("causal mask", readme)
        self.assertIn("not the SECOND visual-token selection mask", readme)

def test_generate_requires_exact_default_qids(self):
    with tempfile.TemporaryDirectory() as directory:
        with self.assertRaisesRegex(ValueError, "exactly six"):
            generate_figures(Path(directory), Path(directory) / "out", ("887",))

def test_explicit_smoke_mode_allows_only_a_default_subset(self):
    with tempfile.TemporaryDirectory() as directory:
        raw = Path(directory) / "raw"
        write_capture(raw, "887")
        generated = generate_figures(
            raw, Path(directory) / "out", ("887",), allow_subset=True
        )
        self.assertEqual([path.suffix for path in generated], [".png", ".md"])
        with self.assertRaisesRegex(ValueError, "default QID"):
            generate_figures(
                raw, Path(directory) / "bad", ("9999",), allow_subset=True
            )
```

- [ ] **Step 5: Run and verify rendering tests fail**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_render_second_causal_attention -v`

Expected: FAIL because figure and README generation are incomplete.

- [ ] **Step 6: Implement the renderer and CLI**

Render each sample with:

```python
masked = np.ma.array(head_attention, mask=causal_mask(pool_size))
axis.imshow(
    masked,
    cmap=cmap_with_bad_black,
    norm=matplotlib.colors.PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax),
    extent=(0, sequence_length, sequence_length, 0),
    interpolation="nearest",
)
axis.add_patch(matplotlib.patches.Rectangle(
    (image_start, image_start),
    image_end - image_start,
    image_end - image_start,
    fill=False,
    edgecolor="red",
    linewidth=0.8,
))
```

Use a `16 x 9` figure at `240 dpi`, 4 rows by 8 columns, head titles `Head 0` through `Head 31`, shared norm, compact per-head colorbars, axes labeled in English, and a figure title that states `Upper triangle (causal mask) shown in black`. Save through a common `_save_figure` helper that closes the figure in `finally`.

The CLI requires `--raw-root` and `--output`, accepts a comma-separated `--question-ids` defaulting to the exact six QIDs, prints every generated path, rejects symlink output targets, and refuses output paths inside `raw_root`. Add a default-false `--allow-subset` flag: without it, the QID tuple must equal all six defaults in order; with it, one or more unique QIDs may be selected only from the default set. The README states `smoke subset` when this flag is active.

- [ ] **Step 7: Run renderer and regression tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_render_second_causal_attention -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

Expected: all tests PASS.

- [ ] **Step 8: Commit the renderer**

```bash
git add tools/render_second_causal_attention.py tests/test_render_second_causal_attention.py
git commit -m "feat: render masked causal attention head grids"
```

---

### Task 4: Six-QID Manifests and Safe D-Only Runner

**Files:**
- Create: `audit/manifests/pope_pop_causal_attention_6.json`
- Create: `audit/manifests/pope_pop_causal_attention_smoke_887.json`
- Create: `scripts/run_second_causal_attention.sh`
- Create: `tests/test_second_causal_attention_script.py`

**Interfaces:**
- Consumes: model arguments from Task 2 and renderer CLI from Task 3.
- Produces: one command that captures D and renders the final six figures; accepts `MANIFEST` and `OUTPUT_ROOT` positional overrides for smoke testing.

- [ ] **Step 1: Write failing runner-contract tests**

```python
import json
from pathlib import Path
import subprocess
import unittest

REPO_ROOT = Path(__file__).parents[1]
SCRIPT = REPO_ROOT / "scripts/run_second_causal_attention.sh"
MANIFEST = REPO_ROOT / "audit/manifests/pope_pop_causal_attention_6.json"


class SecondCausalAttentionScriptTests(unittest.TestCase):
    def test_manifest_has_exact_ordered_qids(self):
        payload = json.loads(MANIFEST.read_text())
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(
            [str(row["question_id"]) for row in payload["samples"]],
            ["887", "1859", "2254", "2284", "2679", "2920"],
        )

    def test_runner_is_d_only_and_passes_capture_contract(self):
        text = SCRIPT.read_text()
        self.assertIn("set -euo pipefail", text)
        self.assertIn("audit_condition=D", text)
        self.assertNotIn("audit_condition=A", text)
        self.assertNotIn("audit_condition=B", text)
        self.assertNotIn("audit_condition=C", text)
        for value in (
            "causal_attention_token_idx=0", "causal_attention_stage=1",
            "causal_attention_layer=0", "causal_attention_pool_size=384",
            "SECOND_AUDIT_MANIFEST", "NCCL_P2P_DISABLE", "NCCL_IB_DISABLE",
        ):
            self.assertIn(value, text)

    def test_no_argument_help_does_not_create_output(self):
        completed = subprocess.run(
            ["bash", str(SCRIPT), "--help"], cwd=REPO_ROOT,
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertIn("Usage:", completed.stdout)
```

- [ ] **Step 2: Run and verify missing-file failure**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_second_causal_attention_script -v`

Expected: FAIL because manifests and runner do not exist.

- [ ] **Step 3: Create exact manifests**

Write `audit/manifests/pope_pop_causal_attention_6.json` exactly as:

```json
{
  "schema_version": 1,
  "seed": 42,
  "purpose": "SECOND causal-attention matrix figure set",
  "samples": [
    {"doc_id": 27, "question_id": "887", "image_id": "3886", "question": "Is there a car in the image?", "ground_truth": "yes"},
    {"doc_id": 52, "question_id": "1859", "image_id": "4858", "question": "Is there a laptop in the image?", "ground_truth": "yes"},
    {"doc_id": 62, "question_id": "2254", "image_id": "5253", "question": "Is there a dining table in the image?", "ground_truth": "no"},
    {"doc_id": 64, "question_id": "2284", "image_id": "5283", "question": "Is there a car in the image?", "ground_truth": "no"},
    {"doc_id": 70, "question_id": "2679", "image_id": "5678", "question": "Is there a toilet in the image?", "ground_truth": "yes"},
    {"doc_id": 75, "question_id": "2920", "image_id": "5919", "question": "Is there a dining table in the image?", "ground_truth": "no"}
  ]
}
```

Write `audit/manifests/pope_pop_causal_attention_smoke_887.json` with the same top-level fields, purpose `SECOND causal-attention QID 887 smoke test`, and only the first sample object. Both must pass `pope_process_audit_docs` validation.

- [ ] **Step 4: Implement the safe runner**

Runner contract:

```bash
usage() {
    echo "Usage: bash scripts/run_second_causal_attention.sh [MANIFEST] [OUTPUT_ROOT]"
}

manifest=${1:-"${repo_dir}/audit/manifests/pope_pop_causal_attention_6.json"}
output_root=${2:-"${repo_dir}/audit/causal_attention/second_causal_attention_v1"}
gpu_ids=${CUDA_VISIBLE_DEVICES:-0,1}
main_port=${SECOND_CAUSAL_PORT:-29891}
num_processes=${SECOND_CAUSAL_PROCESSES:-2}
python_bin=${PYTHON_BIN:-python3}
```

Before mutation, require a readable manifest and reject commas in paths. Refuse a non-empty output root, then create only:

```text
$output_root/results
$output_root/audit_artifacts
$output_root/raw
$output_root/figures
```

Launch exactly two processes by default and pass:

```text
pretrained=liuhaotian/llava-v1.6-vicuna-7b
audit_condition=D
audit_output_dir=$output_root/audit_artifacts
audit_token_idx=0
causal_attention_output_dir=$output_root/raw
causal_attention_token_idx=0
causal_attention_stage=1
causal_attention_layer=0
causal_attention_pool_size=384
```

Use `--num_processes "$num_processes"` and the existing recursion/grid/threshold/stage/alpha values. Read the ordered QIDs from the manifest after validating that every ID belongs to the six-QID default set. After Accelerate succeeds, call `tools/render_second_causal_attention.py --raw-root "$output_root/raw" --output "$output_root/figures" --question-ids "$comma_separated_qids"`. Add `--allow-subset` only when the manifest contains fewer than six QIDs; do not pretend a smoke output is the final six-figure set.

- [ ] **Step 5: Run shell and test verification**

Run:

```bash
bash -n scripts/run_second_causal_attention.sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_second_causal_attention_script -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit manifests and runner**

```bash
git add \
  audit/manifests/pope_pop_causal_attention_6.json \
  audit/manifests/pope_pop_causal_attention_smoke_887.json \
  scripts/run_second_causal_attention.sh \
  tests/test_second_causal_attention_script.py
git commit -m "feat: run six-sample causal attention capture"
```

---

### Task 5: Local Verification, Machine-31 Smoke Test, and Six-Sample Run

**Files:**
- Verify only: all Task 1-4 source and test files.
- Generate (untracked experiment output): `audit/causal_attention/second_causal_attention_v1/**`.

**Interfaces:**
- Consumes: the complete runner and machine-31 SSH access.
- Produces: six final 4K PNGs, six NPZ/JSON pairs, README, and verification evidence.

- [ ] **Step 1: Run local compilation, shell, focused, and full tests**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/second-causal-pycache python3 -m compileall -q \
  lmms_eval/causal_attention.py \
  lmms_eval/models/llava.py \
  LLaVA-NeXT/llava/model/language_model/llava_recursion.py \
  tools/render_second_causal_attention.py \
  tests/test_causal_attention.py \
  tests/test_causal_attention_plumbing.py \
  tests/test_render_second_causal_attention.py \
  tests/test_second_causal_attention_script.py
bash -n scripts/run_second_causal_attention.sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

Expected: exit `0`; all tests PASS.

- [ ] **Step 2: Run a local synthetic render and inspect dimensions**

Use the renderer test fixture to generate six synthetic captures beneath a `mktemp -d` directory, invoke the CLI, and verify all six PNGs are exactly `3840 x 2160`. Open one synthetic QID 887 image with the image viewer and confirm the head order, black upper triangle, and red rectangle.

- [ ] **Step 3: Hash existing D audit sources before remote work**

Run:

```bash
find audit/runs/pope_pop_audit_v1/D/artifacts/D \
  -type f -print0 | sort -z | xargs -0 sha256sum \
  > /tmp/second_causal_sources_before.sha256
```

Expected: a non-empty manifest; record its line count.

- [ ] **Step 4: Prepare an isolated user environment on machine 31**

Connect with:

```bash
ssh -i /home/yxq/.ssh/id_ed25519_second_31 -p 10003 yxq@10.10.65.31
```

Create `/home/yxq/.venvs/second_causal_attention` with `python3 -m venv --system-site-packages`. Install only missing project imports into that environment, beginning with `einops`, and verify:

```bash
PYTHONPATH="$PWD/LLaVA-NeXT:$PWD" \
  /home/yxq/.venvs/second_causal_attention/bin/python -c \
  'import torch, lmms_eval, llava; print(torch.cuda.is_available(), torch.cuda.device_count())'
```

Expected: `True 6`. Do not modify `/root/miniconda3` or `/home/yxq/.venvs/second_hat`.

- [ ] **Step 5: Run QID 887 smoke test on one GPU**

Use a fresh smoke output root and:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHON_BIN=/home/yxq/.venvs/second_causal_attention/bin/python \
SECOND_CAUSAL_PROCESSES=1 \
SECOND_CAUSAL_PORT=29891 \
bash scripts/run_second_causal_attention.sh \
  audit/manifests/pope_pop_causal_attention_smoke_887.json \
  audit/causal_attention/smoke_887
```

Expected: one NPZ/JSON pair and one 3840-by-2160 PNG; metadata says 32 heads, layer 0, stage 1, token 0; GPU run finishes without OOM.

- [ ] **Step 6: Inspect smoke artifact before the full run**

Copy or access the smoke PNG, open it at original resolution, and verify:

- all heads 0-31 appear once in row-major order;
- upper triangle is exactly black;
- lower triangle contains finite heatmap values;
- red image-token rectangle is within sequence bounds;
- title explicitly says black is the causal mask.

If any condition fails, stop and fix before running the remaining samples.

- [ ] **Step 7: Run all six samples on GPUs 0 and 1**

Run with a new empty final output root:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PYTHON_BIN=/home/yxq/.venvs/second_causal_attention/bin/python \
SECOND_CAUSAL_PROCESSES=2 \
SECOND_CAUSAL_PORT=29892 \
bash scripts/run_second_causal_attention.sh \
  audit/manifests/pope_pop_causal_attention_6.json \
  audit/causal_attention/second_causal_attention_v1
```

Monitor `nvidia-smi`, process status, and output count without modifying the running job. Expected: exactly six unique QIDs across rank directories, six figures, and README.

- [ ] **Step 8: Verify final artifact contracts**

Run a Python verification that asserts:

- figure filenames match all six QIDs exactly;
- every PNG is `3840 x 2160`;
- every NPZ contains finite, non-negative float16 `[32,N,N]`, `N <= 384`;
- every JSON matches its path and records valid original/pool shapes and image interval;
- README contains all QIDs, the shared `99.9`-percentile scale, rendering command, and causal-vs-selection warning.

Expected: exit `0` with a concise six-sample summary.

- [ ] **Step 9: Re-hash protected sources and visually inspect real outputs**

Run the same D-artifact hashing command into `/tmp/second_causal_sources_after.sha256`, then:

```bash
cmp /tmp/second_causal_sources_before.sha256 /tmp/second_causal_sources_after.sha256
```

Expected: exit `0`. Open real QID 887 and QID 2254 figures at original resolution and confirm no clipped titles, overlapping panels, unreadable axes, or incorrect masks.

- [ ] **Step 10: Run final regression verification and commit any review fixes**

After any visual-review correction, rerun compilation, shell syntax, focused tests, and the full test suite. Commit only tracked source/test/docs changes; leave generated experiment outputs untracked:

```bash
git status --short
git add \
  lmms_eval/causal_attention.py \
  lmms_eval/models/llava.py \
  LLaVA-NeXT/llava/model/language_model/llava_recursion.py \
  tools/render_second_causal_attention.py \
  audit/manifests/pope_pop_causal_attention_6.json \
  audit/manifests/pope_pop_causal_attention_smoke_887.json \
  scripts/run_second_causal_attention.sh \
  tests/test_causal_attention.py \
  tests/test_causal_attention_plumbing.py \
  tests/test_render_second_causal_attention.py \
  tests/test_second_causal_attention_script.py
git commit -m "fix: address causal attention figure review"
```

Expected: clean status for tracked implementation files while preserving the user's pre-existing modified/untracked files and the new untracked experiment outputs.

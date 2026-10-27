# SECOND Causal Attention Matrix Figure Design

**Date:** 2026-08-17
**Status:** Approved for implementation planning
**Project:** `/home/yxq/Downloads/SECOND-main/lmms-eval-vicuna`

## 1. Goal

Generate six PPT-ready figures that match the structure of the supplied causal-attention example: one figure per selected POPE sample, a 4-by-8 grid containing all 32 LLM attention heads, the causal upper triangle shown in black, and the expanded image-token self-attention region marked with a red rectangle.

The selected question IDs are, in display order:

```text
887, 1859, 2254, 2284, 2679, 2920
```

The figures are descriptive views of LLM self-attention. They do not replace the existing 48-by-48 SECOND spatial attention and token-selection figures.

## 2. Semantic Boundary

The black upper triangle represents the autoregressive LLM causal mask: query position `q` cannot attend to a future key position `k > q`.

It does **not** represent the SECOND visual-token selection mask. SECOND selection remains documented by the existing `applied_mask` artifacts. Figure titles and the companion README must state this distinction explicitly so the two masks cannot be confused in a presentation.

## 3. Experimental Scope

Run only the following configuration:

- Dataset task: `pope_pop_audit`, filtered by a six-sample manifest.
- Audit condition: `D` (`Full SECOND`).
- Model: `liuhaotian/llava-v1.6-vicuna-7b`.
- Recursive stages: `-2, -1, 0, 1`.
- Captured recursive stage: final stage `1`.
- Captured generated-token index: `0` (the first answer token).
- Captured LLM layer: `0`.
- Captured heads: all 32 heads.
- GPUs on machine 31: GPU `0,1`, two Accelerate processes, three samples per process.
- Random seed: `42`.

No full 3,000-sample POPE run and no A/B/C rerun are required.

## 4. Capture Design

### 4.1 Opt-in model arguments

Add an opt-in causal-attention capture configuration to the LLaVA model adapter. The normal evaluation path must remain unchanged when no capture output directory is supplied.

The configuration exposes:

- output directory;
- generated-token index, default `0`;
- recursive stage, default `1`;
- LLM layer index, default `0`;
- downsample size, default `384`.

Arguments are validated at model construction. The output directory is mandatory when capture is enabled; token, stage, layer, and downsample size must be non-negative and within runtime bounds.

### 4.2 Capture point

The final-stage one-token `generate` call currently disables attention output. Enable `output_attentions` only when the configured token and stage are reached. Read:

```text
generation_output.attentions[0][layer]
```

Expected runtime shape:

```text
[batch=1, heads=32, query_length=S, key_length=S]
```

Reject missing attentions, a non-square matrix, batch sizes other than one, non-finite values, negative values, or an out-of-range layer index. A failed capture must stop that sample instead of silently producing a misleading figure.

### 4.3 Image-token interval

Require exactly one `IMAGE_TOKEN_INDEX` placeholder in the unexpanded input IDs.

Let:

```text
image_start = position of IMAGE_TOKEN_INDEX
text_token_count = unexpanded_input_length - 1
image_token_count = S - text_token_count
image_end = image_start + image_token_count
```

Validate `0 <= image_start < image_end <= S`. Store `[image_start, image_end)` in metadata and use it for the red image-token rectangle.

### 4.4 Memory-bounded representation

Do not persist the full `[32,S,S]` tensor because a roughly 3,000-token prompt can require hundreds of megabytes per sample even in float16.

For each head:

1. keep the original attention only long enough to validate and summarize it;
2. downsample to at most `384 x 384` with area/mean pooling;
3. store the downsampled matrix as float16;
4. store original shape, pooled shape, image-token interval, layer, stage, token index, head count, minimum, maximum, and selected percentiles in JSON;
5. release capture tensors before continuing generation.

The pooled causal mask is applied explicitly during rendering, so the upper triangle remains exactly black instead of becoming gray through interpolation.

## 5. Output Contract

Write new outputs beneath a dedicated directory and never inside the original A/B/C/D audit artifact trees:

```text
audit/causal_attention/second_causal_attention_v1/
├── raw/
│   └── rank_<n>/<question_id>/
│       ├── layer_00_stage_1_token_000.npz
│       └── layer_00_stage_1_token_000.json
├── figures/
│   ├── q887_layer0_stage1_causal_attention_all_heads.png
│   ├── q1859_layer0_stage1_causal_attention_all_heads.png
│   ├── q2254_layer0_stage1_causal_attention_all_heads.png
│   ├── q2284_layer0_stage1_causal_attention_all_heads.png
│   ├── q2679_layer0_stage1_causal_attention_all_heads.png
│   └── q2920_layer0_stage1_causal_attention_all_heads.png
└── README.md
```

Output directories must be absent or empty before a run. Writers use rank-specific paths and atomic file replacement to avoid distributed-process collisions and partial files.

## 6. Figure Design

Each PNG is 3840-by-2160 with:

- a 4-by-8 head grid in row-major order (`Head 0` through `Head 31`);
- x-axis `Key Token Position`;
- y-axis `Query Token Position`;
- explicit black upper triangle (`k > q`);
- a red rectangle for `[image_start, image_end)` on both axes;
- a common colormap and a shared scale across all six samples and all heads;
- a concise title containing QID, layer, stage, token index, sequence length, and the statement that black is the causal mask;
- no question text, ground truth, or prediction inside the PNG.

Use a shared global upper limit derived from the predeclared `99.9` percentile of all finite valid pooled values across the complete six-sample capture. Use the same monotonic contrast transform for every panel. Record the exact scale and transform in the README. This preserves comparison across samples while preventing isolated outliers from flattening the plots.

The README records the selected QIDs, experiment configuration, source raw files, image-token ranges, shared scale, semantic distinction between causal and selection masks, and the rendering command.

## 7. Runner and Machine 31

Provide a dedicated six-sample manifest and runner rather than modifying `run.sh`.

The runner must:

- refuse a non-empty output directory;
- export the six-sample manifest through `SECOND_AUDIT_MANIFEST`;
- preserve `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1`;
- use configurable GPU IDs and main-process port, defaulting to GPUs `0,1`;
- run only D;
- capture only token `0`, stage `1`, layer `0`;
- render only after all six raw captures exist and validate;
- leave existing audit runs and PPT figures untouched.

Machine 31 is reached as `yxq@10.10.65.31:10003` with the dedicated local key `/home/yxq/.ssh/id_ed25519_second_31`. The project is already present at the same absolute path. The machine has six idle NVIDIA L20 GPUs. The user-level Python has CUDA PyTorch and the main evaluation packages but requires an isolated environment or targeted dependency completion before LLaVA imports cleanly. Environment preparation must not modify the inaccessible root Conda environment.

## 8. Testing and Verification

### 8.1 Unit tests

Cover:

- disabled capture leaves generation behavior unchanged;
- argument validation;
- layer/head/shape/finite/non-negative validation;
- exact causal-mask construction;
- image-token interval calculation;
- bounded area pooling and deterministic output;
- atomic rank-specific output paths;
- renderer requires exactly the six selected QIDs;
- shared color scale is computed once and reused;
- PNG dimensions and expected filenames;
- runner refuses non-empty output and contains the required D-only arguments.

### 8.2 Local verification

Run the focused tests, the full existing test suite, Python compilation, shell syntax checks, and a synthetic render before touching machine 31.

### 8.3 Remote smoke test

Run one selected sample first on one GPU. Verify:

- exactly 32 heads;
- square prompt matrix;
- finite, non-negative values;
- black causal upper triangle in the rendered image;
- red rectangle bounds agree with metadata;
- no unexpected GPU out-of-memory event.

Only then run all six samples on GPUs 0 and 1.

### 8.4 Final acceptance

Completion requires:

- six validated 3840-by-2160 PNGs;
- six NPZ/JSON raw-summary pairs;
- a complete README;
- no modification of original audit inputs or previous figures;
- source-file hashes unchanged for the existing six-sample D artifacts;
- visual inspection of at least QID 887 plus one CD-harm sample.

## 9. Known Interpretation Limits

- A single layer does not characterize the whole model; Layer 0 was selected to match the requested example, not because it is inherently the most explanatory layer.
- Attention weights are descriptive and are not causal attribution scores.
- Pooling is necessary for memory-bounded, PPT-readable figures and removes token-level detail; the original sequence length and pooling factor are disclosed.
- The red rectangle denotes the expanded multimodal image-token interval, not the subset retained by SECOND selection.

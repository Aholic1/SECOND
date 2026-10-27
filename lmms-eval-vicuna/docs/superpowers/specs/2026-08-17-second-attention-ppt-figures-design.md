# SECOND Attention Distribution PPT Figure Set Design

## Goal

Create a PPT-ready scientific visualization set from the completed SECOND audit artifacts. The deliverable consists of six independent 16:9 PNG figures and one overview PNG. The figures must support visual comparison across samples and stages without relying on the misleading per-image min-max normalization used by the existing overlays.

## Inputs

- Audit root: `audit/runs/pope_pop_audit_v1/D/artifacts/D`
- Each selected sample provides:
  - `input_image.png`
  - three `token_000_stage_*.npz` files containing `ret_attn` and masks
  - three transition JSON files containing keep-ratio diagnostics
- The figure generator reads these files without modifying them.

## Selected Samples

The six samples cover the main causal outcomes observed in the paired audit:

| Question ID | Role in the audit |
| --- | --- |
| 887 | C-to-D token-selection benefit |
| 2254 | B-to-C contrastive-decoding harm |
| 2284 | B-to-C contrastive-decoding harm |
| 2920 | B-to-C contrastive-decoding harm |
| 2679 | B-to-C contrastive-decoding benefit |
| 1859 | A-to-B recursive-pathway benefit |

The causal descriptions are recorded in the companion index, not printed inside the figures.

## Scientific Attention Scale

For every attention map:

1. Convert `ret_attn` to float and require all values to be finite and non-negative.
2. Normalize the map to unit total mass.
3. Divide by the uniform attention density `1 / N`, where `N` is the number of grid cells.
4. Display the resulting relative density with one shared logarithmic color scale across all samples and stages.

A displayed value of `1×` therefore means uniform attention, values above `1×` indicate above-uniform attention, and values below `1×` indicate below-uniform attention. The shared scale makes color meaning comparable across every exported figure. Values outside the displayed range are clipped only for rendering; the source arrays remain unchanged.

## Independent Figure Layout

Each sample is exported as a white-background 16:9 PNG with five equally structured panels:

1. `输入图像 / Input`
2. `阶段 -2→-1 / Stage -2→-1`
3. `阶段 -1→0 / Stage -1→0`
4. `阶段 0→1 / Stage 0→1`
5. `最终删除区域 / Removed Tokens`

The three attention panels show a semi-transparent relative-density heatmap over the exact saved model-input image. The final panel highlights removed grid cells in red over the image; retained regions remain visually unchanged. A single shared colorbar is labeled `相对均匀注意力 / Relative-to-uniform attention`.

The figure contains no question, ground-truth answer, condition prediction, causal conclusion, or prose interpretation. Only bilingual panel/stage labels and the sample ID are retained.

## Overview Figure

The overview uses one row per sample and four columns:

1. Input image
2. Final-stage attention distribution
3. Final removed-token overlay
4. Compact three-stage keep-ratio plot

It uses the same attention color scale and visual conventions as the independent figures. The overview contains sample IDs and bilingual structural labels only.

## Companion Index

Create a UTF-8 Markdown index outside the images containing:

- output filename and question ID;
- audit role for each sample;
- question, ground truth, and A/B/C/D predictions;
- stage keep ratios;
- source artifact paths;
- a short suggested PPT speaker note.

This keeps the images visually clean while preserving the information needed to write slide captions.

## Output Layout

Write generated files under a new directory:

```text
audit/ppt_figures/second_attention_v1/
├── q887_attention_distribution.png
├── q1859_attention_distribution.png
├── q2254_attention_distribution.png
├── q2284_attention_distribution.png
├── q2679_attention_distribution.png
├── q2920_attention_distribution.png
├── second_attention_overview.png
└── README.md
```

Existing audit artifacts and run outputs are read-only inputs and must not be overwritten.

## Rendering and Robustness

- Use Matplotlib, NumPy, and Pillow already available in the analysis environment.
- Select a locally installed CJK-capable font when available; otherwise fall back to English-only structural labels while retaining all Chinese text in `README.md`.
- Preserve the original image aspect inside every panel.
- Use nearest-neighbor interpolation for binary masks and bilinear interpolation for attention overlays.
- Reject missing transitions, inconsistent grid shapes, NaN/Inf, negative attention, or sample-ID mismatches with an explicit error.
- Make the generator deterministic and safe to rerun into the dedicated output directory.

## Acceptance Criteria

- Exactly six independent PNGs, one overview PNG, and one Markdown index are produced.
- Every independent PNG is 16:9 and at least 3000 pixels wide.
- All attention panels share the same relative-density color scale.
- Each figure contains the input, all three transitions, and the final removed-token view.
- No question, GT, prediction, or causal claim appears inside the PNGs.
- The Markdown index accurately maps all figures to source data and audit outcomes.
- Existing experiment outputs remain byte-for-byte untouched.

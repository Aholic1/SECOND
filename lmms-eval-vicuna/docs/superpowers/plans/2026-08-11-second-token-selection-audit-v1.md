# SECOND Token Selection Audit v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible 80-sample POPE audit that runs Standard LLaVA plus three controlled SECOND conditions, records the exact token-selection state, verifies Full SECOND prediction parity, renders attention/mask overlays, and reports paired C-to-D flips.

**Architecture:** A manifest-backed POPE audit task guarantees that every distributed run consumes the same question IDs. Pure audit helpers define condition semantics, selection diagnostics, answer normalization, and paired analysis; the recursive generator calls those helpers while an artifact recorder writes per-rank/per-sample files. A separate shell driver runs A/B/C/D sequentially without modifying the user's existing `run.sh`.

**Tech Stack:** Python 3.10, PyTorch, Hugging Face Datasets, lmms-eval YAML tasks, NumPy, Pillow, OpenCV, pytest, Bash, Accelerate.

## Global Constraints

- Condition A is Standard LLaVA; B is Recursive-AllToken-Raw; C is Recursive-AllToken-CD; D is Full SECOND.
- C and D differ only in whether the proposed selector mask is applied.
- Raw decoding explicitly returns final-stage logits without contrastive combination or candidate truncation.
- AllToken still computes and records `ret_attn`, entropy, cutoff, and proposed mask.
- The audit manifest contains FN=30, FP=10, TP=20, TN=20 sampled with seed 42 from the existing D log.
- Tensor dumping defaults to answer `token_idx=0`.
- Existing logs and the user's modified `run.sh` are never overwritten.
- The old D run has no internal tensors; parity against it is prediction-level, while selector equivalence is test-level.
- New audit reports use explicit `pope_pred_yes_ratio`, `pope_gt_yes_ratio`, and `pope_invalid_ratio` names.

---

### Task 1: Correct and version POPE binary metrics

**Files:**
- Modify: `lmms_eval/tasks/pope/utils.py`
- Modify: `lmms_eval/tasks/pope/pope.yaml`
- Modify: `lmms_eval/tasks/pope/pope_pop.yaml`
- Modify: `lmms_eval/tasks/pope/pope_random.yaml`
- Modify: `lmms_eval/tasks/pope/pope_adv.yaml`
- Create: `tests/test_pope_metrics.py`

**Interfaces:**
- Produces: `normalize_pope_answer(text: str) -> str`, returning `"yes"`, `"no"`, or `"invalid"`.
- Produces: `pope_aggregate_pred_yes_ratio`, `pope_aggregate_gt_yes_ratio`, and `pope_aggregate_invalid_ratio`.
- Preserves: `pope_yes_ratio`, but changes it to an alias of the predicted-Yes ratio for schema version 2.

- [ ] **Step 1: Write failing normalization and aggregation tests**

```python
from lmms_eval.tasks.pope import utils


def _result(prediction, ground_truth):
    return {"prediction": prediction, "ground_truth": ground_truth, "score": 0.0}


def test_explicit_yes_ratios_and_invalid_ratio():
    rows = [_result("yes", "yes"), _result("no", "yes"), _result("invalid", "no")]
    assert utils.pope_aggregate_pred_yes_ratio(rows) == 1 / 3
    assert utils.pope_aggregate_gt_yes_ratio(rows) == 2 / 3
    assert utils.pope_aggregate_invalid_ratio(rows) == 1 / 3
    assert utils.pope_aggregate_yes_ratio(rows) == 1 / 3


def test_process_results_normalizes_first_binary_word():
    metrics = utils.pope_process_results({"question_id": "q", "answer": "yes"}, ["Yes."])
    assert metrics["pope_accuracy"]["prediction"] == "yes"
```

- [ ] **Step 2: Run tests and confirm missing functions fail**

Run: `pytest -q tests/test_pope_metrics.py`

Expected: FAIL because the explicit ratio functions do not exist and `Yes.` is not normalized.

- [ ] **Step 3: Implement one answer parser and consistent confusion accounting**

```python
_BINARY_ANSWER_RE = re.compile(r"^\s*(yes|no)\b", re.IGNORECASE)


def normalize_pope_answer(text):
    match = _BINARY_ANSWER_RE.match(str(text))
    return match.group(1).lower() if match else "invalid"
```

Use this value in `pope_process_results`; count every GT-Yes prediction other than `yes` as a recall miss; add metric payload keys for all explicit ratios.

- [ ] **Step 4: Add schema-2 metrics to every POPE task YAML**

Each YAML keeps `pope_yes_ratio` and adds:

```yaml
  - metric: pope_pred_yes_ratio
    aggregation: !function utils.pope_aggregate_pred_yes_ratio
    higher_is_better: true
  - metric: pope_gt_yes_ratio
    aggregation: !function utils.pope_aggregate_gt_yes_ratio
    higher_is_better: true
  - metric: pope_invalid_ratio
    aggregation: !function utils.pope_aggregate_invalid_ratio
    higher_is_better: false
```

Set task metadata version to `2.0`.

- [ ] **Step 5: Run metric tests**

Run: `pytest -q tests/test_pope_metrics.py`

Expected: PASS.

- [ ] **Step 6: Commit metric correction**

```bash
git add lmms_eval/tasks/pope tests/test_pope_metrics.py
git commit -m "fix: report POPE prediction yes ratio"
```

### Task 2: Build the deterministic audit manifest and filtered task

**Files:**
- Create: `lmms_eval/tasks/pope/audit_utils.py`
- Create: `lmms_eval/tasks/pope/pope_pop_audit.yaml`
- Create: `tools/second_audit.py`
- Create: `tests/test_second_audit_manifest.py`

**Interfaces:**
- Produces: `build_manifest(sample_log: Path, output: Path, seed: int, counts: dict[str, int]) -> dict`.
- Produces: CLI `python tools/second_audit.py sample --samples-jsonl ... --output ...`.
- Produces: `pope_process_audit_docs(dataset)`, reading `SECOND_AUDIT_MANIFEST` and filtering by `question_id`.

- [ ] **Step 1: Write failing tests for strata, reproducibility, and manifest validation**

```python
def test_build_manifest_is_reproducible_and_stratified(tmp_path):
    source = write_synthetic_pope_log(tmp_path, per_stratum=4)
    one = build_manifest(source, tmp_path / "one.json", 42, {"FN": 2, "FP": 1, "TP": 2, "TN": 2})
    two = build_manifest(source, tmp_path / "two.json", 42, {"FN": 2, "FP": 1, "TP": 2, "TN": 2})
    assert one["samples"] == two["samples"]
    assert Counter(row["stratum"] for row in one["samples"]) == {"FN": 2, "FP": 1, "TP": 2, "TN": 2}


def test_filter_rejects_missing_question_id(monkeypatch, tmp_path):
    manifest = write_manifest(tmp_path, question_ids=["missing"])
    monkeypatch.setenv("SECOND_AUDIT_MANIFEST", str(manifest))
    with pytest.raises(ValueError, match="missing"):
        pope_process_audit_docs(fake_dataset([{"question_id": "present"}]))
```

- [ ] **Step 2: Run the manifest tests and confirm failure**

Run: `pytest -q tests/test_second_audit_manifest.py`

Expected: FAIL because the manifest builder and audit filter are absent.

- [ ] **Step 3: Implement the `sample` CLI with standard-library JSONL parsing**

The implementation normalizes logged predictions, classifies TP/TN/FP/FN, rejects duplicate question IDs, checks stratum capacity, samples with `random.Random(seed)`, preserves the sampled original `doc_id`, and writes schema version 1 plus the source path and seed.

- [ ] **Step 4: Implement the environment-backed dataset filter**

```python
def pope_process_audit_docs(dataset):
    manifest_path = os.environ.get("SECOND_AUDIT_MANIFEST")
    if not manifest_path:
        raise ValueError("SECOND_AUDIT_MANIFEST is required for pope_pop_audit")
    wanted = load_and_validate_question_ids(Path(manifest_path))
    filtered = dataset.filter(lambda doc: str(doc["question_id"]) in wanted)
    validate_exact_match(filtered, wanted)
    return filtered
```

- [ ] **Step 5: Add `pope_pop_audit.yaml`**

Copy POPE Popular's dataset, prompt, generation settings, schema-2 metrics, and set:

```yaml
task: "pope_pop_audit"
process_docs: !function audit_utils.pope_process_audit_docs
metadata:
  version: 2.0
```

- [ ] **Step 6: Run manifest tests**

Run: `pytest -q tests/test_second_audit_manifest.py`

Expected: PASS.

- [ ] **Step 7: Generate the real 80-sample manifest and inspect counts**

Run:

```bash
python tools/second_audit.py sample \
  --samples-jsonl logs/liuhaotian__llava-v1.6-vicuna-7b/20260810_001849_samples_pope_pop.jsonl \
  --output audit/manifests/pope_pop_seed42_80.json \
  --seed 42 --fn 30 --fp 10 --tp 20 --tn 20
```

Expected: printed counts `FN=30 FP=10 TP=20 TN=20`, 80 unique question IDs.

- [ ] **Step 8: Commit manifest and task support**

```bash
git add lmms_eval/tasks/pope/audit_utils.py lmms_eval/tasks/pope/pope_pop_audit.yaml tools/second_audit.py tests/test_second_audit_manifest.py audit/manifests/pope_pop_seed42_80.json
git commit -m "feat: add deterministic POPE audit manifest"
```

### Task 3: Expose selector diagnostics without changing its mask

**Files:**
- Modify: `lmms_eval/recursion_utils.py`
- Create: `tests/test_recursion_audit.py`

**Interfaces:**
- Extends: `attn_entropy_topk_based_recursion(..., return_diagnostics: bool = False)`.
- When requested, returns `(mask, requested_keep_ratio, diagnostics)` where diagnostics includes `second_entropy`, `normalized_entropy`, `entropy_2d`, `attention_cutoff`, `requested_keep_ratio`, `actual_keep_ratio`, `selected_token_count`, and `total_token_count`.
- Preserves the original two-value return when `return_diagnostics=False`.

- [ ] **Step 1: Write failing parity, tie, and finite-value tests**

```python
def test_diagnostics_preserve_legacy_mask():
    attn = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)
    legacy_mask, legacy_keep = attn_entropy_topk_based_recursion(attn, torch.zeros_like(attn), 1.0)
    mask, keep, diag = attn_entropy_topk_based_recursion(attn, torch.zeros_like(attn), 1.0, return_diagnostics=True)
    assert torch.equal(mask, legacy_mask)
    assert keep == legacy_keep
    assert diag["selected_token_count"] == int(mask.sum().item())


def test_ties_distinguish_requested_and_actual_keep_ratio():
    attn = torch.ones(4, 4)
    _, requested, diag = attn_entropy_topk_based_recursion(attn, torch.zeros_like(attn), 0.3, return_diagnostics=True)
    assert diag["actual_keep_ratio"] >= requested
```

- [ ] **Step 2: Run selector tests and confirm failure**

Run: `pytest -q tests/test_recursion_audit.py`

Expected: FAIL because `return_diagnostics` is not accepted.

- [ ] **Step 3: Add optional diagnostics around the unchanged selection operations**

Keep row normalization, entropy, exponential keep-ratio mapping, flattened top-k cutoff, `>=` tie behavior, and in-place mask update numerically unchanged. Compute flattened 2D entropy on a separately normalized clone. Raise `ValueError` on non-finite input or derived values.

- [ ] **Step 4: Run selector tests**

Run: `pytest -q tests/test_recursion_audit.py`

Expected: PASS.

- [ ] **Step 5: Commit selector instrumentation**

```bash
git add lmms_eval/recursion_utils.py tests/test_recursion_audit.py
git commit -m "feat: expose SECOND selection diagnostics"
```

### Task 4: Implement audit condition semantics and artifact recording

**Files:**
- Create: `lmms_eval/second_audit.py`
- Create: `tests/test_second_audit_core.py`
- Create: `tests/test_second_audit_recorder.py`

**Interfaces:**
- Produces: `AuditCondition` validation and predicates `uses_recursion`, `uses_contrastive_decoding`, and `applies_selection`.
- Produces: `validate_audit_args(condition, output_dir, token_idx)` and `applied_selection_mask(condition, proposed_mask)`.
- Produces: `select_final_logits(stage_logits, contrastive_alphas, use_cd) -> torch.Tensor`.
- Produces: `resolve_binary_token_ids(tokenizer, prompt: str) -> dict[str, int]` and `binary_logit_record(logits, token_ids) -> dict`.
- Produces: `SecondAuditRecorder(output_dir, condition, rank, token_idx=0)` with `begin_sample`, `write_transition`, and `finish_sample`.

- [ ] **Step 1: Write failing condition and Raw/CD tests**

```python
def test_raw_logits_are_exact_last_stage_logits():
    stages = [torch.tensor([[1.0, 3.0]]), torch.tensor([[2.0, 2.5]])]
    actual = select_final_logits(stages, [0.8], use_cd=False)
    assert torch.equal(actual, stages[-1])


def test_condition_c_and_d_share_cd_but_only_d_applies_selection():
    assert uses_contrastive_decoding("C") and uses_contrastive_decoding("D")
    assert not applies_selection("C") and applies_selection("D")
```

- [ ] **Step 2: Write failing recorder layout test**

```python
def test_recorder_writes_lossless_arrays_and_overlays(tmp_path):
    recorder = SecondAuditRecorder(tmp_path, "D", rank=1)
    recorder.begin_sample("q1", Image.new("RGB", (32, 32)), {"question": "Q?"})
    recorder.write_transition("q1", token_idx=0, source_stage=-2, target_stage=-1,
                              ret_attn=np.ones((4, 4)), proposed_mask=np.eye(4), applied_mask=np.eye(4),
                              metadata={"requested_keep_ratio": 0.5})
    assert (tmp_path / "D" / "rank_1" / "q1" / "token_000_stage_-2_to_-1.npz").exists()
    assert (tmp_path / "D" / "rank_1" / "q1" / "token_000_stage_-2_to_-1_attention.png").exists()
```

- [ ] **Step 3: Run core and recorder tests and confirm failure**

Run: `pytest -q tests/test_second_audit_core.py tests/test_second_audit_recorder.py`

Expected: FAIL because `lmms_eval.second_audit` does not exist.

- [ ] **Step 4: Implement pure condition/logit/token helpers**

For CD=True, move the existing Full SECOND stage-logit calculation into `select_final_logits` without changing its order of operations. For CD=False, return `stage_logits[-1].clone()` immediately. Resolve `Yes` and `No` by comparing tokenization of the exact prompt with tokenization of `prompt + candidate`; accept only a single unambiguous continuation token. If that is not possible, return an explicit unavailable reason rather than choosing arbitrary IDs.

- [ ] **Step 5: Implement collision-free per-rank artifact recording**

Write sample metadata atomically, save raw arrays using `numpy.savez_compressed`, normalize only derived heatmaps, and create proposed/applied mask overlays against the saved model-input image. Reject NaN/Inf arrays and duplicate sample directories.

- [ ] **Step 6: Run core and recorder tests**

Run: `pytest -q tests/test_second_audit_core.py tests/test_second_audit_recorder.py`

Expected: PASS.

- [ ] **Step 7: Commit audit core**

```bash
git add lmms_eval/second_audit.py tests/test_second_audit_core.py tests/test_second_audit_recorder.py
git commit -m "feat: add SECOND audit core and recorder"
```

### Task 5: Wire A/B/C/D into LLaVA generation

**Files:**
- Modify: `lmms_eval/models/llava.py`
- Modify: `LLaVA-NeXT/llava/model/language_model/llava_recursion.py`
- Create: `tests/test_llava_audit_plumbing.py`

**Interfaces:**
- Adds model args: `audit_condition: Optional[str]`, `audit_output_dir: Optional[str]`, `audit_token_idx: int = 0`.
- Extends: `generate_recursive(..., audit_condition=None, audit_recorder=None, audit_sample=None)`.
- Condition A calls ordinary `model.generate(..., generation_type=None)` and does not call `generate_recursive`.

- [ ] **Step 1: Write failing tests around extracted branch helpers and config validation**

```python
def test_audit_requires_output_directory():
    with pytest.raises(ValueError, match="audit_output_dir"):
        validate_audit_args("D", None, 0)


def test_alltoken_applies_ones_but_preserves_proposal():
    proposed = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    applied = applied_selection_mask("C", proposed)
    assert torch.equal(proposed, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    assert torch.equal(applied, torch.ones_like(proposed))
```

- [ ] **Step 2: Run plumbing tests and confirm failure**

Run: `pytest -q tests/test_llava_audit_plumbing.py`

Expected: FAIL because audit arguments and mask helper do not exist.

- [ ] **Step 3: Add model arguments and construct one recorder per process**

Normalize condition IDs to uppercase, validate A/B/C/D, capture Accelerator rank, derive stable `question_id` from the task document, and call `begin_sample` with the actual PIL image passed to that condition's preprocessing.

- [ ] **Step 4: Add Standard LLaVA condition A**

Condition A skips square-forcing and downsampled inputs, calls ordinary LLaVA generation with `generation_type=None`, identical POPE prompt and greedy parameters, decodes the returned continuation, and writes sample-level metadata only.

- [ ] **Step 5: Add recursive conditions B/C/D**

At each non-final stage:

```python
proposed, requested_keep, diagnostics = attn_entropy_topk_based_recursion(
    attn=ret_attn,
    base_top_k=self.attention_threshold[idx_stage],
    image_mask=self.image_mask[stage + 1].clone(),
    return_diagnostics=True,
)
applied = applied_selection_mask(audit_condition, proposed)
self.image_mask[stage + 1] = applied
```

Record source-stage margin immediately and target-stage margin when the next stage logits become available. Only dump transitions for configured `audit_token_idx`; selection itself still runs for every answer token.

- [ ] **Step 6: Route final logits through the pure Raw/CD helper**

B calls `select_final_logits(..., use_cd=False)`; C and D call it with `use_cd=True`. With no audit condition, preserve legacy Full SECOND behavior. Reject `use_noised_for_contrastive=True` in v1 audit mode because it is outside the approved four-condition design.

- [ ] **Step 7: Run audit plumbing and prior unit tests**

Run: `pytest -q tests/test_llava_audit_plumbing.py tests/test_second_audit_core.py tests/test_recursion_audit.py`

Expected: PASS.

- [ ] **Step 8: Commit generation plumbing**

```bash
git add lmms_eval/models/llava.py LLaVA-NeXT/llava/model/language_model/llava_recursion.py tests/test_llava_audit_plumbing.py
git commit -m "feat: add controlled SECOND audit conditions"
```

### Task 6: Add parity and paired-flip analysis

**Files:**
- Modify: `tools/second_audit.py`
- Create: `tests/test_second_audit_analysis.py`

**Interfaces:**
- Adds CLI: `parity --manifest ... --original ... --new ...`.
- Adds CLI: `analyze --manifest ... --a ... --b ... --c ... --d ... --output ...`.
- Produces JSON and Markdown reports joined by `question_id`.

- [ ] **Step 1: Write failing parity and flip tests**

```python
def test_paired_analysis_uses_c_to_d_for_selection_harm():
    rows = paired_rows(c=[True, False, True], d=[False, True, True])
    report = analyze_pairs(rows)
    assert report["selection_harm_count"] == 1
    assert report["selection_benefit_count"] == 1


def test_parity_rejects_any_d_prediction_change():
    with pytest.raises(ValueError, match="prediction parity"):
        verify_prediction_parity(original={"q1": "yes"}, new={"q1": "no"})
```

- [ ] **Step 2: Run analysis tests and confirm failure**

Run: `pytest -q tests/test_second_audit_analysis.py`

Expected: FAIL because analysis subcommands do not exist.

- [ ] **Step 3: Implement strict joins, parity, conditional flips, and Wilson intervals**

Reject missing/duplicate condition rows. Report A-to-B recursive-path changes, B-to-C CD flips, C-to-D selection flips, FN Rescue Rate, per-original-D-stratum counts, and 95% Wilson intervals. Label all rates as diagnostic-set conditional quantities.

- [ ] **Step 4: Run analysis tests**

Run: `pytest -q tests/test_second_audit_analysis.py`

Expected: PASS.

- [ ] **Step 5: Commit analysis support**

```bash
git add tools/second_audit.py tests/test_second_audit_analysis.py
git commit -m "feat: analyze paired SECOND audit runs"
```

### Task 7: Add the four-condition run driver and operating guide

**Files:**
- Create: `scripts/run_second_audit.sh`
- Create: `docs/second_audit_v1.md`
- Create: `tests/test_second_audit_script.py`

**Interfaces:**
- Produces: `bash scripts/run_second_audit.sh SOURCE_D_SAMPLE_LOG [OUTPUT_ROOT]`.
- Outputs: manifest, four lmms-eval run directories, audit artifacts, D parity report, and paired analysis report.

- [ ] **Step 1: Write a failing static command test**

```python
def test_driver_contains_all_conditions_and_safety_guards():
    text = Path("scripts/run_second_audit.sh").read_text()
    for condition in ("A", "B", "C", "D"):
        assert f'run_condition "{condition}"' in text
    assert "SECOND_AUDIT_MANIFEST" in text
    assert "NCCL_P2P_DISABLE" in text
    assert "NCCL_IB_DISABLE" in text
    assert "set -euo pipefail" in text
```

- [ ] **Step 2: Run the script test and confirm failure**

Run: `pytest -q tests/test_second_audit_script.py`

Expected: FAIL because the driver does not exist.

- [ ] **Step 3: Implement the sequential two-GPU driver**

The script validates the source log, refuses an already-populated output root, creates the manifest, exports the manifest path and safe NCCL defaults, launches two Accelerate processes for each condition, locates each condition's unique POPE audit sample log, verifies D parity, then produces the paired report. A uses `fix_grid=none`; B/C/D use the approved stages, threshold, positional embedding, and alphas. Every condition passes its own audit artifact directory through `--model_args`.

- [ ] **Step 4: Document individual and all-in-one commands**

The guide explains output layout, approximate evaluation count (320 sample-condition evaluations), how to inspect overlay PNGs, how to rerun parity, how to interpret C-to-D flips, and the phase-2 gate of at least 5 rescued sampled FN.

- [ ] **Step 5: Run shell syntax and static tests**

Run: `bash -n scripts/run_second_audit.sh && pytest -q tests/test_second_audit_script.py`

Expected: PASS.

- [ ] **Step 6: Commit driver and guide**

```bash
git add scripts/run_second_audit.sh docs/second_audit_v1.md tests/test_second_audit_script.py
git commit -m "docs: add SECOND audit run workflow"
```

### Task 8: Full verification without starting the GPU experiment

**Files:**
- Review all files modified in Tasks 1-7.

**Interfaces:**
- Produces: a verified command handoff; does not launch the 320-evaluation GPU experiment without a separate explicit run request.

- [ ] **Step 1: Run the focused audit suite**

Run:

```bash
pytest -q \
  tests/test_pope_metrics.py \
  tests/test_second_audit_manifest.py \
  tests/test_recursion_audit.py \
  tests/test_second_audit_core.py \
  tests/test_second_audit_recorder.py \
  tests/test_llava_audit_plumbing.py \
  tests/test_second_audit_analysis.py \
  tests/test_second_audit_script.py
```

Expected: all tests pass.

- [ ] **Step 2: Run syntax and import checks**

Run:

```bash
python -m compileall -q lmms_eval tools/second_audit.py
bash -n scripts/run_second_audit.sh
python tools/second_audit.py --help
```

Expected: exit status 0 for every command.

- [ ] **Step 3: Check the real manifest and task registration**

Run:

```bash
SECOND_AUDIT_MANIFEST="$PWD/audit/manifests/pope_pop_seed42_80.json" \
python -m lmms_eval --tasks list | grep -F pope_pop_audit
```

Expected: `pope_pop_audit` is listed. This check may access only the already cached task registry; it must not launch model inference.

- [ ] **Step 4: Inspect repository diff and preserve unrelated work**

Run: `git status --short && git diff --check`

Expected: the user's pre-existing `run.sh`, logs, and unrelated tests remain untouched; no whitespace errors.

- [ ] **Step 5: Report verification evidence**

Record the exact passing test count, manifest counts, shell syntax result, and
remaining unrelated working-tree entries in the handoff. If verification
required a code fix, rerun the failing task's focused test and the full command
from Step 1 before reporting success.

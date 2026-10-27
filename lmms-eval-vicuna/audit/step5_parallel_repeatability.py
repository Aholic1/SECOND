#!/usr/bin/env python3
"""Two-round cross-GPU C/D study: C1/D1 in parallel, then D2/C2 in parallel."""
from __future__ import annotations

import argparse
import hashlib
import html
import importlib.metadata
import json
import os
import random
import socket
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "audit/manifests/step5_changed30_matched_v1"
ORDER = ("C1", "D1", "D2", "C2")
# The default preserves the original four-GPU plan for report-only users.
# Run/round commands replace it with the explicitly selected physical pairs.
GPU_LAYOUT = {"C1": "0,1", "D1": "2,3", "D2": "0,1", "C2": "2,3"}
TARGET_GPU_INDICES = ("0", "1", "2", "3")
TRANSITIONS = ((-2, -1), (-1, 0), (0, 1))
SEEDS = "0,1234,1234,1234"
PARAMS = {"generation_type": "recursion", "fix_grid": "2x2",
          "attention_thresholding_type": "attn_topk", "attention_threshold": 1.0,
          "positional_embedding_type": "bilinear_interpolation", "stages": [-2, -1, 0, 1],
          "contrastive_alphas": [0.8, 0.8, 0.8]}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            result.update(data)
    return result.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def parse_gpu_pair(value):
    parts = [part.strip() for part in str(value).split(",")]
    check(len(parts) == 2 and all(part.isdigit() for part in parts),
          f"GPU pair must be two comma-separated indices, got {value!r}")
    pair = tuple(str(int(part)) for part in parts)
    check(pair[0] != pair[1], f"GPU pair contains a duplicate index: {value!r}")
    return pair


def set_gpu_layout(pair_a, pair_b):
    """Configure the physical pairs used by the two-round schedule."""
    global GPU_LAYOUT, TARGET_GPU_INDICES
    a = parse_gpu_pair(pair_a)
    b = parse_gpu_pair(pair_b)
    check(set(a).isdisjoint(b), f"GPU pairs overlap: {pair_a!r} and {pair_b!r}")
    TARGET_GPU_INDICES = tuple(a + b)
    GPU_LAYOUT = {"C1": ",".join(a), "D1": ",".join(b),
                  "D2": ",".join(a), "C2": ",".join(b)}


def layout_from_snapshot(snapshot_state):
    layout = snapshot_state.get("gpu_layout")
    check(isinstance(layout, dict) and set(layout) == set(ORDER),
          "study snapshot has no complete GPU layout")
    return layout


def load_bundle(folder, verify_images=False):
    folder = Path(folder).resolve()
    hashes = read(folder / "sha256.json")
    for name, expected in hashes.items():
        check(Path(name).name == name, "unsafe bundle filename")
        check(digest(folder / name) == expected, f"frozen file modified: {name}")
    required = {"changed30.json", "controls.json", "combined.json", "pairs.json", "freeze_report.json", "freeze_provenance.json"}
    check(required <= set(hashes), "incomplete frozen hash index")
    combined = read(folder / "combined.json")
    check(combined["provenance_sha256"] == hashes["freeze_provenance.json"], "provenance hash differs")
    rows = combined["samples"]
    ids = [str(r["question_id"]) for r in rows]
    check(len(ids) == len(set(ids)) == combined["expected_count"], "duplicate/missing combined IDs")
    cases = read(folder / "changed30.json")["samples"]
    controls = read(folder / "controls.json")["samples"]
    check(len(cases) == 30, "expected original 30 cases")
    check({r["question_id"] for r in cases + controls} == set(ids), "combined membership differs")
    check(len(cases) + len(controls) == len(ids), "case/control overlap")
    check({r["question_id"]: r for r in cases + controls} == {r["question_id"]: r for r in rows}, "combined row contents differ from case/control manifests")
    check(sum(r["change_direction"] == "corrected" for r in cases) == 13, "expected 13 historical corrections")
    check(sum(r["change_direction"] == "degraded" for r in cases) == 17, "expected 17 historical degradations")
    freeze = read(folder / "freeze_report.json")
    check(freeze["status"] in ("PASS", "NEEDS_REVIEW"), "invalid freeze status")
    check(freeze["control_count"] == len(controls), "control count differs")
    # Partial matching is permitted for repeated predictions; it is NOT reclassified as complete matching.
    if verify_images:
        checked = set()
        for row in rows:
            if row["image_path"] not in checked:
                check(digest(row["image_path"]) == row["image_sha256"], f"image changed: {row['question_id']}")
                checked.add(row["image_path"])
    return rows, {"folder": str(folder), "sha256_index": digest(folder / "sha256.json"),
                  "combined_sha256": hashes["combined.json"], "count": len(rows),
                  "historical_matching_status": freeze["status"], "control_count": len(controls)}


def prepare_review(bundle, output):
    rows, identity = load_bundle(bundle, verify_images=True)
    output = Path(output).resolve()
    check(not output.exists(), f"review directory already exists: {output}")
    output.mkdir(parents=True)
    (output / "images").mkdir()
    # Hide prediction, original GT, case/control role and annotation flags uniformly for ALL rows.
    order = sorted(rows, key=lambda r: int(r["question_id"]))
    random.Random(20261013).shuffle(order)
    cards, key, empty = [], {}, []
    for i, row in enumerate(order, 1):
        blind = f"R{i:03d}"
        image_name = f"images/{blind}.jpg"
        shutil.copyfile(row["image_path"], output / image_name)
        key[blind] = {"question_id": row["question_id"], "image_sha256": row["image_sha256"]}
        empty.append({"review_id": blind, "presence": "unreviewed", "localizable": "unreviewed", "notes": ""})
        cards.append(f'''<article data-id="{blind}"><h2>{blind}</h2><p>{html.escape(row['question'])}</p>
<a href="{image_name}" target="_blank"><img src="{image_name}" alt="Review image {blind}"></a>
<label>图像中目标是否可见 <select class="presence"><option value="unreviewed">未复核</option><option value="yes">可见</option><option value="no">未见</option><option value="uncertain">无法确定</option></select></label>
<label>能否明确定位目标 <select class="localizable"><option value="unreviewed">未复核</option><option value="yes">可以</option><option value="no">不可以</option><option value="uncertain">无法确定</option></select></label>
<textarea placeholder="说明可见证据、遮挡或类别歧义。不要根据上下文推断画面外目标。"></textarea></article>''')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>POPE 视觉标签独立复核</title>
<style>body{font:17px sans-serif;max-width:980px;margin:30px auto;background:#fafafa}article{background:white;border:1px solid #ddd;padding:20px;margin:20px 0}img{max-width:100%;max-height:680px}label{display:block;margin:12px 0}select,input,button,textarea{font:inherit}textarea{width:95%;height:90px}header{position:sticky;top:0;background:#eef5ff;padding:12px}h2{font-size:20px}</style>
<h1>视觉标签独立复核</h1><p>只根据图像回答；目标类别不等于同名颜色、相关物品或画面外物品。可点击图片查看原始大小。无法确定时选择“无法确定”。本页隐藏原标签及模型预测，不会自动替你判断。</p>
<p>建议两人独立复核。若你已经看过模型结果，请在复核者说明中注明。页面不自动保存，关闭前请导出；可导入已导出的文件继续。</p>
<header>复核者 <input id="reviewer" placeholder="姓名或代号"><button onclick="save()">导出复核 JSON</button><input type="file" accept=".json" onchange="restore(this)"></header>
''' + "\n".join(cards) + '''
<script>
const cohort = COHORT_ID;
function save(){let rows=[...document.querySelectorAll('article')].map(a=>({review_id:a.dataset.id,presence:a.querySelector('.presence').value,localizable:a.querySelector('.localizable').value,notes:a.querySelector('textarea').value}));
let reviewer=document.getElementById('reviewer').value.trim();if(!reviewer){alert('请填写复核者');return;}
let bad=rows.find(r=>r.presence==='no'&&r.localizable==='yes');if(bad){alert(bad.review_id+'：未见目标却标为可定位，请复核');return;}
let data={schema_version:1,cohort_sha256:cohort,reviewer,exported_at:new Date().toISOString(),reviews:rows};let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));a.download='visual_review.json';a.click();URL.revokeObjectURL(a.href);}
async function restore(input){try{let data=JSON.parse(await input.files[0].text());if(data.cohort_sha256!==cohort)throw Error('清单哈希不匹配');document.getElementById('reviewer').value=data.reviewer||'';for(let r of data.reviews){let a=[...document.querySelectorAll('article')].find(a=>a.dataset.id===r.review_id);if(!a)throw Error('未知 review_id');a.querySelector('.presence').value=r.presence;a.querySelector('.localizable').value=r.localizable;a.querySelector('textarea').value=r.notes||'';}}catch(e){alert(e.message);}}
</script></html>'''
    page = page.replace("COHORT_ID", json.dumps(identity["combined_sha256"]))
    (output / "index.html").write_text(page, encoding="utf-8")
    write(output / "review_key_private.json", {"cohort": identity, "review_id_to_question": key})
    write(output / "review_template.json", {"schema_version": 1, "cohort_sha256": identity["combined_sha256"], "reviewer": "", "reviews": empty})
    import zipfile
    with zipfile.ZipFile(output / "blind_review.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(output / "index.html", "index.html")
        for file in sorted((output / "images").iterdir()):
            archive.write(file, "images/" + file.name)
    write(output / "review_package_report.json", {"status": "PASS", "count": len(rows), "cohort": identity,
          "blind_zip": str(output / "blind_review.zip"), "warning": "Do not give review_key_private.json or original manifests to independent reviewers; original labels are unchanged."})
    print(f"Review package prepared for {len(rows)} questions: {output / 'blind_review.zip'}")


def git(*args):
    return subprocess.check_output(["git", "-c", f"safe.directory={ROOT.parent}", "-C", str(ROOT), *args], text=True).strip()


def snapshot():
    import yaml
    # Task equivalence is checked while allowing only the intended subset filter/name difference.
    task_dir = ROOT / "lmms_eval/tasks/pope"
    task_full = yaml.load((task_dir / "pope_pop.yaml").read_text(), Loader=yaml.BaseLoader)
    task_subset = yaml.load((task_dir / "pope_pop_audit.yaml").read_text(), Loader=yaml.BaseLoader)
    for task in (task_full, task_subset):
        task.pop("task", None)
        task.pop("process_docs", None)
        if "token" in task.get("dataset_kwargs", {}):
            task["dataset_kwargs"]["token"] = str(task["dataset_kwargs"]["token"]).lower()
    check(task_full == task_subset, "full and audit task generation/prompt configurations differ")
    source_files = {}
    for folder in (ROOT / "lmms_eval", ROOT / "LLaVA-NeXT/llava"):
        for p in sorted(folder.rglob("*")):
            if p.is_file() and p.suffix in (".py", ".yaml", ".yml", ".json"):
                source_files[str(p.relative_to(ROOT))] = digest(p)
    for name in ("step5_parallel_repeatability.py", "validate_step2_capture.py", "run_step5_parallel.sh"):
        source_files["audit/" + name] = digest(ROOT / "audit" / name)
    packages = {}
    for name in ("torch", "transformers", "accelerate", "datasets", "numpy", "Pillow", "safetensors", "tokenizers", "flash-attn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    # Record local HF model resolution and file metadata, without reading 13+GB weights repeatedly.
    from huggingface_hub.constants import HF_HUB_CACHE
    model_dir = Path(HF_HUB_CACHE) / "models--liuhaotian--llava-v1.6-vicuna-7b"
    models = []
    for p in sorted(model_dir.rglob("*")):
        if p.is_file() and ("snapshots" in p.parts or "refs" in p.parts):
            st = p.stat()
            item = {"relative_path": str(p.relative_to(model_dir)), "resolved": str(p.resolve()), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}
            if st.st_size < 20_000_000:
                item["sha256"] = digest(p)
            models.append(item)
    check(bool(models), f"local model cache not found: {model_dir}")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip()
    inventory = {line.split(",")[0].strip(): [p.strip() for p in line.split(",")] for line in gpu.splitlines()}
    check(all(index in inventory for index in TARGET_GPU_INDICES),
          f"required GPU(s) not found: {','.join(TARGET_GPU_INDICES)}")
    check(len({inventory[index][2] for index in TARGET_GPU_INDICES}) == 1,
          "selected GPU models differ")
    check(len({inventory[index][4] for index in TARGET_GPU_INDICES}) == 1,
          "selected GPU memory capacities differ")
    return {"git_commit": git("rev-parse", "HEAD"), "source_files": source_files,
            "source_fingerprint": canonical(source_files), "python": sys.version, "executable": sys.executable,
            "packages": packages, "gpu_identity": gpu, "model_cache": str(model_dir), "model_files": models,
            "runtime": {k: os.environ.get(k) for k in ("CONDA_DEFAULT_ENV", "CUDA_VISIBLE_DEVICES", "NCCL_P2P_DISABLE", "NCCL_IB_DISABLE", "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED")},
            "orchestrator_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "num_processes_per_run": 2, "gpu_layout": GPU_LAYOUT,
            "gpu_uuid_by_index": {index: item[1] for index, item in inventory.items()},
            "seeds": SEEDS, "second_parameters": PARAMS,
            "limitations": "Weight bytes are not fully hashed; metadata/blob paths are checked. Historical source dirty state and numeric behavior cannot be established retroactively."}


def study_setup(bundle, study):
    rows, identity = load_bundle(bundle, verify_images=True)
    state = snapshot()
    provenance = read(Path(bundle) / "freeze_provenance.json")
    historical = provenance["recorded_run_manifests"]
    check(state["git_commit"] == historical["C"]["git_commit"], "HEAD changed since historical C/D; do not silently change experimental code")
    for condition in ("C", "D"):
        check(historical[condition]["second_parameters"] == PARAMS, "historical SECOND parameters differ")
        runtime = historical[condition]["runtime"]
        check(str(runtime["cuda_visible_devices"]) == "0,1" and int(runtime["num_processes"]) == 2, "historical GPU/process layout differs")
    for name, source in provenance["source_files"].items():
        if name in ("C_samples", "D_samples", "C_manifest", "D_manifest"):
            check(digest(source["path"]) == source["sha256"], f"historical source modified: {name}")
    plan_path = study / "study_plan.json"
    if plan_path.exists():
        plan = read(plan_path)
        check(canonical(plan["snapshot"]) == plan["snapshot_sha256"], "study snapshot checksum differs")
        check(plan["cohort"] == identity, "cohort changed after study was prepared")
        check(plan["snapshot"] == state, "source/runtime/model metadata changed; preserve old study and investigate")
        return rows, plan
    check(not study.exists(), f"study exists without a plan; inspect it instead of overwriting: {study}")
    plan = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(), "cohort": identity,
            "order": list(ORDER), "snapshot": state, "snapshot_sha256": canonical(state),
            "recorded_historical_runs": historical, "git_status_at_setup": git("status", "--short"),
            "sampling": "Same seeds in separate processes on different GPU pairs; this checks cross-pair consistency, not same-device repeatability or seed robustness.",
            "rounds": [["C1", "D1"], ["D2", "C2"]],
            "design_limit": "GPU pair and round vary together within each condition. Disagreement requires follow-up repeats on the SAME pair; no separate hardware/time effect is identifiable here.",
            "scope": "All frozen cases are retained, including unmatched and disputed-label cases. Original labels remain unchanged."}
    historical_records(plan, {r["question_id"]: r for r in rows})
    study.mkdir(parents=True)
    write(plan_path, plan)
    (study / "working_tree.patch").write_text(git("diff", "HEAD", "--binary"), encoding="utf-8")
    return rows, plan


def sample_log(run):
    found = list((run / "results").glob("*/*_samples_*.jsonl"))
    check(len(found) == 1, f"expected one sample log under {run}, found {len(found)}")
    result = {}
    for line in found[0].read_text().splitlines():
        raw = json.loads(line)
        metric, doc = raw["pope_accuracy"], raw["doc"]
        qid = str(metric["question_id"])
        check(qid not in result and qid == str(doc["question_id"]), "duplicate or inconsistent logged ID")
        result[qid] = {"prediction": metric["prediction"], "ground_truth": metric["ground_truth"], "question": doc["question"]}
    return result


def array_hash(array):
    return hashlib.sha256(str(array.shape).encode() + str(array.dtype).encode() + array.tobytes()).hexdigest()


def capture_records(run, condition, expected_rows):
    import numpy as np
    records = {}
    logs = sample_log(run)
    for path in sorted((run / "artifacts" / condition).glob("rank_*/*/sample.json")):
        sample = read(path)
        qid = str(sample["question_id"])
        check(qid in expected_rows and qid not in records, f"unexpected/duplicate artifact ID {qid}")
        row = expected_rows[qid]
        for field in ("question", "ground_truth"):
            check(sample[field] == row[field], f"{qid}: frozen {field} differs")
            check(logs[qid][field] == row[field], f"{qid}: logged {field} differs")
        check(sample["prediction"] in ("yes", "no"), f"{qid}: invalid prediction")
        check(sample["prediction"] == logs[qid]["prediction"], f"{qid}: artifact/log prediction differs")
        masks, counts, attention = {}, {}, {}
        for source, target in TRANSITIONS:
            key = f"{source}_to_{target}"
            with np.load(path.parent / f"token_000_stage_{key}.npz", allow_pickle=False) as arrays:
                for name in ("applied_mask", "consumed_mask"):
                    masks[f"{key}/{name}"] = array_hash(arrays[name])
                counts[key] = int(arrays["consumed_mask"].sum())
                attention[key] = array_hash(arrays["ret_attn"])
        transition = read(path.parent / "token_000_stage_0_to_1.json")
        margin = transition["final_decoding_margin"]
        records[qid] = {"prediction": sample["prediction"], "final_token_ids": sample["final_token_ids"],
            "prompt": sample["prompt"], "generation_config": sample["generation_config"], "image_transform": sample["image_transform"],
            "mask_hashes": masks, "consumed_counts": counts, "attention_hashes": attention,
            "yes_no_margin": margin["yes_no_margin"], "rank": sample.get("rank")}
    check(set(records) == set(expected_rows) == set(logs), "artifact/log/frozen question sets differ")
    return records


def stream_command(command, logfile, env):
    with logfile.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(f"[{logfile.parent.name}] {line}", end="", flush=True)
                log.write(line)
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    check(code == 0, f"command failed with exit {code}; inspect {logfile}")


def assert_gpu_pairs_idle():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
        text=True,
    )
    usage = {}
    for line in output.splitlines():
        index, used, total = (part.strip() for part in line.split(","))
        usage[index] = (int(used), int(total))
    errors = []
    for index in TARGET_GPU_INDICES:
        check(index in usage, f"required GPU {index} not found")
        used, total = usage[index]
        # CUDA contexts and display allocations can consume a small amount; a model already loaded cannot.
        if used > 2048:
            errors.append(f"GPU {index}: {used}/{total} MiB in use")
    check(not errors, "GPU pair preflight failed before launch: " + "; ".join(errors))


def verify_completed(run, label, rows, plan):
    complete = read(run / "run_complete.json")
    check(complete["status"] == "PASS", f"{label}: completion status failed")
    check(complete["snapshot_sha256"] == plan["snapshot_sha256"], "completed run provenance differs")
    for name, sha in complete["output_hashes"].items():
        check(digest(run / name) == sha, f"completed output changed: {run / name}")
    manifest = read(run / "run_manifest.json")
    check(manifest["repeat_label"] == label and manifest["condition"] == label[0], "run label differs")
    layout = layout_from_snapshot(plan["snapshot"])
    check(manifest["physical_gpu_pair"] == layout[label], "completed GPU assignment differs")
    expected_uuids = ",".join(plan["snapshot"]["gpu_uuid_by_index"][i] for i in layout[label].split(","))
    check(manifest["cuda_visible_devices"] == expected_uuids, "completed UUID assignment differs")
    check(manifest["num_processes"] == 2 and manifest["seed"] == SEEDS, "completed run settings differ")
    check(capture_records(run, label[0], rows) == read(run / "records.json"), "completed artifact content changed")


def run_one(study, label, rows, plan, port, gpu_pair):
    run = study / label
    expected = {r["question_id"]: r for r in rows}
    if (run / "run_complete.json").exists():
        verify_completed(run, label, expected, plan)
        print(f"Verified completed run, skipping: {label}")
        return
    check(not run.exists(), f"incomplete run exists: {run}. Preserve/rename it before retrying; do not overwrite failed evidence.")
    check(snapshot() == plan["snapshot"], "source/runtime changed before next run")
    load_bundle(plan["cohort"]["folder"], verify_images=True)
    layout = layout_from_snapshot(plan["snapshot"])
    check(gpu_pair == layout[label], "GPU assignment differs from frozen schedule")
    gpu_uuids = ",".join(plan["snapshot"]["gpu_uuid_by_index"][i] for i in gpu_pair.split(","))
    run.mkdir()
    (run / "artifacts").mkdir()
    (run / "results").mkdir()
    condition = label[0]
    model = f"pretrained=liuhaotian/llava-v1.6-vicuna-7b,audit_condition={condition},audit_output_dir={run / 'artifacts'},audit_token_idx=0,audit_artifact_mode=compact"
    command = [sys.executable, "-m", "accelerate.commands.launch", "--num_processes=2", "--main_process_port", str(port),
               "-m", "lmms_eval", "--model", "llava", "--model_args", model, "--tasks", "pope_pop_audit",
               "--batch_size", "1", "--log_samples", "--log_samples_suffix", f"step5_{label}",
               "--output_path", str(run / "results"), "--generation_type", "recursion", "--fix_grid", "2x2",
               "--attention_thresholding_type", "attn_topk", "--attention_threshold", "1.0",
               "--positional_embedding_type", "bilinear_interpolation", "--stages", "-2", "-1", "0", "1",
               "--verbosity", "DEBUG", "--contrastive_alphas", "0.8", "0.8", "0.8", "--seed", SEEDS]
    manifest = {"condition": condition, "repeat_label": label, "expected_count": len(rows), "git_commit": plan["snapshot"]["git_commit"],
                "snapshot_sha256": plan["snapshot_sha256"], "combined_sha256": plan["cohort"]["combined_sha256"],
                "command": command, "main_process_port": port, "physical_gpu_pair": gpu_pair,
                "num_processes": 2, "seed": SEEDS, "cuda_visible_devices": gpu_uuids}
    write(run / "run_manifest.json", manifest)
    env = dict(os.environ)
    # UUIDs prevent differences between CUDA enumeration and nvidia-smi indices.
    env["CUDA_VISIBLE_DEVICES"] = gpu_uuids
    env["NUM_PROCESSES"] = "2"
    env["SECOND_AUDIT_MANIFEST"] = str(Path(plan["cohort"]["folder"]) / "combined.json")
    stream_command(command, run / "console.log", env)
    check(snapshot() == plan["snapshot"], "source/runtime/model metadata changed during run")
    # Validate structure and consumed masks. Do not use a prediction-equality gate here: prediction changes are the result being measured.
    validation = [sys.executable, str(ROOT / "audit/validate_step2_capture.py"), "--artifact-root", str(run / "artifacts"),
                  "--baseline-predictions", str(ROOT / "audit/step1_full/pope_pop_3000_baseline/baseline_predictions.jsonl"),
                  "--expected-manifest", env["SECOND_AUDIT_MANIFEST"], "--expected-count", str(len(rows)),
                  "--condition", condition, "--artifact-mode", "compact", "--report", str(run / "validation_report.json")]
    stream_command(validation, run / "validation.log", env)
    records = capture_records(run, condition, expected)
    write(run / "records.json", records)
    write(run / "run_complete.json", {"status": "PASS", "snapshot_sha256": plan["snapshot_sha256"],
          "output_hashes": {name: digest(run / name) for name in ("records.json", "validation_report.json", "run_manifest.json")},
          "semantics": "Artifact and sample validation passed; prediction repeatability is assessed separately."})


def historical_records(plan, rows):
    """Read selected historical artifacts to check prompts/config; never compare subset doc_id to full doc_id."""
    provenance = read(Path(plan["cohort"]["folder"]) / "freeze_provenance.json")
    result = {}
    for condition in ("C", "D"):
        run = Path(provenance["source_files"][f"{condition}_manifest"]["path"]).parent
        # Existing complete logs contain 3000 rows; subset artifact loading is intentional.
        records = {}
        for path in (run / "artifacts" / condition).glob("rank_*/*/sample.json"):
            qid = path.parent.name
            if qid not in rows:
                continue
            sample = read(path)
            qid = str(sample["question_id"])
            check(qid not in records, "duplicate historical artifact")
            check(sample["prediction"] == rows[qid][f"{condition}_prediction"], "historical artifact/frozen prediction differs")
            records[qid] = {k: sample[k] for k in ("prompt", "generation_config", "image_transform")}
        check(set(records) == set(rows), f"{condition}: selected historical artifacts missing")
        result[condition] = records
    return result


def analyze_records(rows, records, historical=None):
    details = []
    for row in rows:
        qid = row["question_id"]
        values = {k: records[k][qid] for k in ORDER}
        for field in ("prompt", "generation_config", "image_transform"):
            check(all(v[field] == values["C1"][field] for v in values.values()), f"{qid}: repeat C/D {field} differs")
            if historical:
                check(all(historical[c][qid][field] == values["C1"][field] for c in ("C", "D")), f"{qid}: historical {field} differs")
        stable = {c: values[c + "1"]["prediction"] == values[c + "2"]["prediction"] for c in ("C", "D")}
        matches = {c: all(values[c + str(i)]["prediction"] == row[c + "_prediction"] for i in (1, 2)) for c in ("C", "D")}
        masks = {c: values[c + "1"]["mask_hashes"] == values[c + "2"]["mask_hashes"] for c in ("C", "D")}
        tokens = {c: values[c + "1"]["final_token_ids"] == values[c + "2"]["final_token_ids"] for c in ("C", "D")}
        both_stable = all(stable.values())
        current_changed = values["C1"]["prediction"] != values["D1"]["prediction"] if both_stable else None
        role = row["role"]
        if not both_stable:
            outcome = "repeat_unstable"
        elif all(matches.values()):
            outcome = "historical_change_reproduced" if role == "changed_case" else "historical_equal_control_reproduced"
        else:
            outcome = "repeat_stable_but_historical_mismatch"
        details.append({"question_id": qid, "coco_image_id": row["coco_image_id"], "role": role,
            "historical_change_direction": row.get("change_direction"), "original_ground_truth": row["ground_truth"],
            "annotation_review_flag": bool(row["annotation_issues"]), "historical_predictions": {c: row[c + "_prediction"] for c in ("C", "D")},
            "predictions": {k: v["prediction"] for k, v in values.items()}, "outcome": outcome,
            "prediction_repeat_equal": stable, "historical_prediction_match": matches, "first_answer_masks_repeat_equal": masks,
            "final_token_ids_repeat_equal": tokens, "currently_CD_different": current_changed,
            "first_answer_attention_bitwise_equal": {c: values[c + "1"]["attention_hashes"] == values[c + "2"]["attention_hashes"] for c in ("C", "D")},
            "yes_no_margins": {k: v["yes_no_margin"] for k, v in values.items()},
            "margin_absolute_repeat_difference": {c: abs(values[c + "1"]["yes_no_margin"] - values[c + "2"]["yes_no_margin"]) for c in ("C", "D")},
            "mechanism_screen_candidate_pending_label_review": role == "changed_case" and outcome == "historical_change_reproduced" and all(masks.values()) and all(tokens.values()) and not row["annotation_issues"]})
    unstable = [r["question_id"] for r in details if r["outcome"] == "repeat_unstable"]
    mismatch = [r["question_id"] for r in details if not all(r["historical_prediction_match"].values())]
    mask_changed = [r["question_id"] for r in details if not all(r["first_answer_masks_repeat_equal"].values())]
    tokens_changed = [r["question_id"] for r in details if not all(r["final_token_ids_repeat_equal"].values())]
    summary = {"data_validation_status": "PASS",
               "cross_gpu_pair_prediction_consistency_status": "PASS" if not unstable else "MISMATCH",
               "historical_prediction_reproduction_across_gpu_pairs_status": "PASS" if not mismatch else "MISMATCH",
               "cross_gpu_pair_first_answer_mask_consistency_status": "PASS" if not mask_changed else "MISMATCH",
               "cross_gpu_pair_final_token_sequence_consistency_status": "PASS" if not tokens_changed else "MISMATCH",
               "case_count": sum(r["role"] == "changed_case" for r in details),
               "control_count": sum(r["role"] != "changed_case" for r in details),
               "outcome_counts": dict(Counter(r["outcome"] for r in details)),
               "unstable_prediction_ids": unstable, "historical_prediction_mismatch_ids": mismatch,
               "unstable_first_answer_mask_ids": mask_changed, "unstable_final_token_sequence_ids": tokens_changed,
               "annotation_review_ids": [r["question_id"] for r in details if r["annotation_review_flag"]],
               "mechanism_screen_candidate_ids_pending_label_review": [r["question_id"] for r in details if r["mechanism_screen_candidate_pending_label_review"]],
               "limitations": ["Each condition runs once per GPU pair; this tests cross-pair consistency, not within-pair repeatability or universal determinism.", "Attention/masks are recorded only for first answer token, all three stage transitions.",
                              "Labels are unchanged; no automated report establishes target presence or causality.", "Same-image questions are correlated; this outcome-selected set cannot estimate population accuracy.",
                              "Subset scheduling differs from the historical full run; stable historical mismatch needs investigation, not baseline replacement."]}
    return {"summary": summary, "samples": details}


def report_study(study):
    plan = read(study / "study_plan.json")
    rows, identity = load_bundle(plan["cohort"]["folder"])
    check(identity == plan["cohort"], "frozen cohort changed")
    records = {}
    for label in ORDER:
        run = study / label
        verify_completed(run, label, {r["question_id"]: r for r in rows}, plan)
        complete = read(run / "run_complete.json")
        check(complete["status"] == "PASS" and complete["snapshot_sha256"] == plan["snapshot_sha256"], "run gate did not pass")
        for name, sha in complete["output_hashes"].items():
            check(digest(run / name) == sha, f"completed file changed: {label}/{name}")
        records[label] = capture_records(run, label[0], {r["question_id"]: r for r in rows})
        check(records[label] == read(run / "records.json"), "capture record changed after completion")
    result = analyze_records(rows, records, historical_records(plan, {r["question_id"]: r for r in rows}))
    result["summary"]["experiment_design"] = "two-round cross-GPU-pair consistency; not same-GPU repeatability"
    layout = layout_from_snapshot(plan["snapshot"])
    result["summary"]["gpu_layout"] = layout
    result["summary"]["limitations"].append(plan["design_limit"])
    for sample in result["samples"]:
        qid = sample["question_id"]
        pair_a = layout["C1"].replace(",", "_")
        pair_b = layout["D1"].replace(",", "_")
        sample["same_pair_CD_comparisons"] = {
            f"GPU_{pair_a}_C1_vs_D2": {"C": records["C1"][qid]["prediction"], "D": records["D2"][qid]["prediction"]},
            f"GPU_{pair_b}_C2_vs_D1": {"C": records["C2"][qid]["prediction"], "D": records["D1"][qid]["prediction"]},
        }
    result["study_plan_sha256"] = digest(study / "study_plan.json")
    result["cohort"] = identity
    write(study / "repeatability_report.json", result)
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    print(f"Report: {study / 'repeatability_report.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    review = sub.add_parser("review")
    review.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    review.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    run.add_argument("--study", type=Path, required=True)
    run.add_argument("--pair-a", default="0,1", help="physical GPU pair for C1/D2")
    run.add_argument("--pair-b", default="2,3", help="physical GPU pair for D1/C2")
    run.add_argument("--prepare-only", action="store_true", help="Preflight and write plan only; no model launch")
    round_run = sub.add_parser("round")
    round_run.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    round_run.add_argument("--study", type=Path, required=True)
    round_run.add_argument("--round", choices=("1", "2"), required=True)
    round_run.add_argument("--port-base", type=int, default=29931)
    round_run.add_argument("--pair-a", default="0,1", help="physical GPU pair for C1/D2")
    round_run.add_argument("--pair-b", default="2,3", help="physical GPU pair for D1/C2")
    round_run.add_argument("--prepare-only", action="store_true", help="Prepare/check only; no model launch")
    report = sub.add_parser("report")
    report.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "review":
        prepare_review(args.bundle, args.output)
    elif args.action == "report":
        report_study(args.study.resolve())
    elif args.action in ("run", "round"):
        check(os.environ.get("CONDA_DEFAULT_ENV") == "SECOND_clean", "activate SECOND_clean first")
        set_gpu_layout(args.pair_a, args.pair_b)
        planned_gpus = ",".join(TARGET_GPU_INDICES)
        check(os.environ.get("CUDA_VISIBLE_DEVICES", "") == "",
              "unset CUDA_VISIBLE_DEVICES before starting the scheduler")
        check(os.environ.get("NUM_PROCESSES", "2") == "2", "use NUM_PROCESSES=2 per run")
        os.environ["CUDA_VISIBLE_DEVICES"] = planned_gpus
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        os.environ.setdefault("NCCL_IB_DISABLE", "1")
        study = args.study.resolve()
        import fcntl
        study.parent.mkdir(parents=True, exist_ok=True)
        study_lock = (study.parent / ("." + study.name + ".scheduler.lock")).open("a")
        try:
            fcntl.flock(study_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("another scheduler is already using this study directory")
        assert_gpu_pairs_idle()
        rows, plan = study_setup(args.bundle.resolve(), study)
        print(f"Cohort: {len(rows)}; historical matching status: {plan['cohort']['historical_matching_status']}", flush=True)
        if args.action == "run":
            check(args.prepare_only, "serial --action run is disabled; use explicit round 1/2 scheduling")
            print("Preflight passed; NO MODEL RUN. Use the two round commands after checking GPU availability.")
            return
        check(1024 <= args.port_base <= 65534, "port base must be in 1024..65534")
        if args.round == "2":
            for previous in ("C1", "D1"):
                check((study / previous / "run_complete.json").is_file(), "round 1 must finish and pass before round 2")
                verify_completed(study / previous, previous, {r["question_id"]: r for r in rows}, plan)
        if args.prepare_only:
            print(f"Round {args.round} preflight passed; NO MODEL RUN. Planned GPUs: {GPU_LAYOUT}.")
            return
        for port in (args.port_base, args.port_base + 1):
            with socket.socket() as probe:
                probe.bind(("", port))
        labels = ("C1", "D1") if args.round == "1" else ("D2", "C2")
        failures = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(run_one, study, label, rows, plan, args.port_base + offset,
                            layout_from_snapshot(plan["snapshot"])[label]): label
                for offset, label in enumerate(labels)
            }
            for future in as_completed(futures):
                label = futures[future]
                try:
                    future.result()
                    print(f"Completed {label} on GPU {GPU_LAYOUT[label]}", flush=True)
                except Exception as exc:
                    failures.append((label, str(exc)))
                    print(f"FAILED {label}: {exc}", file=sys.stderr, flush=True)
        if failures:
            raise ValueError(f"round {args.round} had failures {failures}; completed peer runs were preserved")
        if args.round == "1":
            print("Round 1 complete. Inspect C1 and D1 validation reports, then run round 2.")
        else:
            if all((study / label / "run_complete.json").is_file() for label in ORDER):
                report_study(study)
            else:
                print("Round 2 complete, but one or more other round artifacts are absent; inspect before reporting.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"FAIL: {exc}")

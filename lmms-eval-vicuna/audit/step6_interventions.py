"""Staged first-token, fixed-mask causal tests. GPU inference only in explicit run commands."""
import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "audit"))
from step5_repeatability import (read, write, check, digest, canonical, snapshot,
    load_bundle, run_one, capture_records)
from step5_unstable_repeatability import configure_runtime, gpu_preflight, ensure_port_free
from step5_visualize_mechanism_candidates import (source_records, load_transition,
    target_rectangles, rect_intersects, nearest_resize)

DEFAULT = ROOT / "audit/runs/step6_interventions_v1"
BUNDLE = ROOT / "audit/manifests/step5_changed30_matched_v1"
PARENT = ROOT / "audit/runs/step5_repeatability_v1"
FOCAL = ("2253", "2559")
CONTEXT = ("311", "645", "1025", "1043")
STAGES = (-1, 0, 1)
SEEDS = (17, 29, 43)


def state():
    value = snapshot()
    value["step6_scripts"] = {p.name: digest(p) for p in
        (Path(__file__), ROOT / "audit/install_step6_hook.py")}
    return value


def lift_additions(applied, consumed, indices):
    """Edit only nearest-resize sampling locations; match actual consumed-token budget."""
    result = applied.copy()
    h, w = consumed.shape
    for index in indices:
        r, c = divmod(int(index), w)
        result[r * result.shape[0] // h, c * result.shape[1] // w] = 1
    expected = consumed.copy().reshape(-1)
    expected[list(indices)] = 1
    check(np.array_equal(nearest_resize(result, (h, w)).reshape(-1), expected),
          "consumed-grid intervention mapping failed")
    return result


def setup(study):
    check(not study.exists(), "study already exists; do not overwrite")
    check("STEP6_MASK_INTERVENTION" in (ROOT / "LLaVA-NeXT/llava/model/language_model/llava_recursion.py").read_text(),
          "install the hook first")
    rows, identity = load_bundle(BUNDLE, verify_images=True)
    parent = read(PARENT / "study_plan.json")
    check(identity == parent["cohort"], "48-question cohort differs from parent")
    current = state()
    study.mkdir(parents=True)
    write(study / "study_plan.json", {"cohort": identity, "snapshot": snapshot(),
        "snapshot_sha256": canonical(snapshot()), "step6_state": current,
        "parent_plan_sha256": digest(PARENT / "study_plan.json"),
        "scope": "First generated token only; other stage masks frozen; later tokens adaptive. All 48 questions retained."})
    print("Prepared; no inference. New source snapshot; historical provenance is not overwritten.")


def verify(study):
    plan = read(study / "study_plan.json")
    check(state() == plan["step6_state"], "source/runtime changed since prepare")
    rows, identity = load_bundle(BUNDLE, verify_images=True)
    check(identity == plan["cohort"], "cohort changed")
    return plan, rows


def execute(study, label, port, intervention=None):
    plan, rows = verify(study)
    gpu_preflight()
    ensure_port_free(port)
    os.environ.pop("SECOND_INTERVENTION_PLAN", None)
    if intervention:
        os.environ["SECOND_INTERVENTION_PLAN"] = str(intervention.resolve())
    try:
        run_one(study, label, rows, plan, port)
    finally:
        os.environ.pop("SECOND_INTERVENTION_PLAN", None)
    if intervention:
        masks = read(intervention)["masks"]
        for row in rows:
            qid = str(row["question_id"])
            paths = list((study / label / "artifacts/D").glob(f"rank_*/{qid}/sample.json"))
            check(len(paths) == 1, "missing intervention artifact")
            for target in STAGES:
                _, arrays, _ = load_transition(paths[0].parent, target - 1, target)
                check(np.array_equal(arrays["applied_mask"], masks[qid][str(target)]),
                      f"{label}/{qid}/{target}: requested mask was not applied")
        write(study / label / "intervention_validation.json", {
            "status": "PASS", "plan_sha256": digest(intervention), "scope": "All requested first-token applied/consumed masks checked"})


def freeze(study):
    plan, rows = verify(study)
    records = {}
    expected = {str(r["question_id"]): r for r in rows}
    for label in ("C1", "D1", "D2", "C2"):
        complete = read(study / label / "run_complete.json")
        check(complete["snapshot_sha256"] == plan["snapshot_sha256"], "baseline snapshot mismatch")
        for name, sha in complete["output_hashes"].items():
            check(digest(study / label / name) == sha, "baseline output changed")
        records[label] = capture_records(study / label, label[0], expected)
        check(records[label] == read(study / label / "records.json"), "baseline artifacts changed")
    for qid in FOCAL + CONTEXT:
        for condition in ("C", "D"):
            a, b = records[condition + "1"][qid], records[condition + "2"][qid]
            check(a == b, f"{qid}/{condition}: repeat records differ; stop before causal tests")
        check(records["C1"][qid]["prediction"] == "yes" and records["D1"][qid]["prediction"] == "no",
              f"{qid}: expected degradation did not reproduce; do not force historical interpretation")
    masks, geometry, arrays_by_id = {}, {}, {}
    for qid, row in expected.items():
        path = list((study / "D1/artifacts/D").glob(f"rank_*/{qid}/sample.json"))[0].parent
        masks[qid], geometry[qid], arrays_by_id[qid] = {}, {}, {}
        for target in STAGES:
            _, arrays, _ = load_transition(path, target - 1, target)
            masks[qid][str(target)] = arrays["applied_mask"].tolist()
            arrays_by_id[qid][target] = arrays
            if qid not in FOCAL + CONTEXT:
                continue
            cells = source_records(path, qid, target, target)
            selected = arrays["consumed_mask"].reshape(-1).astype(bool)
            check(len(cells) == len(selected), "provenance size mismatch")
            check([c["selected"] for c in cells] == selected.tolist(), "provenance selection mismatch")
            boxes = target_rectangles(row)
            check(boxes, f"{qid}: missing target annotation")
            hit = np.array([any(rect_intersects(c["original_rect_clipped"], b) for b in boxes) for c in cells])
            content = np.array([c["original_rect_clipped"][2] > c["original_rect_clipped"][0] and
                                c["original_rect_clipped"][3] > c["original_rect_clipped"][1] for c in cells])
            geometry[qid][target] = {"target_dropped": np.flatnonzero(hit & ~selected).tolist(),
                "non_target_dropped": np.flatnonzero(~hit & ~selected & content).tolist(),
                "per_instance": [{"candidate": sum(rect_intersects(c["original_rect_clipped"], b) for c in cells),
                    "kept": sum(rect_intersects(c["original_rect_clipped"], b) and bool(selected[i]) for i, c in enumerate(cells))} for b in boxes]}
            if qid in CONTEXT:
                check(not geometry[qid][target]["target_dropped"], f"{qid}: target loss now present; context hypothesis requires revision")
    out = study / "plans"
    check(not out.exists(), "plans already frozen")
    out.mkdir()
    write(out / "geometry.json", geometry)
    def save(name, value, details):
        write(out / (name + ".json"), {"masks": value, "design": details,
            "baseline_records_sha256": digest(study / "D1/records.json")})
    save("D_replay", masks, "All 48 original first-token stage masks replayed")
    for stage_name, stages in [("sneg1", [-1]), ("s0", [0]), ("s1", [1]), ("all", list(STAGES))]:
        for kind, seed in [("target", None)] + [("random", s) for s in SEEDS]:
            value = copy.deepcopy(masks)
            detail = {}
            for qid in FOCAL:
                for target in stages:
                    g = geometry[qid][target]
                    indices = g["target_dropped"]
                    check(indices, f"{qid}/{target}: no target loss to rescue")
                    if kind == "random":
                        pool = g["non_target_dropped"]
                        check(len(pool) >= len(indices), "insufficient non-target controls; do not silently reduce budget")
                        rng = np.random.default_rng(seed + int(qid) * 10 + target)
                        indices = sorted(rng.choice(pool, len(indices), replace=False).tolist())
                    a = arrays_by_id[qid][target]
                    value[qid][str(target)] = lift_additions(a["applied_mask"], a["consumed_mask"], indices).tolist()
                    detail[f"{qid}/{target}"] = indices
            suffix = "" if seed is None else f"_{seed}"
            save(f"D_{kind}_{stage_name}{suffix}", value, {"added_consumed_indices": detail,
                 "control": "Non-target content cells, NOT guaranteed semantic background; 3 seeds are exploratory"})
        value = copy.deepcopy(masks)
        for qid in FOCAL + CONTEXT:
            for target in stages:
                value[qid][str(target)] = np.ones_like(arrays_by_id[qid][target]["applied_mask"]).tolist()
        save(f"D_full_{stage_name}", value, "Restore selected stages for six cases; all other first-token masks fixed")
    write(study / "frozen_plans.json", {p.name: digest(p) for p in out.glob("*.json")})
    print("Frozen geometry and interventions. Next: replay gate.")


def replay_gate(study):
    a = read(study / "D1/records.json")
    b = read(study / "D_replay/records.json")
    check(read(study / "D_replay/intervention_validation.json")["status"] == "PASS", "replay masks unverified")
    failures = [qid for qid in a if a[qid] != b[qid]]
    write(study / "replay_gate.json", {"status": "FAIL" if failures else "PASS", "different_record_ids": failures})
    check(not failures, f"replay differs for {failures}; stop, inspect reproducibility")


def interventions(study, group, port):
    frozen = read(study / "frozen_plans.json")
    for name, sha in frozen.items():
        check(digest(study / "plans" / name) == sha, "frozen plan changed")
    if group != "replay":
        replay_gate(study)
    prefix = {"replay": "D_replay", "rescue": "D_target_", "controls": "D_random_", "context": "D_full_"}[group]
    for i, name in enumerate(sorted(n for n in frozen if n.startswith(prefix))):
        execute(study, Path(name).stem, port + i, study / "plans" / name)
    if group == "replay":
        replay_gate(study)


def report(study):
    verify(study)
    replay_gate(study)
    frozen = read(study / "frozen_plans.json")
    result = []
    missing = []
    base = read(study / "D1/records.json")
    for name, sha in sorted(frozen.items()):
        if not name.startswith("D_"):
            continue
        label = Path(name).stem
        run = study / label
        if not (run / "intervention_validation.json").exists():
            missing.append(label)
            continue
        check(digest(study / "plans" / name) == sha, "plan changed")
        check(read(run / "intervention_validation.json")["plan_sha256"] == sha, "run/plan mismatch")
        rows, _ = load_bundle(BUNDLE)
        actual = capture_records(run, "D", {str(r["question_id"]): r for r in rows})
        check(actual == read(run / "records.json"), "run artifacts changed")
        for qid in FOCAL + CONTEXT:
            r = actual[qid]
            result.append({"run": label, "qid": qid, "prediction": r["prediction"],
                "margin": r["yes_no_margin"], "margin_delta_from_D": r["yes_no_margin"] - base[qid]["yes_no_margin"],
                "consumed_counts": r["consumed_counts"]})
    write(study / "intervention_report.json", {"status": "INCOMPLETE" if missing else "COMPLETE",
        "missing_runs": missing, "results": result,
        "limitations": "Selected cases; first-token frozen masks; oracle bbox intervention; not population proof or adaptive-stage proof."})
    print("Report:", study / "intervention_report.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "baseline", "freeze", "replay", "rescue", "controls", "context", "report"])
    parser.add_argument("--study", type=Path, default=DEFAULT)
    parser.add_argument("--port-base", type=int, default=29951)
    args = parser.parse_args()
    configure_runtime()
    check(not os.environ.get("SECOND_INTERVENTION_PLAN"), "unset SECOND_INTERVENTION_PLAN before invoking driver")
    study = args.study.resolve()
    if args.action == "prepare":
        setup(study)
    elif args.action == "baseline":
        for i, label in enumerate(("C1", "D1", "D2", "C2")):
            execute(study, label, args.port_base + i)
    elif args.action == "freeze":
        freeze(study)
    elif args.action == "report":
        report(study)
    else:
        interventions(study, args.action, args.port_base)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"FAIL: {exc}")

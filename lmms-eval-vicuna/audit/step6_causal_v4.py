"""Token-budget-controlled target reallocation tests; preserves prior failed studies."""
import argparse
import copy
import os
from pathlib import Path

import numpy as np
import step6_interventions as core

ROOT = core.ROOT
DEFAULT = ROOT / 'audit/runs/step6_causal_v4'
BASELINE = ROOT / 'audit/runs/step6_interventions_v1'
PRIMARY = ('2253', '2559', '311', '645', '1043')
TARGETS = ('2253', '2559')
CONTEXT = ('311', '645', '1043')
STAGES = (-1, 0, 1)
SEEDS = (17, 29, 43)
BASE_VERIFY = core.verify


def prep(study):
    core.check(not study.exists(), 'v3 study path exists; preserve it and choose a new version')
    check_study = BASELINE
    source_plan = core.read(check_study / 'study_plan.json')
    core.check(core.state() == source_plan['step6_state'], 'source/runtime differs from original Step 6 baseline')
    rows, identity = core.load_bundle(core.BUNDLE, verify_images=True)
    core.check(identity == source_plan['cohort'], 'frozen 48-question cohort differs')
    expected = {str(r['question_id']): r for r in rows}
    baseline = {}
    completion = {}
    for label in ('C1', 'D1', 'D2', 'C2'):
        run = check_study / label
        complete = core.read(run / 'run_complete.json')
        core.check(complete['status'] == 'PASS' and complete['snapshot_sha256'] == source_plan['snapshot_sha256'], f'{label}: baseline incomplete')
        for name, sha in complete['output_hashes'].items():
            core.check(core.digest(run / name) == sha, f'{label}: output hash mismatch')
        baseline[label] = core.capture_records(run, label[0], expected)
        core.check(baseline[label] == core.read(run / 'records.json'), f'{label}: captured records differ')
        completion[label] = core.digest(run / 'run_complete.json')
    for qid in PRIMARY:
        for condition in ('C', 'D'):
            core.check(baseline[condition+'1'][qid] == baseline[condition+'2'][qid], f'{qid}/{condition}: not repeatable in primary cohort')
        core.check(baseline['C1'][qid]['prediction'] == 'yes' and baseline['D1'][qid]['prediction'] == 'no', f'{qid}: expected degradation not reproduced')
    study.mkdir(parents=True)
    core.write(study / 'study_plan.json', source_plan)
    core.write(study / 'v3_plan.json', {
        'driver_sha256': core.digest(Path(__file__)), 'parent_study': str(check_study.resolve()),
        'parent_completion_sha256': completion, 'primary_ids': list(PRIMARY),
        '1025': 'Retained in the 48-row execution but excluded from primary mechanism inference because its D1/D2 masks differed.',
        'design': 'Same-token-budget reallocation. Target arm adds dropped target cells and removes equal-count retained non-target content cells. Feasible-dose controls add dropped non-target cells and remove the same number of retained non-target cells. Full target dose is compared with baseline if a same-dose non-target control is unavailable.'})
    for label in completion:
        (study / label).symlink_to((check_study / label).resolve(), target_is_directory=True)
    print('Prepared v3 by verifying and linking the immutable original baseline; no inference.')


def verify(study):
    plan = core.read(study / 'v3_plan.json')
    core.check(core.digest(Path(__file__)) == plan['driver_sha256'], 'v3 driver changed after prepare')
    for label, expected in plan['parent_completion_sha256'].items():
        core.check(core.digest(study / label / 'run_complete.json') == expected, f'{label} baseline changed')
    return BASE_VERIFY(study)


core.verify = verify


def map_consumed_edits(applied, consumed, add_indices, remove_indices):
    result = applied.copy()
    h, w = consumed.shape
    expected = consumed.astype(bool).reshape(-1).copy()
    for index, value in [(i, True) for i in add_indices] + [(i, False) for i in remove_indices]:
        index = int(index)
        r, c = divmod(index, w)
        result[r * result.shape[0] // h, c * result.shape[1] // w] = 1 if value else 0
        expected[index] = value
    observed = core.nearest_resize(result, (h, w)).astype(bool).reshape(-1)
    core.check(np.array_equal(observed, expected), 'consumed-grid edit mapping did not preserve requested operations')
    return result


def freeze(study):
    plan, rows = verify(study)
    root = study / 'plans'
    core.check(not root.exists(), 'partial plans directory exists; preserve study and use a new versioned path')
    root.mkdir()
    row_map = {str(r['question_id']): r for r in rows}
    masks, geometry, arrays = {}, {}, {}
    for qid, row in row_map.items():
        sample = list((study / 'D1/artifacts/D').glob(f'rank_*/{qid}/sample.json'))
        core.check(len(sample) == 1, f'{qid}: missing/duplicate D1 sample')
        directory = sample[0].parent
        masks[qid], geometry[qid], arrays[qid] = {}, {}, {}
        for stage in STAGES:
            _, item, _ = core.load_transition(directory, stage - 1, stage)
            masks[qid][str(stage)] = item['applied_mask'].tolist()
            arrays[qid][stage] = item
            if qid not in TARGETS + CONTEXT:
                continue
            provenance = core.source_records(directory, qid, stage, stage)
            kept = item['consumed_mask'].reshape(-1).astype(bool)
            core.check(len(provenance) == len(kept), f'{qid}/{stage}: geometry/mask size mismatch')
            core.check([r['selected'] for r in provenance] == kept.tolist(), f'{qid}/{stage}: provenance/mask mismatch')
            boxes = core.target_rectangles(row)
            core.check(boxes, f'{qid}: no target boxes')
            target = np.array([any(core.rect_intersects(r['original_rect_clipped'], b) for b in boxes) for r in provenance])
            content = np.array([r['original_rect_clipped'][2] > r['original_rect_clipped'][0] and r['original_rect_clipped'][3] > r['original_rect_clipped'][1] for r in provenance])
            geometry[qid][str(stage)] = {
                'target_cells': np.flatnonzero(target).tolist(),
                'target_dropped': np.flatnonzero(target & ~kept).tolist(),
                'target_kept': np.flatnonzero(target & kept).tolist(),
                'non_target_dropped': np.flatnonzero(~target & ~kept & content).tolist(),
                'non_target_kept': np.flatnonzero(~target & kept & content).tolist(),
                'candidate_count': int(len(kept)), 'kept_count': int(kept.sum()),
                'per_instance': [{'candidate': int(sum(core.rect_intersects(r['original_rect_clipped'], b) for r in provenance)),
                                  'kept': int(sum(core.rect_intersects(r['original_rect_clipped'], b) and bool(kept[i]) for i, r in enumerate(provenance)))} for b in boxes]}
            if qid in CONTEXT:
                core.check(not geometry[qid][str(stage)]['target_dropped'], f'{qid}/{stage}: target-loss group changed; revise cohort')

    core.write(root / 'geometry.json', geometry)
    plans = {'D_replay': (copy.deepcopy(masks), {'type': 'replay'})}
    # Full-dose target reallocation: restore all dropped target cells while
    # removing the same number of selected non-target content cells.
    for stage_label, stages in [('sneg1', [-1]), ('s0', [0]), ('s1', [1]), ('all', list(STAGES))]:
        additive = copy.deepcopy(masks)
        additive_details = {}
        target_plan = copy.deepcopy(masks)
        target_details = {}
        feasible_target = True
        for qid in TARGETS:
            for stage in stages:
                g = geometry[qid][str(stage)]
                add = g['target_dropped']
                remove_pool = g['non_target_kept']
                if not add:
                    feasible_target = False
                    break
                item = arrays[qid][stage]
                additive[qid][str(stage)] = core.lift_additions(item['applied_mask'], item['consumed_mask'], add).tolist()
                additive_details[f'{qid}/{stage}'] = {'added_target': add, 'token_count_increased': True}
                if len(remove_pool) < len(add):
                    feasible_target = False
                    break
                chosen = sorted(np.random.default_rng(711 + int(qid)*7 + stage).choice(remove_pool, len(add), replace=False).tolist())
                target_plan[qid][str(stage)] = map_consumed_edits(item['applied_mask'], item['consumed_mask'], add, chosen).tolist()
                target_details[f'{qid}/{stage}'] = {'added_target': add, 'removed_non_target': chosen, 'token_count_preserved': True}
        if additive_details:
            plans[f'D_target_add_{stage_label}'] = (additive, {'type': 'full_target_addition', 'edits': additive_details,
                'interpretation': 'Descriptive restoration; no equal-count control is claimed.'})
        if feasible_target:
            plans[f'D_target_{stage_label}'] = (target_plan, {'type': 'full_target_reallocation', 'edits': target_details})
        # Matched feasible dose, paired seeds: target vs non-target additions
        # use the same K and identical removed non-target cells.
        for seed in SEEDS:
            tgt = copy.deepcopy(masks)
            ctl = copy.deepcopy(masks)
            details = {'target': {}, 'control': {}}
            for qid in TARGETS:
                for stage in stages:
                    g = geometry[qid][str(stage)]
                    k = min(len(g['target_dropped']), len(g['non_target_dropped']), len(g['non_target_kept']))
                    if k == 0:
                        details['target'][f'{qid}/{stage}'] = {'dose': 0, 'estimable': False}
                        details['control'][f'{qid}/{stage}'] = {'dose': 0, 'estimable': False}
                        continue
                    rng = np.random.default_rng(seed + int(qid)*100 + stage)
                    target_add = sorted(rng.choice(g['target_dropped'], k, replace=False).tolist())
                    control_add = sorted(rng.choice(g['non_target_dropped'], k, replace=False).tolist())
                    remove = sorted(rng.choice(g['non_target_kept'], k, replace=False).tolist())
                    item = arrays[qid][stage]
                    tgt[qid][str(stage)] = map_consumed_edits(item['applied_mask'], item['consumed_mask'], target_add, remove).tolist()
                    ctl[qid][str(stage)] = map_consumed_edits(item['applied_mask'], item['consumed_mask'], control_add, remove).tolist()
                    details['target'][f'{qid}/{stage}'] = {'dose': k, 'added_target': target_add, 'removed_non_target': remove, 'estimable': True}
                    details['control'][f'{qid}/{stage}'] = {'dose': k, 'added_non_target': control_add, 'removed_non_target': remove, 'estimable': True}
            plans[f'D_target_matched_{stage_label}_{seed}'] = (tgt, {'type': 'matched_dose_target_swap', 'seed': seed, **details})
            plans[f'D_random_matched_{stage_label}_{seed}'] = (ctl, {'type': 'matched_dose_non_target_swap', 'seed': seed, **details})
        # Restore complete stage(s) only for target-retained cases.
        full = copy.deepcopy(masks)
        for qid in CONTEXT:
            for stage in stages:
                full[qid][str(stage)] = np.ones_like(arrays[qid][stage]['applied_mask']).tolist()
        plans[f'D_full_{stage_label}'] = (full, {'type': 'full_stage_restoration', 'modified_ids': list(CONTEXT), 'stages': stages})

    for name, (value, details) in plans.items():
        core.write(root / f'{name}.json', {'masks': value, 'design': details,
            'baseline_records_sha256': core.digest(study / 'D1/records.json')})
    core.write(study / 'frozen_plans.json', {p.name: core.digest(p) for p in sorted(root.glob('*.json'))})
    print(f'Frozen {len(plans)} plans; no inference. See plans/geometry.json and plans/*.json.')


def check_gate(study):
    core.check(core.read(study / 'D_replay/intervention_validation.json')['status'] == 'PASS', 'replay masks were not validated')
    _, rows = verify(study)
    expected = {str(r['question_id']): r for r in rows}
    base = core.capture_records(study / 'D1', 'D', expected)
    replay = core.capture_records(study / 'D_replay', 'D', expected)
    failed = []
    differences = {}
    for qid in base:
        fields = [k for k in base[qid] if base[qid][k] != replay[qid].get(k)]
        if fields:
            differences[qid] = fields
        if base[qid]['mask_hashes'] != replay[qid]['mask_hashes']:
            failed.append(qid)
        if qid in PRIMARY and any(base[qid].get(k) != replay[qid].get(k) for k in ('prediction','final_token_ids','prompt','generation_config','image_transform','rank')):
            failed.append(qid)
    result = {'status': 'FAIL' if failed else 'PASS', 'failed_ids': sorted(set(failed)), 'differences': differences,
              'primary_ids': list(PRIMARY), 'rule': 'Exact all-row masks; exact primary input/prediction/token records; margins and selection signals reported separately.'}
    core.write(study / 'replay_gate.json', result)
    core.check(result['status'] == 'PASS', f'replay gate failed for {result["failed_ids"]}')


def run_group(study, prefix, port):
    # Replay creates the artifact required by check_gate, so gate it afterward.
    if prefix != 'D_replay':
        check_gate(study)
    frozen = core.read(study / 'frozen_plans.json')
    selected = sorted(Path(n).stem for n in frozen if Path(n).stem.startswith(prefix))
    for index, label in enumerate(selected):
        intervention = study / 'plans' / f'{label}.json'
        core.execute(study, label, port + index, intervention)
    if prefix == 'D_replay':
        check_gate(study)


def report(study):
    check_gate(study)
    _, rows = verify(study)
    expected = {str(r['question_id']): r for r in rows}
    baseline = core.capture_records(study / 'D_replay', 'D', expected)
    result, missing = [], []
    frozen = core.read(study / 'frozen_plans.json')
    for filename, sha in sorted(frozen.items()):
        label = Path(filename).stem
        if label == 'D_replay':
            continue
        run = study / label
        validation = run / 'intervention_validation.json'
        if not validation.exists():
            missing.append(label)
            continue
        core.check(core.digest(study / 'plans' / filename) == sha, f'{label}: plan hash changed')
        core.check(core.read(validation)['plan_sha256'] == sha, f'{label}: validation plan hash mismatch')
        records = core.capture_records(run, 'D', expected)
        complete = core.read(run / 'run_complete.json')
        core.check(complete.get('status') == 'PASS', f'{label}: run completion missing')
        for output, output_sha in complete['output_hashes'].items():
            core.check(core.digest(run / output) == output_sha, f'{label}: output changed: {output}')
        core.check(records == core.read(run / 'records.json'), f'{label}: record/artifact mismatch')
        plan = core.read(study / 'plans' / filename)
        for qid in PRIMARY:
            row = records[qid]
            result.append({'run': label, 'qid': qid, 'ground_truth': expected[qid]['ground_truth'],
                'baseline_prediction': baseline[qid]['prediction'], 'prediction': row['prediction'],
                'correct': row['prediction'] == expected[qid]['ground_truth'],
                'baseline_margin': baseline[qid]['yes_no_margin'], 'margin': row['yes_no_margin'],
                'margin_delta': row['yes_no_margin'] - baseline[qid]['yes_no_margin'],
                'baseline_mask_counts': baseline[qid]['consumed_counts'], 'mask_counts': row['consumed_counts'],
                'design': plan['design']})
    by_run_qid = {(r['run'], r['qid']): r for r in result}
    paired = []
    for stage_label, stages in [('sneg1', [-1]), ('s0', [0]), ('s1', [1]), ('all', list(STAGES))]:
        for seed in SEEDS:
            target_label = f'D_target_matched_{stage_label}_{seed}'
            control_label = f'D_random_matched_{stage_label}_{seed}'
            for qid in TARGETS:
                target_row = by_run_qid.get((target_label, qid))
                control_row = by_run_qid.get((control_label, qid))
                if target_row is None or control_row is None:
                    continue
                paired.append({'stage_set': stage_label, 'seed': seed, 'qid': qid,
                    'target_dose': target_row['design']['target'].get(f'{qid}/{stages[0]}', {}).get('dose')
                        if len(stages) == 1 else {str(s): target_row['design']['target'].get(f'{qid}/{s}', {}).get('dose') for s in stages},
                    'target_prediction': target_row['prediction'], 'target_correct': target_row['correct'],
                    'control_prediction': control_row['prediction'], 'control_correct': control_row['correct'],
                    'target_margin_delta': target_row['margin_delta'], 'control_margin_delta': control_row['margin_delta'],
                    'target_minus_control_margin_delta': target_row['margin_delta'] - control_row['margin_delta']})
    core.write(study / 'intervention_report.json', {'status': 'INCOMPLETE' if missing else 'COMPLETE',
        'missing_runs': missing, 'primary_ids': list(PRIMARY), 'instability_case_1025': 'retained in runs, excluded from primary causal conclusions',
        'results': result, 'paired_matched_dose_comparisons': paired,
        'limitations': ['Selected cases only.', 'Target boxes are imperfect object proxies.',
        'Matched-dose non-target swaps are feasible-dose controls; dose zero is reported not estimable.',
        'Full-dose target reallocation is compared to baseline if non-target additions are insufficient.',
        'First generated token masks are intervened; later tokens remain adaptive.',
        'Control additions are not matched on spatial position or selector score.']})
    print('Report:', study / 'intervention_report.json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'freeze', 'replay', 'rescue', 'controls', 'context', 'report'])
    p.add_argument('--study', type=Path, default=DEFAULT)
    p.add_argument('--port-base', type=int, default=29971)
    args = p.parse_args()
    core.configure_runtime()
    core.check(not os.environ.get('SECOND_INTERVENTION_PLAN'), 'unset SECOND_INTERVENTION_PLAN')
    study = args.study.resolve()
    if args.action == 'prepare': prep(study)
    elif args.action == 'freeze': freeze(study)
    elif args.action == 'replay': run_group(study, 'D_replay', args.port_base)
    elif args.action == 'rescue': run_group(study, 'D_target_', args.port_base)
    elif args.action == 'controls': run_group(study, 'D_random_', args.port_base)
    elif args.action == 'context': run_group(study, 'D_full_', args.port_base)
    elif args.action == 'report': report(study)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f'FAIL: {exc}')

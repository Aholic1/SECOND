"""Separate stable-case causal tests from 1025 instability; reuse immutable baselines."""
import argparse
import os
from pathlib import Path
import step6_interventions as core

ROOT = core.ROOT
DEFAULT = ROOT / 'audit/runs/step6_causal_v2'
OLD = ROOT / 'audit/runs/step6_interventions_v1'
PRIMARY = ('2253', '2559', '311', '645', '1043')
core.CONTEXT = ('311', '645', '1043')
original_verify = core.verify


def compare_replay(a, b):
    """No numerical tolerance: exact primary outcomes/masks; signal hashes reported separately."""
    core.check(set(a) == set(b), 'replay question set differs')
    failures, differences = [], {}
    for qid in a:
        fields = sorted(k for k in set(a[qid]) | set(b[qid]) if a[qid].get(k) != b[qid].get(k))
        if fields:
            differences[qid] = fields
        required_primary = {'prediction', 'final_token_ids', 'prompt',
                            'generation_config', 'image_transform', 'rank'}
        if qid in PRIMARY and any(a[qid].get(k) != b[qid].get(k) for k in required_primary):
            failures.append(qid)
        if a[qid]['mask_hashes'] != b[qid]['mask_hashes']:
            failures.append(qid)
        for key in ('prompt', 'generation_config', 'image_transform', 'rank'):
            if a[qid].get(key) != b[qid].get(key):
                failures.append(qid)
    return {'status': 'FAIL' if failures else 'PASS',
            'failed_ids': sorted(set(failures)), 'different_fields_by_id': differences,
            'primary_ids': list(PRIMARY), 'separate_instability_case': '1025',
            'rule': 'Exact primary predictions/final tokens; exact replay masks and input metadata for all 48; margin and signal differences disclosed separately.'}


def verify(study):
    plan, rows = original_verify(study)
    amendment = core.read(study / 'causal_amendment.json')
    core.check(core.digest(Path(__file__)) == amendment['driver_sha256'], 'v2 driver changed after prepare')
    for label, expected in amendment['baseline_completion_sha256'].items():
        core.check(core.digest(study / label / 'run_complete.json') == expected, 'baseline completion changed')
    return plan, rows


def prepare(study, baseline):
    core.check(not study.exists(), 'new study already exists; do not overwrite')
    plan = core.read(baseline / 'study_plan.json')
    core.check(core.state() == plan['step6_state'], 'source/runtime differs from existing baseline; investigate before reuse')
    rows, identity = core.load_bundle(core.BUNDLE, verify_images=True)
    core.check(identity == plan['cohort'], 'frozen cohort changed')
    expected = {str(r['question_id']): r for r in rows}
    records, hashes = {}, {}
    for label in ('C1', 'D1', 'D2', 'C2'):
        run = baseline / label
        complete = core.read(run / 'run_complete.json')
        core.check(complete.get('status') == 'PASS', f'{label}: incomplete')
        core.check(complete['snapshot_sha256'] == plan['snapshot_sha256'], 'baseline snapshot differs')
        for name, sha in complete['output_hashes'].items():
            core.check(core.digest(run / name) == sha, f'{label}: output changed: {name}')
        records[label] = core.capture_records(run, label[0], expected)
        core.check(records[label] == core.read(run / 'records.json'), 'baseline artifacts differ')
        hashes[label] = core.digest(run / 'run_complete.json')
    for qid in PRIMARY:
        for condition in ('C', 'D'):
            core.check(records[condition+'1'][qid] == records[condition+'2'][qid], f'{qid}/{condition}: baseline is not repeatable')
        core.check(records['C1'][qid]['prediction'] == 'yes' and records['D1'][qid]['prediction'] == 'no', f'{qid}: degradation not reproduced')
    study.mkdir(parents=True)
    core.write(study / 'study_plan.json', plan)
    core.write(study / 'causal_amendment.json', {
        'driver_sha256': core.digest(Path(__file__)), 'baseline_study': str(baseline.resolve()),
        'baseline_completion_sha256': hashes, 'primary_ids': list(PRIMARY),
        'excluded_from_primary_interpretation': {'1025': 'Observed actual mask instability; retained in full 48-question execution cohort.'},
        'design': 'First-token fixed masks; later tokens adaptive; three predeclared random seeds; selected-case inference only.'})
    for label in hashes:
        (study / label).symlink_to((baseline / label).resolve(), target_is_directory=True)
    print('Prepared independent v2 study; baseline directories linked read-only by convention. No inference.')


def replay_gate(study):
    core.check(core.read(study / 'D_replay/intervention_validation.json')['status'] == 'PASS', 'requested replay masks were not validated')
    _, rows = verify(study)
    expected = {str(r['question_id']): r for r in rows}
    a = core.capture_records(study / 'D1', 'D', expected)
    b = core.capture_records(study / 'D_replay', 'D', expected)
    result = compare_replay(a, b)
    core.write(study / 'replay_gate.json', result)
    core.check(result['status'] == 'PASS', f'replay failed for {result["failed_ids"]}; stop before causal interpretation')


def freeze(study):
    core.freeze(study)
    plans = study / 'plans'
    # Context-rescue runs must alter only the three target-complete cases.
    baseline_masks = core.read(plans / 'D_replay.json')['masks']
    for path in plans.glob('D_full_*.json'):
        value = core.read(path)
        for qid in core.FOCAL:
            value['masks'][qid] = baseline_masks[qid]
        value['design'] = {
            'restored_stages_for_context_cases': list(core.STAGES),
            'edited_question_ids': list(core.CONTEXT),
            'target-loss cases unchanged': list(core.FOCAL),
            'description': 'Restore the named stage masks only for target-complete cases; other cohort rows replay D1.'}
        core.write(path, value)
    core.write(study / 'frozen_plans.json', {p.name: core.digest(p) for p in sorted(plans.glob('*.json'))})
    print('V2 context plans frozen; target-loss cases remain unchanged in context runs.')


def report(study):
    core.report(study)
    output = study / 'intervention_report.json'
    result = core.read(output)
    rows, _ = core.load_bundle(core.BUNDLE)
    expected = {str(r['question_id']): r for r in rows}
    baseline = core.capture_records(study / 'D_replay', 'D', expected)
    for item in result['results']:
        base = baseline[item['qid']]
        item['baseline_prediction'] = base['prediction']
        item['baseline_margin'] = base['yes_no_margin']
        item['margin_delta_from_replay'] = item['margin'] - base['yes_no_margin']
        item['prediction_changed_from_replay'] = item['prediction'] != base['prediction']
        item['ground_truth'] = expected[item['qid']]['ground_truth']
        item['correct_after_intervention'] = item['prediction'] == item['ground_truth']
    result['primary_ids'] = list(PRIMARY)
    result['excluded_instability_case'] = {
        'question_id': '1025',
        'handling': 'Retained in all 48-item runs and D1 mask replay; excluded from primary mechanism comparisons because D1/D2 masks differed.'}
    result['replay_gate'] = core.read(study / 'replay_gate.json')
    result['interpretation_limit'] = (
        'A target-rescue benefit over equal-count controls supports a target-location effect in these selected cases. '
        'Random controls are not spatial/score matched; three seeds are exploratory, not a population estimate.')
    core.write(output, result)


core.verify = verify
core.replay_gate = replay_gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'freeze', 'replay', 'rescue', 'controls', 'context', 'report'])
    parser.add_argument('--study', type=Path, default=DEFAULT)
    parser.add_argument('--baseline-study', type=Path, default=OLD)
    parser.add_argument('--port-base', type=int, default=29951)
    args = parser.parse_args()
    core.configure_runtime()
    core.check(not os.environ.get('SECOND_INTERVENTION_PLAN'), 'unset SECOND_INTERVENTION_PLAN first')
    study = args.study.resolve()
    if args.action == 'prepare':
        prepare(study, args.baseline_study.resolve())
    elif args.action == 'freeze':
        freeze(study)
    elif args.action == 'report':
        report(study)
    else:
        core.interventions(study, args.action, args.port_base)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f'FAIL: {exc}')

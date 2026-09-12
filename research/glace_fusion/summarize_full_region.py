import argparse
import json
from pathlib import Path

import numpy as np


def groups(rows):
    return dict(all_spatial=[r for r in rows],
        orientation_supported=[r for r in rows if r['in_orientation_support']],
        new_supported=[r for r in rows if r['in_orientation_support'] and not r['previous_probe']],
        previous_probe=[r for r in rows if r['previous_probe']],
        outside_orientation_support=[r for r in rows if not r['in_orientation_support']])


def paired_intervals(rows, method):
    base = np.asarray([r['errors']['v1_two_stage'] for r in rows])
    candidate = np.asarray([r['errors'][method] for r in rows])
    b = (base[:, 0] < 1) & (base[:, 1] < 2)
    c = (candidate[:, 0] < 1) & (candidate[:, 1] < 2)
    blocks = np.array_split(np.arange(len(rows)), min(8, len(rows)))
    draws = np.random.default_rng(2089).integers(0, len(blocks), (10000, len(blocks)))
    delta = c.astype(int) - b.astype(int)
    sums = np.array([delta[x].sum() for x in blocks])
    counts = np.array([len(x) for x in blocks])
    bootstrap = sums[draws].sum(1) / counts[draws].sum(1)
    return dict(recall_1m_2deg_delta=float(delta.mean()), ci95=np.quantile(bootstrap, [.025,.975]).tolist(),
        rescued=int((~b & c).sum()), harmed=int((b & ~c).sum()),
        better_translation=int((candidate[:, 0] < base[:, 0] - .001).sum()),
        worse_translation=int((candidate[:, 0] > base[:, 0] + .001).sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    refined_folder = args.root / 'same_v1_refinement_so3'
    if not refined_folder.exists():
        refined_folder = args.root / 'same_v1_refinement'
    rows = json.loads((args.root / 'rows.json').read_text())
    result = dict(plan=json.loads((args.root / 'plan.json').read_text()), synthetic={}, real={})
    intervals = [('translation','0.2:0.5'), ('translation','0.5:1.0'), ('rotation','0.5:1.0')]
    for label in ['stage1','improved']:
        records = json.loads((args.root / label / 'records.json').read_text())
        by_ts = {str(r['timestamp_us']): r for r in records}
        result['synthetic'][label] = {}
        for group, selected in groups(rows).items():
            if not selected:
                continue
            samples = [by_ts[r['image']] for r in selected]
            out = dict(frames=len(samples), q10=float(np.mean([r['gt_evidence']['q_C'] for r in samples])), intervals={})
            for mode, key in intervals:
                counts = {k: sum(r['modalities'][mode]['adjacent'][key][k] for r in samples) for k in ['correct','ties','pairs']}
                counts['strict'] = counts['correct'] / counts['pairs']
                counts['half_credit'] = (counts['correct'] + .5 * counts['ties']) / counts['pairs']
                out['intervals'][mode+'_'+key] = counts
            result['synthetic'][label][group] = out
    for version, report_path, records_path in [
        ('direct', args.root / 'balanced_report.json', args.root / 'balanced_records.json'),
        ('same_v1_refinement', refined_folder / 'report.json', refined_folder / 'records.json')]:
        report = json.loads(report_path.read_text())
        records = json.loads(records_path.read_text())
        result['real'][version] = report
        result['real'][version]['paired_intervals'] = {}
        for group, selected in groups(records).items():
            if not selected:
                continue
            methods = ['camera_improved', 'balanced_improved'] if version == 'direct' else [
                'camera_improved_then_v1', 'balanced_improved_then_v1']
            result['real'][version]['paired_intervals'][group] = {m: paired_intervals(selected,m) for m in methods}
    result['uncertainty_scope'] = '8 contiguous time blocks, 10000 paired bootstrap draws; one region/date, not whole NCLT uncertainty'
    (args.root / 'summary.json').write_text(json.dumps(result, indent=2))
    print('SUMMARY_COMPLETE')


if __name__ == '__main__':
    main()

import argparse
import json
from pathlib import Path

import numpy as np


def auc(labels, scores):
    labels, scores = np.asarray(labels, bool), np.asarray(scores, float)
    positive, negative = scores[labels], scores[~labels]
    if not len(positive) or not len(negative):
        return None
    return float(((positive[:, None] > negative[None]).astype(float)
                  + .5 * (positive[:, None] == negative[None])).mean())


def evaluate(rows, protocol, methods=('feature_cosine', 'depth_agreement', 'composite')):
    keys = protocol['keys_hard'].astype(str)
    truth = dict(zip(keys, protocol['truth_hard']))
    by_date = {}
    for date in ['all'] + sorted({row['key'].split('/')[0] for row in rows}):
        subset = rows if date == 'all' else [row for row in rows if row['key'].startswith(date + '/')]
        report = dict(frames=len(subset), methods={})
        baseline = np.array([np.linalg.norm(np.asarray(row['source_pose']).reshape(4, 4)[:3, 3]
                                           - truth[row['key']]) for row in subset])
        report['original_fair'] = dict(correct_5m=int((baseline < 5).sum()),
            mean_error_m=float(baseline.mean()), total_error_m=float(baseline.sum()))
        report['oracle_candidate_correct_frames'] = sum(any(
            np.linalg.norm(np.asarray(h['original']['pose']).reshape(4, 4)[:3, 3] - truth[row['key']]) < 5
            for h in row['hypotheses']) for row in subset)
        for method in methods:
            labels, scores, frame_auc, errors, times = [], [], [], [], []
            covered, evidence_good, evidence_frames = 0, 0, 0
            for row in subset:
                candidates = []
                fl, fs = [], []
                for hypothesis in row['hypotheses']:
                    value = hypothesis['evidence'][method]
                    if value is None:
                        continue
                    error = float(np.linalg.norm(np.asarray(hypothesis['original']['pose']).reshape(4, 4)[:3, 3]
                                                 - truth[row['key']]))
                    candidates.append((float(value), float(hypothesis['original']['score']), error))
                    labels.append(error < 5)
                    scores.append(float(value))
                    fl.append(error < 5)
                    fs.append(float(value))
                    times.append(hypothesis['evidence']['milliseconds'])
                if candidates:
                    covered += 1
                    errors.append(max(candidates, key=lambda pair: pair[:2])[2])
                    evidence_good += any(fl)
                    evidence_frames += 1
                else:
                    errors.append(float(np.linalg.norm(np.asarray(row['source_pose']).reshape(4, 4)[:3, 3]
                                                       - truth[row['key']])))
                paired = auc(fl, fs)
                if paired is not None:
                    frame_auc.append(paired)
            errors = np.asarray(errors)
            report['methods'][method] = dict(evidence_frames=covered,
                valid_candidates=len(scores), good_valid_candidates=int(sum(labels)),
                evidence_oracle_correct_frames=evidence_good,
                candidate_auc=auc(labels, scores),
                mean_within_frame_auc=float(np.mean(frame_auc)) if frame_auc else None,
                mixed_evidence_frames=len(frame_auc), correct_5m=int((errors < 5).sum()),
                mean_error_m=float(errors.mean()), total_error_m=float(errors.sum()),
                saved_vs_original_fair_m=float((baseline - errors).sum()),
                newly_correct=int(((baseline >= 5) & (errors < 5)).sum()),
                newly_wrong=int(((baseline < 5) & (errors >= 5)).sum()),
                worsened_over_5m=int((errors - baseline > 5).sum()),
                mean_scoring_ms_per_candidate=float(np.mean(times)) if times else None)
        by_date[date] = report
    return by_date


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--methods', nargs='+', default=['feature_cosine', 'depth_agreement', 'composite'])
    args = parser.parse_args()
    directory = args.directory.resolve(strict=True)
    if directory.parent != Path('/data2/zhangcy') or not directory.name.startswith(('e2e-projection-probe-', 'e2e-roma-probe-')):
        raise ValueError('isolated probe directory required')
    rows = [json.loads(line) for line in (directory / 'results' / 'evidence.jsonl').read_text().splitlines()]
    protocol = np.load('/home/zhangcy/dsh-work/vpr/out/protocol.npz', allow_pickle=False)
    report = dict(scope='offline evaluator only; fixed visual candidates, no MoE arbitration, no threshold fitting',
                  tie_policy='new score ties resolved by original fair score',
                  results=evaluate(rows, protocol, args.methods))
    with (directory / 'results' / 'evaluation.json').open('x') as output:
        json.dump(report, output, indent=2, allow_nan=False)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

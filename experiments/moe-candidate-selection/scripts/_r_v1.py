"""V1: run the live visual branch on 5 frames that have frozen visual candidates and
compare the live pose/score with the frozen best candidate."""
import json
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = '1'
PROJECT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
VERIFY = '/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/verify-frozen_nopruneoptREF'
sys.path.insert(0, PROJECT)

import numpy as np


def rot_deg(a, b):
    r = np.asarray(a, float)[:3, :3] @ np.asarray(b, float)[:3, :3].T
    c = np.clip((np.trace(r) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(c)))


def trans(a, b):
    return float(np.linalg.norm(np.asarray(a, float)[:3, 3] - np.asarray(b, float)[:3, 3]))


FRAMES = json.load(open('/home/zhangcy/.dsh_scratch/vb/v1_candidates.json'))[:5]
queries = json.loads(open('/data2/zhangcy/sp-lightglue-hard1502-20261013-v1/queries-provider.json').read())['queries']
by_key = {q['key']: q['image_path'] for q in queries}

import visual_branch as VB
branch = VB.VisualBranch(device='cuda', arm='frozen_noprune', verbose=True)
print('[boot] branch ready', flush=True)

out = []
for row in FRAMES:
    i, key = int(row['i']), row['key']
    frozen = json.loads(open(os.path.join(VERIFY, '%06d.json' % i)).read())
    poses = frozen['poses']
    base = poses['baseline_V7']
    bs = (base['fair'] or {})['geometry_score']
    best = None
    for name, cand in poses.items():
        if not name.startswith('candidate_'):
            continue
        s = (cand['fair'] or {}).get('geometry_score')
        if s is None:
            continue
        if best is None or s > best[0]:
            best = (float(s), name, cand)
    image_path = by_key[key]
    hyp = branch.hypothesize(key, image_path, budget=16,
                             pose_prior=np.asarray(base['pose']))
    rec = dict(index=i, key=key, frozen_best_score=best[0], frozen_best_name=best[1],
               frozen_base_score=bs,
               frozen_best_trans_vs_base=trans(best[2]['pose'], base['pose']),
               live=None)
    if hyp is None:
        rec['live'] = 'NO_HYPOTHESIS'
    else:
        det = hyp['detail']
        rec['live'] = dict(score=hyp['score'], verified=bool(hyp['verified']),
                           channel=hyp['channel'], n_candidates=hyp['n_candidates'],
                           seconds=round(hyp['seconds'], 3),
                           trans_vs_frozen_best=trans(hyp['pose'], best[2]['pose']),
                           rot_vs_frozen_best=rot_deg(hyp['pose'], best[2]['pose']),
                           trans_vs_frozen_base=trans(hyp['pose'], base['pose']),
                           dscore=round(hyp['score'] - best[0], 6),
                           frozen_refs=frozen['legacy_excluded_references'],
                           live_retrieval=det['retrieval']['visual'] + det['retrieval']['position'],
                           live_refs=hyp['reference_ids'],
                           seconds_breakdown=det['seconds'],
                           capped=det['capped'], pnp_calls=det['pnp_calls'])
    out.append(rec)
    print('[v1] ' + json.dumps(rec), flush=True)

json.dump(out, open('/home/zhangcy/.dsh_scratch/vb/v1_result.json', 'w'), indent=1)
print('[v1-done]')

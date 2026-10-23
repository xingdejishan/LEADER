"""For the 55 dead gated frames: do hypotheses exist at all, and do they fail why?"""
import json
import os

import numpy as np

MOE = '/home/zhangcy/projects/e2e-moe-20261020-v1'
MATCH = ('/data2/zhangcy/sp-lightglue-hard1502-20261013-v1/prov/'
         'match-frozen_nopruneopt/%06d.json')
VERIFY = ('/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/'
          'verify-frozen_nopruneoptREF/%06d.json')


def main():
    diag = [json.loads(l) for l in open(os.path.join(MOE, 'results', 'three_route_diag.jsonl'))]
    ratio = np.array([d.get('ratio', np.nan) for d in diag])
    visual = np.array([bool(d.get('visual')) for d in diag])
    gated = np.isfinite(ratio) & (ratio < 0.21)
    dead = gated & ~visual

    nh, nq, npair, nfull = [], [], [], []
    for i in np.flatnonzero(dead):
        m = json.load(open(MATCH % i))
        v = json.load(open(VERIFY % i))['poses']
        hyps = m.get('hypotheses') or []
        nh.append(len(hyps))
        nq.append(int(m.get('per_keypoint') or 0))
        npair.append(len(m.get('pair_references') or []))
        nfull.append(len([n for n in v if n.startswith('candidate_')]))

    nh = np.array(nh)
    print('dead frames: %d' % len(nh))
    print('hypotheses per frame: median %d   frames with 0: %d   with >=1: %d'
          % (int(np.median(nh)), int((nh == 0).sum()), int((nh >= 1).sum())))
    print('pair references per frame: median %d' % int(np.median(npair)))
    print('candidates written to the verify table: median %d' % int(np.median(nfull)))

    # among those with hypotheses, how good are they?
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    gt = np.asarray(w['hard_gt'], float)
    pos = {k: i for i, k in enumerate(w['hard_keys'])}

    from scipy.spatial.transform import Rotation
    best_pos, best_rot, nq_list = [], [], []
    for i in np.flatnonzero(dead):
        m = json.load(open(MATCH % i))
        hyps = m.get('hypotheses') or []
        if not hyps:
            continue
        bp, br = 1e9, 1e9
        for h in hyps:
            P = np.asarray(h['pose'], float)
            bp = min(bp, float(np.linalg.norm(P[:3, 3] - gt[i][:3, 3])))
            Rd = P[:3, :3].T @ gt[i][:3, :3]
            br = min(br, float(np.degrees(np.linalg.norm(Rotation.from_matrix(Rd).as_rotvec()))))
        best_pos.append(bp); best_rot.append(br); nq_list.append(len(hyps))
    if best_pos:
        a, b = np.array(best_pos), np.array(best_rot)
        print()
        print('among the %d dead frames that DID have hypotheses,' % len(a))
        print('   best hypothesis position error: median %.1f m   <2 m on %d   <5 m on %d'
              % (np.median(a), int((a < 2).sum()), int((a < 5).sum())))
        print('   best hypothesis rotation error: median %.1f deg  <2 deg on %d'
              % (np.median(b), int((b < 2).sum())))


if __name__ == '__main__':
    main()

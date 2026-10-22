"""EXP S: is the image-rotation error a property of the place (i.e. of the map)?

The image estimate correlates 0.85 with the front end's error, which is odd because
the front end's own error does NOT repeat at the same place on another day.  So the
shared part must be pinned down.

For pairs of frames from different days that sit within 1.5 m of each other, compare
  * the image-rotation error of the two frames,
  * the front-end error of the same two frames.
A map-borne error repeats at a place; a frame-borne error does not.
"""
import json
import os
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '3')

from collections import OrderedDict

import numpy as np
from scipy.spatial import cKDTree

PROJECT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001'
TRUTH_DIR = os.path.join(DATA, 'evaluator-only-truth')
LIB = '/data2/zhangcy/sp-lightglue-hard1502-20261013-v1/library.json'
sys.path.insert(0, PROJECT)
import visual_branch as VB      # noqa: E402

N_PAIRS = int(os.environ.get('N', '40'))
R_BC = np.asarray(json.load(open(LIB))['T_BC'], float)[:3, :3]


def logv(R):
    a = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = np.sin(a)
    return v * a / (2 * s) if s > 1e-8 else v / 2


def ra(A, B):
    R = np.swapaxes(A, -1, -2) @ B
    return np.degrees(np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


def kabsch(b, d, w):
    H = (b * w[:, None]).T @ d
    U, S, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


class Branch(VB.VisualBranch):
    def __init__(self, *a, **kw):
        kw.setdefault('device', 'cuda')
        super(Branch, self).__init__(*a, **kw)
        self._features = OrderedDict()
        self._projections = OrderedDict()

    def _cached_features(self, image_path, tag):
        cache = self._features
        if tag in cache:
            value = cache.pop(tag)
            cache[tag] = value
            return value
        value = self.extract(image_path)
        cache[tag] = value
        while len(cache) > 64:
            cache.popitem(last=False)
        return value


def image_error(branch, i, t0, gt, K, rng):
    key = None
    date, stamp = None, None
    return None


def main():
    truth = {}
    for f in os.listdir(TRUTH_DIR):
        d = f[:-5]
        for ts, v in json.load(open(os.path.join(TRUTH_DIR, f))).items():
            truth[(d, str(ts))] = np.asarray(v['T_WB'], float)
    Z = np.load(PROJECT + '/results/full-online-final.npz', allow_pickle=True)
    keys = [str(k) for k in Z['keys']]
    t0 = np.asarray(Z['t0'], float)
    gt = np.stack([truth[tuple(k.split('/'))] for k in keys])
    e0 = np.linalg.norm(t0[:, :3, 3] - gt[:, :3, 3], axis=1)
    dates = np.array([k.split('/')[0] for k in keys])

    # candidate pairs: close in space, different day, both well localised
    pos2 = t0[:, :2, 3]
    tree = cKDTree(pos2)
    nbrs = tree.query_ball_point(pos2, 1.5)
    rng0 = np.random.default_rng(3)
    pairs = []
    for i in rng0.permutation(len(keys))[:8000]:
        if e0[i] > 0.5:
            continue
        for j in nbrs[i]:
            if j <= i or dates[j] == dates[i] or e0[j] > 0.5:
                continue
            pairs.append((int(i), int(j)))
            break
        if len(pairs) >= N_PAIRS * 3:
            break
    pairs = pairs[:N_PAIRS]
    print('pairs %d' % len(pairs), flush=True)

    branch = Branch(device='cuda')
    K = branch.C.K
    rng = np.random.default_rng(7)

    def solve_one(i):
        date, stamp = keys[i].split('/')
        image_path = os.path.join(DATA, 'NCLT', date, 'cam5', stamp + '.jpg')
        T0 = t0[i]
        refs, _ = branch.retrieve(image_path, T0)
        q_model, candidates, _, _ = branch._match_frame(image_path, refs)
        capped = branch._cap(candidates)
        if len(capped) < 8:
            return None
        ids = np.array([c[0] for c in capped])
        world = np.array([branch._projection(branch.references[c[2]])['world'][c[3]]
                          for c in capped])
        pixels = branch.C.model_to_raw(branch.matrix, q_model[ids])
        C_cw0 = np.linalg.inv(T0 @ branch.T_BC)
        R0c = C_cw0[:3, :3].copy()
        centre = -R0c.T @ C_cw0[:3, 3]
        v = world - centre[None, :]
        nn = np.linalg.norm(v, axis=1)
        keep = nn > 1e-6
        v = v[keep] / nn[keep, None]
        pixels = pixels[keep]
        rays = np.linalg.inv(K) @ np.column_stack([pixels, np.ones(len(pixels))]).T
        rays /= np.maximum(np.linalg.norm(rays, axis=0), 1e-12)
        d = rays.T
        Rc = R0c.copy()
        for gate in (6.0, 4.0, 2.5, 1.5, 1.0, 1.0, 1.0):
            ang = np.degrees(np.arccos(np.clip(((v @ Rc.T) * d).sum(1), -1, 1)))
            w = (ang <= gate).astype(float)
            if w.sum() < 8:
                return None
            Rc = kabsch(v, d, w)
        Rb = (R_BC @ Rc).T
        return Rb

    rows = []
    for n, (i, j) in enumerate(pairs):
        Ri = solve_one(i)
        Rj = solve_one(j)
        if Ri is None or Rj is None:
            continue
        ei = logv(Ri.T @ gt[i, :3, :3])
        ej = logv(Rj.T @ gt[j, :3, :3])
        pi = logv(t0[i, :3, :3].T @ gt[i, :3, :3])
        pj = logv(t0[j, :3, :3].T @ gt[j, :3, :3])
        rows.append(dict(i=i, j=j,
                         d_img=float(np.degrees(np.linalg.norm(ei - ej))),
                         d_prior=float(np.degrees(np.linalg.norm(pi - pj))),
                         m_img=float(np.degrees(0.5 * (np.linalg.norm(ei) + np.linalg.norm(ej)))),
                         m_prior=float(np.degrees(0.5 * (np.linalg.norm(pi) + np.linalg.norm(pj))))))
        if (n + 1) % 10 == 0:
            print('  %d/%d pairs, usable %d' % (n + 1, len(pairs), len(rows)), flush=True)

    if not rows:
        print('no usable pairs')
        return
    d_img = np.array([r['d_img'] for r in rows])
    d_pri = np.array([r['d_prior'] for r in rows])
    m_img = np.array([r['m_img'] for r in rows])
    m_pri = np.array([r['m_prior'] for r in rows])
    print('\n## %d pairs at the same place on different days' % len(rows))
    print('   front end : |e_i - e_j| %.3f deg, level %.3f deg  (independent noise => %.3f)'
          % (d_pri.mean(), m_pri.mean(), np.sqrt(2) * m_pri.mean()))
    print('   image     : |e_i - e_j| %.3f deg, level %.3f deg  (independent noise => %.3f)'
          % (d_img.mean(), m_img.mean(), np.sqrt(2) * m_img.mean()))
    if d_img.mean() < 0.7 * np.sqrt(2) * m_img.mean():
        print('   -> the image error repeats at a place: it comes from the reference map')
    else:
        print('   -> the image error does not repeat at a place: it is frame-borne')
    with open('/home/zhangcy/.dsh_scratch/moe_rot_20261022/expS_place.json', 'w') as fh:
        json.dump(rows, fh, indent=1)


if __name__ == '__main__':
    main()

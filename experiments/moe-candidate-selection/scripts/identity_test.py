"""Can a truth-free selector capture the identity-shortcut headroom (0.229 deg)?

Features per frame (all available at inference, none uses truth):
  dcost   map cost of LEADER's pose minus map cost of V7's pose
  dR      rotation difference between the two poses
  dt      translation difference
  date

Trained with leave-one-date-out logistic regression: the model never sees the date it
is evaluated on.  Applied as "keep LEADER's rotation when the model says LEADER is
better", so MPE and everything else stay as the deployed V7 output.
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

MOE = '/home/zhangcy/projects/e2e-moe-20261020-v1'
HIST = '/data2/zhangcy/native1502-eval-20261015-v1/historical-predictions.npz'
sys.path.insert(0, os.path.join(MOE, 'stages/fast_search'))
sys.path.insert(0, MOE)
import fast_lib as F
import pipeline as P

PTS = 1500
_G = {}


def init():
    frames = json.load(open(os.path.join(MOE, 'inputs/frames.json')))
    refs = json.load(open(os.path.join(MOE, 'inputs/refs.json')))
    cfg = P.Config()
    _G.update(frames=frames, cfg=cfg,
              maps=F.MapSource(refs, np.asarray([np.asarray(r['world_from_body'], float)[:3, 3]
                                                 for r in refs]), radius=cfg.crop_radius))


def cost(tree, scan, T):
    q = scan if len(scan) <= PTS else scan[::max(1, len(scan) // PTS)][:PTS]
    return float(np.minimum(tree.query(q @ T[:3, :3].T + T[:3, 3], k=1, workers=1)[0], 1.0).mean())


def one(args):
    i, lead, v7 = args
    cfg = _G['cfg']
    scan = P.prepare_scan(_G['frames'][i], cfg)
    t = v7[:3, 3]
    mp = _G['maps'].local(t, _G['maps'].nearby(t)[:cfg.reference_count])
    tree = cKDTree(mp)
    cl, cv = cost(tree, scan, lead), cost(tree, scan, v7)
    dR = float(np.degrees(np.linalg.norm(
        Rotation.from_matrix(v7[:3, :3].T @ lead[:3, :3]).as_rotvec())))
    dt = float(np.linalg.norm(v7[:3, 3] - lead[:3, 3]))
    return i, cl, cv, dR, dt


def logistic(X, y, iters=600, lr=0.5):
    X1 = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(X1.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X1 @ w, -30, 30)))
        g = X1.T @ (p - y) / len(y) + 1e-3 * w
        w -= lr * g / max(1.0, np.abs(g).max())
    return w


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    gt = np.asarray(w['hard_gt'], float)
    v7 = np.asarray(w['hard_base'], float)
    keys = [str(k) for k in w['hard_keys']]
    dates = np.asarray(w['dates_hard'])

    z = np.load(HIST, allow_pickle=True)
    hk = [str(k) for k in z['keys']]
    pos = {k: i for i, k in enumerate(hk)}
    lead = np.stack([np.asarray(z['historical_LEADER'][pos[str(k)]], float) for k in keys])

    init()
    with Pool(10, initializer=init) as pool:
        out = pool.map(one, [(i, lead[i], v7[i]) for i in range(len(keys))], chunksize=8)

    cl = np.array([r[1] for r in out]); cv = np.array([r[2] for r in out])
    dR = np.array([r[3] for r in out]); dt = np.array([r[4] for r in out])

    def rot_err(p):
        Rd = np.swapaxes(p[:, :3, :3], -1, -2) @ gt[:, :3, :3]
        return np.degrees(np.linalg.norm(Rotation.from_matrix(Rd).as_rotvec(), axis=1))

    el = rot_err(lead); ev = rot_err(v7)
    y = (ev < el).astype(float)
    print('frames %d   V7 rotation better on %.0f%%' % (len(y), 100 * y.mean()))
    print('LEADER MOE %.4f   V7 MOE %.4f   oracle %.4f' % (el.mean(), ev.mean(), np.minimum(el, ev).mean()))

    X = np.column_stack([cl - cv, dR, dt, np.log1p(np.abs(cl - cv)), np.log1p(dR)])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd

    p = np.zeros(len(y))
    for d in sorted(set(dates)):
        te = dates == d
        tr = ~te
        w = logistic(Xs[tr], y[tr])
        p[te] = 1.0 / (1.0 + np.exp(-np.clip(np.hstack([np.ones((te.sum(), 1)), Xs[te]]) @ w, -30, 30)))
    print('leave-one-date-out accuracy %.3f' % ((p > 0.5) == (y > 0.5)).mean())

    chosen = np.where(p > 0.5, ev, el)
    print('MOE with the learned selector: %.4f  (V7 %.4f, oracle %.4f)' % (chosen.mean(), ev.mean(), np.minimum(el, ev).mean()))
    for thr in (0.3, 0.5, 0.7):
        c = np.where(p > thr, ev, el)
        print('  threshold %.1f -> MOE %.4f  (keeps LEADER on %d frames)'
              % (thr, c.mean(), int((p <= thr).sum())))


if __name__ == '__main__':
    main()

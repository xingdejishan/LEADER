"""EXP Q: is the image rotation genuinely correlated with the front end's error?

The previous test anchored the image solve to the front-end rotation, which can
manufacture correlation by construction.  This run stores the raw 2D-3D
correspondences per frame and then solves the rotation two ways offline:

  anchored   graduated gating seeded from the front-end rotation
  prior-free a plain RANSAC over correspondence triples (no prior at all)

If the prior-free solution is still correlated with the front-end error, the two
share their error and fusion cannot help.  If it is independent, fusion is worth
engineering even when the image is noisier on its own.
"""
import json
import os
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '3')

from collections import OrderedDict

import numpy as np

PROJECT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001'
TRUTH_DIR = os.path.join(DATA, 'evaluator-only-truth')
LIB = '/data2/zhangcy/sp-lightglue-hard1502-20261013-v1/library.json'
sys.path.insert(0, PROJECT)
import visual_branch as VB      # noqa: E402

OUT = '/home/zhangcy/.dsh_scratch/moe_rot_20261022/expQ_corr.npz'
N = int(os.environ.get('N', '80'))
R_BC = np.asarray(json.load(open(LIB))['T_BC'], float)[:3, :3]


def logv(R):
    a = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = np.sin(a)
    return v * a / (2 * s) if s > 1e-8 else v / 2


def expv(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def ra(A, B):
    R = np.swapaxes(A, -1, -2) @ B
    return np.degrees(np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


def kabsch(b, d, w):
    H = (b * w[:, None]).T @ d
    U, S, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def prepare(world, pixels, centre, K):
    v = np.asarray(world, float) - centre[None, :]
    n = np.linalg.norm(v, axis=1)
    keep = n > 1e-6
    v, pixels = v[keep] / n[keep, None], np.asarray(pixels, float)[keep]
    rays = np.linalg.inv(K) @ np.column_stack([pixels, np.ones(len(pixels))]).T
    rays /= np.maximum(np.linalg.norm(rays, axis=0), 1e-12)
    return v, rays.T


def anchored(v, d, R0, gates=(6.0, 4.0, 2.5, 1.5, 1.0, 1.0, 1.0)):
    R = R0.copy()
    for gate in gates:
        ang = np.degrees(np.arccos(np.clip(((v @ R.T) * d).sum(1), -1, 1)))
        w = (ang <= gate).astype(float)
        if w.sum() < 8:
            return None, 0
        R = kabsch(v, d, w)
    ang = np.degrees(np.arccos(np.clip(((v @ R.T) * d).sum(1), -1, 1)))
    return R, int((ang <= 2.0).sum())


def prior_free(v, d, rng, trials=600, gate=2.0):
    n = len(v)
    if n < 12:
        return None, 0
    best = (0, None)
    idx = rng.integers(0, n, size=(trials, 3))
    for t in range(trials):
        a, b, c = idx[t]
        if a == b or b == c or a == c:
            continue
        # the three bearings must not be nearly coplanar-degenerate
        if abs(np.linalg.det(np.stack([d[a], d[b], d[c]]))) < 0.05:
            continue
        if abs(np.linalg.det(np.stack([v[a], v[b], v[c]]))) < 0.05:
            continue
        R = kabsch(v[[a, b, c]], d[[a, b, c]], np.ones(3))
        ang = np.degrees(np.arccos(np.clip(((v @ R.T) * d).sum(1), -1, 1)))
        cnt = int((ang <= gate).sum())
        if cnt > best[0]:
            best = (cnt, R)
    if best[1] is None or best[0] < 8:
        return None, best[0]
    ang = np.degrees(np.arccos(np.clip(((v @ best[1].T) * d).sum(1), -1, 1)))
    R = kabsch(v, d, (ang <= gate).astype(float))
    ang = np.degrees(np.arccos(np.clip(((v @ R.T) * d).sum(1), -1, 1)))
    return R, int((ang <= gate).sum())


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

    rng_sel = np.random.default_rng(4242)
    pick = rng_sel.choice(np.where(e0 < 0.5)[0], N, replace=False)
    branch = Branch(device='cuda')
    K = branch.C.K
    rng = np.random.default_rng(7)

    store = []
    for n, i in enumerate(pick):
        key = keys[i]
        date, stamp = key.split('/')
        T0 = t0[i]
        worlds, pixels = None, None
        try:
            refs, _ = branch.retrieve(
                os.path.join(DATA, 'NCLT', date, 'cam5', stamp + '.jpg'), T0)
            q_model, candidates, _, _ = branch._match_frame(
                os.path.join(DATA, 'NCLT', date, 'cam5', stamp + '.jpg'), refs)
            capped = branch._cap(candidates)
            if len(capped) >= 8:
                ids = np.array([c[0] for c in capped])
                worlds = np.array([branch._projection(branch.references[c[2]])['world'][c[3]]
                                   for c in capped])
                pixels = branch.C.model_to_raw(branch.matrix, q_model[ids])
        except Exception as exc:                       # noqa: BLE001
            print('  %s: %s' % (key, exc))
        if worlds is None:
            continue
        C_cw0 = np.linalg.inv(T0 @ branch.T_BC)
        R0c = C_cw0[:3, :3].copy()
        centre = -R0c.T @ C_cw0[:3, 3]
        v, d = prepare(worlds, pixels, centre, K)
        R_anchor, n_anchor = anchored(v, d, R0c)
        R_free, n_free = prior_free(v, d, rng)
        store.append(dict(i=int(i), R0c=R0c, cen=centre, n=len(v),
                          Ran=R_anchor, nan=n_anchor, Rfr=R_free, nfr=n_free))
        if (n + 1) % 10 == 0:
            print('  %d/%d' % (n + 1, len(pick)), flush=True)

    def body(Rc):
        return (R_BC @ np.asarray(Rc)).T

    rows = []
    for r in store:
        i = r['i']
        Rb0 = t0[i, :3, :3]
        out = dict(i=i, moe0=float(ra(Rb0, gt[i, :3, :3])))
        if r['Ran'] is not None:
            Rb = body(r['Ran'])
            out['moe_an'] = float(ra(Rb, gt[i, :3, :3]))
            out['db_an'] = logv(Rb0.T @ Rb)
            out['n_an'] = r['nan']
        if r['Rfr'] is not None:
            Rb = body(r['Rfr'])
            out['moe_fr'] = float(ra(Rb, gt[i, :3, :3]))
            out['db_fr'] = logv(Rb0.T @ Rb)
            out['n_fr'] = r['nfr']
        rows.append(out)

    print('\n## %d frames with correspondences' % len(rows))
    for tag, keym, keyd in (('anchored', 'moe_an', 'db_an'), ('prior-free', 'moe_fr', 'db_fr')):
        sel = [r for r in rows if keym in r]
        m = np.array([r[keym] for r in sel])
        p = np.array([r['moe0'] for r in sel])
        print('   %-11s n=%3d  MOE %.3f (prior %.3f)  median %.2f  better on %d'
              % (tag, len(sel), m.mean(), p.mean(), np.median(m), int((m < p).sum())))
        ep = np.degrees([logv(t0[r['i'], :3, :3].T @ gt[r['i'], :3, :3]) for r in sel])
        ei = np.degrees([logv(expv(r[keyd]) @ t0[r['i'], :3, :3].T @ gt[r['i'], :3, :3])
                         for r in sel])
        print('      corr(prior err, image err) = %+.3f   mean|image err| %.2f vs %.2f'
              % (np.corrcoef(ep.ravel(), ei.ravel())[0, 1],
                 np.linalg.norm(ei, axis=1).mean(), np.linalg.norm(ep, axis=1).mean()))
        for lam in (0.0, 0.2, 0.35, 0.5):
            e = []
            for r in sel:
                i = r['i']
                R = t0[i, :3, :3] @ expv(lam * r[keyd])
                e.append(ra(R, gt[i, :3, :3]))
            print('      lam %.2f -> MOE %.3f' % (lam, np.mean(e)))
    np.savez(OUT, rows=np.array(rows, dtype=object))
    print('\nwrote %s' % OUT)


if __name__ == '__main__':
    main()

"""EXP R: validate the single-frame image/front-end attitude fusion on 400 frames.

Per frame: one image, one scan, no neighbours, no temporal filter.  The LiDAR
branch supplies the translation; the image supplies a second, independent opinion
on the rotation; the two are blended by a fixed factor.

Reports the paired statistics per position-error band, for two independent image
solvers (prior-anchored and prior-free), so the effect cannot be blamed on the
blend being anchored to the front end.
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

OUT = '/home/zhangcy/.dsh_scratch/moe_rot_20261022/expR_validate.npz'
LAMBDAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
R_BC = np.asarray(json.load(open(LIB))['T_BC'], float)[:3, :3]


def logv(R):
    a = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = np.sin(a)
    return v * a / (2 * s) if s > 1e-8 else v / 2


def _arg(name, default):
    tag = '--%s=' % name
    for a in sys.argv[1:]:
        if a.startswith(tag):
            return a.split('=', 1)[1]
    return default


N = int(_arg('n', os.environ.get('N', '400')))
ARM = _arg('arm', 'frozen_noprune')
BUDGET = int(_arg('budget', '256'))
CHANNELS = int(_arg('channels', '5'))
TAG = _arg('tag', 'default')
OUT = '/home/zhangcy/.dsh_scratch/moe_rot_20261022/expR_%s.npz' % TAG


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


def anchored(v, d, R0):
    R = R0.copy()
    for gate in (6.0, 4.0, 2.5, 1.5, 1.0, 1.0, 1.0):
        ang = np.degrees(np.arccos(np.clip(((v @ R.T) * d).sum(1), -1, 1)))
        w = (ang <= gate).astype(float)
        if w.sum() < 8:
            return None, 0
        R = kabsch(v, d, w)
    ang = np.degrees(np.arccos(np.clip(((v @ R.T) * d).sum(1), -1, 1)))
    return R, int((ang <= 2.0).sum())


def prior_free(v, d, rng, trials=500, gate=2.0):
    n = len(v)
    if n < 12:
        return None, 0
    idx = rng.integers(0, n, size=(trials, 3))
    det_d = np.abs(np.linalg.det(np.stack([d[idx[:, 0]], d[idx[:, 1]], d[idx[:, 2]]], 1)))
    det_v = np.abs(np.linalg.det(np.stack([v[idx[:, 0]], v[idx[:, 1]], v[idx[:, 2]]], 1)))
    idx = idx[(det_d > 0.05) & (det_v > 0.05)]
    best = (0, None)
    for a, b, c in idx:
        if a == b or b == c or a == c:
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
    channels = 5

    def retrieve(self, image_path, pose_prior=None, channels=None):
        return super(Branch, self).retrieve(image_path, pose_prior, channels or self.channels)
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
    moe0 = ra(t0[:, :3, :3], gt[:, :3, :3])

    rng_sel = np.random.default_rng(90210)
    pick = rng_sel.choice(len(keys), min(N, len(keys)), replace=False)
    print('frames %d ; sample MPE %.3f MOE %.3f (whole set %.3f / %.3f)'
          % (len(pick), e0[pick].mean(), moe0[pick].mean(), e0.mean(), moe0.mean()),
          flush=True)
    arm, budget, channels = ARM, BUDGET, CHANNELS
    print('variant %s arm=%s budget=%d channels=%d' % (TAG, arm, budget, channels),
          flush=True)
    branch = Branch(device='cuda', arm=arm, keypoint_budget=budget)
    branch.channels = channels
    K = branch.C.K
    rng = np.random.default_rng(7)

    rec = []
    for n, i in enumerate(pick):
        key = keys[i]
        date, stamp = key.split('/')
        image_path = os.path.join(DATA, 'NCLT', date, 'cam5', stamp + '.jpg')
        T0 = t0[i]
        row = dict(i=int(i))
        try:
            refs, _ = branch.retrieve(image_path, T0)
            q_model, candidates, _, _ = branch._match_frame(image_path, refs)
            capped = branch._cap(candidates)
            if len(capped) >= 8:
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
                rays = np.linalg.inv(K) @ np.column_stack(
                    [pixels, np.ones(len(pixels))]).T
                rays /= np.maximum(np.linalg.norm(rays, axis=0), 1e-12)
                dcam = rays.T
                Ra, na = anchored(v, dcam, R0c)
                Rf, nf = prior_free(v, dcam, rng)
                row['n_corr'] = int(len(v))
                if Ra is not None:
                    row['db_an'] = R_BC @ logv(R0c.T @ Ra)
                    row['n_an'] = na
                if Rf is not None:
                    row['db_fr'] = R_BC @ logv(R0c.T @ Rf)
                    row['n_fr'] = nf
        except Exception as exc:                      # noqa: BLE001
            row['error'] = str(exc)
        rec.append(row)
        if (n + 1) % 25 == 0:
            print('  %d/%d  with image %d' % (n + 1, len(rec),
                                              sum(1 for r in rec if 'db_an' in r)),
                  flush=True)

    i_arr = np.array([r['i'] for r in rec], int)
    bands = np.digitize(e0[i_arr], [0.2, 0.5, 2.0])
    for tag, key in (('anchored', 'db_an'), ('prior-free', 'db_fr')):
        sel = np.array([key in r for r in rec])
        ii = i_arr[sel]
        dd = np.array([rec[k][key] for k in range(len(rec)) if sel[k]], float)
        base = moe0[ii]
        print('\n## %s  (n=%d of %d, coverage %.0f%%)'
              % (tag, len(ii), len(rec), 100.0 * len(ii) / len(rec)))
        print('   %-6s %8s %8s %8s %10s %10s' % ('lam', 'mean', 'median', 'p90',
                                                 'improved%', 'changed mean'))
        for lam in LAMBDAS:
            R = np.stack([t0[ii[j], :3, :3] @ expv(lam * dd[j]) for j in range(len(ii))])
            e = ra(R, gt[ii, :3, :3])
            print('   %-6.2f %8.3f %8.3f %8.3f %9.1f%% %+10.3f'
                  % (lam, e.mean(), np.median(e), np.percentile(e, 90),
                     100 * (e < base).mean(), e.mean() - base.mean()))
        print('   by position-error band at lam=0.3:')
        R = np.stack([t0[ii[j], :3, :3] @ expv(0.3 * dd[j]) for j in range(len(ii))])
        e = ra(R, gt[ii, :3, :3])
        for b, name in enumerate(('MPE<0.2', '0.2-0.5', '0.5-2', '>2')):
            m = bands[sel] == b
            if m.sum() < 3:
                continue
            print('      %-8s n=%3d  MOE %.3f -> %.3f  (improved %d/%d)'
                  % (name, int(m.sum()), base[m].mean(), e[m].mean(),
                     int((e[m] < base[m]).sum()), int(m.sum())))
    np.savez(OUT, rec=np.array(rec, dtype=object), pick=pick)
    print('\nwrote %s' % OUT)


if __name__ == '__main__':
    main()

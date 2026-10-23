"""Step 1b: same diagnostic as step1 with the cache bug fixed and BOTH retrieval sets.

Two reference sets per frame, each matched once:
  A anchored   : CLS top-5 union the 5 references nearest the geometry pose T3
                 (the frozen production rule; used for the solver comparison s1/s2/s3)
  B pose-free  : CLS top-10 over the whole 4752-frame library, position channel removed
                 (the retrieval the pose-free expert will actually use; solver s3 only)

Solvers, identical correspondences within a set:
  s1 anchored rotation  centre from T3, graduated Wahba seeded at the T3 rotation
  s2 prior-free rotation centre from T3, RANSAC triples, no prior rotation
  s3 prior-free 6-DoF    EPnP + RANSAC + LM, no prior at all
"""
import os, sys, json, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '1'
from collections import OrderedDict
from multiprocessing import Pool
import numpy as np

PROJECT = '/home/zhangcy/projects/e2e-hypothesis-arbitration-20261023-v1'
SCORED = '/data2/zhangcy/rpge-crossvote-hard1502-20261012-v1/scored.npz'
DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001'
sys.path.insert(0, PROJECT)
import visual_branch as VB
VB.SCRATCH = '/tmp/dsh-layered-vb'
OUT = '/tmp/dsh-layered-frames.jsonl'
_G = {}
TOPK_FREE = 10


def ra(A, B):
    R = np.swapaxes(A, -1, -2) @ B
    return np.degrees(np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


def logv(R):
    a = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = np.sin(a)
    return v * a / (2 * s) if s > 1e-8 else v / 2


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
    """visual_branch keeps plain dicts and calls popitem(last=False) on them, which
    raises once the cache passes 64 entries; OrderedDict is what that call expects."""

    def __init__(self, *a, **kw):
        kw.setdefault('device', 'cpu')
        super(Branch, self).__init__(*a, **kw)
        self._features = OrderedDict()
        self._projections = OrderedDict()


def init():
    _G['branch'] = Branch(scratch='/tmp/dsh-layered-vb', torch_threads=2)
    _G['rng'] = np.random.default_rng(7)


def solve_set(b, key, img, T3, gt_pose, refs, tag, row):
    """s1 s2 s3 on one reference set; writes into row under suffix tag."""
    out = {}
    try:
        q_model, candidates, per_pair, batched = b._match_frame(img, refs)
        capped = b._cap(candidates)
    except Exception as exc:                                   # noqa: BLE001
        row[tag + '_error'] = repr(exc)
        return
    row[tag + '_nref'] = len(refs)
    row[tag + '_ncand'] = len(candidates)
    row[tag + '_ncapped'] = len(capped)
    if len(capped) < 8:
        row[tag + '_error'] = 'few correspondences'
        return
    ids = np.array([c[0] for c in capped])
    world = np.array([b._projection(b.references[c[2]])['world'][c[3]] for c in capped])
    pixels = b.C.model_to_raw(b.matrix, q_model[ids])
    R_gt = gt_pose[:3, :3]
    C_cw = np.linalg.inv(T3 @ b.T_BC)
    R0c = C_cw[:3, :3].copy()
    centre = -R0c.T @ C_cw[:3, 3]
    v, d = prepare(world, pixels, centre, b.C.K)
    R_an, n_an = anchored(v, d, R0c)
    R_fr, n_fr = prior_free(v, d, _G['rng'])
    if R_an is not None:
        Rb = (b.T_BC[:3, :3] @ R_an).T
        out[tag + '_rot_an'] = float(ra(Rb, R_gt))
        out[tag + '_n_an'] = int(n_an)
        out[tag + '_dv_an'] = [float(x) for x in logv(Rb @ R_gt.T)]
    if R_fr is not None:
        Rb = (b.T_BC[:3, :3] @ R_fr).T
        out[tag + '_rot_fr'] = float(ra(Rb, R_gt))
        out[tag + '_n_fr'] = int(n_fr)
        out[tag + '_dv_fr'] = [float(x) for x in logv(Rb @ R_gt.T)]
    try:
        sol = b.C.solve_pnp(world, pixels, ids, b.T_BC)
    except Exception as exc:                                   # noqa: BLE001
        sol = None
        row[tag + '_error6'] = repr(exc)
    if sol is not None:
        T6 = np.asarray(sol['pose'], float)
        out[tag + '_rot_6d'] = float(ra(T6[:3, :3], R_gt))
        out[tag + '_trans_6d'] = float(np.linalg.norm(T6[:3, 3] - gt_pose[:3, 3]))
        out[tag + '_inl_6d'] = int(sol['unique_query_inliers'])
        out[tag + '_qual_6d'] = bool(sol['qualified'])
        out[tag + '_dv_6d'] = [float(x) for x in logv(T6[:3, :3] @ R_gt.T)]
    # retrieval quality: distance from the true place to the nearest selected reference
    pos = _G['branch'].positions
    out[tag + '_ref_hit'] = float(np.min(np.linalg.norm(pos[refs] - gt_pose[:3, 3], axis=1)))
    row.update(out)


def work(i):
    b = _G['branch']
    key = _G['keys'][i]
    date, stamp = key.split('/')
    img = os.path.join(DATA, 'NCLT', date, 'cam5', stamp + '.jpg')
    T3 = _G['t3'][i]
    gt = _G['gt'][i]
    row = dict(i=int(i), key=key, e_geo=float(_G['e_geo'][i]),
               e_geo_rot=float(ra(T3[:3, :3], gt[:3, :3])), e_out=float(_G['e_out'][i]),
               gate=bool(i in _G['gate']))
    try:
        refs_a, det = b.retrieve(img, T3)
    except Exception as exc:                                   # noqa: BLE001
        row['error'] = repr(exc)
        return row
    embed = b.embed(img)
    sim = b.reference_cls @ embed.astype(np.float32)
    refs_b = [int(x) for x in np.argsort(-sim)[:TOPK_FREE]]
    solve_set(b, key, img, T3, gt, refs_a, 'A', row)
    solve_set(b, key, img, T3, gt, refs_b, 'B', row)
    return row


def main():
    z = np.load(SCORED, allow_pickle=True)
    gt_keys = [str(k) for k in z['keys']]
    gt_all = np.asarray(z['reference'], float)
    lut = {k: a for a, k in enumerate(gt_keys)}
    Z = np.load(PROJECT + '/inputs/initial.npz', allow_pickle=True)
    keys = [str(k) for k in Z['keys']]
    gt = np.stack([gt_all[lut[k]] for k in keys])
    out = np.load(PROJECT + '/results/three_route_kcalib.npz', allow_pickle=True)
    out_pose = np.asarray(out['poses'], float)
    t3 = np.asarray(np.load(PROJECT + '/results/three_route_route3_recheck.npz',
                            allow_pickle=True)['poses'], float)
    e_out = np.linalg.norm(out_pose[:, :3, 3] - gt[:, :3, 3], axis=1)
    e_geo = np.linalg.norm(t3[:, :3, 3] - gt[:, :3, 3], axis=1)
    rows = [json.loads(l) for l in
            open(PROJECT + '/results/three_route_kcalib_diag.jsonl')]
    gate = set(a for a, r in enumerate(rows) if r.get('ratio', 9) < 0.21)
    worst = set(np.argsort(-e_out)[:100].tolist())
    sample = sorted(gate | worst)
    print('gate %d worst100 %d overlap %d union %d' % (len(gate), len(worst),
                                                       len(gate & worst), len(sample)))
    _G.update(keys=keys, gt=gt, t3=t3, e_geo=e_geo, e_out=e_out, gate=gate)
    t = time.time()
    with Pool(6, initializer=init) as p:
        res = p.map(work, sample, chunksize=1)
    print('ran %d frames in %.1f s' % (len(res), time.time() - t), flush=True)
    with open(OUT, 'w') as fh:
        for r in res:
            fh.write(json.dumps(r) + '\n')
    report(res, gt, t3)


def report(res, gt, t3):
    edges = [0.0, 0.2, 1.0, 5.0, np.inf]
    cols = [('A_rot_an', 'anchor'), ('A_rot_fr', 'freeR-A'), ('A_rot_6d', 'free6D-A'),
            ('B_rot_6d', 'free6D-B'), ('B_trans_6d', 'trans6D-B'), ('A_trans_6d', 'trans6D-A')]
    print('\n%-12s %4s ' % ('geom err', 'n') +
          ' '.join('%10s' % c[1] for c in cols))
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in res if lo <= r['e_geo'] < hi]
        if not sel:
            continue
        cells = []
        for k, _ in cols:
            v = [r[k] for r in sel if k in r]
            cells.append('%10s' % ('%.2f(%d)' % (np.median(v), len(v)) if v else '-'))
        print('%-12s %4d ' % ('[%g,%g)' % (lo, hi) if hi != np.inf else '[%g,inf)' % lo,
                             len(sel)) + ' '.join(cells))
    print('\nretrieval hit (m, distance from the true place to the nearest selected ref):')
    for tag, name in (('A', 'anchored CLS5+pos5'), ('B', 'pose-free CLS10')):
        v = [r['%s_ref_hit' % tag] for r in res if '%s_ref_hit' % tag in r]
        print('  %-20s n=%3d  median %.1f  <=25m on %d  <=50m on %d'
              % (name, len(v), np.median(v), sum(x <= 25 for x in v), sum(x <= 50 for x in v)))
    print('\nsolver comparison, frames where both exist:')
    for a, b_ in (('A_rot_an', 'A_rot_fr'), ('A_rot_an', 'A_rot_6d'), ('A_rot_6d', 'B_rot_6d')):
        both = [r for r in res if a in r and b_ in r]
        if not both:
            continue
        d = np.array([r[a] - r[b_] for r in both])
        print('  %-11s better than %-11s on %3d/%3d  median diff %+.2f deg'
              % (b_, a, int((d < 0).sum()), len(both), float(np.median(d))))
    print('\ncorrelation of each visual rotation error with the geometry rotation error:')
    for k, dv in (('A_rot_an', 'A_dv_an'), ('A_rot_fr', 'A_dv_fr'), ('A_rot_6d', 'A_dv_6d'),
                  ('B_rot_6d', 'B_dv_6d')):
        both = [r for r in res if k in r]
        if len(both) < 8:
            continue
        ep = np.array([logv(t3[r['i'], :3, :3] @ gt[r['i'], :3, :3].T) for r in both])
        ei = np.array([r[dv] for r in both])
        print('  %-11s n=%3d  corr %+.3f' % (k, len(both), np.corrcoef(ep.ravel(), ei.ravel())[0, 1]))


if __name__ == '__main__':
    main()

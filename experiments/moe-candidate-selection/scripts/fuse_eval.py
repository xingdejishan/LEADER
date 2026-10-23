"""Robust rotation averaging over laser-verified visual candidates vs best-score pick.

Reads the frozen candidate table and the pure route-3 poses (three_route_unified
--mode route3 written to scratch), re-runs the project's own arbitration cost for
every hypothesis, and scores end to end with the project scorer.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '1'

import numpy as np

PROJ = '/home/zhangcy/projects/e2e-moe-20261020-v1'
sys.path.insert(0, PROJ)
sys.path.insert(0, '/home/zhangcy/projects/r-unified-eval-20261018-v1')
import three_route_unified as TRU  # noqa: E402
import score_unified as S  # noqa: E402

VERIFY = ('/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/'
          'verify-frozen_nopruneoptREF/%06d.json')
RES = os.path.join(PROJ, 'results')
R3NPZ = '/home/zhangcy/.dsh_scratch/route3_1502.npz'
MERGED_DIAG = os.path.join(RES, 'three_route_merged_diag.jsonl')
MERGED_NPZ = os.path.join(RES, 'three_route_merged.npz')

EPS = 1e-3
ITERS = 10
GUARDS = [('none', None), ('5deg_1.0m', (5.0, 1.0)),
          ('10deg_2.0m', (10.0, 2.0)), ('20deg_5.0m', (20.0, 5.0))]
VARIANTS = ['best_all', 'best_ver'] + ['fuse_' + g[0] for g in GUARDS]
FUSED = [v for v in VARIANTS if v.startswith('fuse_')]
_G = {}


def read_candidates(i):
    rec = json.load(open(VERIFY % i))['poses']
    out = []
    for name, row in rec.items():
        if not name.startswith('candidate_'):
            continue
        fair = row.get('fair') or {}
        sc = fair.get('geometry_score')
        if sc is None:
            continue
        out.append(dict(name=name, score=float(sc),
                        verified=bool(fair.get('geometry_verified') is True),
                        pose=np.asarray(row['pose'], float)))
    return out


def rot_angle_deg(A, B):
    r = A.T @ B
    c = (np.trace(r) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def project_so3(M):
    U, _, Vt = np.linalg.svd(M)
    D = np.eye(3)
    D[-1, -1] = np.linalg.det(U @ Vt)
    return U @ D @ Vt


def fuse(cands, guard):
    """Chordal-L1 rotation mean by IRLS + translation mean with the same weights."""
    P = np.stack([c['pose'] for c in cands])
    R, t = P[:, :3, :3], P[:, :3, 3]
    n = len(cands)
    sr = st = 0.0
    if n == 1:
        return P[0].copy(), sr, st
    sr = max(rot_angle_deg(R[a], R[b]) for a in range(n) for b in range(a + 1, n))
    st = max(float(np.linalg.norm(t[a] - t[b])) for a in range(n) for b in range(a + 1, n))
    if guard is not None and (sr > guard[0] or st > guard[1]):
        return None, sr, st
    Rc = project_so3(R.sum(0))
    w = np.ones(n)
    for _ in range(ITERS):
        ang = np.array([np.deg2rad(rot_angle_deg(Rc, R[k])) for k in range(n)])
        w = 1.0 / np.maximum(ang, EPS)
        Rc = project_so3((w[:, None, None] * R).sum(0))
    T = np.eye(4)
    T[:3, :3] = Rc
    T[:3, 3] = (w[:, None] * t).sum(0) / w.sum()
    return T, sr, st


def hypotheses(i):
    cands = read_candidates(i)
    hyp = {k: None for k in VARIANTS}
    meta = dict(n_cand=len(cands), n_ver=sum(1 for c in cands if c['verified']),
                spread_r=float('nan'), spread_t=float('nan'),
                fuse_moved=float('nan'), fb={})
    if not cands:
        return hyp, meta
    best_all = max(cands, key=lambda c: c['score'])
    hyp['best_all'] = best_all['pose']
    ver = [c for c in cands if c['verified']]
    if not ver:
        return hyp, meta
    best_ver = max(ver, key=lambda c: c['score'])
    hyp['best_ver'] = best_ver['pose']
    for name, guard in GUARDS:
        T, sr, st = fuse(ver, guard)
        hyp['fuse_' + name] = best_ver['pose'] if T is None else T
        meta['fb'][name] = int(T is None)
        if name == 'none':
            meta['spread_r'], meta['spread_t'] = sr, st
            if T is not None:
                meta['fuse_moved'] = rot_angle_deg(T[:3, :3], best_ver['pose'][:3, :3])
    return hyp, meta


def init():
    TRU._OPT.update(mode='unified', gate='merged', gate2=-1.0, delta=0.05, rot=True,
                    solver='adaptive', coarse_points=1000, fine_points=1500, threads=4)
    TRU.init()
    z = np.load(R3NPZ, allow_pickle=True)
    diag = {d['key']: d for d in (json.loads(l) for l in open(MERGED_DIAG))}
    keys = [str(k) for k in z['keys']]
    _G['r3'] = np.asarray(z['poses'], float)
    _G['ratio'] = np.array([float(diag[k]['ratio']) for k in keys])


def one(i):
    cfg, maps = TRU._G['cfg'], TRU._G['maps']
    T3 = _G['r3'][i]
    ratio = _G['ratio'][i]
    hyp, meta = hypotheses(i)
    scan = TRU.P.prepare_scan(TRU._G['frames'][i], cfg)
    q = scan[:min(TRU.ARB_POINTS, len(scan))]
    c3 = TRU.cost_at(T3, q, maps, cfg)
    gate = bool(np.isfinite(ratio) and ratio < TRU.AMB_THRESHOLD)
    out = dict(i=i, ratio=float(ratio), c_route3=float(c3), meta=meta,
               pose={}, cv={}, acc={})
    for v in VARIANTS:
        P = hyp[v]
        if P is None or not gate:
            out['pose'][v] = T3.tolist()
            out['cv'][v] = float('nan')
            out['acc'][v] = False
            continue
        cv = TRU.cost_at(P, q, maps, cfg)
        ok = bool(cv < c3 - 0.05)
        out['cv'][v] = float(cv)
        out['acc'][v] = ok
        out['pose'][v] = (P if ok else T3).tolist()
    return out


def _w(i):
    return one(i)


def metrics(poses, gt):
    t = S.pos_err(poses, gt)
    r = S.rot_deg(poses[:, :3, :3], gt[:, :3, :3])
    return dict(MPE=float(t.mean()), MOE=float(r.mean()),
                ok=int(((t <= 1) & (r <= 2)).sum()), over5=int((t > 5).sum()),
                over10=int((t > 10).sum())), t, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default='/home/zhangcy/.dsh_scratch/fuse_out.json')
    a = ap.parse_args()
    t_start = time.time()
    init()
    n = len(_G['r3'])
    idx = list(range(n if a.limit <= 0 else min(a.limit, n)))
    t0 = time.time()
    if a.workers > 1:
        with Pool(a.workers, initializer=init) as p:
            res = p.map(_w, idx, chunksize=8)
    else:
        res = [_w(i) for i in idx]
    t_arb = time.time() - t0
    w = S.load_world()
    gt = w['hard_gt']
    r3 = _G['r3']

    full = {}
    for v in VARIANTS:
        A = r3.copy()
        for k, r in enumerate(res):
            A[idx[k]] = np.asarray(r['pose'][v], float)
        full[v] = A
    base = full['best_all']
    m, tb, rb = metrics(base, gt)

    # baseline sanity: does it equal the shipped merged output?
    mg = np.load(MERGED_NPZ, allow_pickle=True)
    kg = [str(k) for k in mg['keys']]
    lk = {k: i for i, k in enumerate(kg)}
    if len(idx) == n and [str(k) for k in np.load(R3NPZ, allow_pickle=True)['keys']] == kg:
        dm = np.abs(base - np.asarray(mg['poses'], float)).max(axis=(1, 2))
        print('baseline vs three_route_merged.npz: differ on %d frames (max %.3f m)'
              % (int((dm > 1e-9).sum()), float(dm.max())))

    rows = {}
    for v in VARIANTS:
        mm, tt, rr = metrics(full[v], gt)
        acc = np.array([r['acc'][v] for r in res] + [False] * (n - len(res)))
        row = dict(mm, accepted=int(acc.sum()))
        if v != 'best_all':
            dpos = np.abs(full[v] - base).max(axis=(1, 2))
            ch = dpos > 1e-9
            accb = np.array([r['acc']['best_all'] for r in res] + [False] * (n - len(res)))
            row['changed'] = int(ch.sum())
            row['changed_both_accepted'] = int((ch & acc & accb).sum())
            row['newly_accepted'] = int((acc & ~accb).sum())
            row['dropped'] = int((accb & ~acc).sum())
            row['dMOE_on_changed'] = float((rr - rb)[ch].mean()) if ch.any() else 0.0
            row['dMPE_on_changed'] = float((tt - tb)[ch].mean()) if ch.any() else 0.0
            row['sum_dMOE_accepted'] = float((rr - rb)[ch & acc & accb].sum() / n)
        rows[v] = row

    # ---------- the 783-frame band: raw hypothesis accuracy, no arbitration
    band = []
    for i in range(n):
        hp, mt = hypotheses(i)
        if mt['n_ver'] > 0:
            band.append((i, (hp, mt)))
    bi = np.array([b[0] for b in band])
    gt_b = gt[bi]
    print('\n[783 band] frames with >=1 laser-verified candidate: %d   raw pose vs GT, no arbitration'
          % len(band))
    print('%-16s %8s %8s %6s %6s %6s' % ('variant', 'MPE', 'MOE', 'ok', '>5m', '>10m'))
    for v in VARIANTS:
        P = np.stack([(hp[v] if hp[v] is not None else r3[i]) for i, (hp, mt) in band])
        mb, _, _ = metrics(P, gt_b)
        print('%-16s %8.4f %8.4f %6d %6d %6d'
              % (v, mb['MPE'], mb['MOE'], mb['ok'], mb['over5'], mb['over10']))
    sr = np.array([mt['spread_r'] for _, (hp, mt) in band])
    st = np.array([mt['spread_t'] for _, (hp, mt) in band])
    fm = np.array([mt['fuse_moved'] for _, (hp, mt) in band])
    nver = np.array([mt['n_ver'] for _, (hp, mt) in band])
    print('n_ver hist: %s' % {int(k): int((nver == k).sum()) for k in sorted(set(nver))})
    print('spread: rot med %.2f p90 %.2f deg (>.5/10/20: %d/%d/%d); '
          'trans med %.3f p90 %.3f m (>1/2/5: %d/%d/%d)'
          % (np.median(sr), np.percentile(sr, 90), int((sr > 5).sum()),
             int((sr > 10).sum()), int((sr > 20).sum()),
             np.median(st), np.percentile(st, 90), int((st > 1).sum()),
             int((st > 2).sum()), int((st > 5).sum())))
    print('fused-vs-best rotation delta: med %.3f mean %.3f p90 %.3f deg'
          % (np.median(fm), fm.mean(), np.percentile(fm, 90)))

    # ---------- 1502 end to end
    print('\n[1502 end to end]')
    print('%-16s %8s %8s %5s %5s %6s %8s %8s %8s %7s %7s'
          % ('variant', 'MPE', 'MOE', 'ok', '>5m', '>10m', 'accepted', 'changed',
             'chg&acc', 'new', 'lost'))
    for v in VARIANTS:
        r = rows[v]
        print('%-16s %8.4f %8.4f %5d %5d %6d %8d %8d %8d %7d %7d'
              % (v, r['MPE'], r['MOE'], r['ok'], r['over5'], r['over10'],
                 r['accepted'], r.get('changed', 0), r.get('changed_both_accepted', 0),
                 r.get('newly_accepted', 0), r.get('dropped', 0)))
    m3, _, _ = metrics(r3, gt)
    print('%-16s %8.4f %8.4f %5d %5d %6d' % ('route3_alone', m3['MPE'], m3['MOE'],
                                             m3['ok'], m3['over5'], m3['over10']))
    for v in FUSED:
        name = v[5:]
        fb = sum(int(r['meta']['fb'].get(name, 0)) for r in res)
        print('[%s] guard fallbacks=%d  changed=%d  dMOE_on_changed=%+.4f deg  '
              'sum_dMOE/n1502=%+.5f'
              % (v, fb, rows[v]['changed'], rows[v]['dMOE_on_changed'],
                 rows[v]['sum_dMOE_accepted']))
    json.dump({v: {k: val for k, val in r.items()} for v, r in rows.items()},
              open(a.out, 'w'), indent=1)
    print('\nruntime: arbitration+scoring %.1f s wall (workers=%d), total %.1f s'
          % (t_arb, a.workers, time.time() - t_start))


if __name__ == '__main__':
    main()

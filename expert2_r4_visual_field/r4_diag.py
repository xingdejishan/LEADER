"""R4 direction diagnostics (truth used ONLY as a measuring stick, never in selection).

    python r4_diag.py --tag main

Answers three questions asked of the run:
  1. geometric reach: does the plane spanned by (u, e) plus |s|<=10, |d|<=2 contain the
     truth? Report the truth offset projected on the estimated axes, and the vertical
     contamination |s*|*sin(theta) induced by a direction error theta.
  2. strata by where u came from (own trajectory vs reference-library fallback).
  3. strata by the direction error theta itself.
"""
import argparse
import glob
import json
import sys
from collections import Counter

import numpy as np

OUT = '/data2/zhangcy/r4-full-20261019-v1'
sys.path.insert(0, '/home/zhangcy/projects/r4-full-20261019-v1')
import r4_lib as L  # noqa: E402

THETAS = [(0.0, 1.0), (1.0, 3.0), (3.0, 10.0), (10.0, 180.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='main')
    ap.add_argument('--out', default='')
    A = ap.parse_args()
    w = L.World()
    diag = {}
    for p in glob.glob(OUT + '/diagnostics_%s.jsonl' % A.tag):
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                diag[r['key']] = r
    keys = w.keys
    n = len(keys)
    assert all(k in diag for k in keys), 'missing frames'

    rows = []
    for i, k in enumerate(keys):
        r = diag[k]
        T0 = w.deploy[i]
        G = w.gt[i]
        u = np.asarray(r['u'], float)
        e = np.asarray(r['lateral'], float)
        ut, mt = w.u_truth(i)
        et = L.lateral_axis(ut)
        dvec = G[:3, 3] - T0[:3, 3]
        c = np.clip(abs(float(u @ ut)), -1.0, 1.0)
        theta = float(np.degrees(np.arccos(c)))
        s_true = float(dvec @ ut)
        b_true = float(dvec @ et)
        a_main = float(dvec @ u)
        b_main = float(dvec @ e)
        fallback = r['u_source'].startswith('ref_')
        sel = np.asarray(r['pose'], float)
        err = sel[:3, 3] - G[:3, 3]
        rows.append(dict(
            key=k, date=r['date'], replaced=r['replaced'],
            u_source=r['u_source'], fallback=fallback,
            theta=0.0 if fallback else theta,
            s_true=s_true, b_true=b_true, a_main=a_main, b_main=b_main,
            contam=abs(s_true) * np.sin(np.radians(theta)),
            reach=bool(abs(a_main) <= 10.0 + 1e-9 and abs(b_main) <= 2.0 + 1e-9),
            reach_ideal=bool(abs(s_true) <= 10.0 + 1e-9 and abs(b_true) <= 2.0 + 1e-9),
            pos_err=float(np.linalg.norm(err)),
            along_res=float(err @ u), lat_res=float(err @ e),
            map_points=r['map_points'], s=r['s'], d=r['d'],
            base_err=float(np.linalg.norm(T0[:3, 3] - G[:3, 3]))))
    keep = ~np.array([r['replaced'] for r in rows])
    fb = np.array([r['fallback'] for r in rows])
    th = np.array([r['theta'] for r in rows])
    pe = np.array([r['pos_err'] for r in rows])
    be = np.array([r['base_err'] for r in rows])
    contam = np.array([r['contam'] for r in rows])
    reach = np.array([r['reach'] for r in rows])
    reach_ideal = np.array([r['reach_ideal'] for r in rows])

    print('=== frames: %d (kept %d) ===' % (n, int(keep.sum())))
    print('u source:', dict(Counter(r['u_source'] for r in rows)))
    print('\n--- 1. geometric reach (axes u, e with |s|<=10 m, |d|<=2 m) ---')
    meas = ~fb
    print('direction measurable from own trajectory: %d frames (%.1f%%)'
          % (meas.sum(), 100.0 * meas.mean()))
    print('truth reachable on the estimated axes+grid   : %d (%.1f%%)'
          % (reach.sum(), 100.0 * reach.mean()))
    print('truth reachable if u were exact (ideal axes) : %d (%.1f%%)'
          % (reach_ideal.sum(), 100.0 * reach_ideal.mean()))
    print('  |a_main|=|truth offset . u|: med=%.2f p90=%.2f  (limit 10)'
          % (np.median(np.abs([r['a_main'] for r in rows])),
             np.percentile(np.abs([r['a_main'] for r in rows]), 90)))
    print('  |b_main|=|truth offset . e|: med=%.2f p90=%.2f  (limit 2)  >2 m: %d'
          % (np.median(np.abs([r['b_main'] for r in rows])),
             np.percentile(np.abs([r['b_main'] for r in rows]), 90),
             int((np.abs([r['b_main'] for r in rows]) > 2).sum())))
    print('  vertical contamination |s*|sin(theta) on the %d measurable frames:'
          % meas.sum())
    for thr in (0.5, 1.0, 2.0):
        print('     <=%.1f m: %d (%.1f%% of measurable, %.1f%% of all)'
              % (thr, int((contam[meas] <= thr).sum()),
                 100.0 * (contam[meas] <= thr).mean(), 100.0 * (contam[meas] <= thr).mean() * meas.mean()))
    print('  on the %d measurable frames, theta: med=%.2f p75=%.2f p90=%.2f max=%.1f'
          % (meas.sum(), np.median(th[meas]), np.percentile(th[meas], 75),
             np.percentile(th[meas], 90), th[meas].max()))

    print('\n--- 2. strata: where did u come from ---')
    strata = {}
    for name, sel in (('own trajectory (quad)', ~fb), ('reference-library fallback', fb)):
        m = sel & keep
        strata[name] = dict(n=int(sel.sum()), n_keep=int(m.sum()),
                            base_MPE=float(be[m].mean()), MPE=float(pe[m].mean()),
                            med=float(np.median(pe[m])),
                            ok12=float((pe[m] <= 1).mean()),
                            base_ok12=float((be[m] <= 1).mean()))
        print('  %-28s n=%4d  MPE %.3f -> %.3f  med %.3f -> %.3f  pos<=1m %.1f%% -> %.1f%%'
              % (name, sel.sum(), be[m].mean(), pe[m].mean(), np.median(be[m]),
                 np.median(pe[m]), 100.0 * (be[m] <= 1).mean(),
                 100.0 * (pe[m] <= 1).mean()))

    print('\n--- 3. strata: direction error theta (measurable frames only) ---')
    bins = []
    for lo, hi in THETAS:
        sel = meas & (th >= lo) & (th < hi) & keep
        if not sel.any():
            continue
        bins.append(dict(lo=lo, hi=hi, n=int(sel.sum()),
                         base_MPE=float(be[sel].mean()), MPE=float(pe[sel].mean()),
                         med=float(np.median(pe[sel])),
                         ok12=float((pe[sel] <= 1).mean()),
                         base_ok12=float((be[sel] <= 1).mean()),
                         contam_med=float(np.median(contam[sel]))))
        print('  theta %4.1f-%5.1f deg  n=%4d  MPE %.3f -> %.3f  med %.3f -> %.3f  '
              'pos<=1m %.1f%% -> %.1f%%  med contam %.2f m'
              % (lo, hi, sel.sum(), be[sel].mean(), pe[sel].mean(), np.median(be[sel]),
                 np.median(pe[sel]), 100.0 * (be[sel] <= 1).mean(),
                 100.0 * (pe[sel] <= 1).mean(), np.median(contam[sel])))

    print('\n--- 4. where does the residual sit (kept frames) ---')
    ar = np.abs([r['along_res'] for r in rows])
    lr = np.abs([r['lat_res'] for r in rows])
    bad = keep & (pe > 1.0)
    print('all kept: |along residual| med=%.2f p90=%.2f ; |lateral residual| med=%.2f p90=%.2f'
          % (np.median(ar[keep]), np.percentile(ar[keep], 90),
             np.median(lr[keep]), np.percentile(lr[keep], 90)))
    print('failing kept (%d): |along| med=%.2f ; |lateral| med=%.2f ; '
          'of those, lateral >2 m: %d, along >10 m: %d'
          % (bad.sum(), np.median(ar[bad]), np.median(lr[bad]),
             int((lr[bad] > 2).sum()), int((ar[bad] > 10).sum())))
    print('selected s on failing frames: %s' % Counter(
        [rows[i]['s'] for i in np.flatnonzero(bad)]).most_common(6))

    payload = dict(n=n, kept=int(keep.sum()),
                   measurable=int(meas.sum()), reachable=int(reach.sum()),
                   reachable_ideal=int(reach_ideal.sum()),
                   contam_le2=int((contam[meas] <= 2).sum()),
                   strata=strata, theta_bins=bins,
                   pos_err=[round(float(x), 4) for x in pe],
                   keys=keys)
    out = A.out or (OUT + '/diag_%s.json' % A.tag)
    json.dump(payload, open(out, 'w'), indent=1)
    print('\nwrote', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""EXP B: is the reference map itself tilted relative to the truth world?

The bulk of the front-end rotation error is tilt (~1 deg), not yaw, and the
front-end pose fits the map better than the truth pose does.  One explanation is
that the reference maps are built from poses that are themselves tilted, so the
map is a rotated copy of the world and every fit to it inherits the rotation.

Test: at the ground-truth position, fit the dominant ground plane of
  (a) the local reference map points, and
  (b) the query scan transformed by the truth pose (which must be level),
and compare both normals with world up.
"""
import json
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

PROJECT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
sys.path.insert(0, os.path.join(PROJECT, 'stages/fast_search'))
sys.path.insert(0, os.path.join(PROJECT, 'stages'))
import fast_lib as F      # noqa: E402
import pipeline as P      # noqa: E402

DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001'
TRUTH_DIR = os.path.join(DATA, 'evaluator-only-truth')
OUT = '/home/zhangcy/.dsh_scratch/moe_rot_20261022/expB_map_tilt.npz'
N_PER_DATE = int(os.environ.get('NPD', '50'))
RADIUS = 25.0


def fit_ground(pts):
    """Dominant near-horizontal plane of pts.  Returns (normal, rms, inlier_frac, n_plane)."""
    if len(pts) < 200:
        return None
    z = pts[:, 2]
    lo, hi = np.percentile(z, [2, 40])
    cand = pts[(z >= lo) & (z <= hi)]
    if len(cand) < 100:
        return None
    # coarse: the mode of z in 0.1 m bins
    hist, edges = np.histogram(cand[:, 2], bins=40)
    k = int(np.argmax(hist))
    zc = 0.5 * (edges[k] + edges[k + 1])
    sel = cand[np.abs(cand[:, 2] - zc) < 0.4]
    if len(sel) < 100:
        return None
    n = np.array([0.0, 0.0, 1.0])
    for _ in range(4):
        c = sel.mean(0)
        u, s, vt = np.linalg.svd(sel - c, full_matrices=False)
        n = vt[-1]
        if n[2] < 0:
            n = -n
        r = (sel - c) @ n
        keep = np.abs(r) < 2.5 * max(np.std(r), 0.05) + 0.05
        if keep.all() or keep.sum() < 100:
            break
        sel = sel[keep]
    c = sel.mean(0)
    r = (sel - c) @ n
    return n, float(np.sqrt((r ** 2).mean())), len(sel) / float(len(pts)), len(sel)


def main():
    t0_ = time.time()
    refs = json.load(open(os.path.join(PROJECT, 'inputs/refs.json')))
    pos = np.asarray([r['world_from_body'] for r in refs], np.float64)[:, :3, 3]
    maps = F.MapSource(refs, pos, radius=60.0)
    cfg = P.Config()

    blob = np.load(os.path.join(PROJECT, 'results/full-online-final.npz'), allow_pickle=True)
    keys = [str(k) for k in blob['keys']]
    t0 = np.asarray(blob['t0'], np.float64)
    truth = {}
    for f in os.listdir(TRUTH_DIR):
        date = f[:-5]
        for ts, v in json.load(open(os.path.join(TRUTH_DIR, f))).items():
            truth[(date, str(ts))] = np.asarray(v['T_WB'], np.float64)
    gt = np.stack([truth[tuple(k.split('/'))] for k in keys])

    rng = np.random.default_rng(20261022)
    dates = np.array([k.split('/')[0] for k in keys])
    pick = []
    for d in sorted(set(dates.tolist())):
        idx = np.where((dates == d) & (np.linalg.norm(t0[:, :3, 3] - gt[:, :3, 3], axis=1) < 0.2))[0]
        pick.append(rng.choice(idx, min(N_PER_DATE, len(idx)), replace=False))
    pick = np.concatenate(pick)
    print('sample %d frames over %s' % (len(pick), sorted(set(dates[pick].tolist()))))

    rows = []
    for j, i in enumerate(pick):
        key = keys[i]
        date, ts = key.split('/')
        rec = {'key': key,
               'scan_path': os.path.join(DATA, 'NCLT', date, 'velodyne_sync', ts + '.bin')}
        scan = P.prepare_scan(rec, cfg)
        centre = gt[i][:3, 3]
        mp = maps.local(centre, maps.nearby(centre)[:3])
        d2 = np.linalg.norm(mp[:, :2] - centre[None, :2], axis=1)
        mpn = mp[d2 < RADIUS]
        gm = fit_ground(mpn)
        # the scan, expressed in the world frame by the truth pose
        sw = scan @ gt[i][:3, :3].T + gt[i][:3, 3]
        ds = np.linalg.norm(sw[:, :2] - centre[None, :2], axis=1)
        gs = fit_ground(sw[ds < RADIUS])
        if gm is None or gs is None:
            continue
        n_map, rms_m, frac_m, nm = gm
        n_scan, rms_s, frac_s, ns = gs
        # front-end tilt error: angle between the pose tilts and truth up
        Rt = t0[i][:3, :3].T @ gt[i][:3, :3]
        ang_t0 = np.degrees(np.arccos(np.clip((np.trace(Rt) - 1) / 2, -1, 1)))
        w = np.array([Rt[2, 1] - Rt[1, 2], Rt[0, 2] - Rt[2, 0], Rt[1, 0] - Rt[0, 1]])
        s = np.linalg.norm(w)
        wv = w * (np.arctan2(s, np.trace(Rt) - 1) / s) if s > 1e-9 else w / 2
        tilt_t0 = np.degrees(np.linalg.norm(wv[:2]))
        yaw_t0 = np.degrees(abs(wv[2]))
        tilt_map = np.degrees(np.arccos(np.clip(n_map[2], -1, 1)))
        tilt_scan = np.degrees(np.arccos(np.clip(n_scan[2], -1, 1)))
        rows.append((i, date, tilt_map, tilt_scan, tilt_t0, yaw_t0, ang_t0,
                     float(n_map[0]), float(n_map[1]), rms_m, rms_s))
        if (j + 1) % 20 == 0:
            print('  %d/%d  %.1f s' % (j + 1, len(pick), time.time() - t0_), flush=True)

    rows = np.array(rows, dtype=object)
    tilt_map = np.array(rows[:, 2], float)
    tilt_scan = np.array(rows[:, 3], float)
    tilt_t0 = np.array(rows[:, 4], float)
    yaw_t0 = np.array(rows[:, 5], float)
    ang_t0 = np.array(rows[:, 6], float)
    nmx = np.array(rows[:, 7], float)
    nmy = np.array(rows[:, 8], float)
    rms_m = np.array(rows[:, 9], float)
    rms_s = np.array(rows[:, 10], float)

    print('\n## ground-plane tilt from world up (deg), n=%d' % len(rows))
    for lab, v in (('map normal', tilt_map), ('scan normal (truth pose)', tilt_scan),
                   ('tilt part of front-end err', tilt_t0), ('yaw part', yaw_t0),
                   ('total front-end err', ang_t0),
                   ('map plane rms m', rms_m * 0 + 0),):
        if 'rms' in lab:
            continue
        print('   %-28s mean %.3f  median %.3f  p90 %.3f' % (lab, v.mean(), np.median(v), np.percentile(v, 90)))
    print('   map plane fit rms  %.3f m ; scan plane fit rms %.3f m' % (rms_m.mean(), rms_s.mean()))
    print('   signed map normal tilt: mean nx %+.4f  ny %+.4f' % (nmx.mean(), nmy.mean()))

    print('\n## per date')
    for d in sorted(set(rows[:, 1].tolist())):
        m = rows[:, 1] == d
        print('   %s n=%3d  map tilt %.3f  scan tilt %.3f  front-end tilt %.3f  yaw %.3f'
              % (d, int(m.sum()), tilt_map[m].mean(), tilt_scan[m].mean(),
                 tilt_t0[m].mean(), yaw_t0[m].mean()))

    print('\n## does the map tilt explain the front-end tilt?')
    cc = np.corrcoef(tilt_map - tilt_scan, tilt_t0)[0, 1]
    print('   corr(map tilt - scan tilt, front-end tilt) = %+.3f' % cc)
    diff = tilt_map - tilt_scan
    print('   mean(|map - scan|) = %.3f deg ; mean front-end tilt = %.3f deg' % (diff.mean(), tilt_t0.mean()))
    # vector alignment: is the front-end tilt *in the same direction* as the map tilt?
    print('   mean signed (map tilt) vector components: (%+.4f, %+.4f)' % (nmx.mean(), nmy.mean()))

    np.savez(OUT, pick=np.array(rows[:, 0], int), date=rows[:, 1],
             tilt_map=tilt_map, tilt_scan=tilt_scan, tilt_t0=tilt_t0, yaw_t0=yaw_t0,
             ang_t0=ang_t0, nmx=nmx, nmy=nmy, rms_m=rms_m, rms_s=rms_s)
    print('\nwrote %s  total %.1f s' % (OUT, time.time() - t0_))


if __name__ == '__main__':
    main()

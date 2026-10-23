"""Shared map / shared KD-tree in the two-hypothesis arbitration.

Question: the arbiter scores the route-3 pose and the visual candidate with the
same truncated scan-to-map cost, today on two separately cropped maps with two
KD-trees.  The two poses are often within a metre, so one map may do.

Replays exactly the arbitration block of three_route_unified.one() on the 1502
hard frames, taking the route-3 hypothesis from a deterministic route-3 replay
(scratch/route3_poses.npz) because the frozen npz stores the *chosen* pose.

  B0  per-pose map (deployed)     two crops, two KD-trees
  A   shared map at the midpoint  one crop, one KD-tree
  B   shared map at route-3 pose  one crop, one KD-tree

Cost: mean truncated (1 m) nearest-neighbour distance of the same 4000 scan
points, transformed into the hypothesis frame, against a 60 m crop of the merged
cloud of the 3 nearest references (0.3 m voxel).  Accept the visual pose iff
C_vis < C_route3 - 0.05, on the frames the ambiguity gate lets through.
"""
import json, os, sys, time
import numpy as np
from scipy.spatial import cKDTree

ROOT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
VERIFY = ('/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/'
          'verify-frozen_nopruneoptREF/%06d.json')
OUT = '/home/zhangcy/.dsh_scratch/arb_shared'
ARB_POINTS = 4000
TRUNC = 1.0
DELTA = 0.05
AMB = 0.21

for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '1'
sys.path.insert(0, os.path.join(ROOT, 'stages/fast_search'))
sys.path.insert(0, '/home/zhangcy/projects/r-unified-eval-20261018-v1')
import fast_lib as F          # noqa: E402
import pipeline as P          # noqa: E402
import score_unified as SU    # noqa: E402

os.makedirs(OUT, exist_ok=True)


def cost(tree, pts, trunc=TRUNC):
    d = tree.query(pts, k=1, workers=1, distance_upper_bound=trunc)[0]
    return float(np.where(np.isfinite(d), d, trunc).mean())


def own_map(pose, maps, cfg):
    centre = pose[:3, 3]
    return maps.local(centre, maps.nearby(centre)[:cfg.reference_count])


def score_own(pose, q, maps, cfg):
    centre = pose[:3, 3]
    return cost(cKDTree(own_map(pose, maps, cfg)), q @ pose[:3, :3].T + centre)


def score_shared(pose, q, tree):
    return cost(tree, q @ pose[:3, :3].T + pose[:3, 3])


def load_visual(n):
    out = [None] * n
    for i in range(n):
        rec = json.load(open(VERIFY % i))['poses']
        best = None
        for name, row in rec.items():
            if not name.startswith('candidate_'):
                continue
            sc = (row.get('fair') or {}).get('geometry_score')
            if sc is None:
                continue
            if best is None or sc > best[0]:
                best = (float(sc), np.asarray(row['pose'], float))
        out[i] = None if best is None else best[1]
    return out


def main():
    t0 = time.time()
    frames = json.load(open(os.path.join(ROOT, 'inputs/frames.json')))
    refs = json.load(open(os.path.join(ROOT, 'inputs/refs.json')))
    cfg = P.Config()
    n = len(frames)

    r3 = np.load(os.path.join(OUT, 'route3_poses.npz'), allow_pickle=True)
    T3 = np.asarray(r3['poses'], float)
    assert [str(k) for k in r3['keys']] == [str(f['key']) for f in frames]
    frozen = np.load(os.path.join(ROOT, 'results/three_route_merged.npz'), allow_pickle=True)
    frozen_pose = np.asarray(frozen['poses'], float)

    diag = [json.loads(l) for l in open(os.path.join(ROOT, 'results/three_route_merged_diag.jsonl'))]
    ratio = np.array([d['ratio'] for d in diag], float)
    c3_ref = np.array([d.get('c_route3', np.nan) for d in diag], float)
    cv_ref = np.array([d.get('c_visual', np.nan) for d in diag], float)
    acc_ref = np.array([bool(d.get('visual')) for d in diag])

    vis = load_visual(n)
    has_vis = np.array([v is not None for v in vis])
    mid = np.array([0.5 * (T3[i, :3, 3] + vis[i][:3, 3]) if has_vis[i] else T3[i, :3, 3]
                    for i in range(n)])
    print('[load] %.1f s' % (time.time() - t0), flush=True)

    maps = F.MapSource(refs, np.asarray([r['world_from_body'] for r in refs])[:, :3, 3],
                       radius=cfg.crop_radius, max_refs=800, max_maps=4096)
    scans = [P.prepare_scan(rec, cfg) for rec in frames]
    qs = [s[:min(ARB_POINTS, len(s))] for s in scans]
    print('[scans] %.1f s  median scan %d pts, query %d pts (deployed prefix slice)'
          % (time.time() - t0, int(np.median([len(s) for s in scans])), len(qs[0])), flush=True)

    gated = np.isfinite(ratio) & (ratio < AMB) & has_vis
    elig = np.array([bool(has_vis[i]) and all(
        len(maps.nearby(c)) > 0 for c in (T3[i, :3, 3], mid[i], vis[i][:3, 3]))
        for i in range(n)])
    print('[gate] %d frames enter arbitration (%d have a candidate); %d of 1502 are scorable '
          'in all three variants (a reference within 30 m at each centre)'
          % (int(gated.sum()), int(has_vis.sum()), int(elig.sum())), flush=True)

    sep_all = np.array([np.linalg.norm(vis[i][:3, 3] - T3[i, :3, 3])
                        for i in range(n) if has_vis[i]])
    sep_g = np.array([np.linalg.norm(vis[i][:3, 3] - T3[i, :3, 3])
                      for i in range(n) if gated[i]])

    # ---------------- warm every map the timed passes will touch -----------
    for i in range(n):
        if not elig[i]:
            continue
        score_own(T3[i], qs[i], maps, cfg)
        score_own(vis[i], qs[i], maps, cfg)
        maps.local(mid[i], maps.nearby(mid[i])[:cfg.reference_count])
    maps.hit = maps.miss = 0
    print('[warm] %.1f s' % (time.time() - t0), flush=True)

    # ---------------- three scorings, two alternating rounds ----------------
    def run_b0():
        c3, cv, tm = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n)
        for i in range(n):
            if not elig[i]:
                continue
            q = qs[i]
            t = time.perf_counter()
            c3[i] = score_own(T3[i], q, maps, cfg)
            cv[i] = score_own(vis[i], q, maps, cfg)
            tm[i] = (time.perf_counter() - t) * 1e3
        return c3, cv, tm, np.zeros(n, int)

    def run_a():
        c3, cv, tm, m = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n), np.zeros(n, int)
        for i in range(n):
            if not elig[i]:
                continue
            q = qs[i]
            t = time.perf_counter()
            mp = maps.local(mid[i], maps.nearby(mid[i])[:cfg.reference_count])
            m[i] = len(mp)
            tree = cKDTree(mp)
            c3[i] = score_shared(T3[i], q, tree)
            cv[i] = score_shared(vis[i], q, tree)
            tm[i] = (time.perf_counter() - t) * 1e3
        return c3, cv, tm, m

    def run_b():
        c3, cv, tm, m = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n), np.zeros(n, int)
        for i in range(n):
            if not elig[i]:
                continue
            q = qs[i]
            t = time.perf_counter()
            mp = own_map(T3[i], maps, cfg)
            m[i] = len(mp)
            tree = cKDTree(mp)
            c3[i] = score_shared(T3[i], q, tree)
            cv[i] = score_shared(vis[i], q, tree)
            tm[i] = (time.perf_counter() - t) * 1e3
        return c3, cv, tm, m

    runners = dict(B0_per_pose=run_b0, A_midpoint=run_a, B_route3=run_b)
    vals, times, mapn = {}, {k: [] for k in runners}, {}
    for rnd, order in enumerate((('B0_per_pose', 'A_midpoint', 'B_route3'),
                                 ('B_route3', 'A_midpoint', 'B0_per_pose'))):
        for name in order:
            c3, cv, tm, m = runners[name]()
            vals[name] = (c3, cv)
            times[name].append(tm)
            mapn[name] = m
            print('[round %d %-11s] warm median %.2f ms  p90 %.2f ms'
                  % (rnd, name, np.median(tm[elig]), np.percentile(tm[elig], 90)), flush=True)
    print('[cache] hit/miss during the timed rounds %d/%d' % (maps.hit, maps.miss), flush=True)
    for name in runners:
        print('[%-11s] map %d pts' % (name, int(np.median(mapn[name][elig]))), flush=True)

    # ---------------- where the time goes (gated frames, cold rebuild) ----
    parts = dict(rebuild=[], crop=[], tree=[], query2=[])
    for i in np.where(gated & elig)[0]:
        centre = T3[i, :3, 3]
        triple = maps.nearby(centre)[:cfg.reference_count]
        maps._map_cache.pop(triple, None)          # force a real merged-cloud build
        t = time.perf_counter()
        merged = F.voxel(np.concatenate([maps._ref_cloud(j) for j in triple]), 0.3)
        parts['rebuild'].append((time.perf_counter() - t) * 1e3)
        t = time.perf_counter()
        mp = merged[np.linalg.norm(merged - centre, axis=1) <= maps.radius]
        parts['crop'].append((time.perf_counter() - t) * 1e3)
        t = time.perf_counter()
        tree = cKDTree(mp)
        parts['tree'].append((time.perf_counter() - t) * 1e3)
        t = time.perf_counter()
        score_shared(T3[i], qs[i], tree)
        score_shared(vis[i], qs[i], tree)
        parts['query2'].append((time.perf_counter() - t) * 1e3)
    parts = {k: float(np.median(v)) for k, v in parts.items()}
    print('[breakdown] merged-cloud rebuild %.1f ms | crop %.1f | KD-tree %.1f | two queries %.1f'
          % (parts['rebuild'], parts['crop'], parts['tree'], parts['query2']), flush=True)

    # ---------------- fidelity of the replay ------------------------------
    m = np.isfinite(c3_ref)
    print('[fidelity] gated %d  max|dc3| %.2e  max|dcv| %.2e'
          % (int(m.sum()), np.nanmax(np.abs(c3b[m] - c3_ref[m])),
             np.nanmax(np.abs(cvb[m] - cv_ref[m]))), flush=True)

    w = SU.load_world()
    gt = w['hard_gt']
    assert w['hard_keys'] == [str(k) for k in frozen['keys']]
    err = lambda p, i: SU.pos_err(p[None], gt[i][None])[0]

    def decide(c3, cv):
        return gated & np.isfinite(c3) & np.isfinite(cv) & (cv < c3 - DELTA)

    acc_b = decide(c3b, cvb)
    print('[baseline] accept set identical to deployed: %s (%d frames)'
          % (bool((acc_b == acc_ref).all()), int(acc_b.sum())), flush=True)

    out = {}
    for name in ('B0_per_pose', 'A_midpoint', 'B_route3'):
        c3, cv = vals[name]
        tms = np.median(np.stack(times[name]), axis=0)
        acc = decide(c3, cv)
        pred = np.stack([vis[i] if acc[i] else T3[i] for i in range(n)])
        base_pred = np.stack([vis[i] if acc_b[i] else T3[i] for i in range(n)])
        t = SU.pos_err(pred, gt)
        r = SU.rot_deg(pred[:, :3, :3], gt[:, :3, :3])
        same_as_frozen = bool(np.abs(pred - frozen_pose).max() < 1e-12)
        flipped = np.where(acc != acc_b)[0]
        det = []
        for i in flipped:
            det.append(dict(i=int(i), key=str(frozen['keys'][i]),
                            sep=float(np.linalg.norm(vis[i][:3, 3] - T3[i, :3, 3])),
                            base_accept=bool(acc_b[i]), var_accept=bool(acc[i]),
                            err_before=float(err(base_pred, i)), err_after=float(err(pred, i)),
                            c3_b=float(c3b[i]), cv_b=float(cvb[i]),
                            c3_v=float(c3[i]), cv_v=float(cv[i]), ratio=float(ratio[i]),
                            margin_b=float(cvb[i] - c3b[i]), margin_v=float(cv[i] - c3[i])))
        gt_idx = gated & elig
        row = dict(variant=name, accepted=int(acc.sum()),
                   MPE=float(t.mean()), MOE=float(r.mean()),
                   ok1m2deg=int(((t <= 1) & (r <= 2)).sum()),
                   over5=int((t > 5).sum()), over10=int((t > 10).sum()),
                   ms_med=float(np.median(tms[elig])),
                   ms_p90=float(np.percentile(tms[elig], 90)),
                   ms_med_gated=float(np.median(tms[gt_idx])),
                   ms_p90_gated=float(np.percentile(tms[gt_idx], 90)),
                   flips=int(len(flipped)),
                   flip_on=int((acc & ~acc_b).sum()), flip_off=int((~acc & acc_b).sum()),
                   helped=int(sum(1 for d in det if d['err_after'] < d['err_before'] - 1e-9)),
                   hurt=int(sum(1 for d in det if d['err_after'] > d['err_before'] + 1e-9)),
                   dc3_med_abs=float(np.nanmedian(np.abs(c3 - c3b))),
                   dc3_max_abs=float(np.nanmax(np.abs(c3 - c3b))),
                   frozen_identical=same_as_frozen, flip_detail=det)
        out[name] = row
        print('[%-11s] MPE %.4f MOE %.4f ok %d >5m %d >10m %d | accepted %d | flips %d (+%d/-%d, '
              'helped %d hurt %d) | %.1f/%.1f ms warm (%.1f/%.1f gated)'
              % (name, row['MPE'], row['MOE'], row['ok1m2deg'], row['over5'], row['over10'],
                 row['accepted'], row['flips'], row['flip_on'], row['flip_off'],
                 row['helped'], row['hurt'], row['ms_med'], row['ms_p90'],
                 row['ms_med_gated'], row['ms_p90_gated']), flush=True)
        if same_as_frozen:
            print('             reproduces the frozen npz bit for bit', flush=True)
        for d in det:
            print('             flip %4d sep %8.2f m  err %.2f -> %.2f m  (margin %.3f -> %.3f)'
                  % (d['i'], d['sep'], d['err_before'], d['err_after'],
                     d['margin_b'], d['margin_v']), flush=True)
        with open(os.path.join(OUT, 'per_frame_%s.csv' % name), 'w') as fh:
            fh.write('i,key,gated,ratio,c3,cv,margin,accept,err_chosen,err_route3,err_vis,sep,ms\n')
            for i in range(n):
                fh.write('%d,%s,%d,%.6f,%.6f,%.6f,%.6f,%d,%.6f,%.6f,%.6f,%.3f,%.4f\n'
                         % (i, frozen['keys'][i], int(gated[i]), ratio[i], c3[i], cv[i],
                            cv[i] - c3[i], int(acc[i]), err(pred, i), err(T3[i], i),
                            err(vis[i], i) if has_vis[i] else -1,
                            float(np.linalg.norm(vis[i][:3, 3] - T3[i, :3, 3])) if has_vis[i] else -1,
                            tms[i]))

    t = SU.pos_err(T3, gt)
    r = SU.rot_deg(T3[:, :3, :3], gt[:, :3, :3])
    ctx = dict(MPE=float(t.mean()), MOE=float(r.mean()),
               ok1m2deg=int(((t <= 1) & (r <= 2)).sum()),
               over5=int((t > 5).sum()), over10=int((t > 10).sum()))
    sepq = {str(p): float(np.percentile(sep_all, p)) for p in (5, 10, 25, 50, 75, 90, 95, 99)}
    summary = dict(n=n, gated=int(gated.sum()), has_vis=int(has_vis.sum()), elig=int(elig.sum()),
                   route3_only=ctx, breakdown_ms=parts, sep_quantiles=sepq,
                   sep_cdf={str(k): float(np.mean(sep_all <= k)) for k in (0.5, 1, 2, 3, 5, 10, 20, 30)},
                   sep_gated={str(p): float(np.percentile(sep_g, p)) for p in (5, 25, 50, 75, 90, 95)},
                   sep_gated_cdf={str(k): float(np.mean(sep_g <= k)) for k in (1, 2, 5)},
                   variants=out, seconds=round(time.time() - t0, 1))
    with open(os.path.join(OUT, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=1)
    print('[route3-only]', json.dumps(ctx), flush=True)
    print('[sep]', json.dumps(sepq), flush=True)
    print('[cdf]', json.dumps(summary['sep_cdf']), flush=True)
    print('[sep gated]', json.dumps({k: round(v, 2) for k, v in summary['sep_gated'].items()}), flush=True)
    print('[done] %.1f s' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()

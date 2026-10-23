"""Cross-view consistency features for the geometric along-track correction.

The search scores a candidate pose once, against one merged map, with one query
subset. That single number can improve while the pose moves further from the truth,
because the scan and the map share the same bias. The features here measure the same
improvement several times over disjoint parts of the evidence:

    eight azimuth sectors of the scan, each scored at the incoming and the corrected pose
    each reference cloud scored on its own, plus the spread across the triple

A correction that is real improves most sectors and every reference. One that is driven
by a map bias improves the sectors facing the map's weak direction and hurts the rest.

Everything is computed from the scan, the local map, and the two poses. No truth, no
extra sensor, no second map.

Contract: the caller supplies the same objects the search built, so nothing is
recomputed. `features()` needs

    tree          scipy cKDTree over `map_points`
    map_points    the local reference cloud the search scored against
    scan          the prepared scan (metres, voxel downsampled by the caller)
    T0, T1        4x4 incoming and corrected poses
    along, side   unit along-track and lateral directions used by the search
    surface       17 x 3 coarse cost surface, optional
    s, d          the chosen along-track and lateral offsets, optional
    refs          list of reference clouds, one per reference scan, optional

Missing optional inputs become NaN and are filled with the training mean by the gate.
"""
import numpy as np

TRUNC = 1.0
S_COARSE = np.arange(-8.0, 8.0001, 1.0)
N_FINE = 2000
N_REF = 800
N_SECTORS = 8


def spread_subset(points, n):
    """Evenly spaced subset from the first to the last point of the sweep."""
    if len(points) <= n:
        return points
    idx = np.linspace(0, len(points) - 1, num=n)
    return points[idx.astype(np.int64)]


def _dist(tree, points, workers=4):
    d = tree.query(points, k=1, workers=workers, distance_upper_bound=TRUNC)[0]
    return np.where(np.isfinite(d), d, TRUNC)


def _cost(tree, points, workers=4):
    return float(_dist(tree, points, workers).mean())


def _normals(tree, points, k=10):
    _, idx = tree.query(points, k=min(k, len(points)), workers=4)
    nb = points[idx]
    nb = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', nb, nb) / max(nb.shape[1], 1)
    _, vec = np.linalg.eigh(cov)
    return vec[:, :, 0]


def features(tree, map_points, scan, T0, T1, along, side, surface=None, s=None, d=None,
             refs=None, workers=4):
    """Return the feature dict the gate expects."""
    out = {}
    q = spread_subset(scan, N_FINE)
    rot = np.asarray(T0, float)[:3, :3]
    centre = np.asarray(T0, float)[:3, 3]
    target = np.asarray(T1, float)[:3, 3]
    base = q @ rot.T
    xw0 = base + centre
    xw1 = base + target
    d0 = _dist(tree, xw0, workers)
    d1 = _dist(tree, xw1, workers)
    c0, c1 = float(d0.mean()), float(d1.mean())
    corr = target - centre
    out['cost0'] = c0
    out['cost1'] = c1
    out['gain'] = c0 - c1
    out['shift'] = float(np.linalg.norm(corr))

    # --- residual distribution shift
    out['d_med_0'] = float(np.median(d0)); out['d_med_1'] = float(np.median(d1))
    out['d_p90_0'] = float(np.percentile(d0, 90)); out['d_p90_1'] = float(np.percentile(d1, 90))
    out['d_frac10_0'] = float((d0 < 0.1).mean()); out['d_frac10_1'] = float((d1 < 0.1).mean())

    # --- disjoint halves: does the gain survive on each?
    half = np.zeros(len(xw0), bool); half[::2] = True
    g = []
    for m in (half, ~half):
        g.append(_cost(tree, xw0[m], workers) - _cost(tree, xw1[m], workers))
    out['gain_half_min'] = float(min(g))
    out['gain_half_gap'] = float(abs(g[0] - g[1]))

    # --- azimuth sectors
    ang = np.arctan2(q[:, 1], q[:, 0])
    sec = ((ang + np.pi) / (2 * np.pi) * N_SECTORS).astype(int) % N_SECTORS
    gs = []
    for t in range(N_SECTORS):
        m = sec == t
        if m.sum() < 50:
            continue
        gs.append(_cost(tree, xw0[m], workers) - _cost(tree, xw1[m], workers))
    out['sector_frac_pos'] = float(np.mean(np.array(gs) > 0)) if gs else 0.0
    out['sector_gain_med'] = float(np.median(gs)) if gs else 0.0
    out['sector_gain_min'] = float(np.min(gs)) if gs else 0.0

    # --- point-to-plane information at the corrected pose
    dist, idx = tree.query(xw1, k=1, workers=workers, distance_upper_bound=1.0)
    ok = np.isfinite(dist)
    if ok.sum() >= 30:
        n = _normals(tree, map_points)[idx[ok]]
        r = ((xw1[ok] - map_points[idx[ok]]) * n).sum(1)
        w = (np.abs(r) < 0.3).astype(float)
        H = np.einsum('n,ni,nj->ij', w, n, n) / max(w.sum(), 1)
        ev, evec = np.linalg.eigh(H)
        norm = np.linalg.norm(corr)
        cn = corr / norm if norm > 1e-6 else np.zeros(3)
        out['lambda_min'] = float(ev[0]); out['lambda_max'] = float(ev[-1])
        out['cond'] = float(ev[0] / max(ev[-1], 1e-12))
        out['cos_weak'] = float(abs(cn @ evec[:, 0])) if norm > 1e-6 else 0.0
        out['cos_u'] = float(abs(cn @ along)) if norm > 1e-6 else 0.0
        out['info_u'] = float(along @ H @ along)
        out['info_lat'] = float(side @ H @ side)
        out['frac_inlier'] = float(w.mean())
        out['resid_med'] = float(np.median(np.abs(r)))
        out['resid_frac05'] = float((np.abs(r) > 0.5).mean())
    else:
        for k in ('lambda_min', 'lambda_max', 'cond', 'cos_weak', 'cos_u', 'info_u',
                  'info_lat', 'frac_inlier', 'resid_med', 'resid_frac05'):
            out[k] = np.nan

    # --- per-reference gains
    gains = []
    if refs:
        qs = spread_subset(scan, N_REF)
        bs = qs @ rot.T
        for j, cloud in enumerate(refs[:3]):
            if cloud is None or len(cloud) < 200:
                out['ref%d_gain' % j] = np.nan
                continue
            from scipy.spatial import cKDTree
            tr = cKDTree(cloud)
            out['ref%d_gain' % j] = _cost(tr, bs + centre, workers) - _cost(tr, bs + target, workers)
            gains.append(out['ref%d_gain' % j])
    for j in range(3):
        out.setdefault('ref%d_gain' % j, np.nan)
    fin = [x for x in gains if np.isfinite(x)]
    out['ref_all_pos'] = float(all(x > 0 for x in fin)) if fin else np.nan
    out['ref_min_gain'] = float(min(fin)) if fin else np.nan
    out['ref_gain_spread'] = float(np.std(fin)) if fin else np.nan

    # --- chosen offsets and coarse-surface shape
    out['s'] = float(s) if s is not None else np.nan
    out['d'] = float(d) if d is not None else np.nan
    if surface is not None:
        surf = np.asarray(surface, float)
        smin = float(surf.min())
        a0, b0 = np.unravel_index(int(surf.argmin()), surf.shape)
        col = surf[:, b0]
        mid = len(S_COARSE) // 2
        out['surface_min'] = smin
        out['surface_depth'] = float(
            np.mean([surf[mid, k] for k in range(surf.shape[1])]) - smin)
        out['surface_curv'] = (float(col[a0 + 1] + col[a0 - 1] - 2 * col[a0])
                               if 1 <= a0 < len(col) - 1 else np.nan)
        out['surface_basin'] = float((col <= smin + 0.01).sum())
        out['surface_rivals'] = float((surf.min(axis=1) <= smin + 0.02).sum() - 1)
        out['surface_edge'] = float(a0 == 0 or a0 == len(col) - 1)
    else:
        for k in ('surface_min', 'surface_depth', 'surface_curv', 'surface_basin',
                  'surface_rivals', 'surface_edge'):
            out[k] = np.nan
    return out

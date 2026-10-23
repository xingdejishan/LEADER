"""Learned three-way router: geometry expert, visual expert, or keep the front-end pose.

This replaces the hand-set switching test of arbitration.decide() (C_incumbent - C_best >
k * sigma_C, k = 8.4) with a discrete choice made by a small MLP over cross-modal
evidence.  The output is one of three arms -- T0 (the front-end pose the search
started from), T_geo (the geometry expert), T_vis (the visual expert) -- and the
chosen arm's 4x4 is returned untouched: no pose is ever averaged or blended.

Nothing here reads ground truth.  Weights are fitted offline on per-frame features
of the two experts plus the GT-based per-frame loss, with a leave-one-date-out
protocol; at inference the module only sees the feature dict below.

Feature contract (44 values, all produced by the driver at run time)
    geometry   c_geo, c_t0, c_geo/c_t0, sigma_C, sigma_C std, along-track ambiguity
               ratio rho, coarse descent depth, cost-surface second/best basin ratio,
               flat-cell fraction, basin along-track offset, support fraction, map
               points, fine search ran
    visual     availability, fair score, verified flag, candidate count, inliers,
               inlier ratio, median reprojection px, coverage cells, multi-reference
               consistency count, laser residual median / p90, supported bins,
               top-bin share, fair margin over the runner-up, reference count,
               PnP calls, retrieval cosine (top-1 and mean), PnP channel one-hot
    cross      C_vis - C_geo, C_vis - C_t0, C_geo - C_t0, |t_vis - t_geo|,
               rot(t_vis, t_geo), |t_vis - t0|, rot(t_vis, t0), |t_geo - t0|,
               rot(t_geo, t0), visual support fraction
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, 'results', 'router-weights.npz')

ARMS = ('T0', 'T_geo', 'T_vis')

FEATURES = [
    # geometry expert
    'c_geo', 'c_t0', 'c_geo_ratio_t0', 'sigma_c', 'sigma_std', 'rho', 'gain',
    'surf_ratio2', 'surf_flat', 'surf_ds', 'supp_geo_frac', 'n_map',
    # visual expert
    'has_visual', 'fair', 'verified', 'n_hyp', 'inliers', 'inlier_ratio',
    'reproj_px', 'coverage_cells', 'consensus', 'resid_med', 'resid_p90',
    'supported_bins', 'top_bin_share', 'fair_gap', 'n_refs', 'pnp_calls',
    'cos_top1', 'cos_mean', 'ch_pair', 'ch_full', 'ch_cluster',
    # cross-modal
    'dcost', 'c_vis_minus_t0', 'c_geo_minus_t0', 'dist_vg', 'rot_vg',
    'dist_v0', 'rot_v0', 'dist_g0', 'rot_g0', 'supp_vis_frac',
]

LOG_FEATURES = ('c_geo', 'c_t0', 'sigma_c', 'sigma_std', 'gain', 'n_map', 'inliers',
                'reproj_px', 'resid_med', 'resid_p90', 'dist_vg', 'dist_v0', 'dist_g0',
                'fair_gap')

VISUAL_ONLY = ('fair', 'verified', 'n_hyp', 'inliers', 'inlier_ratio', 'reproj_px',
               'coverage_cells', 'consensus', 'resid_med', 'resid_p90', 'supported_bins',
               'top_bin_share', 'fair_gap', 'n_refs', 'pnp_calls', 'cos_top1', 'cos_mean',
               'dcost', 'c_vis_minus_t0', 'dist_vg', 'rot_vg', 'dist_v0', 'rot_v0',
               'supp_vis_frac')


def _f(v, d=np.nan):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return d
    return v if np.isfinite(v) else d


def extract(diag):
    """One feature dict from the driver's per-frame diagnostics.  NaN = arm absent."""
    def g(k, d=np.nan):
        return _f(diag.get(k, d), np.nan)

    has = 'visual_score' in diag
    ch = str(diag.get('visual_channel') or '')
    c_geo, c_t0, c_vis = g('c_geo'), g('c_t0'), g('c_vis_ev')
    f = dict(c_geo=c_geo, c_t0=c_t0,
             c_geo_ratio_t0=(c_geo / c_t0 if c_t0 > 1e-9 else np.nan),
             sigma_c=g('arb_sigma'), sigma_std=g('arb_sigma_std'), rho=g('ratio'),
             gain=g('gain'), surf_ratio2=g('surf_ratio2'), surf_flat=g('surf_flat'),
             surf_ds=g('surf_ds'),
             supp_geo_frac=g('supp_geo') / max(g('n_query'), 1.0),
             n_map=g('n_map'), ran_fine=1.0 if diag.get('ran_fine') else 0.0,
             has_visual=1.0 if has else 0.0,
             fair=g('visual_score'), verified=1.0 if diag.get('visual_verified') else 0.0,
             n_hyp=g('visual_candidates'), inliers=np.nan,
             inlier_ratio=g('visual_inlier_ratio'), reproj_px=g('visual_reproj_px'),
             coverage_cells=g('visual_coverage_cells'), consensus=np.nan,
             resid_med=np.nan, resid_p90=np.nan, supported_bins=np.nan,
             top_bin_share=np.nan, fair_gap=g('visual_fair_gap'),
             n_refs=g('visual_n_refs'), pnp_calls=g('visual_pnp_calls'),
             cos_top1=g('retr_cos_top1'), cos_mean=g('retr_cos_mean'),
             ch_pair=1.0 if ch == 'pair' else 0.0,
             ch_full=1.0 if ch == 'full' else 0.0,
             ch_cluster=1.0 if ch == 'cluster' else 0.0,
             dcost=(c_vis - c_geo) if np.isfinite(c_vis) and np.isfinite(c_geo) else np.nan,
             c_vis_minus_t0=(c_vis - c_t0) if np.isfinite(c_vis) and np.isfinite(c_t0) else np.nan,
             c_geo_minus_t0=(c_geo - c_t0) if np.isfinite(c_geo) and np.isfinite(c_t0) else np.nan,
             dist_vg=g('dist_vg'), rot_vg=g('rot_vg'), dist_v0=g('dist_v0'),
             rot_v0=g('rot_v0'), dist_g0=g('dist_g0'), rot_g0=g('rot_g0'),
             supp_vis_frac=g('supp_vis') / max(g('n_query'), 1.0))
    scored = diag.get('visual_scored') or []
    if scored:
        top = scored[0]
        f.update(inliers=_f(top.get('inliers')), consensus=_f(top.get('consensus')),
                 resid_med=_f(top.get('residual_median_m')),
                 resid_p90=_f(top.get('residual_p90_m')),
                 supported_bins=_f(top.get('supported_bins')),
                 top_bin_share=_f(top.get('top_bin_share')))
    if not has:
        for k in VISUAL_ONLY:
            f[k] = np.nan
        f['ch_pair'] = f['ch_full'] = f['ch_cluster'] = 0.0
    return f


def to_vector(f, names=None):
    names = FEATURES if names is None else names
    return np.asarray([f.get(n, np.nan) for n in names], float)


class Router(object):
    """The deployed model: standardise, two ReLU layers, mask the absent arm, argmax."""

    def __init__(self, path=WEIGHTS):
        z = np.load(path, allow_pickle=True)
        self.names = [str(x) for x in z['names']]
        self.log = set(str(x) for x in z['log'])
        self.med = np.asarray(z['med'], float)
        self.scale = np.asarray(z['scale'], float)
        self.params = []
        i = 0
        while ('w%d' % i) in z.files:
            self.params.append((np.asarray(z['w%d' % i], float),
                                np.asarray(z['b%d' % i], float)))
            i += 1
        self.nparam = int(sum(w.size + b.size for w, b in self.params))

    def vector(self, diag):
        f = extract(diag)
        v = np.asarray([f.get(n, np.nan) for n in self.names], float)
        for j, n in enumerate(self.names):
            if n in self.log and np.isfinite(v[j]):
                v[j] = np.log1p(max(v[j], 0.0))
        return v

    def logits(self, x):
        z = (np.asarray(x, float) - self.med) / self.scale
        z = np.where(np.isfinite(z), z, 0.0)
        z = np.clip(z, -6.0, 6.0)
        for a, (w, b) in enumerate(self.params):
            z = z @ w.T + b          # torch stores Linear weight as (out, in)
            if a < len(self.params) - 1:
                z = np.maximum(z, 0.0)
        return z

    def probs(self, x, vis_ok=True):
        lg = self.logits(x)
        if not vis_ok:
            lg[2] = -1e9
        lg = lg - lg.max()
        p = np.exp(lg)
        return p / p.sum()

    def decide(self, diag, vis_ok):
        """(arm_index, probabilities, diagnostics). Discrete choice only."""
        v = self.vector(diag)
        p = self.probs(v, vis_ok=bool(vis_ok))
        k = int(np.argmax(p))
        return k, p, dict(router_choice=ARMS[k], router_prob=[float(x) for x in p],
                          router_vis_ok=bool(vis_ok),
                          router_features=[float(np.nan_to_num(x, nan=-999.0))
                                           for x in v])


def save_weights(path, names, log, med, scale, params):
    out = dict(names=np.array(names), log=np.array(sorted(log)),
               med=np.asarray(med, float), scale=np.asarray(scale, float))
    for i, (w, b) in enumerate(params):
        out['w%d' % i] = np.asarray(w, float)
        out['b%d' % i] = np.asarray(b, float)
    np.savez(path, **out)
    return path

"""Hypothesis generation, shared evidence, calibrated switching test.

The pose problem is still the minimisation of one number: the mean truncated
nearest-neighbour distance between the transformed scan and the local reference
cloud (truncation 1 m). What changes is how a second candidate is judged.

Old rule (three_route_unified.py before 2026-10-23): each branch scored its own
pose on its own crop of the map, and the switch was taken when

    C_visual < C_route3 - delta          delta = 0.05 m, set by hand

delta was a hand-fitted constant: large enough not to chase noise, small enough
not to miss the disaster frames, and it says nothing about how much the cost is
allowed to move for reasons unrelated to the candidate being worse.

New rule: one Evidence object per frame (the map and KD-tree the search already
assembled), hypotheses are scored on that same evidence, and the switching
threshold is read off the local roughness of the cost surface:

    sigma_C = calibrate(evidence, scan, incumbent_pose)   # 12 tiny perturbations
    switch  iff  C_incumbent - C_alternate > k * sigma_C

sigma_C is the dispersion of C over 12 perturbations of the incumbent pose
(+-0.1 m on three axes, +-0.5 deg about three axes). It is the amount by which
the cost moves when the pose is known to be fine, i.e. the irreducible jitter of
this statistic on this frame. The switching test then asks whether the measured
advantage of the other hypothesis clears that jitter by a factor k. The only free
number left is k, which is dimensionless and set once for the whole run:

    k = 0.05 m / median(sigma_C) so that k * sigma_C ~= the old delta

Nothing here is trained and nothing is cached across frames.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

COST_TRUNC = 1.0        # truncation of the shared cost, metres (unchanged)
ARB_POINTS = 4000       # query points for scoring a hypothesis (unchanged)
PERT_TRANS = 0.1        # calibration: +-0.1 m on each axis
PERT_ROT_DEG = 0.5      # calibration: +-0.5 deg about each axis
MAD_TO_SIGMA = 1.4826   # normal-consistent scale factor for the MAD
SIGMA_FLOOR = 1e-6      # keeps the threshold positive on a flat cost surface


class Hypothesis:
    """One candidate pose plus where it came from.

    pose    4x4 world_from_body
    source  short tag, e.g. 'route3' (geometry) or 'visual'
    meta    free-form dict; meta['incumbent'] marks the pose to be displaced,
            the rest is carried into the diagnostics untouched
    cost    filled in by decide(), None until then
    """

    __slots__ = ('pose', 'source', 'meta', 'cost')

    def __init__(self, pose, source, meta=None):
        p = np.asarray(pose, dtype=np.float64)
        if p.shape != (4, 4):
            raise ValueError('pose must be 4x4, got %r' % (p.shape,))
        self.pose = p.copy()
        self.source = str(source)
        self.meta = dict(meta or {})
        self.cost = None

    def __repr__(self):
        c = 'None' if self.cost is None else '%.6f' % self.cost
        return 'Hypothesis(%s, cost=%s)' % (self.source, c)


def union_maps(*sets):
    """One evidence cloud from several local-map crops of the same frame.

    A hypothesis can sit far from the initial pose (the visual candidates on the
    frames that need rescuing sit up to 400 m away), and a crop taken around the
    initial pose then contains nothing near it: every scan point falls outside the
    truncation and the cost saturates at 1.0 m, which is not evidence, it is a
    handicap. Taking the union of the crops each hypothesis would have used on its
    own keeps the evidence identical for all hypotheses without deleting any of it.

    Duplicate points are kept: the cost only reads the nearest distance per query
    point, so a repeated map point cannot change its value.
    """
    return np.concatenate([np.asarray(s, dtype=np.float64) for s in sets])


class Evidence:
    """One frame of shared evidence: the local map, its KD-tree, and the query set.

    The map is built from this frame's own crops (see union_maps), so both
    hypotheses are scored against exactly the same points and no frame pays for a
    second map assembly rule. `tree` can be handed in when the caller already
    built one for the same points.

    `score` reproduces the project cost function bit for bit:

        C(pose) = mean( min(|p_i - NN(M, R p_i + t)|, 1.0) )

    with the same truncation, the same query points (the first 4000 prepared scan
    points, exactly as the previous arbitration used) and the same KD-tree upper
    bound trick, which does not change the value.
    """

    def __init__(self, map_points, scan=None, trunc=COST_TRUNC, n_points=ARB_POINTS,
                 tree=None):
        self.points = np.asarray(map_points, dtype=np.float64)
        if len(self.points) == 0:
            raise ValueError('evidence map is empty')
        # reuse the frame's tree when the caller already built one: the map is
        # the search's map, so the tree is the search's tree
        self.tree = cKDTree(self.points) if tree is None else tree
        self.trunc = float(trunc)
        self.n_points = int(n_points)
        self.scan = None if scan is None else np.asarray(scan, dtype=np.float64)
        self.base = self.query(self.scan)
        self.n_map = len(self.points)
        self.n_query = 0 if self.base is None else len(self.base)

    def query(self, scan):
        """The query subset: first n_points points of the prepared scan."""
        if scan is None:
            return None
        scan = np.asarray(scan, dtype=np.float64)
        return scan[:min(self.n_points, len(scan))]

    def score(self, pose, base=None):
        """Mean truncated nearest-neighbour distance of `pose`, in metres."""
        q = self.base if base is None else base
        if q is None:
            raise ValueError('no query points: build the evidence with a scan')
        t = np.asarray(pose, dtype=np.float64)
        pts = q @ t[:3, :3].T + t[:3, 3]
        d = self.tree.query(pts, k=1, workers=1,
                            distance_upper_bound=self.trunc)[0]
        return float(np.where(np.isfinite(d), d, self.trunc).mean())


def perturbations(pose, trans=PERT_TRANS, rot_deg=PERT_ROT_DEG):
    """12 small poses around `pose`: 3 axes x +-translation, 3 axes x +-rotation.

    Translations are world-frame offsets in metres; rotations are world-frame
    rotations in degrees about the pose origin (z is the yaw axis of this map).
    The nominal pose itself is not included, so the dispersion is measured about
    the perturbed cloud, not about the nominal one.
    """
    t = np.asarray(pose, dtype=np.float64)
    out = []
    for a in range(3):
        for sgn in (+1.0, -1.0):
            p = t.copy()
            p[a, 3] += sgn * trans
            out.append(p)
    for a in range(3):
        for sgn in (+1.0, -1.0):
            axis = np.zeros(3)
            axis[a] = 1.0
            R = Rotation.from_rotvec(np.deg2rad(sgn * rot_deg) * axis).as_matrix()
            p = t.copy()
            p[:3, :3] = R @ t[:3, :3]
            out.append(p)
    return out


def calibrate(evidence, scan, pose, trans=PERT_TRANS, rot_deg=PERT_ROT_DEG,
              points=None):
    """Dispersion of C around `pose`, metres.

    Scores the 12 perturbations of `pose` on `evidence` and returns both
    estimators of the scale:

        sigma_mad = 1.4826 * median(|C - median(C)|)     robust, the primary one
        sigma_std = population standard deviation         reported alongside it

    MAD is the default because a perturbation can land on a step of the cost
    surface and drag the standard deviation with it, while the median absolute
    deviation stays on the bulk of the twelve. `scan` is accepted so a caller can
    calibrate on an explicit query set; when it is None the evidence's own query
    points are reused.
    """
    base = evidence.query(scan) if scan is not None else None
    poses = perturbations(pose, trans=trans, rot_deg=rot_deg)
    c = np.asarray([evidence.score(p, base=base) for p in poses], dtype=np.float64)
    centre = float(np.median(c))
    mad = float(np.median(np.abs(c - centre)))
    sigma_mad = MAD_TO_SIGMA * mad
    sigma_std = float(c.std())
    return {
        'n': int(len(c)),
        'c_median': centre,
        'c_mean': float(c.mean()),
        'c_min': float(c.min()),
        'c_max': float(c.max()),
        'mad': mad,
        'sigma_mad': sigma_mad,
        'sigma_std': sigma_std,
        'sigma_c': max(sigma_mad, SIGMA_FLOOR),   # the one the test uses
    }


def decide(hypotheses, evidence, k, scan=None, points=None):
    """Pick a hypothesis with a calibrated switching test.

    All hypotheses are scored on the shared evidence. The first hypothesis marked
    meta['incumbent'] (else the first in the list) is the incumbent: the pose that
    keeps its place unless something clearly better shows up. An alternate is
    adopted only when

        C_incumbent - C_alternate  >  k * sigma_C

    with sigma_C calibrated about the incumbent, so the threshold is in metres of
    the same cost and scales with how rough this frame's cost surface is. The
    ambiguity ratio rho does not enter here at all; it only decides whether the
    other modality is asked in the first place.

    Returns (selected_hypothesis, diagnostics).
    """
    if not hypotheses:
        raise ValueError('decide() needs at least one hypothesis')
    k = float(k)

    costs = [evidence.score(h.pose) for h in hypotheses]
    for h, c in zip(hypotheses, costs):
        h.cost = c
    names = [h.source for h in hypotheses]

    inc = 0
    for a, h in enumerate(hypotheses):
        if h.meta.get('incumbent'):
            inc = a
            break
    win = int(np.argmin(costs))
    margin = float(costs[inc] - costs[win])

    cal = None
    thresh = 0.0
    switched = False
    if len(hypotheses) > 1:
        cal = calibrate(evidence, scan, hypotheses[inc].pose)
        thresh = k * cal['sigma_c']
        switched = bool(win != inc and margin > thresh)

    sel = hypotheses[win] if switched else hypotheses[inc]
    diag = {
        'decided': len(hypotheses) > 1,
        'costs': dict(zip(names, [float(c) for c in costs])),
        'names': names,
        'incumbent': names[inc],
        'best': names[win],
        'margin': margin,
        'k': k,
        'sigma_c': None if cal is None else cal['sigma_c'],
        'sigma_c_std': None if cal is None else cal['sigma_std'],
        'sigma_c_mad_raw': None if cal is None else cal['mad'],
        'sigma_c_n': None if cal is None else cal['n'],
        'thresh': thresh,
        'switched': switched,
        'chosen': sel.source,
    }
    return sel, diag

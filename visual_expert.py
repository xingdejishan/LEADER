"""Visual expert that shares where the frame is, and nothing about which way it looks.

Two variants, kept apart on purpose:

  pose-free (2026-10-23, round 1)  select_references_posefree(): the whole 4752-frame
      library ranked by the DINOv2 CLS descriptor alone.  A reference within 25 m of the
      true position on 138/173 diagnostic frames at K=20, but the gate-open frames lost
      about half of the rescue the cascade gets, because with no position channel the
      retrieval falls into a different place on the hard frames.

  position-gated (this round)  select_references_by_position(): the candidate set is the
      ball of radius R around the position the geometry expert reports, and the ranking
      inside the ball is the same CLS cosine.  That is the entire change.  R = 25 m is the
      radius of the verification crop, i.e. the scale at which the library can still
      answer for a place.

Why this is still not a pose prior.  Three independent guarantees, enforced by code and by
measurement rather than promised:

  1. Signature.  select_references_by_position(query, position) and
     PoseFreeVisualExpert.hypothesize(image_path, position) accept a position.  No
     argument in this module can hold an orientation: require_position() copies its input
     and raises unless the result is a finite 1-D array of length 3, so a (3,3) rotation,
     a (4,4) pose, an (N,3) batch and a homogeneous 4-vector are all refused by shape
     before any arithmetic happens.
  2. Call site.  moe_parallel.one() builds `geo_position = np.array(T3[:3, 3])` -- a fresh
     copy of three numbers -- and passes that name.  The 4x4 pose is never an argument, so
     no view of it exists inside the call.
  3. Measurement.  The same image with the same translation and three different
     orientations (identity, the geometry rotation, a random one) returns bit-identical
     references and a bit-identical pose; the audit prints the maximum absolute difference
     (0.0 when there is no leak).

The solve itself is untouched: EPnP + RANSAC + LM from correspondences, no initial guess,
no camera centre, no rotation handed in (visual_branch.solve -> C.solve_pnp(world, pixels,
ids, T_BC) takes no prior argument at all).  Acceptance is the producer's own qualification
test -- >=15 unique inliers, >=6 of the 16 coverage cells, >=0.2 inlier ratio -- and the
one with the most unique inliers among the qualified solutions wins.
"""
import os
from collections import OrderedDict

import numpy as np

import visual_branch as VB

TOP_K = 20
POSITION_RADIUS = 25.0


class OrderedBranch(VB.VisualBranch):
    """The frozen branch, with its two caches actually usable for more than 64 entries.

    visual_branch builds self._features and self._projections as plain dicts and evicts
    with dict.popitem(last=False), which only exists on OrderedDict: the first full run
    died on 1458 of 1502 frames with

        TypeError: popitem() takes no keyword arguments

    as soon as either cache passed 64 entries.  Nothing else about the branch changes.
    """

    def __init__(self, *args, **kwargs):
        super(OrderedBranch, self).__init__(*args, **kwargs)
        self._features = OrderedDict()
        self._projections = OrderedDict()


def require_position(position):
    """The one prior this module is allowed to see: a point in metres, and only that.

    Raises
    ------
    TypeError   if the argument is a container that could carry a pose (dict, or an object
                such as a pose record) rather than raw numbers.
    ValueError  unless the result is exactly shape (3,), finite.  Every shape a rotation or
                a pose can arrive in is two-dimensional (3x3, 4x4, Nx3) or the wrong
                length (quaternion 4, homogeneous 4), so all of them fail here.
    """
    if isinstance(position, dict):
        raise TypeError('position must be metres, not a dict: %r' % (sorted(position),))
    try:
        arr = np.array(position, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError('position must be convertible to 3 floats: %r' % (exc,))
    if arr.shape != (3,):
        raise ValueError('position must have shape (3,), got %r -- a rotation matrix, a '
                         '4x4 pose, an Nx3 batch or a homogeneous 4-vector cannot pass '
                         'this check' % (arr.shape,))
    if not np.isfinite(arr).all():
        raise ValueError('position contains a non-finite value: %r' % (arr,))
    return arr


def position_candidates(branch, position, radius):
    """Library indices inside the ball of `radius` metres around `position`, and distances."""
    point = require_position(position)
    distance = np.linalg.norm(branch.positions - point[None, :], axis=1)
    return np.flatnonzero(distance <= float(radius)), distance


def select_references_by_position(query, position, branch, top_k=TOP_K,
                                  radius=POSITION_RADIUS):
    """Position decides WHERE to look, the image descriptor decides WHICH frames.

    The candidate set is every library frame within `radius` metres of `position`; inside
    it the frames are ranked by the DINOv2 CLS cosine against `query` and the top `top_k`
    are returned.  If the ball holds fewer than `top_k` frames, all of them are used --
    the radius is not expanded, so the candidate count is whatever the geometry gives.

    `position` is the only thing taken from the geometric pose; require_position() makes
    that literal.  Returns (reference indices, detail).
    """
    image_path = query['image_path'] if isinstance(query, dict) else query
    point = require_position(position)
    cand, distance = position_candidates(branch, point, radius)
    k = int(top_k)
    detail = dict(rule='position-ball-then-cls', radius=float(radius),
                  candidates=int(cand.size), top_k=k)
    if cand.size == 0:
        detail.update(n_selected=0, cosine=[], candidate_distance_m=[])
        return [], detail
    similarity = branch.reference_cls[cand] @ branch.embed(image_path).astype(np.float32)
    order = np.argsort(-similarity)[:k]
    refs = [int(cand[i]) for i in order]
    detail.update(cosine=[float(similarity[i]) for i in order],
                  candidate_distance_m=[float(distance[i]) for i in order],
                  n_selected=len(refs))
    return refs, detail


def select_references_posefree(query, branch, top_k=TOP_K):
    """Round 1's variant: global CLS ranking over the whole library, no position at all.

    Kept because that ablation is what tells us whether the position channel is worth its
    loss of independence.  Round 3 calls this again, and puts the provider's consensus
    acceptance back around it (see candidates_posefree below).
    """
    image_path = query['image_path'] if isinstance(query, dict) else query
    k = int(top_k)
    similarity = branch.reference_cls @ branch.embed(image_path).astype(np.float32)
    order = np.argsort(-similarity)[:k]
    refs = [int(i) for i in order]
    return refs, dict(rule='cls-cosine-top%d' % k,
                      cosine=[float(similarity[i]) for i in refs],
                      candidates=int(len(similarity)), n_selected=len(refs))


class PoseFreeVisualExpert(object):
    """One pose per image, from the image, the library and one point in metres."""

    def __init__(self, branch=None, top_k=TOP_K, radius=POSITION_RADIUS, device='cpu',
                 scratch='/tmp/dsh-posefree-vb'):
        self.branch = branch if branch is not None else OrderedBranch(
            device=device, scratch=scratch, torch_threads=2)
        self.top_k = int(top_k)
        self.radius = float(radius)

    def position_candidates(self, position):
        """Just the ball, no image work: how many references the geometry admits."""
        cand, _ = position_candidates(self.branch, position, self.radius)
        return cand

    def select_references(self, query, position, top_k=None):
        return select_references_by_position(
            query, position, self.branch, self.top_k if top_k is None else top_k,
            self.radius)

    def hypothesize(self, image_path, position):
        """One Hypothesis worth of pose, or None when the image cannot support one.

        `position` picks the reference set.  It never seeds the solver and it is never
        turned into a rotation: the caller holds the 4x4, this call receives 3 numbers.
        """
        branch = self.branch
        refs, retrieval = self.select_references(image_path, position)
        if not refs:
            return None
        q_model, candidates, per_pair, batched = branch._match_frame(image_path, refs)
        if not candidates:
            return None
        by_reference = {}
        for query_index, _, reference, attribute in candidates:
            entry = by_reference.setdefault(reference, ([], []))
            entry[0].append(query_index)
            entry[1].append(attribute)
        by_reference = {ri: (np.array(ids), np.array(aid))
                        for ri, (ids, aid) in by_reference.items()}
        hypotheses, accepted, calls, capped = branch._hypotheses(q_model, candidates,
                                                                 by_reference)
        qualified = [h for h in hypotheses if h['qualified']]
        if not qualified:
            return None
        qualified.sort(key=lambda s: (-s['unique_query_inliers'], -s['coverage_cells'],
                                      s['median_reprojection_px']))
        best = qualified[0]
        return dict(pose=np.asarray(best['pose'], dtype=np.float64),
                    detail=dict(rule=retrieval['rule'], top_k=self.top_k,
                                radius=self.radius,
                                candidates=int(retrieval['candidates']),
                                n_refs=int(retrieval['n_selected']),
                                cosine=retrieval['cosine'], capped=int(capped),
                                hypotheses=len(hypotheses),
                                qualified=len(qualified), accepted=len(accepted),
                                pnp_calls=int(calls),
                                channel=best.get('channel'),
                                inliers=int(best['unique_query_inliers']),
                                inlier_ratio=float(best['inlier_ratio']),
                                coverage_cells=int(best['coverage_cells']),
                                reprojection_px=float(best['median_reprojection_px']),
                                consensus=int(best.get('consensus_reference_count', 0)),
                                reference_ids=list(best.get('reference_ids', [])),
                                batched=bool(batched)))

    # ------------------------------------------------------- round 3 acceptance
    def candidates_posefree(self, image_path, scan=None, top_k=None, score_voxel=None):
        """Pose-free CLS top-k over the whole library, then every candidate the frozen
        provider's own consensus rule keeps, each with its fair score.

        Retrieval is select_references_posefree(): the whole 4752-frame library ranked by
        the DINOv2 CLS descriptor alone.  No position, no pose and no reference position
        enters -- not even as a filter after the ranking.

        The candidate set is exactly what branch._hypotheses() produces: the per-reference
        pair solves, the whole-set solve and the per-cluster solves, each one scored by the
        provider's own qualification test.  Every qualified solution carries
        `consensus_reference_count`, the number of distinct references whose qualified
        solutions agree with it within 5 m / 10 deg; those constants and the count that
        turns agreement into acceptance are the frozen plan's
        (PnP.consensus_translation_m, PnP.consensus_rotation_deg,
        PnP.min_supporting_reference_images) and are read from `branch.pnp_cfg`, never
        re-typed in this module.

        `fair` is the provider's laser verification score (geometry_score): the median per
        azimuth-bin share of the frame's own scan returns within 0.3 m of a map point built
        from references other than the ones that produced the pose, on a 0.4 m query cloud.
        Same function the frozen producer used; the scan handed in is the frame's own raw
        scan, decoded the way the branch decodes it.  No ground truth, no prior.

        Returns evidence, not a decision: the driver's acceptance step picks at most one
        pose out of it.
        """
        branch = self.branch
        k = self.top_k if top_k is None else int(top_k)
        refs, retrieval = select_references_posefree(image_path, branch, k)
        out = dict(retrieval=retrieval, refs=list(refs), candidates=0, batched=False,
                   n_hypotheses=0, n_qualified=0, n_accepted=0, pnp_calls=0, capped=0,
                   qualified_rows=[], accepted_rows=[], query_points=0)
        if not refs:
            return out
        q_model, candidates, per_pair, batched = branch._match_frame(image_path, refs)
        out['candidates'] = int(len(candidates))
        out['batched'] = bool(batched)
        if not candidates:
            return out
        by_reference = {}
        for query_index, _, reference, attribute in candidates:
            entry = by_reference.setdefault(reference, ([], []))
            entry[0].append(query_index)
            entry[1].append(attribute)
        by_reference = {ri: (np.array(ids), np.array(aid))
                        for ri, (ids, aid) in by_reference.items()}
        hypotheses, accepted, calls, capped = branch._hypotheses(q_model, candidates,
                                                                 by_reference)
        out.update(n_hypotheses=len(hypotheses), n_accepted=len(accepted),
                   pnp_calls=int(calls), capped=int(capped))
        for solution in hypotheses:
            if not solution['qualified']:
                continue
            out['qualified_rows'].append(dict(
                pose=np.asarray(solution['pose'], dtype=np.float64),
                consensus=int(solution.get('consensus_reference_count', 0)),
                reference_ids=list(solution['reference_ids']),
                unique_query_inliers=int(solution['unique_query_inliers']),
                inlier_ratio=float(solution['inlier_ratio']),
                coverage_cells=int(solution['coverage_cells']),
                median_reprojection_px=float(solution['median_reprojection_px']),
                channel=solution.get('channel')))
        out['n_qualified'] = len(out['qualified_rows'])
        if scan is None:
            return out
        query = branch._query_cloud(scan, branch.score_voxel if score_voxel is None
                                    else float(score_voxel))
        out['query_points'] = int(len(query))
        for solution in accepted:
            row = branch.verify(solution['pose'], query, solution['reference_ids'])
            out['accepted_rows'].append(dict(
                pose=np.asarray(solution['pose'], dtype=np.float64),
                fair=float(row['geometry_score']), verified=bool(row['geometry_verified']),
                consensus=int(solution.get('consensus_reference_count', 0)),
                reference_ids=list(solution['reference_ids']),
                unique_query_inliers=int(solution['unique_query_inliers']),
                inlier_ratio=float(solution['inlier_ratio']),
                coverage_cells=int(solution['coverage_cells']),
                median_reprojection_px=float(solution['median_reprojection_px']),
                channel=solution.get('channel'),
                residual_median_m=row['residual_median_m'],
                supported_bins=int(row['supported_bins']),
                top_bin_share=row['top_bin_share']))
        return out

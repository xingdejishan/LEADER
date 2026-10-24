import os
import time
from collections import OrderedDict

import numpy as np
from scipy.spatial import cKDTree

import arbitration as ARB
from visual_ordered import OrderedBranch


def shortlist(hypotheses, topk=2):
    eligible = [h for h in hypotheses if h.get('qualified') and h.get('reference_ids')
                and np.isfinite(h.get('median_reprojection_px', np.inf))]
    eligible.sort(key=lambda h: (-int(h.get('consensus_reference_count', 0)),
                                -int(h['unique_query_inliers']), -int(h['coverage_cells']),
                                float(h['median_reprojection_px'])))
    result = []
    for hypothesis in eligible:
        pose = np.asarray(hypothesis['pose'], dtype=np.float64)
        if any(np.allclose(pose, np.asarray(old['pose']), atol=1e-6, rtol=0) for old in result):
            continue
        result.append(dict(hypothesis))
        if len(result) == topk:
            break
    return result


class ProposalBranch(OrderedBranch):
    def propose(self, key, image_path):
        started = time.monotonic()
        references, retrieval = self.retrieve(image_path)
        if not references:
            return dict(hypotheses=[], top2=[], prefilter_seconds=0, n_candidates=0,
                        detail=dict(note='no_references'))
        retrieval_seconds = time.monotonic() - started
        tick = time.monotonic()
        q_model, candidates, per_pair, batched = self._match_frame(image_path, references)
        match_seconds = time.monotonic() - tick
        by_reference = {}
        for candidate in candidates:
            entry = by_reference.setdefault(candidate[2], ([], []))
            entry[0].append(candidate[0])
            entry[1].append(candidate[3])
        by_reference = {ri: (np.asarray(ids), np.asarray(aid))
                        for ri, (ids, aid) in by_reference.items()}
        tick = time.monotonic()
        hypotheses, accepted, calls, capped = self._hypotheses(q_model, candidates, by_reference)
        pnp_seconds = time.monotonic() - tick
        tick = time.monotonic()
        top2 = shortlist(hypotheses)
        prefilter_seconds = time.monotonic() - tick
        return dict(hypotheses=hypotheses, top2=top2, n_candidates=len(hypotheses),
                    prefilter_seconds=prefilter_seconds,
                    detail=dict(key=key, retrieval=retrieval, batched=bool(batched),
                                pnp_calls=int(calls), candidates=len(candidates),
                                qualified=sum(bool(h['qualified']) for h in hypotheses),
                                consensus_accepted=len(accepted), shortlisted=len(top2),
                                seconds=dict(retrieval=retrieval_seconds, match=match_seconds,
                                             pnp=pnp_seconds, prefilter=prefilter_seconds, verify=0.0)))


class LightSelector:
    def __init__(self, refs, visual_refs, points=1000, map_points_per_ref=15000, k=8.4):
        self.refs = refs
        self.positions = np.asarray([r['world_from_body'] for r in refs], dtype=np.float64)[:, :3, 3]
        self.tree = cKDTree(self.positions)
        index = {r['key']: i for i, r in enumerate(refs)}
        self.visual_to_map = {int(r['index']): index.get(r['key']) for r in visual_refs}
        self.points, self.map_points_per_ref, self.k = int(points), int(map_points_per_ref), float(k)
        self.clouds = OrderedDict()
        self.dtype = np.dtype([('x', '<u2'), ('y', '<u2'), ('z', '<u2'), ('i', 'u1'), ('l', 'u1')])

    def cloud(self, index):
        row = self.refs[index]
        stat = os.stat(row['scan_path'])
        fingerprint = (stat.st_size, stat.st_mtime_ns)
        entry = self.clouds.pop(index, None)
        if entry is not None and entry[0] == fingerprint:
            self.clouds[index] = entry
            return entry[1]
        if not stat.st_size or stat.st_size % self.dtype.itemsize or stat.st_size > 16000000:
            raise ValueError('invalid or oversized reference scan')
        raw = np.memmap(row['scan_path'], dtype=self.dtype, mode='r')
        ids = np.linspace(0, len(raw) - 1, min(len(raw), self.map_points_per_ref)).astype(np.int64)
        xyz = np.column_stack([raw[a][ids] for a in ('x', 'y', 'z')]).astype(np.float64) * 0.005 - 100.0
        transform = np.asarray(row['world_from_body'], dtype=np.float64)
        cloud = xyz @ transform[:3, :3].T + transform[:3, 3]
        del raw
        self.clouds[index] = (fingerprint, cloud)
        while len(self.clouds) > 32:
            self.clouds.popitem(last=False)
        return cloud

    def nearby(self, centre, excluded):
        ids = [i for i in self.tree.query_ball_point(centre, 30.0) if i not in excluded]
        ids.sort(key=lambda i: float(np.linalg.norm(self.positions[i] - centre)))
        return ids[:3]

    def candidates(self, proposal):
        return proposal.get('top2', [])[:2]

    def evidence(self, points, query, hypotheses):
        return ARB.Evidence(points, query, n_points=self.points)

    def select(self, geometry, proposal, scan):
        from e2e_online import valid_pose

        started = time.monotonic()
        incumbent = valid_pose(geometry['pose'])
        diag = dict(chosen='route3', decided=False, switched=False, selection_mode='light',
                    k=self.k, sigma_c=None, thresh=None, costs={}, visual_available=False,
                    excluded_references=[], dropped=[], shortlist_size=0, candidates_scored=0,
                    n_query=0, n_map=0)

        def finish(pose, visual=None):
            diag['selector_seconds'] = time.monotonic() - started
            return pose, diag, visual

        candidates, excluded = [], set()
        for candidate in self.candidates(proposal):
            mapped = [self.visual_to_map.get(int(i)) for i in candidate['reference_ids']]
            if not mapped or any(i is None for i in mapped):
                diag['dropped'].append('unknown_candidate_source')
                continue
            candidate = dict(candidate, pose=valid_pose(candidate['pose']))
            candidates.append(candidate)
            excluded.update(mapped)
        diag['excluded_references'] = sorted(excluded)
        diag['shortlist_size'] = len(candidates)
        if not candidates:
            diag['note'] = 'no_qualified_visual_proposal'
            return finish(incumbent)
        geo_refs = self.nearby(incumbent[:3, 3], excluded)
        if not geo_refs:
            diag['note'] = 'no_independent_geometry_reference'
            return finish(incumbent)
        sources, centres, supported = set(geo_refs), [incumbent[:3, 3]], []
        for candidate in candidates:
            refs = self.nearby(candidate['pose'][:3, 3], excluded)
            if not refs:
                diag['dropped'].append('no_independent_visual_reference')
                continue
            sources.update(refs)
            centres.append(candidate['pose'][:3, 3])
            supported.append(candidate)
        if not supported:
            diag['note'] = 'no_independent_visual_reference'
            return finish(incumbent)
        parts = []
        for index in sorted(sources):
            cloud = self.cloud(index)
            keep = np.zeros(len(cloud), dtype=bool)
            ordered_centres = sorted(centres, key=lambda c: float(np.sum((self.positions[index]-c)**2)))
            for centre in ordered_centres:
                remaining = np.flatnonzero(~keep)
                if not len(remaining):
                    break
                keep[remaining] = np.sum((cloud[remaining] - centre) ** 2, axis=1) <= 60.0 ** 2
            if keep.any():
                parts.append(cloud[keep])
        if not parts or sum(len(p) for p in parts) < 500 or len(scan) < 100:
            diag['note'] = 'insufficient_independent_evidence'
            return finish(incumbent)
        ids = np.linspace(0, len(scan) - 1, min(len(scan), self.points)).astype(np.int64)
        query = np.asarray(scan, dtype=np.float64)[ids]
        hypotheses = [ARB.Hypothesis(incumbent, 'route3', {'incumbent': True})]
        hypotheses.extend(ARB.Hypothesis(c['pose'], 'visual_%d' % i) for i, c in enumerate(supported))
        evidence = self.evidence(np.concatenate(parts), query, hypotheses)
        chosen, decision = ARB.decide(hypotheses, evidence, self.k)
        if not all(np.isfinite(v) for v in decision['costs'].values()):
            raise ValueError('nonfinite light-selection cost')
        diag.update(decision)
        diag.update(getattr(evidence, 'diagnostics', {}))
        diag.update(chosen_hypothesis=chosen.source,
                    chosen='route3' if chosen.source == 'route3' else 'visual',
                    visual_available=True, n_query=len(query), n_map=evidence.n_map,
                    candidates_scored=len(supported), evidence_references=sorted(sources))
        winner = min(range(len(supported)), key=lambda i: decision['costs']['visual_%d' % i])
        visual = dict(supported[winner], score=None, verified=None,
                      n_candidates=proposal['n_candidates'], detail=proposal.get('detail'))
        return finish(valid_pose(chosen.pose), visual)


class BatchedEvidence(ARB.Evidence):
    def __init__(self, points, query, hypotheses, n_points):
        super().__init__(points, query, n_points=n_points)
        poses = [h.pose for h in hypotheses] + ARB.perturbations(hypotheses[0].pose)
        poses = np.asarray(poses, dtype=np.float64)
        transformed = np.einsum('nj,bkj->bnk', self.base, poses[:, :3, :3]) + poses[:, None, :3, 3]
        distances = self.tree.query(transformed.reshape(-1, 3), k=1, workers=1,
                                    distance_upper_bound=self.trunc)[0]
        costs = np.minimum(distances, self.trunc).reshape(len(poses), -1).mean(axis=1)
        self.scores = {pose.tobytes(): float(cost) for pose, cost in zip(poses, costs)}

    def score(self, pose, base=None):
        if base is None:
            key = np.asarray(pose, dtype=np.float64).tobytes()
            if key in self.scores:
                return self.scores[key]
        return super().score(pose, base)


class SparseSelector(LightSelector):
    def candidates(self, proposal):
        result = []
        for hypothesis in proposal.get('hypotheses', [])[:8]:
            pose = np.asarray(hypothesis['pose'], dtype=np.float64)
            if not hypothesis.get('reference_ids') or not np.isfinite(pose).all():
                continue
            if any(np.allclose(pose, old['pose'], atol=1e-6, rtol=0) for old in result):
                continue
            result.append(hypothesis)
        return result

    def evidence(self, points, query, hypotheses):
        return BatchedEvidence(points, query, hypotheses, self.points)

    def select(self, geometry, proposal, scan):
        pose, diagnostics, visual = super().select(geometry, proposal, scan)
        diagnostics['selection_mode'] = 'sparse'
        if diagnostics.get('note') == 'no_qualified_visual_proposal':
            diagnostics['note'] = 'no_visual_proposal'
        return pose, diagnostics, visual

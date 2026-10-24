from collections import OrderedDict

import numpy as np
from scipy.spatial import cKDTree
from visual_ordered import OrderedBranch
from visual_branch import NEIGHBOURS


class AcceleratedBranch(OrderedBranch):
    def __init__(self, *args, **kwargs):
        self.verify_backend = kwargs.pop('verify_backend', 'tree')
        super().__init__(*args, **kwargs)
        self._verification_maps = OrderedDict()
        self._verification_refs = OrderedDict()

    def _map_cloud(self, index):
        cloud = self._verification_refs.pop(index, None)
        if cloud is None:
            cloud = super()._map_cloud(index)
        self._verification_refs[index] = cloud
        while len(self._verification_refs) > 32:
            self._verification_refs.popitem(last=False)
        return cloud

    def _cluster_cloud(self, centre):
        radius = self.verify_cfg['map_radius_m']
        key = tuple(np.flatnonzero(np.linalg.norm(self.positions - centre, axis=1) <= radius))
        if not key:
            return None
        entry = self._verification_maps.pop(key, None)
        if entry is None:
            cloud, labels = super()._cluster_cloud(centre)
            if getattr(self, 'verify_backend', 'tree') == 'kernel':
                from bounded_kernel import BoundedKernel

                limit = max(self.verify_cfg['nn_thresholds_m'] + [self.verify_cfg['residual_window_m']])
                if limit > 1.0:
                    raise ValueError('bounded kernel cannot score thresholds above 1 metre')
                index = BoundedKernel(cloud, labels)
            else:
                index = cKDTree(cloud)
            entry = (cloud, labels, index)
        self._verification_maps[key] = entry
        while len(self._verification_maps) > 2:
            self._verification_maps.popitem(last=False)
        return entry[:2]

    def _verify_tree(self, cloud):
        for points, labels, tree in self._verification_maps.values():
            if points is cloud:
                return tree
        raise RuntimeError('verification map tree missing')

    def verify(self, pose, query_cloud, excluded):
        cfg = self.verify_cfg
        pose = np.asarray(pose, dtype=np.float64)
        placed = self.C.transform(query_cloud, pose)
        built = self._cluster_cloud(pose[:3, 3])
        out = dict(valid=False, map_references=0, map_points=0, residual_median_m=None, residual_p90_m=None, supported_bins=0, bin_overlap_median=None, top_bin_share=None, geometry_score=0.0, geometry_verified=False, nearest_unresolved=0, excluded=len(excluded))
        if built is None:
            out['nearest_unresolved'] = int(len(placed))
            return out
        cloud, labels = built
        tree = self._verify_tree(cloud)
        out['map_points'] = int(len(cloud))
        available = np.ones(len(self.positions), dtype=bool)
        excluded = sorted(set((int(x) for x in excluded)))
        if excluded:
            available[np.asarray(excluded, dtype=np.int64)] = False
        own = np.linalg.norm(self.positions - pose[:3, 3], axis=1)
        near = own <= cfg['map_radius_m']
        if excluded:
            near = near.copy()
            near[np.asarray(excluded, dtype=np.int64)] = False
        out['map_references'] = int(near.sum())
        disallowed = ~available[labels]
        n = len(placed)
        if hasattr(tree, 'query_allowed'):
            nearest = tree.query_allowed(placed, available)
            done = np.full(n, bool(np.any(available[labels])))
        else:
            nearest = np.full(n, np.inf)
            done = np.zeros(n, dtype=bool)
            rows_all = np.arange(n)
            limit = len(cloud)
            for k in (NEIGHBOURS, NEIGHBOURS * 4, NEIGHBOURS * 16):
                todo = rows_all[~done]
                if not len(todo):
                    break
                distance, index = tree.query(placed[todo], k=min(k, limit), workers=1)
                if distance.ndim == 1:
                    distance = distance[:, None]
                    index = index[:, None]
                ok = ~disallowed[index]
                any_ok = ok.any(1)
                if any_ok.any():
                    first = np.argmax(ok, axis=1)
                    nearest[todo[any_ok]] = distance[np.arange(len(todo)), first][any_ok]
                    done[todo[any_ok]] = True
        out['nearest_unresolved'] = int((~done).sum())
        out['valid'] = True
        primary = cfg['primary_nn_threshold_m']
        for threshold in cfg['nn_thresholds_m']:
            out['overlap_%g' % threshold] = float(np.mean(nearest <= threshold))
        inside = nearest <= cfg['residual_window_m']
        if int(inside.sum()) >= 20:
            out['residual_median_m'] = float(np.median(nearest[inside]))
            out['residual_p90_m'] = float(np.percentile(nearest[inside], 90))
        azimuth = np.arctan2(query_cloud[:, 1], query_cloud[:, 0])
        bins = int(cfg['azimuth_bins'])
        bin_index = np.floor((azimuth + np.pi) / (2 * np.pi) * bins).astype(int).clip(0, bins - 1)
        overlaps, supported = ([], 0)
        points = np.zeros(bins)
        for b in range(bins):
            mask = bin_index == b
            if int(mask.sum()) < int(cfg['min_points_per_bin']):
                continue
            fraction = float(np.mean(nearest[mask] <= primary))
            overlaps.append(fraction)
            points[b] = int((nearest[mask] <= primary).sum())
            if fraction >= cfg['bin_support_overlap']:
                supported += 1
        total = int(points.sum())
        out['supported_bins'] = supported
        out['bin_overlap_median'] = float(np.median(overlaps)) if overlaps else None
        out['top_bin_share'] = float(points.max() / total) if total else None
        out['geometry_score'] = out['bin_overlap_median'] if out['bin_overlap_median'] is not None else 0.0
        out['geometry_verified'] = bool(out['bin_overlap_median'] is not None and out['bin_overlap_median'] >= cfg['bin_support_overlap'] and (supported >= cfg['min_supported_bins']) and (out['residual_median_m'] is not None) and (out['residual_median_m'] <= cfg['residual_median_max_m']) and (out['residual_p90_m'] is not None) and (out['residual_p90_m'] <= cfg['residual_p90_max_m']) and (out['top_bin_share'] is not None) and (out['top_bin_share'] <= cfg['max_top_bin_share']))
        return out

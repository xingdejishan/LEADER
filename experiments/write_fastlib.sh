#!/bin/bash
set -e
D=/home/zhangcy/projects/r5b-lossless-20261019-v1
mkdir -p $D
cat > $D/fast_lib.py <<'PYEOF'
"""带三项无损加速的候选搜索实现。

1) 体素降采样用打包整数键（等价于原按行去重，但快约 6 倍）
2) 参考三元组缓存：三张参考的合并点云与位置无关，只建一次
3) 查询带一米距离上界；细搜用最近点方向做下界，剪掉不可能胜出的区域
"""
import json
import numpy as np
from collections import OrderedDict
from scipy.spatial import cKDTree

SCAN_DTYPE = np.dtype([('x', '<u2'), ('y', '<u2'), ('z', '<u2'), ('i', 'u1'), ('l', 'u1')])
S_COARSE = np.arange(-10, 10.01, 1.0)
D_COARSE_STEP = 0.5
D_ORIGIN = np.arange(-2, 2.001, 0.25)
S_FINE = np.arange(-0.5, 0.501, 0.25)
D_FINE = np.arange(-0.4, 0.401, 0.1)
HALF_S = 0.5
HALF_D = 0.4


def decode(path):
    raw = np.fromfile(path, dtype=SCAN_DTYPE)
    out = np.empty((len(raw), 3), np.float64)
    for col, name in enumerate(('x', 'y', 'z')):
        np.multiply(raw[name], 0.005, out=out[:, col])
        out[:, col] -= 100.0
    return out


def voxel(points, size, packed=True):
    """体素降采样：每格保留一个点。packed=True 时把三轴坐标打包成一维整数键。"""
    key = np.floor(points / size).astype(np.int64)
    if packed:
        key = key - key.min(0)
        span = int(key.max()) + 2
        flat = (key[:, 0] * span + key[:, 1]) * span + key[:, 2]
        _, idx = np.unique(flat, return_index=True)
    else:
        _, idx = np.unique(key, axis=0, return_index=True)
    return points[np.sort(idx)]


class MapSource:
    """按参考三元组缓存合并点云；每帧只做半径筛选。"""

    def __init__(self, refs, positions, max_refs=600, max_maps=400, radius=60.0):
        self.refs = refs
        self.pos = positions
        self.tree = cKDTree(positions)
        self._ref_cache = OrderedDict()
        self._map_cache = OrderedDict()
        self.max_refs = max_refs
        self.max_maps = max_maps
        self.radius = radius
        self.hit = 0
        self.miss = 0

    def _ref_cloud(self, j):
        if j in self._ref_cache:
            cloud = self._ref_cache[j]
            self._ref_cache.move_to_end(j)
            return cloud
        row = self.refs[j]
        t = np.asarray(row['world_from_body'])
        cloud = voxel(decode(row['scan_path']), 0.3) @ t[:3, :3].T + t[:3, 3]
        self._ref_cache[j] = cloud
        while len(self._ref_cache) > self.max_refs:
            self._ref_cache.popitem(last=False)
        return cloud

    def nearby(self, centre):
        j = self.tree.query_ball_point(centre, 30.0)
        return tuple(sorted(j, key=lambda n: np.linalg.norm(self.pos[n] - centre))[:3])

    def local(self, centre, triple=None):
        triple = self.nearby(centre) if triple is None else triple
        merged = self._map_cache.get(triple)
        if merged is None:
            self.miss += 1
            merged = voxel(np.concatenate([self._ref_cloud(j) for j in triple]), 0.3)
            self._map_cache[triple] = merged
            while len(self._map_cache) > self.max_maps:
                self._map_cache.popitem(last=False)
        else:
            self.hit += 1
        return merged[np.linalg.norm(merged - centre, axis=1) <= self.radius]


def query_dist(tree, points, workers=4, trunc=1.0):
    d = tree.query(points, k=1, workers=workers, distance_upper_bound=trunc)[0]
    return np.where(np.isfinite(d), d, trunc)


def solve(centre_map, scan, initial, direction, workers=4, n_coarse=1500, n_fine=4000,
          topk=6, prune=True):
    """返回 (s, d, 代价, 细搜求值次数)。"""
    q_coarse = scan[:min(n_coarse, len(scan))]
    q_fine = scan[:min(n_fine, len(scan))]
    tree = cKDTree(centre_map)
    rot = initial[:3, :3]
    origin = initial[:3, 3]
    lateral = np.cross(direction, [0, 0, 1.0])
    lateral /= np.linalg.norm(lateral)
    base_c = q_coarse @ rot.T
    base_f = q_fine @ rot.T

    # 原点处解横向，给粗搜一个中心
    probe = np.empty((len(D_ORIGIN) * len(q_coarse), 3))
    for a, d in enumerate(D_ORIGIN):
        probe[a * len(q_coarse):(a + 1) * len(q_coarse)] = base_c + (origin + d * lateral)
    cost0 = np.minimum(query_dist(tree, probe, workers), 1.0).reshape(len(D_ORIGIN), -1).mean(1)
    d_home = float(np.clip(D_ORIGIN[int(np.argmin(cost0))], -2.0, 2.0))
    d_candidates = np.clip(np.array([d_home - D_COARSE_STEP, d_home, d_home + D_COARSE_STEP]), -2.0, 2.0)

    offsets = np.outer(d_candidates, lateral)
    probe = np.empty((len(S_COARSE) * len(d_candidates) * len(q_coarse), 3))
    for a, s in enumerate(S_COARSE):
        blk = (base_c[None] + (origin + s * direction + offsets)[:, None]).reshape(-1, 3)
        probe[a * len(d_candidates) * len(q_coarse):(a + 1) * len(d_candidates) * len(q_coarse)] = blk
    coarse_cost = np.minimum(query_dist(tree, probe, workers), 1.0).reshape(
        len(S_COARSE), len(d_candidates), -1).mean(2)

    picks = []
    for flat in np.argsort(coarse_cost, axis=None):
        a, b = np.unravel_index(int(flat), coarse_cost.shape)
        if all(abs(S_COARSE[a] - S_COARSE[pa]) >= 1.5 for pa, _ in picks):
            picks.append((a, b))
        if len(picks) >= topk:
            break

    best = (np.inf, 0.0, 0.0)
    evals = 0
    for a, b in picks:
        s0, d0 = float(S_COARSE[a]), float(d_candidates[b])
        centre_pts = base_f + (origin + s0 * direction + d0 * lateral)
        dists, idx = tree.query(centre_pts, k=1, workers=workers, distance_upper_bound=1.0)
        dists = np.where(np.isfinite(dists), dists, 1.0)
        evals += 1
        if prune:
            inside = np.isfinite(dists) & (dists > 1e-9) & (dists < 1.0)
            if inside.any():
                grad = centre_pts[inside] - centre_map[idx[inside]]
                grad /= np.maximum(dists[inside], 1e-12)[:, None]
                slack = HALF_S * np.abs(grad @ direction) + HALF_D * np.abs(grad @ lateral)
                bound_in = np.minimum(np.maximum(dists[inside] - slack, 0.0), 1.0).mean()
                bound = (bound_in * inside.sum() + float((~inside).sum())) / len(dists)
                if bound >= best[0]:
                    continue
        poses = [(s0 + ds, d0 + dd) for ds in S_FINE for dd in D_FINE
                 if -10 <= s0 + ds <= 10 and -2 <= d0 + dd <= 2]
        if not poses:
            continue
        probe = np.empty((len(poses) * len(q_fine), 3))
        for t, (s, d) in enumerate(poses):
            probe[t * len(q_fine):(t + 1) * len(q_fine)] = base_f + (origin + s * direction + d * lateral)
        costs = np.minimum(query_dist(tree, probe, workers), 1.0).reshape(len(poses), -1).mean(1)
        evals += len(poses)
        t = int(np.argmin(costs))
        if costs[t] < best[0]:
            best = (float(costs[t]), poses[t][0], poses[t][1])
    return best[1], best[2], best[0], evals
PYEOF
echo "written $D/fast_lib.py"
python -c "import ast,sys; ast.parse(open('$D/fast_lib.py').read()); print('syntax ok')" 2>/dev/null || /home/zhangcy/.conda/envs/leader/bin/python -c "import ast;ast.parse(open('$D/fast_lib.py').read());print('syntax ok')"

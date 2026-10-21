"""Compare the exact fine-grid enumeration against the GPU distance-field
alternatives: continuous refinement, and field-ranking plus exact re-score.

    CUDA_VISIBLE_DEVICES=<free gpu> python df_run.py <frames> <voxel>

The first argument is the total number of frames sampled evenly across five
deployed-error bands; the second is the distance-field voxel size.
"""
import json
import sys
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, '/home/zhangcy/projects/r5b-lossless-20261019-v1')
sys.path.insert(0, '/home/zhangcy/projects/r5c-distancefield-20261019-v1')
import fast_lib as F
import df_core as DF

H = '/data2/zhangcy/v7-raw-geometry-hard1502-20261013-v1/'
sel = json.load(open(H + 'selection.json'))
refs, qs = sel['references'], sel['queries']
z = np.load(H + 'scored.npz', allow_pickle=True)
keys = [str(k) for k in z['keys']]
v7 = z['refined_V7_poses']
gt = z['reference']
rep = z['replaced'].astype(bool)
qmap = {str(q['key']): q for q in qs}
rpos = np.asarray([r['world_from_body'] for r in refs])[:, :3, 3]

S_COARSE = np.arange(-8.0, 8.0001, 1.0)
D_HOME = np.arange(-1.0, 1.0001, 0.25)
D3 = np.array([-0.5, 0.0, 0.5])
S_FINE = np.arange(-2, 3) * 0.25
D_FINE = np.arange(-0.4, 0.4001, 0.1)
TOPK = 6


def sync(dev):
    if dev.type == 'cuda':
        torch.cuda.synchronize(dev)


def run_frame(i, maps, dev, voxel_field=0.4):
    out = {}
    c = v7[i][:3, 3]
    M = maps.local(c)
    scan = F.decode(qmap[keys[i]]['scan_path'])
    scan = scan[(np.linalg.norm(scan, axis=1) > 1) & (np.linalg.norm(scan, axis=1) < 60)]
    scan = F.voxel(scan, 0.3)
    u = v7[i][:3, :3][:, 0].copy(); u /= np.linalg.norm(u)
    lat = np.cross(u, [0, 0, 1.0]); lat /= np.linalg.norm(lat)
    R0 = v7[i][:3, :3]; o = v7[i][:3, 3]
    tree = cKDTree(M)
    qc = scan[:1500]; qf = scan[:4000]
    bc = qc @ R0.T; bf = qf @ R0.T

    t0 = time.perf_counter(); sync(dev)
    fld = DF.Field(M, voxel=voxel_field, trunc=1.0, dev=dev)
    sync(dev); out['t_field'] = time.perf_counter() - t0

    # field accuracy against exact KD, on points drawn from the actual scan
    rng = np.random.default_rng(0)
    idx = rng.choice(len(scan), size=min(4000, len(scan)), replace=False)
    probe = scan[idx] @ R0.T + o + rng.normal(0, 1.2, (len(idx), 3))
    exact = np.minimum(tree.query(probe, k=1, workers=4)[0], 1.0)
    approx, _ = fld.sample(torch.from_numpy(probe).to(dev).float())
    err = np.abs(approx.cpu().numpy() - exact)
    out['field_mae'] = float(err.mean()); out['field_p95'] = float(np.percentile(err, 95))

    # coarse, exact
    t0 = time.perf_counter()
    P = np.empty((len(D_HOME) * len(qc), 3))
    for a, d in enumerate(D_HOME):
        P[a * len(qc):(a + 1) * len(qc)] = bc + (o + d * lat)
    c0 = np.minimum(F.query_dist(tree, P), 1.0).reshape(len(D_HOME), -1).mean(1)
    dh = float(D_HOME[int(np.argmin(c0))])
    DC = np.clip(dh + D3, -2, 2)
    grid = [(s, float(d)) for s in S_COARSE for d in DC]
    P = np.empty((len(grid) * len(qc), 3))
    for a, (s, d) in enumerate(grid):
        P[a * len(qc):(a + 1) * len(qc)] = bc + (o + s * u + d * lat)
    exact_coarse = np.minimum(F.query_dist(tree, P), 1.0).reshape(len(grid), -1).mean(1)
    out['t_coarse_cpu'] = time.perf_counter() - t0

    t0 = time.perf_counter(); sync(dev)
    val, _ = fld.sample(torch.from_numpy(P).to(dev).float())
    field_coarse = val.view(len(grid), len(qc)).mean(1).cpu().numpy()
    sync(dev); out['t_coarse_gpu'] = time.perf_counter() - t0

    def nms(order_source):
        picks = []
        for t in np.argsort(order_source):
            s = grid[t][0]
            if all(abs(s - grid[p][0]) >= 1.5 for p in picks):
                picks.append(t)
            if len(picks) >= TOPK:
                break
        return picks

    pick_exact = nms(exact_coarse)
    out['pick_agree'] = len(set(pick_exact) & set(nms(field_coarse)))

    # fine A: exact enumeration of the 45 poses per candidate
    t0 = time.perf_counter()
    best = (1e9, 0.0, 0.0)
    for t in pick_exact:
        s0, d0 = grid[t]
        poses = [(s0 + ds, d0 + dd) for ds in S_FINE for dd in D_FINE
                 if -8 <= s0 + ds <= 8 and -2 <= d0 + dd <= 2]
        Q = np.empty((len(poses) * len(qf), 3))
        for a, (s, d) in enumerate(poses):
            Q[a * len(qf):(a + 1) * len(qf)] = bf + (o + s * u + d * lat)
        costs = np.minimum(F.query_dist(tree, Q), 1.0).reshape(len(poses), -1).mean(1)
        a = int(np.argmin(costs))
        if costs[a] < best[0]:
            best = (float(costs[a]), poses[a][0], poses[a][1])
    out['t_fine_cpu'] = time.perf_counter() - t0
    T = v7[i].copy(); T[:3, 3] = o + best[1] * u + best[2] * lat
    out['err_grid'] = float(np.linalg.norm(T[:3, 3] - gt[i][:3, 3]))

    # fine B: continuous refinement on the field, then exact re-score of the six
    t0 = time.perf_counter(); sync(dev)
    s0 = torch.tensor([grid[t][0] for t in pick_exact], device=dev).float()
    d0 = torch.tensor([grid[t][1] for t in pick_exact], device=dev).float()
    s_r, d_r, _ = DF.refine_batch(fld, torch.from_numpy(bf).to(dev).float(),
                                  torch.from_numpy(o).to(dev).float(),
                                  torch.from_numpy(u).to(dev).float(),
                                  torch.from_numpy(lat).to(dev).float(), s0, d0, iters=6)
    sync(dev); out['t_fine_gpu'] = time.perf_counter() - t0
    s_np = s_r.cpu().numpy(); d_np = d_r.cpu().numpy()

    t0 = time.perf_counter()
    cand = [float(np.minimum(F.query_dist(tree, bf + (o + s_np[a] * u + d_np[a] * lat)),
                              1.0).mean()) for a in range(len(s_np))]
    out['t_rescore'] = time.perf_counter() - t0
    a = int(np.argmin(cand))
    T = v7[i].copy(); T[:3, 3] = o + s_np[a] * u + d_np[a] * lat
    out['err_gpu'] = float(np.linalg.norm(T[:3, 3] - gt[i][:3, 3]))
    return out


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    VOXEL = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
    dep = np.linalg.norm(v7[:, :3, 3] - gt[:, :3, 3], axis=1)
    frames = []
    for lo, hi in [(0.5, 1), (1, 2), (2, 3), (3, 5), (5, 8)]:
        idx = np.where((dep >= lo) & (dep < hi) & (~rep))[0]
        if len(idx):
            frames.extend(int(x) for x in idx[np.linspace(0, len(idx) - 1,
                                                          min(max(n // 5, 1), len(idx)), dtype=int)])
    dev = DF.pick_device()
    print('device:', dev, 'voxel', VOXEL, flush=True)
    maps = F.MapSource(refs, rpos)
    rows = []
    for k, i in enumerate(frames):
        r = run_frame(i, maps, dev, voxel_field=VOXEL)
        r['i'] = i; r['dep'] = float(dep[i]); rows.append(r)
        print('%2d/%d frame %4d dep=%5.2f | field mae=%.4f p95=%.4f | pick=%d | '
              'grid=%.2f gpu=%.2f | t_fld=%.0f coarse %.0f/%.0f fine %.0f/%.0f ms'
              % (k + 1, len(frames), i, dep[i], r['field_mae'], r['field_p95'],
                 r['pick_agree'], r['err_grid'], r['err_gpu'], r['t_field'] * 1e3,
                 r['t_coarse_cpu'] * 1e3, r['t_coarse_gpu'] * 1e3,
                 r['t_fine_cpu'] * 1e3, r['t_fine_gpu'] * 1e3), flush=True)
    eg = np.array([r['err_grid'] for r in rows]); eu = np.array([r['err_gpu'] for r in rows])
    print('\n=== summary over %d frames ===' % len(rows))
    print('exact grid : med=%.2f  <=1m=%.0f%%' % (np.median(eg), 100 * (eg <= 1).mean()))
    print('gpu refine : med=%.2f  <=1m=%.0f%%' % (np.median(eu), 100 * (eu <= 1).mean()))
    for k in ('t_field', 't_coarse_cpu', 't_coarse_gpu', 't_fine_cpu', 't_fine_gpu', 't_rescore'):
        a = np.array([r[k] for r in rows]) * 1e3
        print('  %-14s median %6.1f ms' % (k, np.median(a)))
    print('  pick agreement median %.0f / 6' % np.median([r['pick_agree'] for r in rows]))

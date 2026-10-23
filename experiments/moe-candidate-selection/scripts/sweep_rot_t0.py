"""Pass 2: the rotation block applied to the front-end pose, swept over eps.

Why a separate pass: the first sweep stored the rotation block only at eps = 1e-6
when starting from t0. The objective is now "minimise orientation error without
moving translation", so the front-end pose is the only starting point that matters
(its MPE is the front end's by construction) and the margin needs its own sweep.

Per frame: decode, prepare, local map, then the project's `rot_only` from t0 for
eps in (1e-6, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1). The seven chains are run
together with a shared prefix: they start from the same pose, and a chain only
diverges from the others when one accepts a step and another rejects it, so the
candidate step is computed once per distinct pose. The result is bit-identical to
running the seven separately (verified against pass 1 at eps = 1e-6).

    python sweep_rot_t0.py --workers 16      # run (resumable)
    python sweep_rot_t0.py --merge
"""
import argparse
import json
import os
import sys
import time

for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '1'

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

PROJECT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
DATASET = '/data2/dataset/NCLT'
ONLINE = os.path.join(PROJECT, 'results', 'full-online.npz')
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'sweep_rot')
SNAP = os.path.join(HERE, 'rot_only_snapshot.py')
EPS = (1e-6, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1)
BLOCK = 400
MAX_IT = 6          # the project's cap
TRUNC = 1.2         # the project's objective truncation
HIT = 0.6           # the project's inlier radius

sys.path.insert(0, os.path.join(PROJECT, 'stages/fast_search'))
sys.path.insert(0, os.path.join(PROJECT, 'stages'))
import fast_lib as F          # noqa: E402
import pipeline as P          # noqa: E402


def scan_path(key):
    date, ts = key.split('/')
    return '%s/%s/velodyne_sync/%s.bin' % (DATASET, date, ts)


def rot_chain(tree, geom, q, T, eps_list=EPS):
    """rot_only from T for every eps in one pass, shared prefix, exact same rules.

    Same split, same objective, same 0.5 deg step cap, same acceptance test
    (held-out truncated cost improves by more than eps and the 0.6 m inlier count
    does not fall) as the project's function. Returns (poses, moved, steps).
    """
    par = (np.floor(q / 1.0).astype(np.int64) @ np.array([73856093, 19349663, 83492791])) % 2
    train, val = q[par == 0], q[par == 1]

    def obj(M):
        d = tree.query(val @ M[:3, :3].T + M[:3, 3], k=1, distance_upper_bound=TRUNC)[0]
        return float(np.minimum(d, TRUNC).mean()), int((d < HIT).sum())

    def step(t, centre):
        W = train @ t[:3, :3].T + centre
        dist, idx = tree.query(W, k=1, distance_upper_bound=TRUNC)
        ok = np.isfinite(dist)
        if ok.sum() < 30:
            return None
        A, B = W[ok] - centre, geom[idx[ok]] - centre
        a, b = A.mean(0), B.mean(0)
        U, s, Vt = np.linalg.svd((A - a).T @ (B - b))
        if s[1] < 1e-8:
            return None
        corr = np.eye(3)
        corr[-1, -1] = np.linalg.det(Vt.T @ U.T)
        vec = Rotation.from_matrix(Vt.T @ corr @ U.T).as_rotvec()
        scale = min(1.0, np.deg2rad(0.5) / max(np.linalg.norm(vec), 1e-12))
        cand = t.copy()
        cand[:3, :3] = Rotation.from_rotvec(vec * scale).as_matrix() @ t[:3, :3]
        cand[:3, 3] = centre
        return cand

    T = np.asarray(T, np.float64)
    centre = T[:3, 3].copy()
    out = np.repeat(T[None], len(eps_list), 0)
    steps = np.zeros(len(eps_list), np.uint8)
    value0, hit0 = obj(T)
    cur = {m: (T, value0, hit0) for m in range(len(eps_list))}
    for _ in range(MAX_IT):
        groups = {}
        for m, st in cur.items():
            if st is None:
                continue
            groups.setdefault(st[0].tobytes(), []).append(m)
        if not groups:
            break
        nxt = {}
        for members in groups.values():
            t, value, hit = cur[members[0]]
            cand = step(t, centre)
            if cand is None:
                for m in members:
                    nxt[m] = None
                continue
            nv, nh = obj(cand)
            for m in members:
                if nv < value - eps_list[m] and nh >= hit:
                    out[m] = cand
                    steps[m] += 1
                    nxt[m] = (cand, nv, nh)
                else:
                    nxt[m] = None
        cur = nxt
    moved = np.array([np.abs(out[m][:3, :3] - T[:3, :3]).max() > 1e-12
                      for m in range(len(eps_list))])
    return out, moved, steps


# --------------------------------------------------------------- worker
_G = {}


def _init(refs, positions, keys, t0):
    cfg = P.Config(solver='adaptive', coarse_points=1000, fine_points=1500,
                   workers=1, min_scan_range=1.0, max_scan_range=60.0,
                   voxel=0.3, crop_radius=60.0, reference_count=3)
    _G.update(cfg=cfg, keys=keys, t0=t0,
              maps=F.MapSource(refs, positions, radius=60.0))


def _one(i):
    T0 = np.asarray(_G['t0'][i], np.float64)
    key = _G['keys'][i]
    rec = {'key': key, 'scan_path': scan_path(key)}
    cfg, maps = _G['cfg'], _G['maps']
    scan = P.prepare_scan(rec, cfg)
    if len(scan) < 50:
        raise RuntimeError('scan too small (%d)' % len(scan))
    centre = T0[:3, 3]
    triple = maps.nearby(centre)[:cfg.reference_count]
    mp = maps.local(centre, triple) if triple else None
    if mp is None or len(mp) < 100:
        raise RuntimeError('local map too small')
    tree = cKDTree(mp)
    q = scan[np.linalg.norm(scan, axis=1) <= 50]
    cells = np.floor(q / 0.3).astype(np.int64)
    _, ids = np.unique(cells, axis=0, return_index=True)
    q = q[np.sort(ids)]
    return rot_chain(tree, mp, q, T0)


def run_block(job):
    k0, k1, shard = job
    if os.path.exists(shard):
        return shard, k1 - k0, 0.0, 0
    n = k1 - k0
    rot = np.full((n, len(EPS), 4, 4), np.nan)
    moved = np.zeros((n, len(EPS)), bool)
    steps = np.zeros((n, len(EPS)), np.uint8)
    bad = np.zeros(n, np.uint8)
    t = time.time()
    for j in range(n):
        try:
            rot[j], moved[j], steps[j] = _one(k0 + j)
        except Exception as exc:
            bad[j] = 1
            sys.stderr.write('frame %d %s FAILED %r\n' % (k0 + j, _G['keys'][k0 + j], exc))
    np.savez(shard, idx=np.arange(k0, k1), rot=rot, moved=moved, steps=steps, bad=bad)
    return shard, n, time.time() - t, int(bad.sum())


def merge(out, dst, n):
    rot = np.full((n, len(EPS), 4, 4), np.nan)
    moved = np.zeros((n, len(EPS)), bool)
    steps = np.zeros((n, len(EPS)), np.uint8)
    bad = np.zeros(n, np.uint8)
    shards = sorted(f for f in os.listdir(out) if f.startswith('blk_') and f.endswith('.npz'))
    for name in shards:
        z = np.load(os.path.join(out, name))
        idx = z['idx']
        rot[idx] = z['rot']
        moved[idx] = z['moved']
        steps[idx] = z['steps']
        bad[idx] = z['bad']
    np.savez(dst, keys=np.array(_keys), rot=rot, moved=moved, steps=steps, bad=bad,
             eps=np.array(EPS))
    print('merged %d shards -> %s  bad %d' % (len(shards), dst, int(bad.sum())))


_keys = None


def main():
    global _keys
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--block', type=int, default=BLOCK)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--merge', action='store_true')
    ap.add_argument('--merge-out', default=os.path.join(HERE, 'sweep_rot.npz'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    z = np.load(ONLINE, allow_pickle=True)
    keys = [str(k) for k in z['keys']]
    t0 = np.asarray(z['t0'], np.float64)
    _keys = keys
    n = len(keys) if a.limit <= 0 else min(a.limit, len(keys))

    if a.merge:
        return merge(a.out, a.merge_out, n)

    if not os.path.exists(SNAP):
        sys.exit('run sweep_compute.py once first: it writes the rot_only snapshot')
    refs = json.load(open(os.path.join(PROJECT, 'inputs/refs.json')))
    positions = np.asarray([r['world_from_body'] for r in refs])[:, :3, 3]

    blocks = [(k0, min(k0 + a.block, n)) for k0 in range(0, n, a.block)]
    jobs = [(k0, k1, os.path.join(a.out, 'blk_%06d.npz' % (k0 // a.block)))
            for k0, k1 in blocks]
    from multiprocessing import Pool
    t_start = time.time()
    done = frames = bad = 0
    with Pool(a.workers, initializer=_init, initargs=(refs, positions, keys, t0)) as pool:
        for shard, cnt, secs, nbad in pool.imap_unordered(run_block, jobs):
            done += 1
            frames += cnt
            bad += nbad
            el = time.time() - t_start
            print('[%3d/%3d] %5.1f%%  %6d frames  %6.1fs  bad=%d  elapsed %.1f min  eta %.1f min'
                  % (done, len(jobs), 100.0 * done / len(jobs), frames, secs, bad,
                     el / 60.0, el / done * (len(jobs) - done) / 60.0), flush=True)
    print('run finished in %.1f min, bad %d' % ((time.time() - t_start) / 60.0, bad))
    merge(a.out, a.merge_out, n)
    return 0


if __name__ == '__main__':
    sys.exit(main())

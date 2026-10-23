"""Offline sweep over the geometric branch's two decision points, full 104,728-frame set.

Decision points
  (a) compute gate      run the along-track search only if descent depth gain > thr
  (b) rotation block    rotation-only refinement, step accepted when the held-out
                        truncated cost improves by more than a margin eps

Per frame this script records, once, everything the offline table needs:
  gain      descent depth of the coarse cost surface (gate2 statistic)
  t0        front-end pose
  P_search  fine-stage output (translation corrected, rotation of t0 preserved)
  P_rot[m]  rotation block applied to P_search with eps = EPS[m]
  P_rot_t0  rotation block applied to t0 with eps = 1e-6

The rotation block is the project's own `rot_only` (from the pre-2026-10-22
three_route_unified.py, when the geometry branch still rotated; the current tree
has it removed and is translation-only). It is lifted verbatim out of that file
-- `_snapshot_rot_only` copies the function source, exposes the hardcoded
acceptance margin as `eps` and nothing else -- and every worker imports the
snapshot, so what ran is auditable next to this file.

Read-only w.r.t. the project: shards and logs go to --out (default under
/home/zhangcy/.dsh_scratch). Usage:

    python sweep_compute.py --workers 16          # run (resumable)
    python sweep_compute.py --merge               # shards -> one npz
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

PROJECT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
DATASET = '/data2/dataset/NCLT'
ONLINE = os.path.join(PROJECT, 'results', 'full-online.npz')
TRUTH = '/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth'
ROT_SRC = os.path.join(PROJECT, '.backups', 'remove-rotation-20261022-182520',
                       'three_route_unified.py')
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'sweep_full')
SNAP = os.path.join(HERE, 'rot_only_snapshot.py')
EPS = (1e-6, 0.005, 0.01, 0.02, 0.05)
BLOCK = 400

sys.path.insert(0, os.path.join(PROJECT, 'stages/fast_search'))
sys.path.insert(0, os.path.join(PROJECT, 'stages'))
import fast_lib as F          # noqa: E402
import pipeline as P          # noqa: E402


# --------------------------------------------------------------- inputs
def scan_path(key):
    date, ts = key.split('/')
    return '%s/%s/velodyne_sync/%s.bin' % (DATASET, date, ts)


def load_frames():
    z = np.load(ONLINE, allow_pickle=True)
    keys = [str(k) for k in z['keys']]
    t0 = np.asarray(z['t0'], np.float64)
    return keys, t0


def load_truth():
    out = {}
    for f in os.listdir(TRUTH):
        date = f.replace('.json', '')
        for ts, v in json.load(open(os.path.join(TRUTH, f))).items():
            out[(date, str(ts))] = np.asarray(v['T_WB'], np.float64)
    return out


# ------------------------------------------- the project's rotation block
def _snapshot_rot_only():
    """Copy `rot_only` out of the project, exposed acceptance margin, write it."""
    src = open(ROT_SRC).read()
    body = src[src.index('def rot_only('):src.index('def one(')]
    if 'nv < value - 1e-6' not in body:
        raise RuntimeError('rot_only source changed: acceptance margin not found')
    body = body.replace('nv < value - 1e-6', 'nv < value - eps')
    body = body.replace('def rot_only(tree, geom, q, T):',
                        'def rot_only(tree, geom, q, T, eps=1e-6):')
    head = ('"""Snapshot of the project\'s rot_only, eps exposed (generated, do not edit).\n\n'
            'Source: %s\nOnly change: the hardcoded acceptance margin 1e-6 -> `eps`.\n"""\n'
            'import numpy as np\nfrom scipy.spatial.transform import Rotation\n\n\n'
            % ROT_SRC)
    with open(SNAP, 'w') as fh:
        fh.write(head + body)
    return SNAP


def cost(tree, pts, trunc=1.0):
    d = tree.query(pts, k=1, workers=1, distance_upper_bound=trunc)[0]
    return float(np.where(np.isfinite(d), d, trunc).mean())


# --------------------------------------------------------------- worker
_G = {}


def _init(refs, positions, keys, t0):
    from rot_only_snapshot import rot_only
    cfg = P.Config(solver='adaptive', coarse_points=1000, fine_points=1500,
                   workers=1, topk=6, min_scan_range=1.0, max_scan_range=60.0,
                   voxel=0.3, crop_radius=60.0, reference_count=3)
    _G.update(rot_only=rot_only, cfg=cfg, keys=keys, t0=t0,
              maps=F.MapSource(refs, positions, radius=60.0))


def _one(i, keys, t0):
    """All quantities the table needs for one frame. Raises on hard failure."""
    T0 = np.asarray(t0[i], np.float64)
    rec = {'key': keys[i], 'scan_path': scan_path(keys[i])}
    maps, cfg = _G['maps'], _G['cfg']

    centre = T0[:3, 3]
    triple = maps.nearby(centre)[:cfg.reference_count]
    if len(triple) == 0:
        raise RuntimeError('no reference within 30 m')
    mp = maps.local(centre, triple)
    if len(mp) < 100:
        raise RuntimeError('local map too small (%d)' % len(mp))

    scan = P.prepare_scan(rec, cfg)
    if len(scan) < 50:
        raise RuntimeError('scan too small (%d)' % len(scan))

    tree = cKDTree(mp)
    u = P.along_track_direction(T0)

    cs = F.coarse_stage(mp, scan, T0, u, workers=1, n_coarse=cfg.coarse_points,
                        topk=cfg.topk)
    gain = cost(cs['tree'], cs['base_c'] + centre) - float(cs['surface'].min())

    pose, _ = P.localize(rec, T0, maps, cfg, coarse=cs, scan=scan)
    P_search = np.asarray(pose, np.float64)

    q = scan[np.linalg.norm(scan, axis=1) <= 50]
    cells = np.floor(q / 0.3).astype(np.int64)
    _, ids = np.unique(cells, axis=0, return_index=True)
    q = q[np.sort(ids)]

    rot = np.empty((len(EPS), 4, 4), np.float64)
    for m, eps in enumerate(EPS):
        rot[m] = _G['rot_only'](tree, mp, q, P_search, eps=eps)
    rot_t0 = _G['rot_only'](tree, mp, q, T0, eps=1e-6)
    return gain, P_search, rot, rot_t0


def run_block(job):
    k0, k1, shard = job
    keys, t0 = _G['keys'], _G['t0']
    if os.path.exists(shard):
        return shard, k1 - k0, 0.0, 0
    n = k1 - k0
    gain = np.full(n, np.nan)
    search = np.full((n, 4, 4), np.nan)
    rot = np.full((n, len(EPS), 4, 4), np.nan)
    rot_t0 = np.full((n, 4, 4), np.nan)
    bad = np.zeros(n, np.uint8)
    t = time.time()
    for j in range(n):
        i = k0 + j
        try:
            gain[j], search[j], rot[j], rot_t0[j] = _one(i, keys, t0)
        except Exception as exc:            # one bad frame must not kill the block
            bad[j] = 1
            sys.stderr.write('frame %d %s FAILED %r\n' % (i, keys[i], exc))
    np.savez(shard, idx=np.arange(k0, k1), keys=np.array(keys[k0:k1]),
             gain=gain, t0=np.asarray(t0[k0:k1], np.float64), search=search,
             rot=rot, rot_t0=rot_t0, bad=bad)
    return shard, n, time.time() - t, int(bad.sum())


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--block', type=int, default=BLOCK)
    ap.add_argument('--limit', type=int, default=0, help='0 = all frames')
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--merge', action='store_true')
    ap.add_argument('--merge-out', default=os.path.join(HERE, 'sweep_full.npz'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    keys, t0 = load_frames()
    n = len(keys) if a.limit <= 0 else min(a.limit, len(keys))

    if a.merge:
        return merge(a.out, a.merge_out, keys, t0, n)

    _snapshot_rot_only()
    refs = json.load(open(os.path.join(PROJECT, 'inputs/refs.json')))
    positions = np.asarray([r['world_from_body'] for r in refs])[:, :3, 3]

    # --- two passes: blocks with many >0.2 m frames first, then the rest
    truth = load_truth()
    err = np.full(n, np.nan)
    for i in range(n):
        g = truth.get(tuple(keys[i].split('/')))
        if g is not None:
            err[i] = np.linalg.norm(t0[i][:3, 3] - g[:3, 3])
    blocks = [(k0, min(k0 + a.block, n)) for k0 in range(0, n, a.block)]
    share = {b: float(np.nanmean(err[b[0]:b[1]] > 0.2)) for b in blocks}
    blocks.sort(key=lambda b: -share[b])
    print('frames %d  blocks %d  workers %d  (%.1f%% of frames have t0 err > 0.2 m)'
          % (n, len(blocks), a.workers, 100.0 * np.nanmean(err > 0.2)), flush=True)

    jobs = [(k0, k1, os.path.join(a.out, 'blk_%06d.npz' % (k0 // a.block)))
            for k0, k1 in blocks]

    from multiprocessing import Pool
    t_start = time.time()
    done = frames = bad = 0
    with Pool(a.workers, initializer=_init,
              initargs=(refs, positions, keys, t0)) as pool:
        for shard, cnt, secs, nbad in pool.imap_unordered(run_block, jobs):
            done += 1
            frames += cnt
            bad += nbad
            el = time.time() - t_start
            left = len(jobs) - done
            eta = el / done * left
            print('[%3d/%3d] %5.1f%%  %6d frames  %6.1fs  bad=%d  elapsed %.1f min  eta %.1f min  %s'
                  % (done, len(jobs), 100.0 * done / len(jobs), frames, secs, bad,
                     el / 60.0, eta / 60.0, os.path.basename(shard)), flush=True)
    print('run finished in %.1f min, bad frames %d' % ((time.time() - t_start) / 60.0, bad))
    merge(a.out, a.merge_out, keys, t0, n)


def merge(out, dst, keys, t0, n):
    gain = np.full(n, np.nan)
    search = np.full((n, 4, 4), np.nan)
    rot = np.full((n, len(EPS), 4, 4), np.nan)
    rot_t0 = np.full((n, 4, 4), np.nan)
    bad = np.zeros(n, np.uint8)
    shards = sorted(f for f in os.listdir(out) if f.startswith('blk_') and f.endswith('.npz'))
    got = 0
    for name in shards:
        z = np.load(os.path.join(out, name))
        idx = z['idx']
        gain[idx] = z['gain']
        search[idx] = z['search']
        rot[idx] = z['rot']
        rot_t0[idx] = z['rot_t0']
        bad[idx] = z['bad']
        got += len(idx)
    print('merged %d shards, %d frames, %d bad -> %s'
          % (len(shards), got, int(bad.sum()), dst), flush=True)
    np.savez(dst, keys=np.array(keys[:n]), gain=gain, t0=np.asarray(t0[:n], np.float64),
             P_search=search, P_rot=rot, P_rot_t0=rot_t0, bad=bad, eps=np.array(EPS))
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""Live visual expert over the whole frame list: the frozen provider, re-derived.

One frame in, one row out: the retrieved reference list, every PnP candidate the frozen
three channels produce, its fair (laser) score, and the wall-clock cost of each stage.
Nothing is decided here; the arbitration in parallel_hypo.py consumes the row the same
way the cascade consumes the frozen table.

The implementation is not new. `visual_branch.VisualBranch` is the reconstruction of the
producer that wrote the cascade's table, `visual_expert.OrderedBranch` is that same class
with its two caches actually usable, and this module only adds two memoisations that
reuse the producer's own caching policy (`verify_prov.cluster_cloud`, one labelled tree
per reference set) so that a frame with 14 candidates does not rebuild the same 25 m
cloud 14 times. Both return the identical objects the uncached calls return, so the
numbers below are the branch's numbers.

    python visual_live.py --workers 6 --out /tmp/vislive
"""
import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from multiprocessing import Pool

for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '1'
os.environ.pop('CUDA_VISIBLE_DEVICES', None)

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import visual_branch as VB                     # noqa: E402
from visual_expert import OrderedBranch        # noqa: E402

_G = {}
_OPT = {}

# --- memoisation of the two objects the producer also cached ---------------------
# Both caches are bounded exactly the way the producer bounded its own (verify_prov
# keeps 32 labelled clouds, 4096 clouds of reference points): a 25 m labelled cloud is
# ~30 MB and its tree ~40 MB, so an unbounded cache is a memory leak, not a speed-up.
TREES_MAX = 32
CLOUDS_MAX = 32
_TREES = OrderedDict()
_CLOUDS = OrderedDict()
_ORIG_CLUSTER = VB.VisualBranch._cluster_cloud


def _kdtree_cached(points):
    """cKDTree keyed on the cloud's own content fingerprint.

    Identity is not usable as the key once the cloud cache is bounded: an evicted array
    can be freed and a later one can land on the same id. The fingerprint is the length
    plus the sum of every coordinate, which a 1.2 M-point cloud costs ~3 ms to compute
    against ~300 ms to build a tree.
    """
    key = (len(points), float(points.sum()))
    got = _TREES.get(key)
    if got is not None:
        _TREES.move_to_end(key)
        return got
    tree = cKDTree(points)
    _TREES[key] = tree
    if len(_TREES) > TREES_MAX:
        _TREES.popitem(last=False)
    return tree


def _cluster_cached(self, centre):
    """verify_prov's cluster_cloud: one labelled cloud per reference set, reused."""
    radius = self.verify_cfg['map_radius_m']
    near = np.flatnonzero(np.linalg.norm(self.positions - centre, axis=1) <= radius)
    key = tuple(int(x) for x in near)
    got = _CLOUDS.get(key)
    if got is not None:
        _CLOUDS.move_to_end(key)
        return got
    value = _ORIG_CLUSTER(self, centre)
    _CLOUDS[key] = value
    if len(_CLOUDS) > CLOUDS_MAX:
        _CLOUDS.popitem(last=False)
    return value


VB._kdtree = _kdtree_cached
VB.VisualBranch._cluster_cloud = _cluster_cached


def init():
    frames = json.load(open(os.path.join(HERE, 'inputs/frames.json')))
    z = np.load(os.path.join(HERE, 'inputs/initial.npz'), allow_pickle=True)
    poses = np.asarray(z['poses'], dtype=np.float64)
    keys = [str(k) for k in z['keys']]
    _G.update(frames=frames, poses=poses, initial_keys=keys,
              expert=OrderedBranch(device='cpu', arm='frozen_noprune', torch_threads=2,
                                   scratch=os.path.join(_OPT['scratch'],
                                                        'vb%d' % os.getpid())))


def image_of(key):
    return os.path.join(VB.DATA, 'NCLT', key.split('/')[0], 'cam5', key.split('/')[1] + '.jpg')


def one(i):
    expert = _G['expert']
    rec = _G['frames'][i]
    key = str(rec['key'])
    T0 = _G['poses'][i]
    row = dict(index=i, key=key, prior_translation=[float(x) for x in T0[:3, 3]])
    clock = {}
    a = time.time()
    try:
        scan = expert.C.decode(rec['scan_path'])
    except BaseException as exc:                      # noqa: BLE001
        row['error'] = 'decode:%r' % (exc,)
        row['seconds'] = time.time() - a
        return row
    clock['decode'] = time.time() - a

    a = time.time()
    refs, retrieval = expert.retrieve(image_of(key), T0)
    clock['retrieve'] = time.time() - a
    row['references'] = [int(x) for x in refs]
    row['retrieval'] = dict(visual=[int(x) for x in retrieval['visual']],
                            position=[int(x) for x in retrieval['position']],
                            visual_cosine=[float(x) for x in retrieval['visual_cosine']],
                            position_distance_m=[float(x) for x in
                                                 retrieval.get('position_distance_m', [])])
    if not refs:
        row['candidates'] = []
        row['seconds'] = clock
        return row

    a = time.time()
    q_model, cand, per_pair, batched = expert._match_frame(image_of(key), refs)
    clock['match'] = time.time() - a
    row['matches'] = int(len(cand))
    row['batched'] = bool(batched)
    if not cand:
        row['candidates'] = []
        row['seconds'] = clock
        return row

    by_reference = {}
    for query_index, score, reference, attribute in cand:
        entry = by_reference.setdefault(reference, ([], []))
        entry[0].append(query_index)
        entry[1].append(attribute)
    by_reference = {ri: (np.array(ids), np.array(aid)) for ri, (ids, aid) in by_reference.items()}

    a = time.time()
    hypotheses, accepted, calls, capped = expert._hypotheses(q_model, cand, by_reference)
    clock['pnp'] = time.time() - a
    row['pnp_calls'] = int(calls)
    row['capped'] = int(capped)

    a = time.time()
    query = expert._query_cloud(scan, expert.score_voxel)
    clock['query_cloud'] = time.time() - a
    row['query_points'] = int(len(query))

    a = time.time()
    rows = []
    for solution in hypotheses:
        try:
            fair = expert.verify(solution['pose'], query, solution['reference_ids'])
            score = float(fair['geometry_score'])
            verified = bool(fair['geometry_verified'])
        except BaseException as exc:                  # noqa: BLE001
            score, verified = None, None
            row.setdefault('verify_errors', []).append('%s:%r'
                                                       % (solution.get('channel'), (exc,)))
        rows.append(dict(channel=solution.get('channel'),
                         pose=[[float(x) for x in r] for r in solution['pose']],
                         inliers=int(solution['unique_query_inliers']),
                         inlier_ratio=float(solution['inlier_ratio']),
                         coverage_cells=int(solution['coverage_cells']),
                         reprojection_px=float(solution['median_reprojection_px']),
                         qualified=bool(solution['qualified']),
                         consensus=int(solution.get('consensus_reference_count', 0)),
                         reference_ids=[int(x) for x in solution['reference_ids']],
                         fair=score, verified=verified))
    clock['verify'] = time.time() - a
    row['candidates'] = rows
    row['seconds'] = clock
    row['total'] = float(sum(clock.values()))
    return row


def _w(i):
    row = one(i)
    if 'total' not in row:
        row['total'] = float(sum(row.get('seconds', {}).values())) \
            if isinstance(row.get('seconds'), dict) else float(row.get('seconds') or 0.0)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--scratch', default='/tmp/vislive-cache')
    ap.add_argument('--out', default='/tmp/vislive')
    a = ap.parse_args()
    os.makedirs(a.scratch, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    _OPT.update(scratch=a.scratch)

    n = len(json.load(open(os.path.join(HERE, 'inputs/frames.json'))))
    idx = list(range(0, n if a.limit <= 0 else min(a.limit, n), a.stride))
    path = os.path.join(a.out, 'live.jsonl')
    started = time.time()
    rows = []
    # streamed: one line per finished frame, so a crash costs one frame and not the run
    with open(path + '.tmp', 'w') as fh:
        if a.workers > 1:
            with Pool(a.workers, initializer=init) as pool:
                for row in pool.imap(_w, idx, chunksize=1):
                    fh.write(json.dumps(row) + '\n')
                    fh.flush()
                    rows.append(row)
        else:
            init()
            for i in idx:
                row = _w(i)
                fh.write(json.dumps(row) + '\n')
                fh.flush()
                rows.append(row)
    rows.sort(key=lambda r: r['index'])
    with open(path + '.tmp2', 'w') as fh:
        for row in rows:
            fh.write(json.dumps(row) + '\n')
    os.replace(path + '.tmp2', path)
    try:
        os.remove(path + '.tmp')
    except OSError:
        pass
    tot = np.array([r.get('total', 0.0) for r in rows])
    print('frames=%d wrote=%s wall=%.1fs per-frame median=%.2fs mean=%.2fs sum=%.1f min'
          % (len(rows), path, time.time() - started, float(np.median(tot)),
             float(tot.mean()), float(tot.sum()) / 60), flush=True)
    had = sum(1 for r in rows if r.get('candidates'))
    print('frames with >=1 candidate=%d  errors=%d  total candidates=%d'
          % (had, sum(1 for r in rows if 'error' in r),
             sum(len(r.get('candidates') or []) for r in rows)), flush=True)


if __name__ == '__main__':
    main()

"""V3: drive MoE over all 1502 keys, 8 processes, 4 query threads, gate2=-1.

usage: v3_run.py <out.npz> [visual 0|1]
"""
import json
import os
import sys
import time
from multiprocessing import Pool

for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '1'

import numpy as np

ROOT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
sys.path.insert(0, ROOT)
from algorithms import MoE  # noqa: E402

VERIFY = ('/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/'
          'verify-frozen_nopruneoptREF/%06d.json')
_G = {}


def visual_table(n):
    """The frozen visual candidates, exactly as the reference driver reads them."""
    out = []
    for i in range(n):
        rec = json.load(open(VERIFY % i))['poses']
        best = None
        for name, row in rec.items():
            if not name.startswith('candidate_'):
                continue
            sc = (row.get('fair') or {}).get('geometry_score')
            if sc is None:
                continue
            if best is None or sc > best[0]:
                best = (float(sc), np.asarray(row['pose'], float))
        out.append(None if best is None else best[1])
    return out


def init(visual, workers):
    refs = json.load(open(os.path.join(ROOT, 'inputs/refs.json')))
    positions = np.asarray([r['world_from_body'] for r in refs])[:, :3, 3]
    frames = json.load(open(os.path.join(ROOT, 'inputs/frames.json')))
    z = np.load(os.path.join(ROOT, 'inputs/initial.npz'), allow_pickle=True)
    _G.update(frames=frames, keys=[str(k) for k in z['keys']], visual=visual,
              poses=np.asarray(z['poses'], np.float64),
              moe=MoE(refs, positions, dict(workers=workers, gate2=-1.0, delta=0.05)))


def one(i):
    moe = _G['moe']
    scan = moe.prepare_scan(_G['frames'][i])
    vis = _G['visual'][i] if _G['visual'] is not None else None
    pose, diag = moe.localize(scan, _G['poses'][i], visual_pose=vis)
    return i, pose, diag


def main():
    out = sys.argv[1]
    use_visual = bool(int(sys.argv[2])) if len(sys.argv) > 2 else False
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    threads = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    n = 1502
    vis = visual_table(n) if use_visual else None

    t0 = time.time()
    with Pool(workers, initializer=init, initargs=(vis, threads)) as p:
        res = p.map(one, range(n), chunksize=4)
    res.sort(key=lambda r: r[0])
    poses = np.stack([r[1] for r in res])
    keys = np.array(_G['keys'] if _G.get('keys') else
                    [str(k) for k in np.load(os.path.join(ROOT, 'inputs/initial.npz'),
                                             allow_pickle=True)['keys']])
    np.savez(out, keys=keys[:n], poses=poses)
    with open(os.path.splitext(out)[0] + '.jsonl', 'w') as fh:
        for i, _, dg in res:
            dg = dict(dg)
            dg['key'] = str(keys[i])
            fh.write(json.dumps(dg) + '\n')
    dt = time.time() - t0
    nvis = sum(1 for _, _, d in res if d['chosen'] == 'visual')
    nfine = sum(1 for _, _, d in res if d['ran_fine'])
    print('visual=%d frames=%d fine=%d visual_chosen=%d  %.1f s (%.3f s/frame, %d procs)'
          % (use_visual, n, nfine, nvis, dt, dt / n, workers))
    print('wrote', out)


if __name__ == '__main__':
    main()

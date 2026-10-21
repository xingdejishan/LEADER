"""Run the chain over a frame list and write corrected poses.

Inputs (all paths are arguments, nothing is hard-coded):

  --frames   JSON: [{"key": ..., "scan_path": ...}, ...]
  --refs     JSON: [{"key": ..., "scan_path": ..., "world_from_body": 4x4}, ...]
  --poses    NPZ:  {"keys": [...], "poses": (N,4,4)}   initial poses, same order as frames
  --out      NPZ:  {"keys", "poses"} corrected poses
  --diag     JSONL: one diagnostics record per frame

Example:

  python run_frames.py --frames frames.json --refs refs.json \
      --poses initial.npz --out corrected.npz --diag diag.jsonl --workers 12
"""
import argparse
import json
import multiprocessing as mp

import numpy as np

import fast_lib as F
import pipeline as P

_STATE = {}


def _init(refs, positions, cfg):
    _STATE['maps'] = F.MapSource(refs, positions, radius=cfg.crop_radius)
    _STATE['cfg'] = cfg


def _one(item):
    record, initial = item
    pose, diag = P.localize(record, initial, _STATE['maps'], _STATE['cfg'])
    diag['key'] = record['key']
    return pose, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', required=True)
    ap.add_argument('--refs', required=True)
    ap.add_argument('--poses', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--diag', required=True)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--topk', type=int, default=6)
    ap.add_argument('--coarse-points', type=int, default=1500)
    ap.add_argument('--fine-points', type=int, default=4000)
    ap.add_argument('--no-prune', action='store_true')
    args = ap.parse_args()

    frames = json.load(open(args.frames))
    refs = json.load(open(args.refs))
    positions = np.asarray([r['world_from_body'] for r in refs])[:, :3, 3]

    blob = np.load(args.poses, allow_pickle=True)
    keys = [str(k) for k in blob['keys']]
    poses = np.asarray(blob['poses'], dtype=np.float64)
    order = {k: i for i, k in enumerate(keys)}
    initial = [poses[order[str(f['key'])]] for f in frames]

    cfg = P.Config(coarse_points=args.coarse_points, fine_points=args.fine_points,
                   topk=args.topk, prune=not args.no_prune)
    items = list(zip(frames, initial))

    if args.workers > 1:
        with mp.Pool(args.workers, initializer=_init,
                     initargs=(refs, positions, cfg)) as pool:
            results = pool.map(_one, items)
    else:
        _init(refs, positions, cfg)
        results = [_one(it) for it in items]

    out_poses = np.stack([r[0] for r in results])
    np.savez(args.out, keys=np.array([f['key'] for f in frames], dtype=object),
             poses=out_poses)
    with open(args.diag, 'w') as fh:
        for _, diag in results:
            fh.write(json.dumps({k: v for k, v in diag.items()
                                 if not isinstance(v, np.ndarray)} | {
                's': float(diag['s']), 'd': float(diag['d']),
                'cost': float(diag['cost']),
            }) + '\n')
    print('wrote %s (%d frames) and %s' % (args.out, len(results), args.diag))


if __name__ == '__main__':
    main()

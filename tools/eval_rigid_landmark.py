import argparse
import json
import time
from pathlib import Path

import torch

from models.rigid_landmark import RigidLandmarkFusion
from tools.eval_landmark_memory import reference_tensors
from tools.rigid_landmark_data import frames, collate, predict
from tools.train_local905 import digest


def main():
    parser = argparse.ArgumentParser()
    for name in ('cache', 'pose_cache', 'checkpoint', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--subset', choices=('val', 'test'), required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    manifest = json.loads((args.cache / 'manifest.json').read_text())
    if (checkpoint['cache_manifest_sha256'] != digest(args.cache / 'manifest.json') or
            checkpoint['pose_manifest_sha256'] != digest(args.pose_cache / 'manifest.json') or
            manifest['reference_map_sha256'] != digest(args.cache / 'reference_map.npz')):
        raise ValueError('Model and frozen data differ')
    model = RigidLandmarkFusion().cuda().eval()
    model.load_state_dict(checkpoint['model'])
    started = time.perf_counter()
    reference = reference_tensors(args.cache)
    queries = frames(args.cache, args.pose_cache, args.subset)
    output = []
    with torch.no_grad():
        for query in queries:
            try:
                pose = predict(model, reference, collate([query]))[0]
                if not torch.isfinite(pose).all():
                    raise FloatingPointError('Nonfinite pose')
                output.append(dict(scan=query['scan'], status='ok', T_world_body=pose.cpu().tolist()))
            except Exception as error:
                output.append(dict(scan=query['scan'], status='failed', error=str(error)))
    args.out.mkdir(parents=True, exist_ok=False)
    path = args.out / 'predictions.json'
    payload = dict(protocol='local905_gt_isolated_online_v1', subset=args.subset, predictions=output,
                   checkpoint_sha256=digest(args.checkpoint), split_sha256=manifest['split_sha256'],
                   elapsed_seconds=time.perf_counter()-started, magic=True,
                   reference_map_sha256=manifest['reference_map_sha256'],
                   timing_scope='Includes cache loading and rigid correction; excludes cached LEADER pose and SAM generation')
    path.write_text(json.dumps(payload, indent=2) + '\n')
    path.with_suffix('.sha256').write_text(digest(path) + '\n')


if __name__ == '__main__':
    main()

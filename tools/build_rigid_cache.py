import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from models.sc2pcr import Matcher
from utils.full_pool_robust_v1 import full_pool_refine
from tools.train_local905 import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--data_root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.cache / 'manifest.json').read_text())
    matcher = Matcher(inlier_threshold=2.0, d_thre=2, num_iterations=10, ratio=.15,
                      nms_radius=.1, max_points=3000, k1=30)
    center = torch.tensor(manifest['center_t'], device='cuda')
    records = {}
    for subset in ('train', 'val', 'test'):
        records[subset] = []
        with torch.no_grad():
            for i, record in enumerate(manifest['frames'][subset]):
                tick = time.perf_counter()
                with np.load(args.cache / record['path']) as data:
                    index = data['selected']
                    source = torch.tensor(data['source'][index], device='cuda')
                    target = torch.tensor(data['predicted'][index, :3], device='cuda') - center
                initial = matcher.estimator(source[None], target[None])[0]
                pose = full_pool_refine(initial, source, target)
                pose[:3, 3] += center
                records[subset].append(dict(scan=record['scan'], baseline=pose.cpu().tolist(),
                                           seconds=time.perf_counter() - tick))
                if (i+1) % 100 == 0 or i+1 == len(manifest['frames'][subset]):
                    print(subset, i+1, flush=True)
    output = args.out / 'baseline_poses.json'
    output.write_text(json.dumps(records, indent=2) + '\n')
    scene = args.data_root / 'train_scene'
    meta = json.loads((scene / 'scene_meta.json').read_text())
    camera_from_body = np.linalg.inv(np.array(meta['T_BC_camera_to_body']))
    truth = [np.loadtxt(scene / 'train/poses' / (Path(record['scan']).stem + '.txt')) @ camera_from_body
             for record in manifest['frames']['train']]
    np.save(args.out / 'training_poses.npy', np.asarray(truth, dtype=np.float32))
    (args.out / 'manifest.json').write_text(json.dumps(dict(
        feature_manifest_sha256=digest(args.cache / 'manifest.json'),
        baseline_poses_sha256=digest(output),
        training_poses_sha256=digest(args.out / 'training_poses.npy'),
        query_GT_in_online_cache=False, training_labels_separate=True), indent=2) + '\n')


if __name__ == '__main__':
    main()

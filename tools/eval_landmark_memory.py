import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from models.landmark_memory import LandmarkMemory
from models.sc2pcr import Matcher
from utils.full_pool_robust_v1 import full_pool_refine
from tools.train_local905 import digest


def reference_tensors(cache, device='cuda'):
    with np.load(cache / 'reference_map.npz', allow_pickle=False) as data:
        return {key: torch.as_tensor(data[key].astype(np.float32), device=device)
                for key in ('lidar', 'image', 'world', 'error', 'confidence')}


def predict(model, reference, batch):
    index = batch['candidate_index'].long()
    return model(batch['lidar'].float(), batch['image'].float(), batch['predicted'][..., :3].float(),
                 reference['lidar'][index], reference['image'][index], reference['world'][index],
                 reference['error'][index], reference['confidence'][index].tanh(),
                 batch['candidate_valid'].bool())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--subset', choices=('val', 'test'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.cache / 'manifest.json').read_text())
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if (checkpoint['cache_manifest_sha256'] != digest(args.cache / 'manifest.json') or
            digest(args.cache / 'reference_map.npz') != manifest['reference_map_sha256']):
        raise ValueError('Frozen reference/input cache changed')
    model = LandmarkMemory().cuda().eval()
    model.load_state_dict(checkpoint['model'])
    reference = reference_tensors(args.cache)
    center = torch.tensor(manifest['center_t'], device='cuda')
    matcher = Matcher(inlier_threshold=2.0, d_thre=2, num_iterations=10, ratio=0.15,
                      nms_radius=0.1, max_points=3000, k1=30)
    rows = []
    started = time.perf_counter()
    allowed = {'source', 'lidar', 'image', 'predicted', 'valid', 'selected',
               'candidate_index', 'candidate_valid'}
    with torch.no_grad():
        for record in manifest['frames'][args.subset]:
            tick = time.perf_counter()
            path = args.cache / record['path']
            try:
                if digest(path) != record['sha256']:
                    raise ValueError('Query cache fingerprint mismatch')
                with np.load(path, allow_pickle=False) as data:
                    if set(data.files) != allowed:
                        raise ValueError('Unexpected query fields; GT boundary violation')
                    batch = {key: torch.as_tensor(data[key], device='cuda') for key in data.files}
                selected = batch['selected'].long()
                delta = predict(model, reference, {key: value[selected] for key, value in batch.items()
                                                   if key != 'selected'})
                source = batch['source'][selected].float()
                target = batch['predicted'][selected, :3] - center + delta
                initial = matcher.estimator(source[None], target[None])[0]
                transform = full_pool_refine(initial, source, target)
                transform[:3, 3] += center
                torch.cuda.synchronize()
                if not torch.isfinite(transform).all():
                    raise FloatingPointError('Nonfinite final transform')
                rows.append(dict(scan=record['scan'], status='ok', T_world_body=transform.cpu().tolist(),
                                 seconds=time.perf_counter() - tick))
            except Exception as error:
                rows.append(dict(scan=record['scan'], status='failed', error=str(error),
                                 seconds=time.perf_counter() - tick))
    payload = dict(protocol='local905_gt_isolated_online_v1', checkpoint_sha256=digest(args.checkpoint),
                   split_sha256=manifest['split_sha256'], subset=args.subset, magic=True,
                   expected_frames=len(rows), predictions=rows,
                   elapsed_seconds=time.perf_counter() - started,
                   timing_scope='Cached frozen query features; excludes feature extraction and SAM preprocessing',
                   frozen_feature_extraction_seconds=sum(r['seconds'] for r in manifest['frames'][args.subset]),
                   reference_map_sha256=manifest['reference_map_sha256'],
                   sam_manifest_sha256=manifest['sam_manifest_sha256'])
    args.out.mkdir(parents=True, exist_ok=False)
    output = args.out / 'predictions.json'
    output.write_text(json.dumps(payload, indent=2) + '\n')
    output.with_suffix('.sha256').write_text(digest(output) + '\n')
    print(json.dumps(dict(frames=len(rows), failed=sum(r['status'] != 'ok' for r in rows),
                         elapsed_seconds=payload['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()

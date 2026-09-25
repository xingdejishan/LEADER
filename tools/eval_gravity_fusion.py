import argparse
import json
import time
from pathlib import Path

import torch

from models.gravity_fusion import MultimodalGravityFusion
from tools.gravity_fusion_data import frames, collate
from tools.train_local905 import digest


def main():
    parser = argparse.ArgumentParser()
    for name in ('cache', 'checkpoint', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--subset', choices=('val', 'test'), required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    manifest = json.loads((args.cache/'manifest.json').read_text())
    if checkpoint['cache_manifest_sha256'] != digest(args.cache/'manifest.json'):
        raise ValueError('Frozen surface cache differs')
    model = MultimodalGravityFusion().cuda().eval()
    model.load_state_dict(checkpoint['model'])
    started = time.perf_counter()
    queries = frames(args.cache, args.subset)
    output = []
    with torch.no_grad():
        for query in queries:
            try:
                pose = model(collate([query]))[0]
                if not torch.equal(pose[:3, 3], query['baseline'][:3, 3]):
                    raise RuntimeError('Fixed translation changed')
                if not torch.isfinite(pose).all():
                    raise FloatingPointError('Nonfinite final pose')
                output.append(dict(scan=query['scan'], status='ok', T_world_body=pose.cpu().tolist()))
            except Exception as error:
                output.append(dict(scan=query['scan'], status='failed', error=str(error)))
    args.out.mkdir(parents=True, exist_ok=False)
    path = args.out/'predictions.json'
    payload = dict(protocol='local905_gt_isolated_online_v1', subset=args.subset, predictions=output,
                   checkpoint_sha256=digest(args.checkpoint), split_sha256=manifest['split_sha256'],
                   elapsed_seconds=time.perf_counter()-started, multimodal=True, translation_frozen=True,
                   timing_scope='Gravity fusion with cached GeoCalib and LiDAR normals; excludes image and normal extraction',
                   sam_manifest_sha256=manifest['sam_manifest_sha256'])
    path.write_text(json.dumps(payload, indent=2)+'\n')
    path.with_suffix('.sha256').write_text(digest(path)+'\n')


if __name__ == '__main__':
    main()


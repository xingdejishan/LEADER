import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from tools.train_local905 import digest


def main():
    parser = argparse.ArgumentParser()
    for name in ('surface_cache', 'sam_manifest', 'vendor', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.vendor))
    from geocalib import GeoCalib
    model = GeoCalib().cuda().eval()
    surface = json.loads((args.surface_cache/'manifest.json').read_text())
    images = json.loads(args.sam_manifest.read_text())['frames']
    args.out.mkdir(parents=True, exist_ok=False)
    result = {key: surface[key] for key in ('split_sha256', 'sam_manifest_sha256', 'reference_training_poses_sha256')}
    result.update(frames={}, query_GT_present=False, visual_model='GeoCalib pinhole pretrained',
                  weights_sha256=digest(Path(torch.hub.get_dir())/'geocalib/pinhole.tar'),
                  input_surface_manifest_sha256=digest(args.surface_cache/'manifest.json'),
                  image_warp='Knew @ clockwise90 @ inv(K); 480x640 f300 principal point240,320')
    camera_rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    intrinsics = np.array([[300., 0., 240.], [0., 300., 320.], [0., 0., 1.]])
    tick = time.perf_counter()
    for subset, records in surface['frames'].items():
        result['frames'][subset] = []
        (args.out/subset).mkdir()
        for index, record in enumerate(records):
            image_record = images[record['scan']]
            image_path = (args.sam_manifest.parent/image_record['image']).resolve()
            if digest(image_path) != image_record['image_sha256']:
                raise ValueError('Source image changed')
            image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
            homography = intrinsics @ camera_rotation @ np.linalg.inv(image_record['K'])
            upright = cv2.warpPerspective(image, homography, (480, 640))
            tensor = torch.from_numpy(upright.copy()).permute(2, 0, 1).float().cuda()/255
            output = model.calibrate(tensor, priors={'focal': torch.tensor(300., device='cuda')})
            camera_up = output['gravity'].vec3d[0].cpu().numpy()
            body_up = np.asarray(image_record['T_camera_lidar'])[:3, :3].T @ camera_rotation.T @ camera_up
            uncertainty = np.array([float(output[key].flatten()[0]) for key in ('roll_uncertainty', 'pitch_uncertainty')], np.float32)
            with np.load(args.surface_cache/record['path']) as pair:
                baseline = pair['baseline'].copy()
            with np.load(args.surface_cache/record['cloud']) as cloud:
                count = min(len(cloud['source']), 512)
                chosen = np.linspace(0, len(cloud['source'])-1, count, dtype=np.int64)
                payload = dict(source=cloud['source'][chosen], normal=cloud['normal'][chosen],
                               image=cloud['image'][chosen], baseline=baseline,
                               visual_up=body_up.astype(np.float32), uncertainty=uncertainty)
            if not all(np.isfinite(value).all() for value in payload.values()):
                raise ValueError('Invalid gravity input')
            path = args.out/subset/f'{index:04d}.npz'
            np.savez_compressed(path, **payload)
            result['frames'][subset].append(dict(scan=record['scan'], path=str(path.relative_to(args.out)), sha256=digest(path)))
            if index % 25 == 0:
                print(json.dumps(dict(subset=subset, frame=index, seconds=time.perf_counter()-tick)), flush=True)
            if subset == 'train' and index == 0:
                cv2.imwrite(str(args.out/'upright_sample.jpg'), cv2.cvtColor(upright, cv2.COLOR_RGB2BGR))
    result['elapsed_seconds'] = time.perf_counter()-tick
    (args.out/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()

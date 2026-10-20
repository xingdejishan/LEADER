import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


RAW_DTYPE = np.dtype([('x', '<u2'), ('y', '<u2'), ('z', '<u2'),
                      ('intensity', 'u1'), ('ring', 'u1')])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    keys = json.loads(args.split.read_text(encoding='utf-8'))['splits']['train']
    key = keys[0]
    stem = Path(key).stem
    scene = args.data_root / 'train_scene'
    metadata = json.loads((scene / 'scene_meta.json').read_text(encoding='utf-8'))
    body_from_camera = np.asarray(metadata['T_BC_camera_to_body'])
    intrinsic = np.loadtxt(scene / 'train' / 'calibration' / (stem + '.txt'))
    raw = np.fromfile(args.data_root / key, dtype=RAW_DTYPE)
    body = np.column_stack((raw['x'], raw['y'], raw['z'])).astype(float) * 0.005 - 100
    camera = (body - body_from_camera[:3, 3]) @ body_from_camera[:3, :3]
    image_path = scene / 'train' / 'rgb' / (stem + '.jpg')
    with Image.open(image_path) as source:
        image = source.convert('RGB')
    pixels = camera @ intrinsic.T
    pixels = pixels[:, :2] / np.maximum(camera[:, 2:3], 1e-6)
    valid = ((camera[:, 2] > 2) & (camera[:, 2] < 80) &
             (pixels[:, 0] >= 0) & (pixels[:, 0] < image.width) &
             (pixels[:, 1] >= 0) & (pixels[:, 1] < image.height))
    selected = np.flatnonzero(valid)
    selected = selected[np.linspace(0, len(selected) - 1, min(5000, len(selected)), dtype=int)]
    draw = ImageDraw.Draw(image)
    for index in selected:
        x, y = pixels[index]
        depth = camera[index, 2]
        color = (int(255 * (1 - depth / 80)), int(255 * depth / 80), 40)
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.out)
    print(json.dumps({'scan': key, 'raw_points': len(body),
                      'in_image': int(valid.sum()), 'drawn': len(selected)}))


if __name__ == '__main__':
    main()

from functools import partial
from pathlib import Path
import sys

import numpy as np


def rgb_feature_extractor(vendor, checkpoint, device='cuda'):
    import torch
    from torch import nn
    from torchvision import transforms

    sys.path.insert(0, str(Path(vendor) / 'datasets'))
    from extract_features import DistilledVisionTransformer

    model = DistilledVisionTransformer(
        img_size=[480, 640], patch_size=16, embed_dim=384, depth=12,
        num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=256)
    saved = torch.load(checkpoint, map_location='cpu')['model_state_dict']
    model.load_state_dict({k.replace('module.backbone.', ''): v for k, v in saved.items()
                           if k.startswith('module.backbone')})
    model.to(device).eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        transforms.Resize([480, 640], antialias=False),
    ])

    def extract(paths):
        from PIL import Image
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(transform(image.convert('RGB')))
        with torch.inference_mode():
            return model(torch.stack(images).to(device)).float().cpu().numpy()

    extract.protocol = 'official_rgb_r2former_480x640'
    return extract


class CachedRGBFeatures:
    def __init__(self, split):
        import json
        split = Path(split)
        manifest = json.loads((split / 'features_manifest.json').read_text())
        paths = sorted(p for p in (split / 'rgb').iterdir()
                       if p.suffix.lower() in ('.jpg', '.png'))
        if manifest['protocol'] != 'official_rgb_r2former_480x640':
            raise ValueError('Expected the RGB R2Former feature protocol')
        if [p.name for p in paths] != manifest['images']:
            raise ValueError('Global feature order does not match scene images')
        self.features = np.load(split / 'features.npy', mmap_mode='r')
        import hashlib
        hasher = hashlib.sha256()
        with (split / 'features.npy').open('rb') as handle:
            for block in iter(lambda: handle.read(1048576), b''):
                hasher.update(block)
        if hasher.hexdigest() != manifest['sha256']:
            raise ValueError('Global feature cache checksum mismatch')
        if self.features.shape != (len(paths), 256):
            raise ValueError('Unexpected global feature shape')
        self.indices = {p.stem: i for i, p in enumerate(paths)}
        self.names = [p.name for p in paths]
        if len(self.indices) != len(paths):
            raise ValueError('Duplicate image stems')

    def __getitem__(self, stem):
        return np.array(self.features[self.indices[stem]], dtype=np.float32, copy=True)


def main():
    import argparse
    import hashlib
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'test'], default='test')
    parser.add_argument('--vendor', required=True)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    split = args.scene / args.split
    if (split / 'features.npy').exists() or (split / 'features_manifest.json').exists():
        raise FileExistsError('Feature cache already exists; refusing to overwrite')
    paths = sorted(p for p in (split / 'rgb').iterdir() if p.suffix.lower() in ('.jpg', '.png'))
    if not paths or len({p.stem for p in paths}) != len(paths):
        raise ValueError('Empty split or duplicate stems')
    meta = json.loads((args.scene / 'scene_meta.json').read_text())
    try:
        from .nclt_camera import validate_dates
    except ImportError:
        from nclt_camera import validate_dates
    validate_dates(meta['splits'].get('train', {}).get('dates', []),
                   meta['splits'].get('test', {}).get('dates', []))
    if {p.stem for p in paths} != {r['image'] for r in meta['splits'][args.split]['pairs']}:
        raise ValueError('Scene manifest and image list differ')
    extract = rgb_feature_extractor(args.vendor, args.checkpoint)
    features = np.empty((len(paths), 256), dtype=np.float32)
    for start in range(0, len(paths), 16):
        result = extract(paths[start:start + 16])
        if not np.isfinite(result).all() or not np.allclose(np.linalg.norm(result, axis=1), 1, atol=1e-5):
            raise ValueError('Invalid global features')
        features[start:start + len(result)] = result
        if start % 160 == 0:
            print(f'RGB {start + len(result)}/{len(paths)}', flush=True)
    temporary = split / 'features.tmp.npy'
    np.save(temporary, features)
    temporary.replace(split / 'features.npy')
    manifest = dict(protocol=extract.protocol, images=[p.name for p in paths],
                    sha256=hashlib.sha256((split / 'features.npy').read_bytes()).hexdigest(),
                    checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest())
    (split / 'features_manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()

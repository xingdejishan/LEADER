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
        if len(self.indices) != len(paths):
            raise ValueError('Duplicate image stems')

    def __getitem__(self, stem):
        return np.array(self.features[self.indices[stem]], dtype=np.float32, copy=True)

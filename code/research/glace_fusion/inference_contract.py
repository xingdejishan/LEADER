import hashlib
import json
from pathlib import Path

try:
    from .glace_adapter import GLACEAdapter, deit_global_feature_fn
    from .nclt_camera import preprocess_image
    from .rgb_features import CachedRGBFeatures, rgb_feature_extractor
except ImportError:
    from glace_adapter import GLACEAdapter, deit_global_feature_fn
    from nclt_camera import preprocess_image
    from rgb_features import CachedRGBFeatures, rgb_feature_extractor


RGB_PROTOCOL = 'official_rgb_r2former_480x640'


def resolve_contract(head, resolution=None):
    path = Path(head).parent / 'config.json'
    config = json.loads(path.read_text()) if path.exists() else {}
    protocol = config.get('global_feature_protocol', 'legacy_gray')
    expected = int(config.get('local_image_resolution', 480 if protocol == RGB_PROTOCOL else 616))
    if resolution is not None and resolution != expected:
        raise ValueError(f'Head expects image resolution {expected}, received {resolution}')
    return dict(global_feature_protocol=protocol, image_resolution=expected)


class InferenceSession:
    def __init__(self, vendor, head, checkpoint, *, split=None, T_BC=None,
                 encoder=None, resolution=None, pose_backend='opencv', hypotheses=None, coordinate_precision=None):
        self.contract = resolve_contract(head, resolution)
        self.rgb = self.contract['global_feature_protocol'] == RGB_PROTOCOL
        self.cache = CachedRGBFeatures(split) if self.rgb and split else None
        if self.rgb:
            config = json.loads((Path(head).parent / 'config.json').read_text())
            expected = config.get('global_backbone_sha256')
            if self.cache:
                actual = json.loads((Path(split) / 'features_manifest.json').read_text()).get('checkpoint_sha256')
            else:
                actual = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
            if not expected or actual != expected:
                raise ValueError('Global feature backbone does not match training checkpoint')
            if config.get('encoder_sha256'):
                encoder_path = Path(encoder) if encoder else Path(vendor) / 'ace_encoder_pretrained.pt'
                if hashlib.sha256(encoder_path.read_bytes()).hexdigest() != config['encoder_sha256']:
                    raise ValueError('Local encoder does not match training checkpoint')
        self.split = Path(split) if split else None
        self.extract = None
        if self.rgb and self.cache is None:
            self.extract = rgb_feature_extractor(vendor, checkpoint)
        legacy = None if self.rgb else deit_global_feature_fn(vendor, checkpoint)
        self.adapter = GLACEAdapter(vendor, head, encoder_path=encoder, T_BC=T_BC,
            global_feature_fn=legacy, pose_backend=pose_backend, coordinate_precision=coordinate_precision,
            pnp_threshold=10. if self.rgb or pose_backend == 'dsacstar' else 4.,
            hypotheses=hypotheses or (3200 if pose_backend == 'dsacstar' else 1000))
        self.contract.update(coordinate_precision=self.adapter.coordinate_precision, pose_backend=pose_backend, threshold_px=self.adapter.pnp_threshold,
                             hypotheses=self.adapter.hypotheses,
                             global_feature_source='cache' if self.cache else 'online',
                             head_sha256=self.adapter.head_sha256)

    def infer(self, path, K):
        gray, scaled_K = preprocess_image(path, K, self.contract['image_resolution'])
        feature = None
        if self.rgb:
            if self.cache:
                name = self.cache.names[self.cache.indices[Path(path).stem]]
                if (self.split / 'rgb' / name).resolve() != Path(path).resolve():
                    raise ValueError('Cached feature does not belong to this image path')
                feature = self.cache[Path(path).stem]
            else:
                feature = self.extract([path])[0]
        return self.adapter.infer(gray, scaled_K, global_feature=feature)

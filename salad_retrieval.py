"""Pure SALAD reference selection: one global cosine channel, K = 5, no pose anywhere.

This module is the whole reference-selection rule of the visual expert.  It replaces
the frozen two-channel rule (DINOv2 CLS cosine top-5 unioned with the 5 library
references nearest the caller's pose prior) with SALAD global cosine top-5 over the
entire 4752-image library.  The caller hands in an image path and gets reference ids
back; there is no second argument and there is no place to put a pose.

K = 5, chosen from the measurement already on record (the 290 hard frames of
`results/band5-nopose-frames.json`): K = 20 puts a <=5 m reference into 42 more
retrieval lists, yet fewer frames end up inside 5 m (29 against 31) and it costs about
a quarter more compute.  The number is a constant here, not a knob: `retrieve` takes
no channel count either.

Fixed assets, all inside this checkout (see the README section "SALAD 参考检索"):

    salad/aggregator.py             the SALAD head, vendored verbatim from
                                    github.com/serizba/salad (MIT)
    salad/weights/dino_salad.ckpt   official GSV-Cities checkpoint, 188/188 parameters
                                    loaded strictly, 352 MB, not tracked by git
    salad/reference-salad.npy       unit-norm 4752 x 8448 descriptors, one row per
                                    library reference in `library.json` order, 160 MB,
                                    not tracked by git; regenerate with
                                    `python salad/make_features.py`

The query side is embedded online with SALAD's official 322x322 evaluation transform
(~22 ms per frame on one card), so nothing about a query frame is precomputed or
looked up by image path.

Pose refusal is a run-time check, not a comment: `reject_pose_like` raises on a 4x4
pose, a 3-vector position and a 3x3 rotation matrix, and `retrieve` calls it on any
extra argument it is handed.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SALAD_ROOT = os.path.join(HERE, 'salad')
SALAD_WEIGHTS = os.path.join(SALAD_ROOT, 'weights', 'dino_salad.ckpt')
SALAD_REFERENCE = os.path.join(SALAD_ROOT, 'reference-salad.npy')
DINOV2_SOURCE = '/home/zhangcy/projects/v7-dinov2-recall-20261012-v1/dinov2-source'
EVAL_SIZE = 322                    # SALAD's official evaluation resize
TOP_K_REFERENCES = 5               # fixed: see the module docstring
DIMENSION = 8448                   # 64 clusters x 128 dims + 256 global token
BACKBONE_BLOCKS = 4                # the checkpoint trains the last four ViT blocks
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def reject_pose_like(value, name='argument'):
    """Refuse anything that carries a pose, a position or a rotation.  Never ignored."""
    if value is None:
        return
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        raise TypeError('%s: refusing non-numeric argument %r, reference selection is'
                        ' pose-free' % (name, value))
    if array.shape == (4, 4):
        raise TypeError('%s: refusing a 4x4 pose matrix, reference selection is pose-free'
                        % name)
    if array.shape in ((3,), (3, 1), (1, 3)):
        raise TypeError('%s: refusing a 3-vector position, reference selection is pose-free'
                        % name)
    if array.shape == (3, 3):
        raise TypeError('%s: refusing a 3x3 rotation matrix, reference selection is'
                        ' pose-free' % name)
    raise TypeError('%s: refusing non-image argument of shape %r, reference selection is'
                    ' pose-free' % (name, array.shape))


def model_image(path):
    """The image SALAD is evaluated on: upright, 322x322, bicubic, RGB uint8."""
    from PIL import Image
    with Image.open(path) as image:
        image = image.convert('RGB').transpose(Image.Transpose.ROTATE_270)
        image = image.resize((EVAL_SIZE, EVAL_SIZE), Image.Resampling.BICUBIC)
        return np.asarray(image).copy()


def _unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


def build_model(device='cuda'):
    """The official SALAD descriptor, rebuilt without `pytorch_lightning`.

    The reconstruction is the one already used for the feature extraction on record:
    the last four ViT blocks stay trainable, the earlier blocks run under no_grad, and
    the SALAD head (64 clusters, 128 dims, 256-token) aggregates the patch grid.  A
    checkpoint that does not load every parameter is a hard error, not a warning.
    """
    import torch

    import sys
    if SALAD_ROOT not in sys.path:
        sys.path.insert(0, SALAD_ROOT)
    from aggregator import SALAD
    if not os.path.isfile(SALAD_WEIGHTS):
        raise RuntimeError('SALAD weights missing at %s (see salad/make_features.py)'
                           % SALAD_WEIGHTS)

    class _Model(torch.nn.Module):
        def __init__(self, backbone, head, blocks):
            super(_Model, self).__init__()
            self.backbone = torch.nn.Module()
            self.backbone.model = backbone
            self.backbone.blocks_trained = blocks
            self.aggregator = head

        def forward(self, x):
            b, _, h, w = x.shape
            tokens = self.backbone.model.prepare_tokens_with_masks(x)
            with torch.no_grad():
                for block in self.backbone.model.blocks[:-self.backbone.blocks_trained]:
                    tokens = block(tokens)
            tokens = tokens.detach()
            for block in self.backbone.model.blocks[-self.backbone.blocks_trained:]:
                tokens = block(tokens)
            tokens = self.backbone.model.norm(tokens)
            token = tokens[:, 0]
            patch = tokens[:, 1:].reshape(b, h // 14, w // 14, 768).permute(0, 3, 1, 2)
            return self.aggregator((patch, token))

    backbone = torch.hub.load(DINOV2_SOURCE, 'dinov2_vitb14', source='local',
                              pretrained=False)
    model = _Model(backbone, SALAD(num_channels=768, num_clusters=64, cluster_dim=128,
                                   token_dim=256, dropout=0.3), BACKBONE_BLOCKS)
    payload = torch.load(SALAD_WEIGHTS, map_location='cpu')
    state = payload.get('state_dict', payload) if isinstance(payload, dict) else payload
    own = dict(model.state_dict())
    usable = {k: v for k, v in state.items() if k in own}
    missing = [k for k in own if k not in usable]
    if missing or len(usable) != len(own):
        raise RuntimeError('SALAD checkpoint does not cover the model: %d/%d parameters,'
                           ' missing %r' % (len(usable), len(own), missing[:5]))
    model.load_state_dict(usable, strict=True)
    model = model.eval().requires_grad_(False)
    return model.cuda() if device == 'cuda' else model


def load_reference(path=SALAD_REFERENCE):
    """The reference descriptor matrix, unit-norm, rows in `library.json` order."""
    if not os.path.isfile(path):
        raise RuntimeError('SALAD reference features missing at %s: run'
                           ' `python salad/make_features.py`' % path)
    matrix = np.load(path).astype(np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != DIMENSION:
        raise RuntimeError('SALAD reference features have shape %r, expected (N, %d)'
                           % (matrix.shape, DIMENSION))
    return _unit(matrix)


class SaladRetrieval(object):
    """Global SALAD cosine over the whole library.  Construct once per process."""

    def __init__(self, device='cuda', reference=SALAD_REFERENCE, verbose=False):
        self.device = device
        self.reference = load_reference(reference)
        self.model = build_model(device)
        self.verbose = verbose

    def embed(self, image_path):
        """One unit-norm SALAD descriptor for one image.  Image in, vector out."""
        import torch
        torch_ = torch
        x = torch_.from_numpy(model_image(image_path)).permute(2, 0, 1).float() / 255
        x = (x - torch_.tensor(MEAN)[:, None, None]) / torch_.tensor(STD)[:, None, None]
        x = x[None].to(self.device)
        with torch_.inference_mode():
            descriptor = self.model(x)[0]
        return _unit(descriptor.float().cpu().numpy())

    def topk(self, image_path, k=TOP_K_REFERENCES):
        """The reference ids, highest SALAD cosine first, plus the values."""
        k = int(k)
        if k != TOP_K_REFERENCES:
            raise ValueError('K is fixed at %d for this expert, got %d'
                             % (TOP_K_REFERENCES, k))
        similarity = self.reference @ self.embed(image_path).astype(np.float32)
        ids = [int(i) for i in np.argsort(-similarity)[:k]]
        detail = dict(mode='salad', k=k, pose_free=True, visual=ids,
                      visual_cosine=[float(similarity[i]) for i in ids])
        if self.verbose:
            print('[salad] %s -> %s' % (image_path, ids), flush=True)
        return ids, detail

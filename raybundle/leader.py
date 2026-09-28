import math

import MinkowskiEngine as ME
import numpy as np
import torch

from models.model_mink import LEADER
from models.sc2pcr import Matcher
from utils.full_pool_robust_v1 import full_pool_refine
from .mapping import sha256, extract, camera_rays


L0_SHA = '49cf3085b507e2f67a0f250394cadd980b0f5da18eb00dcfe37a6c766d3f4d7a'


def load_leader(path):
    if sha256(path) != L0_SHA:
        raise ValueError('official L0 checkpoint SHA differs')
    checkpoint = torch.load(str(path), map_location='cpu')
    model = LEADER(in_channels=3, out_channels=4, feat_channels=512)
    missing, extra = model.load_state_dict(checkpoint['model'], strict=False)
    if extra or any(not x.endswith('num_batches_tracked') for x in missing):
        raise ValueError('LEADER architecture differs')
    model.requires_grad_(False).eval()
    center = torch.as_tensor(checkpoint['center_t'], dtype=torch.float32)
    return model, center


def encode_frame(sample, record, model, center, noise, max_queries=64):
    torch.manual_seed(20)
    with torch.no_grad():
        coordinates = ME.utils.batched_coordinates([sample['coords']])
        sparse = ME.SparseTensor(torch.as_tensor(sample['feats']), coordinates)
        encoded = model.encoder(sparse)
        stride = torch.as_tensor(encoded.tensor_stride, dtype=torch.float32)
        polar = (encoded.C[:, 1:].float() + stride / 2) * .2
        angle = polar[:, 0] * (2 * math.pi / (1024 * .2))
        source = torch.stack((polar[:, 1] * angle.cos(), polar[:, 1] * angle.sin(), polar[:, 2]), -1)
        prediction = model.decoder(encoded.F)
        count = max(min(50, len(prediction)), int(.5 * len(prediction)))
        if not 6 <= count <= 10000:
            raise ValueError('LiDAR point count outside inference budget')
        indices = prediction[:, 3].topk(count).indices
        source, predicted = source[indices].float(), prediction[indices, :3].float()
        matcher = Matcher(inlier_threshold=2., d_thre=2, num_iterations=10, ratio=.15, nms_radius=.1, max_points=3000, k1=30)
        initial = matcher.estimator(source[None], predicted[None])[0]
        centered = full_pool_refine(initial, source, predicted)
        pose = centered.clone()
        pose[:3, 3] += center
        residual = (source @ centered[:3, :3].T + centered[:3, 3] - predicted).norm(dim=-1)
        weights = torch.sigmoid(prediction[indices, 3]) * (1 - (residual / .6).square()).clamp_min(0).square()
    pixels, desc = extract(record.image_path, max_features=512, expected_sha=record.image_sha256)
    if len(desc) == 0:
        raise ValueError('no query image features')
    pixels, desc = pixels[:max_queries], desc[:max_queries]
    rays = camera_rays(pixels) @ record.T_BC[:3, :3].T
    origins = np.repeat(record.T_BC[None, :3, 3], len(rays), axis=0)
    frame = dict(source=source.double(), target=(predicted + center).double(), weights=weights.double(), T0=pose.double(),
                 origins=torch.from_numpy(origins.copy()), directions=torch.from_numpy(rays), descriptors=torch.from_numpy(desc),
                 pixels=torch.from_numpy(pixels), group_ids=torch.arange(len(rays)), camera_ids=torch.full((len(rays),),5),
                 independent_sigma=torch.tensor(noise['independent_sigma'], dtype=torch.float64),
                 shared_covariance=torch.tensor(noise['shared_covariance'], dtype=torch.float64))
    return frame

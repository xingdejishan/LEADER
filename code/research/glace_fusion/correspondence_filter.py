from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from .valid_region import resize_valid_mask, sample_valid_region, validate_mask


class CorrespondenceFilter:
    def __init__(self, valid_mask, head, confidence_folder=None):
        self.mask = validate_mask(np.load(valid_mask))
        self.model = None
        self.protocol = None
        self.mask_hash = hashlib.sha256(Path(valid_mask).read_bytes()).hexdigest()
        if confidence_folder is not None:
            import joblib
            folder = Path(confidence_folder)
            self.protocol = json.loads((folder / 'protocol.json').read_text())
            head_hash = hashlib.sha256(Path(head).read_bytes()).hexdigest()
            if head_hash != self.protocol['head_sha256'] or self.mask_hash != self.protocol['valid_mask_sha256']:
                raise ValueError('Confidence model requires its exact head and FOV mask')
            if self.protocol['inference_contract']['coordinate_precision'] != 'fp32_head':
                raise ValueError('Confidence model requires FP32 correspondence')
            self.model = joblib.load(folder / 'confidence.joblib')

    def apply(self, output):
        height, width = output.image_size_hw
        if self.model is not None and (height, width) != (480, 630):
            raise ValueError('Confidence features require the calibrated 480x630 image')
        mask = resize_valid_mask(self.mask, height, width)
        valid = sample_valid_region(mask, output.uv)
        valid &= np.isfinite(output.xyz_world).all(1)
        indices = np.flatnonzero(valid)
        if not len(indices):
            raise ValueError('No finite correspondence in valid FOV')
        count_valid = len(indices)
        if self.model is not None:
            from .correspondence_confidence import point_probabilities, select
            probability = point_probabilities(self.model, output.xyz_world[indices], output.uv[indices], self.protocol)
            indices = indices[select(probability, self.protocol['retained_fraction'])]
        inliers = None if output.inlier_mask is None else output.inlier_mask[indices]
        diagnostics = dict(output.diagnostics, correspondence_filter=dict(
            input_count=len(output.uv), valid_fov_count=count_valid, output_count=len(indices),
            mask_sha256=self.mask_hash, confidence_filter=self.model is not None))
        return replace(output, uv=output.uv[indices], xyz_world=output.xyz_world[indices],
            inlier_mask=inliers, inlier_count=None if inliers is None else int(inliers.sum()),
            diagnostics=diagnostics)

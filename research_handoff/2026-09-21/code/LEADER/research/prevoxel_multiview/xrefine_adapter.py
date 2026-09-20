import hashlib
import sys
from pathlib import Path

import numpy as np


XREFINE_COMMIT = "5261f65ff64575a31e5330ecd5d72ee5f0ba27bb"
XREFINE_WEIGHT_SHA256 = "f8962ee8d0eddbf4c4a455a3424e18818fdf457d374f924be2cf7abb8f3bb5e8"
PATCH_RADIUS = 5


def xrefine_to_project(pixels):
    return np.asarray(pixels, dtype=np.float64) - .5


def project_to_xrefine(pixels):
    return np.asarray(pixels, dtype=np.float64) + .5


def full_patch_valid(mask, pixels):
    mask = np.asarray(mask, dtype=bool)
    pixels = project_to_xrefine(pixels)
    centre = np.floor(pixels).astype(np.int64)
    valid = ((centre[:, 0] - PATCH_RADIUS >= 0) & (centre[:, 0] + PATCH_RADIUS < mask.shape[1]) &
             (centre[:, 1] - PATCH_RADIUS >= 0) & (centre[:, 1] + PATCH_RADIUS < mask.shape[0]))
    for index in np.flatnonzero(valid):
        x, y = centre[index]
        valid[index] = bool(mask[y - PATCH_RADIUS:y + PATCH_RADIUS + 1, x - PATCH_RADIUS:x + PATCH_RADIUS + 1].all())
    return valid, centre


class XRefineAdapter:
    def __init__(self, source_dir, weights, device="cuda"):
        import torch

        source_dir, weights = Path(source_dir), Path(weights)
        if not source_dir.joinpath("hubconf.py").exists():
            raise FileNotFoundError(source_dir)
        digest = hashlib.sha256(weights.read_bytes()).hexdigest()
        if digest != XREFINE_WEIGHT_SHA256:
            raise ValueError("unexpected XRefine weight SHA256")
        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))
        from hubconf import XRefine

        self.torch = torch
        self.device = torch.device(device)
        self.model = XRefine(pretrained=False, detector="general", variant="small",
                             adjust_only_second_keypoint=True, image_values_are_normalized=True)
        self.model.net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=False)["model"])
        self.model.to(self.device).eval()

    def refine_pair(self, reference_image, query_image, reference_pixels, query_pixels, reference_mask, query_mask,
                    batch_size=512):
        from PIL import Image

        reference_pixels = np.asarray(reference_pixels, dtype=np.float64)
        query_pixels = np.asarray(query_pixels, dtype=np.float64)
        output = query_pixels.copy()
        candidate = query_pixels.copy()
        status = np.full(len(query_pixels), "fallback", dtype="U24")
        ref_valid, ref_centre = full_patch_valid(reference_mask, reference_pixels)
        query_valid, query_centre = full_patch_valid(query_mask, query_pixels)
        valid = ref_valid & query_valid
        status[~ref_valid] = "reference_patch_invalid"
        status[ref_valid & ~query_valid] = "query_patch_invalid"
        if not valid.any():
            return candidate, output, status, {"reference_patch_center": ref_centre, "query_patch_center": query_centre}
        image = lambda path: self.torch.from_numpy(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.).to(self.device)
        reference, query = image(reference_image), image(query_image)
        valid_indices = np.flatnonzero(valid)
        with self.torch.inference_mode():
            for start in range(0, len(valid_indices), batch_size):
                index = valid_indices[start:start + batch_size]
                ref = self.torch.as_tensor(project_to_xrefine(reference_pixels[index]), dtype=self.torch.float32, device=self.device)
                initial = self.torch.as_tensor(project_to_xrefine(query_pixels[index]), dtype=self.torch.float32, device=self.device)
                refined_ref, refined_query = self.model(ref, initial, reference, query)
                if not self.torch.allclose(refined_ref, ref, atol=1e-6, rtol=0.):
                    raise RuntimeError("single-sided XRefine changed reference coordinates")
                refined = xrefine_to_project(refined_query.detach().cpu().numpy())
                finite = np.isfinite(refined).all(axis=1)
                inside = ((refined[:, 0] >= 0) & (refined[:, 0] < query_mask.shape[1]) &
                          (refined[:, 1] >= 0) & (refined[:, 1] < query_mask.shape[0]))
                pixel_x = np.clip(np.floor(refined[:, 0] + .5).astype(np.int64), 0, query_mask.shape[1] - 1)
                pixel_y = np.clip(np.floor(refined[:, 1] + .5).astype(np.int64), 0, query_mask.shape[0] - 1)
                accepted = finite & inside & np.asarray(query_mask, dtype=bool)[pixel_y, pixel_x]
                output[index[accepted]] = refined[accepted]
                status[index[accepted]] = "refined"
                status[index[~accepted]] = "output_invalid"
        return candidate, output, status, {"reference_patch_center": ref_centre, "query_patch_center": query_centre}

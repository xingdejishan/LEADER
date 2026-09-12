import argparse
import importlib
import json
from pathlib import Path
import random
import sys

import numpy as np

from .valid_region import resize_valid_mask, sample_valid_region, grid_valid_region


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vendor', type=Path, required=True)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--mask', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.vendor))
    import torch
    dataset_module = importlib.import_module('dataset')
    mask = np.load(args.mask)
    rows = []
    for augment in [False, True]:
        dataset = dataset_module.CamLocDataset(args.scene / 'train', augment=augment, image_height=480, use_half=False)
        for height in [320, 480, 720]:
            for index in [0, len(dataset) // 2, len(dataset) - 1]:
                def load(region):
                    dataset.valid_region = region
                    random.seed(2089)
                    torch.manual_seed(2089)
                    return dataset._get_single_item(index, height)
                full = load(None)
                masked = load(mask)
                for component in [0, 2, 3, 4, 5, 8]:
                    torch.testing.assert_close(full[component], masked[component], rtol=0, atol=0)
                image_mask = masked[1][None]
                H, W = masked[0].shape[-2:]
                grid = grid_valid_region(image_mask, (H + 7) // 8, (W + 7) // 8)
                yy, xx = np.mgrid[:grid.shape[-2], :grid.shape[-1]]
                uv = np.column_stack([8 * (xx.ravel() + .5), 8 * (yy.ravel() + .5)])
                reference = sample_valid_region(masked[1][0].numpy(), uv)
                np.testing.assert_array_equal(grid.numpy().reshape(-1), reference)
                if not augment:
                    expected = resize_valid_mask(mask, H, W)
                    np.testing.assert_array_equal(reference, sample_valid_region(expected, uv))
                if not grid.any() or grid.all():
                    raise ValueError('Unexpected valid-FOV grid')
                rows.append(dict(augment=augment, height=height, image_index=index,
                    valid_grid_fraction=float(grid.float().mean()), geometry_and_image_unchanged=True))
    report = dict(passed=True, cases=len(rows), records=rows)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

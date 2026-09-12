import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .correspondence_filter import CorrespondenceFilter
from .glace_adapter import GLACEOutput


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    base = Path('/root/rivermind-data')
    selection = json.loads((root / 'selection.json').read_text())
    head = root / selection['arm'] / 'head.pt'
    if hashlib.sha256(head.read_bytes()).hexdigest() != selection['head_sha256']:
        raise ValueError('Selected checkpoint changed')
    train = set(json.loads((root / 'train_stems.json').read_text()))
    validation = set(json.loads((root / 'validation_stems.json').read_text()))
    test = {p.stem for p in (root / 'test/coordinates').glob('*.npz')}
    if train & validation or train & test or validation & test:
        raise ValueError('Data split overlap')
    model = CorrespondenceFilter(base / 'glace_nclt_stage2_local_mask_20260912/valid_mask.npy',
        head, root / 'confidence')
    for stem in sorted(test):
        old = np.load(base / 'glace_stage3_region_full_20260912/improved/coordinates' / (stem + '.npz'))
        new = np.load(root / 'test/coordinates' / (stem + '.npz'))
        filtered = np.load(root / 'confidence/coordinates' / (stem + '.npz'))
        for key in ['uv', 'K', 'GT']:
            np.testing.assert_array_equal(old[key], new[key])
        output = GLACEOutput(None, None, new['uv'], new['xyz'], new['K'], None, None, (480, 630))
        replay = model.apply(output)
        np.testing.assert_array_equal(replay.uv, filtered['uv'])
        np.testing.assert_array_equal(replay.xyz_world, filtered['xyz'])
        np.testing.assert_array_equal(new['xyz'][filtered['original_indices']], filtered['xyz'])
    report = dict(complete=True, frames=len(test), head_sha256=selection['head_sha256'],
        split_disjoint=True, training_frames=len(train), validation_frames=len(validation),
        paired_uv_K_GT_identical=True, deployment_filter_matches_offline_all_frames=True,
        confidence_inference_inputs='predicted xyz and uv only; no test labels or scan depths')
    (root / 'verification.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

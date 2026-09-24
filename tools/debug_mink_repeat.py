import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import MinkowskiEngine as ME
import torch

from models.model_mink import LEADER
from run_mink import get_data_loader


def align(first, second):
    a = {tuple(row): index for index, row in enumerate(first[0].tolist())}
    b = {tuple(row): index for index, row in enumerate(second[0].tolist())}
    if set(a) != set(b):
        return {'same_coordinates': False, 'a': len(a), 'b': len(b)}
    keys = sorted(a)
    af = first[1][[a[key] for key in keys]]
    bf = second[1][[b[key] for key in keys]]
    difference = (af - bf).abs()
    return {'same_coordinates': True, 'max': float(difference.max()),
            'mean': float(difference.mean()),
            'count_gt_1e_4': int((difference > 1e-4).sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--base_checkpoint', type=Path, required=True)
    args = parser.parse_args()
    flags = SimpleNamespace(dataset='Local905', dataset_folder=str(args.data_root),
                            local905_split=str(args.split), local905_max_points=0,
                            voxel_size=0.2, horizontal_res=1024, batch_size=1,
                            val_batch_size=1, num_workers=0, mode='train', magic_manifest='')
    _, val = get_data_loader(flags)
    batch = next(iter(val))
    model = LEADER(in_channels=3, out_channels=4, feat_channels=512, magic=False).cuda().eval()
    model.load_state_dict(torch.load(args.base_checkpoint, map_location='cpu')['model'])
    captured = {}
    names = ('stem', 'encoders.0.downsample', 'encoders.0.res',
             'encoders.0.maxpool', 'encoders.0.fuse',
             'encoders.1.fuse', 'encoders.2.fuse', 'encoders.3.fuse',
             'encoders.4.fuse')
    handles = []
    for name, module in model.encoder.named_modules():
        if name in names:
            def hook(_, __, output, key=name):
                captured.setdefault(key, []).append((output.C.cpu().clone(), output.F.cpu().clone()))
            handles.append(module.register_forward_hook(hook))
    with torch.no_grad():
        sparse = ME.SparseTensor(batch['feats'].cuda(), batch['coords'].cuda())
        model.encoder(sparse)
        model.encoder(sparse)
    for handle in handles:
        handle.remove()
    report = {name: align(*values) for name, values in captured.items()}
    for index in (0, 1):
        down = captured['encoders.0.downsample'][index][0]
        residual = captured['encoders.0.res'][index][0]
        report[f'downsample_residual_row_order_equal_{index}'] = bool(torch.equal(down, residual))
        report[f'downsample_residual_coordinate_set_equal_{index}'] = (
            set(map(tuple, down.tolist())) == set(map(tuple, residual.tolist())))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

import argparse
import json
import time
from pathlib import Path

import MinkowskiEngine as ME
import numpy as np
import torch
from PIL import Image

from data.local905_query import Local905Query
from models.magic_fusion import project_voxel_centers
from run_mink import TRR
from tools.train_relation import setup, write
from tools.train_local905 import batch_loss
from tools.train_magic_revision import freeze_bn_stats


def projection_check(frame, record, manifest_dir):
    points = torch.as_tensor(frame['points'][0], dtype=torch.float64)
    intrinsic = frame['intrinsics'].double()
    extrinsic = frame['camera_from_lidar'].double()
    bounds = frame['image_bounds'].double()
    identity = torch.eye(4, dtype=torch.float64)[None]
    batch = torch.zeros(len(points), dtype=torch.long)
    pixels, valid = project_voxel_centers(points, batch, intrinsic, extrinsic, identity, bounds)
    raw_camera = points.numpy() @ np.asarray(record['T_camera_lidar'])[:3, :3].T + np.asarray(record['T_camera_lidar'])[:3, 3]
    positive = raw_camera[:, 2] > 1e-6
    original = raw_camera[positive] @ np.asarray(record['K']).T
    original = original[:, :2]/original[:, 2:]
    with Image.open(manifest_dir/record['image']) as image:
        original *= np.asarray(record['resized_size'])/np.asarray(image.size)
    good = valid.numpy()[positive]
    difference = np.abs(original[good]-pixels.numpy()[positive][good]).max()
    angle = .37
    augment = torch.tensor([[np.cos(angle),-np.sin(angle),0,.8], [np.sin(angle),np.cos(angle),0,-.4],
                            [0,0,1,.2], [0,0,0,1]], dtype=torch.float64)
    transformed = points @ augment[:3,:3].T + augment[:3,3]
    recovered, _ = project_voxel_centers(transformed, batch, intrinsic, extrinsic, torch.linalg.inv(augment)[None], bounds)
    recovery_error = float((recovered[valid]-pixels[valid]).abs().max())
    if difference > .001 or recovery_error > 1e-8:
        raise ValueError('Projection/augmentation mismatch')
    return dict(valid_points=int(valid.sum()), raw_to_sam_max_pixel_error=float(difference),
                inverse_augmentation_max_pixel_error=recovery_error)


def main():
    parser = argparse.ArgumentParser()
    for name in ('data_root','assets','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    model, base, loader = setup(args)
    manifest_path = args.assets/'sam_cache/manifest_905.json'
    manifest = json.loads(manifest_path.read_text())
    results = []
    for index in (0,319):
        batch = loader.collate_fn([loader.dataset[index]])
        key = loader.dataset.keys[index]
        if not torch.equal(batch['T_corr'],torch.eye(4)[None]):
            raise ValueError('Unexpected current training augmentation')
        results.append(dict(scan=key, **projection_check(batch,manifest['frames'][key],manifest_path.parent)))
    query = Local905Query(args.data_root,args.assets/'split_masked.json',manifest_path,max_points=0,subset='test')
    key = query.keys[0]
    results.append(dict(scan=key, **projection_check(query.load(key),manifest['frames'][key],manifest_path.parent)))
    model.eval()
    sample = loader.collate_fn([loader.dataset[0]])
    sparse = ME.SparseTensor(sample['feats'].cuda(),sample['coords'].cuda())
    with torch.no_grad():
        original = model.encoder(sparse)
        changed = model.interaction.encode(model.encoder,sparse,sample)
        if not torch.equal(original.C,changed.C):
            raise ValueError('Sparse coordinates changed')
        difference = float((original.F-changed.F).abs().max())
        prediction_difference = float((model.decoder(original.F)-model.decoder(changed.F)).abs().max())
        if difference > 1e-5 or prediction_difference > 1e-5:
            raise ValueError('Pretrained initial behavior not preserved')
    model.train()
    freeze_bn_stats(model)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-5)
    center = torch.tensor(base['center_t'],device='cuda')
    torch.cuda.reset_peak_memory_stats()
    tick = time.perf_counter()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        for index in range(step*8,(step+1)*8):
            batch = loader.collate_fn([loader.dataset[index]])
            loss,_ = batch_loss(model,batch,center,TRR(scale=10),True,.2,1024)
            (loss/8).backward()
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('Invalid real training gradient')
        gradients = {prefix:sum(float(p.grad.abs().sum()) for name,p in model.named_parameters()
                                if name.startswith(prefix) and p.grad is not None)
                     for prefix in ('encoder','decoder','interaction.image_encoder','interaction.exchanges.1','interaction.exchanges.3',
                                    'interaction.exchanges.1.transport.edge','interaction.exchanges.3.transport.edge')}
        optimizer.step()
    if any(value <= 0 for value in gradients.values()):
        raise ValueError('Missing modality/backbone gradient')
    report = dict(projection=results, initial_feature_max_difference=difference,
        initial_prediction_max_difference=prediction_difference, gradient_l1=gradients,
        two_updates_seconds=time.perf_counter()-tick,peak_memory_mib=torch.cuda.max_memory_allocated()/1024**2,
        scope='Numerical actual pipeline check, not proof of physical calibration or synchronization accuracy')
    write(args.out/'verification.json',report)
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()

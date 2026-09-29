import argparse
import json
from pathlib import Path

import MinkowskiEngine as ME
import numpy as np
import torch

from data.leader_nclt_manifest import ManifestLEADERDataset, TRAIN_DATES, collate_leader_frames
from tools.magic_nclt_runtime import (batch_forward, configure_training_mode, load_pretrained_model,
                                      model_parameter_groups, safe_json_dump, state_dict_sha256)
from tools.train_magic_nclt import _move_training_state


def ordered(details):
    coordinates = details['encoded_coordinates'].detach().cpu().numpy()
    order = np.lexsort(tuple(coordinates[:, column] for column in (3, 2, 1, 0)))
    return coordinates[order], details['predictions'].detach().cpu()[order]


def check(args):
    dataset = ManifestLEADERDataset([args.data / date / 'manifest.json' for date in TRAIN_DATES],
                                    args.data, split='train', sam_cache_manifest=args.sam, cache_memory_frames=1)
    indices = np.random.default_rng(20).permutation(len(dataset))[:4]
    batch = collate_leader_frames([dataset[int(i)] for i in indices])
    device = torch.device('cuda:0')
    results = {}
    initial = None
    common_hash = None
    for mode in ('lidar_finetune', 'mma_only'):
        torch.manual_seed(20)
        torch.cuda.manual_seed_all(20)
        model, checkpoint = load_pretrained_model(args.l0, True, device, fusion_mode=mode)
        center = torch.as_tensor(checkpoint['center_t'], device=device, dtype=torch.float32)
        del checkpoint
        observed = state_dict_sha256({k: v for k, v in model.state_dict().items()
                                      if not k.startswith('magic_fusion.')})
        if common_hash is None:
            common_hash = observed
        assert observed == common_hash
        assert not any('attention.' in key for key in model.state_dict())
        model.eval()
        with torch.no_grad():
            _, _, details = batch_forward(model, batch, center, device, True)
        if initial is None:
            initial_coords, initial = ordered(details)
        coordinates, predictions = ordered(details)
        assert np.array_equal(initial_coords, coordinates)
        assert torch.equal(initial, predictions)
        if mode == 'mma_only':
            model.eval()
            sparse = ME.SparseTensor(features=batch['feats'].to(device), coordinates=batch['coords'].to(device))
            with torch.no_grad():
                encoded, stages = model.encoder(sparse, return_stages=True)
                merged = model.magic_fusion(encoded.F, torch.zeros_like(encoded.F[:, :3]), encoded.C,
                                             encoded.tensor_stride, None, None, None, None, None, stages=stages)
            assert torch.equal(merged, encoded.F)
        optimizer = torch.optim.Adam(model_parameter_groups(model))
        losses = []
        gradients = {}
        for update in range(2):
            configure_training_mode(model)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = batch_forward(model, batch, center, device, True)
            assert torch.isfinite(loss)
            loss.backward()
            for group in optimizer.param_groups:
                total = sum(float(p.grad.detach().abs().sum()) for p in group['params'] if p.grad is not None)
                if group['name'] != 'fusion' or mode == 'mma_only':
                    assert total > 0
                gradients[group['name']] = total
            if update == 1 and mode == 'mma_only':
                assert float(model.magic_fusion.mma.eq4_f1[0].conv.kernel.grad.abs().sum()) > 0
            losses.append(float(loss.detach().cpu()))
            optimizer.step()
            center = _move_training_state(model, optimizer, center, torch.device('cpu'))
            center = _move_training_state(model, optimizer, center, device)
        model.eval()
        with torch.no_grad():
            _, _, a = batch_forward(model, batch, center, device, True)
            _, _, repeated = batch_forward(model, batch, center, device, True)
            changed = dict(batch)
            for key in ('sam_features', 'intrinsics', 'camera_from_lidar', 'image_bounds', 'image_valid_mask'):
                changed.pop(key, None)
            _, _, b = batch_forward(model, changed, center, device, True)
        ca, pa = ordered(a)
        cr, pr = ordered(repeated)
        cb, pb = ordered(b)
        assert np.array_equal(ca, cr) and np.array_equal(ca, cb)
        differences = {'mode': mode, 'unchanged_repeat_max_abs': float((pa - pr).abs().max()),
                       'image_removed_max_abs': float((pa - pb).abs().max()),
                       'unchanged_repeat_equal': torch.equal(pa, pr), 'image_removed_equal': torch.equal(pa, pb)}
        print(json.dumps(differences), flush=True)
        assert torch.equal(pa, pb)
        results[mode] = {'losses': losses, 'gradient_abs_sum': gradients,
                         'common_initial_state_sha256': common_hash,
                         'zero_residual_prediction_equal': True, 'no_image_input_after_updates': True,
                         'repeated_forward_diagnostics': differences}
        del optimizer, model, a, b, repeated, details
        torch.cuda.empty_cache()
    torch.manual_seed(20)
    original, checkpoint = load_pretrained_model(args.l0, False, device)
    original.eval()
    center = torch.as_tensor(checkpoint['center_t'], device=device, dtype=torch.float32)
    with torch.no_grad():
        _, _, details = batch_forward(original, batch, center, device, False)
    coordinates, predictions = ordered(details)
    assert np.array_equal(initial_coords, coordinates)
    assert torch.equal(initial, predictions)
    results['official_l0_prediction_equal'] = True
    results['frame_keys'] = batch['frame_keys']
    results['gpu_peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
    safe_json_dump(args.report, results)
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--sam', type=Path, required=True)
    parser.add_argument('--l0', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    check(parser.parse_args())

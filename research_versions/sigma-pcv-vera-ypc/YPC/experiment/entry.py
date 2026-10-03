import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / 'source'))
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from ypc_head import YPCHead, best_right_yaw, rz_degrees
from models.far_covolume_standard import CoVolumeStandardFAR
from models.sigma_pcv import enable
from tools.sigma_runtime import network_row, sc2, cpu, budget
from tools.far_runtime import make_dataset, observation, verify_sources
from tools.far_covolume_runtime import visible_pool, project
from tools.magic_nclt_runtime import load_pretrained_model, sha256_file, state_dict_sha256, safe_json_dump


def report(store, **value):
    safe_json_dump(store / 'progress.json', dict(time=time.time(), **value))
    print(json.dumps(value, allow_nan=False), flush=True)


def guard(config):
    forbidden = str(Path(config['parent_config']['gt_root']).resolve())
    def deny(event, args):
        if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)) and forbidden in os.path.abspath(os.fsdecode(args[0])):
            raise PermissionError('development truth forbidden in YPC worker')
    sys.addaudithook(deny)


def setup(config):
    assert sys.prefix == '/home/zhangcy/.conda/envs/leader' and os.getuid() == 1003
    guard(config)
    torch.set_num_threads(1)
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    torch.cuda.set_per_process_memory_fraction(config['cuda_allocator_fraction'])
    device = torch.device('cuda:0')
    verify_sources(config['parent_config'])
    leader, saved = load_pretrained_model(config['parent_config']['base_checkpoint'], False, device)
    center = torch.as_tensor(saved['center_t'], device=device, dtype=torch.float32).reshape(3)
    del saved
    assert sha256_file(config['far_checkpoint']) == config['far_checkpoint_sha256']
    far = CoVolumeStandardFAR(**config['parent_config']['far_covolume']).to(device)
    saved = torch.load(config['far_checkpoint'], map_location='cpu')
    assert saved['config'] == config['parent_config']
    far.load_state_dict(saved['model'], strict=True)
    enable(far)
    leader.requires_grad_(False).eval()
    far.requires_grad_(False).eval()
    return leader, far, center, device


def frozen(leader, far):
    return dict(leader=state_dict_sha256(leader.state_dict()), far=state_dict_sha256(far.state_dict()))


def image(path, device):
    with Image.open(path) as im:
        return torch.from_numpy(np.asarray(im.convert('RGB')).copy()).permute(2, 0, 1).float().to(device) / 255


def raw_visible(sample, limit):
    raw = np.asarray(sample['points'], np.float32)
    uv, depth, cell, valid = project(raw, sample['camera_from_lidar'].numpy())
    ids = np.flatnonzero(valid)
    if not len(ids):
        return torch.from_numpy(raw[:0]), torch.from_numpy(uv[:0].astype(np.float32)), ids
    ids = ids[np.lexsort((ids, depth[ids], cell[ids]))]
    ids = ids[np.r_[True, cell[ids][1:] != cell[ids][:-1]]]
    ids = ids[np.lexsort((ids, -np.square(raw[ids, :2]).sum(-1)))][:limit]
    return torch.from_numpy(raw[ids].copy()), torch.from_numpy(uv[ids].astype(np.float32)), ids


def original(sample, index, config, leader, far, center, device, truth=False):
    assert np.array_equal(sample['T_corr'], np.eye(4, dtype=np.float32))
    row = network_row(sample, leader, device, center, truth=truth)
    n = len(row['source'])
    ids = row['reliability'].topk(max(min(50, n), int(.5 * n))).indices
    source, mu = row['source'][ids], row['mu'][ids]
    initial = sc2(config, dict(row, selected=ids), row['mu'], index)
    warm, _ = far.warm_start(initial, source, mu)
    return row, ids, initial, warm


def queries(far, raw, pose, grid):
    rotations = pose[:3, :3][None] @ rz_degrees(grid)
    positions = torch.einsum('kij,nj->nki', rotations, raw) + pose[:3, 3]
    flat = positions.reshape(-1, 3)
    return torch.cat([far.field(flat[i:i+1024]) for i in range(0, len(flat), 1024)]).reshape(len(raw), len(grid), 32)


def checks(config, store):
    leader, far, center, device = setup(config)
    before = frozen(leader, far)
    angle = torch.tensor([.3], device=device)
    rotation = rz_degrees(angle)[0]
    recovered = best_right_yaw(torch.eye(3, device=device), rotation)
    assert abs(float(recovered) - .3) < 1e-5
    head = YPCHead().to(device)
    rgb = torch.rand(3, 64, 80, device=device)
    pixels = torch.rand(16, 2, device=device) * torch.tensor([79., 63.], device=device)
    observed = F.normalize(torch.rand(16, 32, device=device), dim=-1)
    desc = head.descriptor(rgb, pixels, observed)
    assert torch.allclose(desc, observed, atol=1e-6)
    field = F.normalize(torch.rand(16, 21, 32, device=device), dim=-1)
    raw = torch.rand(16, 3, device=device) * 30
    posterior = head.posterior(desc, field, raw, head.grid)
    F.cross_entropy(posterior['logits'][None], torch.tensor([12], device=device)).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters())
    assert head.image_adapter.weight.grad.abs().sum() > 0
    assert head.field_adapter.weight.grad.abs().sum() > 0
    assert head.stem[-1].weight.grad.abs().sum() > 0
    assert head.weight[-1].weight.grad.abs().sum() > 0
    features = torch.randn(12, 512, device=device)
    permutation = torch.randperm(12, device=device)
    discrepancy = float((leader.decoder(features[permutation]) - leader.decoder(features)[permutation]).abs().max())
    assert discrepancy < 1e-6
    baseline = [json.loads(line) for line in Path(config['baseline_predictions']).read_text().splitlines()]
    assert sha256_file(config['baseline_predictions']) == config['baseline_prediction_sha256']
    data = make_dataset(config['parent_config'], 'dev')
    parity = []
    with torch.no_grad():
        for index in (0, 300, 600, 900):
            budget(store)
            sample = data[index]
            assert 'T_WB' not in sample
            row, ids, initial, warm = original(sample, index, config, leader, far, center, device)
            points, pixels, visible, structure = [v.to(device) for v in visible_pool(sample, row['source'][ids], 2048)]
            observed = observation(far, sample, pixels, image(data.records[index].image_path, device), device)
            pose, _, _ = far(points, observed, initial, row['source'][ids], row['mu'][ids], structure, visible)
            pose[:3, 3] += center
            difference = float((pose.double() - torch.tensor(baseline[index]['T_WB_pred'], device=device)).abs().max())
            assert difference < 1e-5, difference
            raw, pix, raw_ids = raw_visible(sample, config['ypc']['point_max'])
            assert np.array_equal(raw.numpy(), sample['points'][raw_ids])
            parity.append(dict(index=index, max=difference, raw_visible_points=len(raw), T_corr_identity=True))
    assert before == frozen(leader, far)
    safe_json_dump(store / 'checks.json', dict(parity=parity, decoder_permutation_error=discrepancy, frozen=before,
        frontend_limit='A cyclic feature reindex passed to pointwise MMRegressor commutes with that reindex; it alone does not introduce a phase-dependent prediction for the same point. No frontend training claimed.'))
    report(store, phase='checks_complete', parity=parity)


def export(config, store):
    leader, far, center, device = setup(config)
    before = frozen(leader, far)
    data = make_dataset(config['parent_config'], 'train')
    directory = store / 'train-cache'
    directory.mkdir()
    records = []
    started = time.monotonic()
    with torch.no_grad():
        for index in range(len(data)):
            budget(store)
            sample = data[index]
            row, ids, initial, warm = original(sample, index, config, leader, far, center, device, truth=True)
            raw, pixels, raw_ids = raw_visible(sample, config['ypc']['point_max'])
            raw, pixels = raw.to(device), pixels.to(device)
            angle = float(best_right_yaw(warm[:3, :3], row['pose_truth'][:3, :3]))
            rgb = image(data.records[index].image_path, device)
            observed = observation(far, sample, pixels, rgb, device) if len(raw) else raw.new_empty((0, 32))
            field = queries(far, raw, warm, torch.linspace(-4, 4, 21, device=device)) if len(raw) else raw.new_empty((0, 21, 32))
            cached = dict(frame_key=sample['frame_key'], index=index, raw=raw, pixels=pixels, raw_ids=raw_ids, observed=observed,
                field=field.half(), warm=warm, angle_deg=angle, image_path=str(data.records[index].image_path), image_sha256=sha256_file(data.records[index].image_path))
            path = directory / ('%04d.pt' % index)
            torch.save(cpu(cached), path)
            records.append(dict(index=index, frame_key=sample['frame_key'], file=path.name, points=len(raw), angle_deg=angle, in_grid=abs(angle) <= 4, sha256=sha256_file(path), bytes=path.stat().st_size))
            assert sum(v['bytes'] for v in records) < 4 * 1024**3
            if (index + 1) % 20 == 0 or index == 0:
                report(store, phase='export_train', frames=index+1, total=len(data), seconds=time.monotonic()-started, out_of_grid=sum(not v['in_grid'] for v in records))
    assert before == frozen(leader, far)
    safe_json_dump(store / 'train-freeze.json', dict(records=records, frame_keys=list(data.keys), manifest_hashes=data.manifest_hashes, frozen=before, config_sha256=config['self_sha256']))
    report(store, phase='export_complete', frames=len(records), points=sum(v['points'] for v in records))


def cached(store, record):
    path = store / 'train-cache' / record['file']
    assert sha256_file(path) == record['sha256']
    row = torch.load(path, map_location='cpu')
    assert row['frame_key'] == record['frame_key']
    assert sha256_file(row['image_path']) == row['image_sha256']
    return row


def evaluate(head, config, store, records, device, wrong_image=False):
    result = []
    head.eval()
    with torch.no_grad():
        for offset, record in enumerate(records):
            budget(store)
            row = cached(store, record)
            if not len(row['raw']):
                continue
            pixels, observed, field, raw = [row[name].to(device).float() for name in ('pixels', 'observed', 'field', 'raw')]
            if wrong_image:
                other = cached(store, records[(offset + 1) % len(records)])
                observed = observed.roll(max(1, len(observed)//2), 0)
                rgb = image(other['image_path'], device)
            else:
                rgb = image(row['image_path'], device)
            output = head.posterior(head.descriptor(rgb, pixels, observed), field, raw, head.grid)
            target = row['angle_deg']
            label = int((head.grid - target).abs().argmin())
            nearest_rank = int((output['logits'] > output['logits'][label]).sum()) + 1
            result.append(dict(frame_key=row['frame_key'], angle_deg=target, in_grid=abs(target)<=4, mean_deg=float(output['mean']), variance_deg2=float(output['variance']), argmax=int(output['logits'].argmax()), label=label,
                rank=nearest_rank, ce=float(F.cross_entropy(output['logits'][None], torch.tensor([label], device=device))) if abs(target)<=4 else None))
    target = np.array([v['angle_deg'] for v in result])
    mean = np.array([v['mean_deg'] for v in result])
    inside = [v for v in result if v['in_grid']]
    return dict(frames=len(result), in_grid_frames=len(inside), outside_grid_frames=len(result)-len(inside),
        zero_rmse_deg=float(np.sqrt(np.mean(target**2))), measured_rmse_deg=float(np.sqrt(np.mean((target-mean)**2))),
        top1_in_grid=float(np.mean([v['argmax']==v['label'] for v in inside])), mean_rank_in_grid=float(np.mean([v['rank'] for v in inside])),
        ce_in_grid=float(np.mean([v['ce'] for v in inside])), rows=result, wrong_image=wrong_image)


def train(config, store):
    guard(config)
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(config['cuda_allocator_fraction'])
    torch.manual_seed(config['seed'])
    random = np.random.RandomState(config['seed'])
    device = torch.device('cuda:0')
    freeze = json.loads((store / 'train-freeze.json').read_text())
    assert freeze['config_sha256'] == config['self_sha256']
    fit = [v for v in freeze['records'] if v['index']%5 != 0 and v['points']]
    calibration = [v for v in freeze['records'] if v['index']%5 == 0 and v['points']]
    head = YPCHead().to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=config['ypc']['learning_rate'])
    initial = evaluate(head, config, store, calibration, device)
    safe_json_dump(store / 'initial-phase-diagnostic.json', initial)
    history, updates = [], 0
    started = time.monotonic()
    for epoch in range(1, config['ypc']['epochs']+1):
        head.train()
        total = []
        for offset in random.permutation(len(fit)):
            budget(store)
            record = fit[offset]
            if not record['in_grid']:
                continue
            row = cached(store, record)
            pixels, observed, field, raw = [row[name].to(device).float() for name in ('pixels', 'observed', 'field', 'raw')]
            descriptor = head.descriptor(image(row['image_path'], device), pixels, observed)
            output = head.posterior(descriptor, field, raw, head.grid)
            label = (head.grid-row['angle_deg']).abs().argmin().reshape(1)
            loss = F.cross_entropy(output['logits'][None], label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(head.parameters(), 5.)
            assert torch.isfinite(loss) and torch.isfinite(gradient)
            optimizer.step()
            updates += 1
            total.append(float(loss.detach()))
            if updates % 100 == 0 or updates == 1:
                report(store, phase='train_coarse', epoch=epoch, updates=updates, loss=float(loss.detach()), seconds=time.monotonic()-started)
        diagnostic = evaluate(head, config, store, calibration, device)
        summary = {k:v for k,v in diagnostic.items() if k != 'rows'}
        checkpoint = store / ('ypc-coarse-e%02d.pt' % epoch)
        torch.save(dict(head=cpu(head.state_dict()), optimizer=cpu(optimizer.state_dict()), epoch=epoch, updates=updates, config_sha256=config['self_sha256']), checkpoint)
        history.append(dict(epoch=epoch, updates=updates, mean_ce=float(np.mean(total)), checkpoint=checkpoint.name, checkpoint_sha256=sha256_file(checkpoint), calibration=summary))
        safe_json_dump(store / 'training-history.json', history)
        safe_json_dump(store / ('calibration-e%02d.json' % epoch), diagnostic)
        report(store, phase='coarse_epoch_complete', **history[-1])
    wrong = evaluate(head, config, store, calibration, device, wrong_image=True)
    safe_json_dump(store / 'wrong-image-phase-diagnostic.json', wrong)
    measured = diagnostic['measured_rmse_deg']
    passed = measured <= .8 * diagnostic['zero_rmse_deg'] and diagnostic['top1_in_grid'] >= 3/21 and measured < wrong['measured_rmse_deg']
    residual_square = np.array([(v['angle_deg']-v['mean_deg'])**2/v['variance_deg2'] for v in diagnostic['rows']])
    quantile_index = min(len(residual_square)-1, math.ceil((len(residual_square)+1)*.95)-1)
    scale = max(1., float(np.sort(residual_square)[quantile_index]/3.841458820694124))
    safe_json_dump(store / 'phase-gate.json', dict(passed=passed, gate=config['ypc']['gate'], measured_rmse_deg=measured, zero_rmse_deg=diagnostic['zero_rmse_deg'],
        wrong_image_rmse_deg=wrong['measured_rmse_deg'], top1_in_grid=diagnostic['top1_in_grid'], chance_top1=1/21, variance_scale_squared=scale,
        actual_updates=updates, checkpoint_sha256=history[-1]['checkpoint_sha256'], fit_frames=len(fit), calibration_frames=len(calibration),
        outside_grid_fit=sum(not v['in_grid'] for v in fit), outside_grid_calibration=diagnostic['outside_grid_frames'],
        result_scope='phase measurement gate only; no YPC development pose result and no frontend-conditioned training'))
    report(store, phase='phase_gate_complete', passed=passed, measured_rmse_deg=measured, zero_rmse_deg=diagnostic['zero_rmse_deg'], top1_in_grid=diagnostic['top1_in_grid'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['checks', 'export', 'train'], required=True)
    args = parser.parse_args()
    config = json.loads((root / 'config.json').read_text())
    config['self_sha256'] = sha256_file(root / 'config.json')
    assert all(sha256_file(root/name) == digest for name,digest in config['execution_sha256'].items())
    globals()[args.phase](config, Path(config['store']))


if __name__ == '__main__':
    main()

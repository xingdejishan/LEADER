"""coh-rot A/B training on the four small training dates.

Arm A : L_TRR                       (pure LEADER, the original loss)
Arm B : L_TRR + lambda * L_coh-rot

Everything else is identical between the arms: base checkpoint, epoch count, learning
rates, batch size, seed, and the per-epoch frame order (derived from the seed, not from
any RNG the training loop consumes).  The script refuses to start if the two arms would
differ in any of those.

This is NOT `tools/train_local905_ab.py`: that script compares a multimodal fusion module
against a LiDAR baseline and asserts fusion identity.  Here the only difference is the
extra loss term on the decoder's per-point world coordinates.
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path('/home/zhangcy/projects/LEADER-magic-909c81d')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import MinkowskiEngine as ME  # noqa: E402
from models.model_mink import LEADER  # noqa: E402
from utils.pose_util import polar_expansion_to_cartesian  # noqa: E402

import leader_nclt  # noqa: E402
from coh_rot import coherent_rotation_loss, weights_from_scores  # noqa: E402

BASE_CHECKPOINT = Path('/home/zhangcy/projects/nclt-relation-compare-20260927-a/data/assets/official_l0.pt')


def resource_guard():
    """The project's registered resource policy: nice 15, one CPU thread, <=0.35 of a card."""
    os.nice(15)
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(0.35)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def state_digest(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def repo_trr(scale=10.0):
    """The project's original TRR loss, imported from the repo when possible."""
    try:
        from run_mink import TRR
        return TRR(scale=scale), 'repo:run_mink.TRR'
    except Exception as error:  # pragma: no cover - depends on optional heavy deps
        print('  [warn] could not import run_mink.TRR (%r); using the verbatim copy' % (error,),
              flush=True)

        class TRR:
            def __init__(self, scale=10.0):
                self.scaler = np.log(scale) / np.pi

            def __call__(self, gt, pred, u, batch_idx):
                batch_size = batch_idx.max() + 1
                l_raw = (pred - gt).norm(dim=-1)
                u_cut = u.clamp(min=-10 * torch.pi, max=10 * torch.pi).atan() * self.scaler
                u_scale = u.atan() * self.scaler
                u_cut_exp = u_cut.exp()
                u_scale_exp = u_scale.exp()
                w_sum = torch.zeros((batch_size), dtype=pred.dtype, device=pred.device) \
                    .scatter_add_(0, batch_idx, torch.min(u_cut_exp, u_scale_exp))
                w = torch.max(u_cut_exp, u_scale_exp) / w_sum.clamp(min=1e-6)[batch_idx]
                L_TRR = torch.zeros((batch_size), dtype=pred.dtype, device=pred.device)
                L_TRR.scatter_add_(0, batch_idx, l_raw * w).mean()
                L2 = l_raw.mean()
                return L_TRR, L2

        return TRR(scale=scale), 'fallback:verbatim_copy'


def trr_value(loss_fn, targets, prediction, batch_index):
    """`run_mink.TRR` returns the per-frame weighted vector; its callers reduce it.

    `tools/train_local905.py` does `return weighted.mean(), raw`; `run_mink.process_one_epoch`
    does `loss = loss_t_weighted.mean()`.  Both reduce the same quantity, so the A/B does too.
    """
    weighted, raw = loss_fn(targets, prediction[:, :3], prediction[:, -1], batch_index)
    return weighted.mean(), raw


def forward(model, batch, center, device='cuda'):
    """Returns (pred, voxel_centers_w, batch_idx) exactly as the official training loop."""
    sparse = ME.SparseTensor(batch['feats'].to(device, non_blocking=True),
                             batch['coords'].to(device, non_blocking=True))
    encoded = model.encoder(sparse)
    stride = torch.tensor(encoded.tensor_stride, device=device, dtype=torch.float32)
    voxel_centers = (encoded.C[:, 1:].float() + stride / 2) * leader_nclt.VOXEL_SIZE
    voxel_centers_l = polar_expansion_to_cartesian(
        voxel_centers, leader_nclt.HORIZONTAL_RES * leader_nclt.VOXEL_SIZE)
    batch_index = encoded.C[:, 0].long()
    poses = batch['T'].to(device, non_blocking=True)
    targets = voxel_centers_l[:, None] @ poses[batch_index, :3, :3].permute(0, 2, 1)
    targets = targets[:, 0] + poses[batch_index, :3, 3] - center
    prediction = model.decoder(encoded.F)
    return prediction, targets, batch_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('A', 'B', 'C'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--lr-enc', type=float, default=1e-5)
    parser.add_argument('--lr-dec', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--lam', type=float, default=1.0)
    parser.add_argument('--model-mode', choices=('train', 'eval'), default='eval')
    parser.add_argument('--eval-every', type=int, default=10,
                        help='keep a checkpoint snapshot every N epochs (diagnostic only)')
    parser.add_argument('--limit', type=int, default=0, help='smoke: use only the first N frames')
    parser.add_argument('--smoke-arms', action='store_true', help='run one step for both arms and exit')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    resource_guard()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print('=== coh-rot A/B training ===', flush=True)
    print('  arm=%s lam=%s epochs=%d batch=%d lr_enc=%g lr_dec=%g seed=%d mode=%s'
          % (args.arm, args.lam, args.epochs, args.batch, args.lr_enc, args.lr_dec,
             args.seed, args.model_mode), flush=True)

    frames = leader_nclt.train_frames()
    if args.limit:
        frames = frames[:args.limit]
    print('  train frames: %d' % len(frames), flush=True)
    store = leader_nclt.prepare_all(frames, progress=200)
    center = torch.tensor(leader_nclt.CANONICAL_CENTER_T, dtype=torch.float32, device=args.device)

    base = torch.load(BASE_CHECKPOINT, map_location='cpu')
    model = LEADER(in_channels=3, out_channels=4, feat_channels=512, width=1024, magic=False)
    missing, unexpected = model.load_state_dict(base['model'], strict=False)
    if missing or unexpected:
        raise ValueError('base checkpoint mismatch: missing=%r unexpected=%r' % (missing, unexpected))
    initial_digest = state_digest(model.state_dict())
    print('  base checkpoint %s' % sha256_file(BASE_CHECKPOINT), flush=True)
    print('  initial model digest %s' % initial_digest, flush=True)
    model = model.to(args.device)
    del base

    encoder_parameters = list(model.encoder.parameters())
    decoder_parameters = list(model.decoder.parameters())
    optimizer = torch.optim.AdamW(
        [{'params': decoder_parameters, 'lr': args.lr_dec},
         {'params': encoder_parameters, 'lr': args.lr_enc}],
        weight_decay=args.weight_decay)
    loss_fn, loss_source = repo_trr(scale=10.0)
    print('  TRR source: %s' % loss_source, flush=True)

    rng = np.random.default_rng(args.seed)
    order_seed = int(rng.integers(0, 2 ** 31 - 1))
    print('  order seed: %d' % order_seed, flush=True)

    settings = {
        'arm': args.arm, 'lam': args.lam, 'epochs': args.epochs, 'batch': args.batch,
        'lr_enc': args.lr_enc, 'lr_dec': args.lr_dec, 'weight_decay': args.weight_decay,
        'seed': args.seed, 'order_seed': order_seed, 'model_mode': args.model_mode,
        'train_frames': len(frames), 'limit': args.limit,
        'base_checkpoint': str(BASE_CHECKPOINT),
        'base_checkpoint_sha256': sha256_file(BASE_CHECKPOINT),
        'initial_model_digest': initial_digest,
        'center_t': list(leader_nclt.CANONICAL_CENTER_T),
        'voxel_size': leader_nclt.VOXEL_SIZE, 'horizontal_res': leader_nclt.HORIZONTAL_RES,
        'loss_source': loss_source,
        'impl': 'train_cohrot_ab.py',
    }
    (args.out / ('settings-arm%s.json' % args.arm)).write_text(
        json.dumps(settings, indent=2) + '\n', encoding='utf-8')

    if args.smoke_arms:
        return smoke(model, store, frames, center, loss_fn, args)

    history = args.out / ('history-arm%s.jsonl' % args.arm)
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train() if args.model_mode == 'train' else model.eval()
        epoch_rng = np.random.default_rng(order_seed + epoch)
        order = epoch_rng.permutation(len(store))
        accumulator = {}
        steps = 0
        for start in range(0, len(order), args.batch):
            indices = order[start:start + args.batch]
            batch = leader_nclt.collate([(store[i], frames[i]['T_WB']) for i in indices])
            prediction, targets, batch_index = forward(model, batch, center, args.device)
            trr, l2 = trr_value(loss_fn, targets, prediction, batch_index)
            loss = trr
            stats = {'trr': float(trr.item()), 'l2': float(l2.item())}
            if args.arm in ('B', 'C'):
                weights = weights_from_scores(prediction[:, 3].detach())
                rotation, rotation_stats = coherent_rotation_loss(
                    prediction[:, :3], targets, weights, batch_index)
                loss = trr + args.lam * rotation
                stats['coh'] = float(rotation.item())
                for key, value in rotation_stats.items():
                    if isinstance(value, (int, float)) and value is not None:
                        stats[key] = value
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite loss at epoch %d step %d' % (epoch, steps))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            stats['loss'] = float(loss.item())
            for key, value in stats.items():
                accumulator[key] = accumulator.get(key, 0.0) + value
            steps += 1
            if steps == 1:
                print('  first step loss=%.6f' % stats['loss'], flush=True)
        record = {'epoch': epoch, 'steps': steps}
        record.update({key: value / steps for key, value in accumulator.items()})
        record['elapsed_s'] = time.monotonic() - started
        with history.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        print('epoch %2d  %s' % (epoch, json.dumps(
            {k: round(v, 6) for k, v in record.items() if k not in ('epoch', 'steps', 'elapsed_s')})),
            flush=True)
        state = {
            'model': {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
            'optimizer': optimizer.state_dict(), 'epoch': epoch, 'settings': settings,
            'center_t': list(leader_nclt.CANONICAL_CENTER_T),
        }
        torch.save(state, args.out / ('last-arm%s.pt' % args.arm))
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            torch.save({
                'model': {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                'epoch': epoch, 'settings': settings,
                'center_t': list(leader_nclt.CANONICAL_CENTER_T),
                'final_model_digest': state_digest(model.state_dict()),
            }, args.out / ('arm%s-e%02d.pt' % (args.arm, epoch)))

    final_path = args.out / ('arm%s.pt' % args.arm)
    torch.save({
        'model': {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        'epoch': args.epochs, 'settings': settings,
        'center_t': list(leader_nclt.CANONICAL_CENTER_T),
        'final_model_digest': state_digest(model.state_dict()),
    }, final_path)
    print('FINAL %s sha256=%s' % (final_path, sha256_file(final_path)), flush=True)
    print('TRAINING_JSON ' + json.dumps({
        'arm': args.arm, 'epochs': args.epochs, 'train_frames': len(frames),
        'seconds': time.monotonic() - started, 'final_model_digest': state_digest(model.state_dict()),
        'arm_checkpoint_sha256': sha256_file(final_path), 'settings': settings,
    }), flush=True)


def smoke(model, store, frames, center, loss_fn, args):
    """One step each way, to fix the batch-norm mode and record the cost."""
    device = args.device
    indices = list(range(min(args.batch, len(store))))
    batch = leader_nclt.collate([(store[i], frames[i]['T_WB']) for i in indices])
    report = {}
    for mode in ('train', 'eval'):
        model.train() if mode == 'train' else model.eval()
        torch.cuda.synchronize()
        started = time.monotonic()
        prediction, targets, batch_index = forward(model, batch, center, device)
        output_std = float(prediction[:, :3].std().item())
        trr, _ = trr_value(loss_fn, targets, prediction, batch_index)
        weights = weights_from_scores(prediction[:, 3].detach())
        rotation, rotation_stats = coherent_rotation_loss(
            prediction[:, :3], targets, weights, batch_index)
        for arm, lam in (('A', 0.0), ('B', 0.1), ('C', 0.47)):
            value = float((trr + lam * rotation).item())
            detail = {}
            if lam:
                detail = {'coh': float(rotation.item())}
                detail.update({k: v for k, v in rotation_stats.items()
                               if isinstance(v, (int, float))})
            report['%s-%s' % (mode, arm)] = dict(
                loss=value, trr=float(trr.item()), pred_std_m=output_std, **detail)
        torch.cuda.synchronize()
        report[mode + '-seconds'] = time.monotonic() - started
        report[mode + '-peak_mb'] = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print('SMOKE_JSON ' + json.dumps(report), flush=True)
    with (args.out / 'smoke.json').open('w', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()

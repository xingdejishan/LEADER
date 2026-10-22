"""Evaluate a LEADER checkpoint on the official 300-per-date NCLT test split.

The pose readout is the project's own: top-50 % of the decoder's per-point confidence,
SC2-PCR (`models.sc2pcr.Matcher`) on the selected local/世界 correspondence pairs, then
`utils.full_pool_robust_v1.full_pool_refine` at 1.2 m / 0.6 m.  That is exactly
`run_mink.process_one_epoch`'s validation path, at the original batch of one frame.

Scoring goes through the frozen reference-policy helper and is hash-checked first:

    helper      c1f5ca48759acd79d2bfc1bea1e6671b059d686da279c97682e939986058a4b1
    policy      3e36356edb50284a3cde017188b3283bd01fb41bfe41fb1ddd201b01a0114a4a
    replacement 07416ebc5791d631ffb82db10cde577d209878a6cfe8a781810537e23c9aa2f4

Run the same checkpoint twice with different --pass-index values to measure the noise
floor of the readout itself (Minkowski forward passes are not bit-reproducible on CUDA).
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = Path('/home/zhangcy/projects/LEADER-magic-909c81d')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'policy'))

import MinkowskiEngine as ME  # noqa: E402
from models.model_mink import LEADER  # noqa: E402
from models.sc2pcr import Matcher  # noqa: E402
from utils.full_pool_robust_v1 import full_pool_refine  # noqa: E402
from utils.pose_util import polar_expansion_to_cartesian  # noqa: E402

import leader_nclt  # noqa: E402
from coh_rot import kabsch_forward, linear_omega, weights_from_scores  # noqa: E402

POLICY_SHA256 = '3e36356edb50284a3cde017188b3283bd01fb41bfe41fb1ddd201b01a0114a4a'
HELPER_SHA256 = 'c1f5ca48759acd79d2bfc1bea1e6671b059d686da279c97682e939986058a4b1'
REPLACEMENT_SHA256 = '07416ebc5791d631ffb82db10cde577d209878a6cfe8a781810537e23c9aa2f4'


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


def verify_policy():
    details = {}
    for name, expected in (('nclt_reference_policy.py', HELPER_SHA256),
                           ('nclt-reference-policy.json', POLICY_SHA256),
                           ('nclt-2012-02-12-suspect255-odometry.npz', REPLACEMENT_SHA256)):
        path = HERE / 'policy' / name
        got = sha256_file(path)
        if got != expected:
            raise ValueError('reference artefact %s has SHA256 %s, expected %s' % (name, got, expected))
        details[name] = got
    return details


@torch.no_grad()
def predict_frame(model, coords, feats, center, device, matcher, t_wb):
    """One frame, exactly as `run_mink.process_one_epoch` does it for validation."""
    # a single-frame batch: the encoder requires the leading batch column
    batched = torch.cat([torch.zeros(coords.shape[0], 1, dtype=torch.int32), coords], dim=1)
    sparse = ME.SparseTensor(feats.to(device), batched.to(device))
    encoded = model.encoder(sparse)
    stride = torch.tensor(encoded.tensor_stride, device=device, dtype=torch.float32)
    voxel_centers = (encoded.C[:, 1:].float() + stride / 2) * leader_nclt.VOXEL_SIZE
    centers_local = polar_expansion_to_cartesian(
        voxel_centers, leader_nclt.HORIZONTAL_RES * leader_nclt.VOXEL_SIZE)
    prediction = model.decoder(encoded.F)
    # ground-truth world coordinates of exactly the decoded voxel centres
    pose = torch.from_numpy(np.asarray(t_wb, dtype=np.float32)).to(device)
    centers_w = centers_local @ pose[:3, :3].T + pose[:3, 3] - center

    scores = prediction[:, 3]
    count = scores.shape[0]
    k = max(min(50, count), int(0.5 * count))
    _, indices = scores.topk(k=k)
    # The pose readout runs on the CPU on purpose.  `models.sc2pcr.Matcher` and
    # `utils.full_pool_robust_v1.full_pool_refine` both call `torch.det` on 3x3 matrices,
    # and on CUDA that JIT-compiles a reduction through nvrtc, which this deployment
    # (torch 1.12.0+cu116, sm_89) rejects.  The matcher returns CPU tensors anyway, and
    # both arms go through exactly the same readout, so this is a level playing field.
    local = centers_local[indices].float().cpu()
    predicted = prediction[indices, :3].float().cpu()
    transform = matcher.estimator(local[None], predicted[None])[0]
    transform = full_pool_refine(transform, local, predicted)
    transform = transform.clone()
    transform[:3, 3] += center.cpu()
    return transform, prediction, centers_w


def diagnostics(prediction, centers_w, weights):
    """Rotation-bias readout: how much the whole predicted field still needs rotating."""
    weight = weights / weights.sum().clamp_min(1e-12)
    x = prediction - (weight[:, None] * prediction).sum(0, keepdim=True)
    y = centers_w - (weight[:, None] * centers_w).sum(0, keepdim=True)
    q, _ = kabsch_forward(x, y, weight)
    cos_q = ((q.diagonal().sum() - 1) / 2).clamp(-1, 1)
    angle = math.degrees(torch.acos(cos_q).item())
    omega = math.degrees(torch.linalg.norm(linear_omega(x, y, weight)).item())
    return angle, omega


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pass-index', type=int, default=1)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    resource_guard()
    artefacts = verify_policy()
    print('=== reference artefacts verified ===', flush=True)
    for name, digest in artefacts.items():
        print('  %-46s %s' % (name, digest), flush=True)

    from nclt_reference_policy import score_poses  # noqa: E402

    frames = leader_nclt.dev_frames()
    if args.limit:
        frames = frames[:args.limit]
    print('  test frames: %d over %d dates' % (len(frames), len(set(f['date'] for f in frames))), flush=True)

    state = torch.load(args.checkpoint, map_location='cpu')
    model = LEADER(in_channels=3, out_channels=4, feat_channels=512, width=1024, magic=False)
    model.load_state_dict(state['model'])
    model = model.to(args.device).eval()
    center = torch.tensor(leader_nclt.CANONICAL_CENTER_T, dtype=torch.float32, device=args.device)
    matcher = Matcher(inlier_threshold=2., d_thre=2, num_iterations=10, ratio=.15,
                      nms_radius=.1, max_points=3000, k1=30)
    print('  checkpoint %s sha256=%s' % (args.checkpoint, sha256_file(args.checkpoint)), flush=True)

    store = leader_nclt.prepare_all(frames)

    rows = []
    started = time.monotonic()
    for index, (frame, (coords, feats)) in enumerate(zip(frames, store)):
        transform, prediction, centers_w = predict_frame(
            model, coords, feats, center, args.device, matcher, frame['T_WB'])
        pose = transform.detach().cpu().numpy().astype(np.float64)
        gt_pose = np.asarray(frame['T_WB'], dtype=np.float64)
        mpe = float(np.linalg.norm(pose[:3, 3] - gt_pose[:3, 3]))
        cos = (np.einsum('ij,ij->', pose[:3, :3], gt_pose[:3, :3]) - 1.0) / 2.0
        moe = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        # mechanism readout: how much the whole predicted coordinate field still needs
        # rotating to line up with the geometric target (this is what L_coh-rot penalises)
        weights = weights_from_scores(prediction[:, 3].detach())
        field_angle_deg, field_omega_deg = diagnostics(
            prediction[:, :3].detach(), centers_w.detach(), weights)
        rows.append({
            'frame_key': frame['frame_key'], 'date': frame['date'],
            'timestamp_us': frame['timestamp_us'],
            'MPE_m': mpe, 'MOE_deg': moe,
            'field_rotation_deg': field_angle_deg, 'field_omega_deg': field_omega_deg,
            'voxels': int(prediction.shape[0]),
            'prediction': pose.tolist(),
        })
        if (index + 1) % 100 == 0:
            elapsed = time.monotonic() - started
            print('  %d/%d frames  %.1fs  (%.3f s/frame)  MPE=%.4f MOE=%.4f'
                  % (index + 1, len(frames), elapsed, elapsed / (index + 1),
                     float(np.mean([r['MPE_m'] for r in rows])),
                     float(np.mean([r['MOE_deg'] for r in rows]))), flush=True)

    predictions = np.stack([np.asarray(r['prediction'], dtype=np.float64) for r in rows])
    dates = np.asarray([r['date'] for r in rows])
    stamps = np.asarray([r['timestamp_us'] for r in rows], dtype=np.int64)
    gt_poses = np.stack([np.asarray(f['T_WB'], dtype=np.float64) for f in frames])
    report, position_errors, orientation_errors, replaced = score_poses(
        dates, stamps, predictions, gt_poses,
        policy_path=HERE / 'policy' / 'nclt-reference-policy.json')

    for row, pe, oe in zip(rows, position_errors, orientation_errors):
        row['reference_MPE_m'] = float(pe)
        row['reference_MOE_deg'] = float(oe)

    tag = 'pass%d-%s' % (args.pass_index, args.label)
    frame_path = args.out / ('%s-frames.jsonl' % tag)
    with frame_path.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')

    per_date = {}
    field_rotation = np.asarray([r['field_rotation_deg'] for r in rows], dtype=np.float64)
    for date in sorted(set(dates.tolist())):
        mask = dates == date
        per_date[date] = {
            'frames': int(mask.sum()),
            'MPE_m': float(position_errors[mask].mean()),
            'MOE_deg': float(orientation_errors[mask].mean()),
            'MPE_m_median': float(np.median(position_errors[mask])),
            'MOE_deg_median': float(np.median(orientation_errors[mask])),
            'field_rotation_deg': float(field_rotation[mask].mean()),
        }

    summary = dict(
        label=args.label, pass_index=args.pass_index,
        checkpoint=str(args.checkpoint), checkpoint_sha256=sha256_file(args.checkpoint),
        frames=len(rows), per_date=per_date,
        MPE_m=float(position_errors.mean()), MOE_deg=float(orientation_errors.mean()),
        MPE_m_median=float(np.median(position_errors)),
        MOE_deg_median=float(np.median(orientation_errors)),
        field_rotation_deg=float(field_rotation.mean()),
        field_omega_deg=float(np.mean([r['field_omega_deg'] for r in rows])),
        reference=report,
        artefacts=artefacts,
        frames_file=frame_path.name,
        frames_file_sha256=sha256_file(frame_path),
        process_seconds=time.monotonic() - started,
        peak_allocated_mb=torch.cuda.max_memory_allocated() / (1024 ** 2),
    )
    (args.out / ('%s-report.json' % tag)).write_text(json.dumps(summary, indent=2) + '\n',
                                                     encoding='utf-8')
    print('EVAL_JSON ' + json.dumps({k: v for k, v in summary.items()
                                     if k not in ('per_date', 'artefacts')}), flush=True)
    print('  replaced frames under the reference policy: %d' % int(replaced.sum()), flush=True)


if __name__ == '__main__':
    main()

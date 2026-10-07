import hashlib
import json
from pathlib import Path

import numpy as np


DEFAULT_POLICY = Path(__file__).resolve().parents[1] / 'evaluation' / 'nclt-reference-policy.json'


def apply_reference_policy(dates, timestamps_us, gt_poses, policy_path=DEFAULT_POLICY):
    policy_path = Path(policy_path).resolve()
    policy_bytes = policy_path.read_bytes()
    policy = json.loads(policy_bytes)
    dates = np.asarray(dates, dtype=str)
    timestamps_us = np.asarray(timestamps_us)
    references = np.array(gt_poses, dtype=np.float64, copy=True)
    if timestamps_us.dtype.kind not in 'iu':
        raise ValueError('timestamps_us must contain integer microsecond timestamps')
    if dates.ndim != 1 or timestamps_us.shape != dates.shape or references.shape != (len(dates), 4, 4):
        raise ValueError('dates, timestamps_us and gt_poses must describe the same frames')
    replaced = np.zeros(len(dates), dtype=bool)
    for segment in policy['segments']:
        selected = (dates == segment['date']) & (timestamps_us >= segment['start_utime_inclusive']) & (timestamps_us <= segment['end_utime_inclusive'])
        if not selected.any():
            continue
        if np.any(replaced & selected):
            raise ValueError('Overlapping reference replacement rules')
        reference_path = policy_path.parent / segment['reference_file']
        if hashlib.sha256(reference_path.read_bytes()).hexdigest() != segment['reference_sha256']:
            raise ValueError('Replacement reference SHA256 mismatch')
        with np.load(reference_path, allow_pickle=False) as source:
            times = source['timestamps_us']
            poses = source['poses']
        if times.shape != (segment['frame_count'],) or poses.shape != (len(times), 4, 4) or np.any(np.diff(times) <= 0):
            raise ValueError('Invalid replacement reference identity or shape')
        requested = timestamps_us[selected]
        indices = np.searchsorted(times, requested)
        if np.any(indices >= len(times)) or not np.array_equal(times[indices], requested):
            raise ValueError('Missing odometry reference inside suspect segment; GT fallback forbidden')
        references[selected] = poses[indices]
        replaced[selected] = True
    if not np.isfinite(references).all():
        raise ValueError('Non-finite evaluation reference')
    metadata = {
        'reference_protocol': policy['protocol'],
        'reference_policy_sha256': hashlib.sha256(policy_bytes).hexdigest(),
        'reference_replaced_frames': int(replaced.sum()),
        'reference_original_gt_frames': int((~replaced).sum()),
        'official_gt_benchmark_compatible': False,
    }
    return references, replaced, metadata


def score_poses(dates, timestamps_us, predictions, gt_poses, policy_path=DEFAULT_POLICY):
    references, replaced, metadata = apply_reference_policy(dates, timestamps_us, gt_poses, policy_path)
    predictions = np.asarray(predictions, dtype=np.float64)
    if len(references) == 0 or predictions.shape != references.shape or not np.isfinite(predictions).all():
        raise ValueError('Invalid or empty prediction array')
    position_errors = np.linalg.norm(predictions[:, :3, 3] - references[:, :3, 3], axis=1)
    cos_angle = (np.einsum('nij,nij->n', predictions[:, :3, :3], references[:, :3, :3]) - 1.0) / 2.0
    orientation_errors = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    report = dict(metadata, frames=len(references), MPE_m=float(position_errors.mean()), MOE_deg=float(orientation_errors.mean()))
    return report, position_errors, orientation_errors, replaced

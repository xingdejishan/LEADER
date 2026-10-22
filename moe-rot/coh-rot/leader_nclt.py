"""NCLT v2 frame store for the coh-rot A/B.

Preprocessing is a line-for-line copy of the project's canonical loader
(`data/leader_nclt_manifest.py :: ManifestLEADERDataset.__getitem__`), so a network
trained here sees exactly the tensors the frozen V5 evaluation cache was built from:

    get_velo(scan)                      -> points (metres), intensity labels
    keep 1 m < range < 100 m
    polar = cartesian_to_polar_expansion(points, voxel_size * horizontal_res)
    feats = (polar.z, polar.range, intensity)
    sparse_quantize(coords=polar, feats, quantization_size=voxel_size)

Only the manifest validation differs: the training manifests are the four `train_small`
manifests (1445 frames, not the frozen 300-frame dev manifests), and the dev manifests
are the official 300-per-date split that the project's reference policy was defined on.
"""

import json
from pathlib import Path

import MinkowskiEngine as ME
import numpy as np
import torch

from data.robotcar_sdk.python.velodyne import get_velo
from utils.pose_util import cartesian_to_polar_expansion

CANONICAL_DATA_ROOT = Path('/data2/zhangcy/leader-v2-full-trajectory-20261001/NCLT')
SCAN_ROOT = Path('/data2/dataset/NCLT')
TRAIN_SMALL = Path('/data2/zhangcy/leader-hard-subsets-20261008-v2/manifests/train_small')
OFFICIAL_300 = Path('/home/zhangcy/projects/nclt_official_300_20260926/NCLT')
EVALUATOR_TRUTH = Path('/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth')

TRAIN_DATES = ('2012-01-22', '2012-02-02', '2012-02-18', '2012-05-11')
DEV_DATES = ('2012-02-12', '2012-02-19', '2012-03-31', '2012-05-26')

# The map centre embedded in the official LEADER checkpoint (`official_l0.pt`).
# The README for the hard subsets is explicit: never recompute the map centre from a
# small training subset, keep the original one.
CANONICAL_CENTER_T = (-104.24640471432245, -381.28044425519795, -203.66971518771015)

VOXEL_SIZE = 0.2
HORIZONTAL_RES = 1024
MIN_RANGE = 1.0
MAX_RANGE = 100.0


def load_manifest_frames(manifest_paths, data_root):
    """Return a list of frame dicts in manifest order."""
    frames = []
    for manifest_path in manifest_paths:
        payload = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
        date = payload['date']
        t_bc = np.asarray(payload['roi']['T_BC'], dtype=np.float64)
        t_cb = np.linalg.inv(t_bc)
        for row in payload['frames']:
            timestamp = int(row['timestamp_us'])
            record = {
                'frame_key': '{}/{}'.format(date, timestamp),
                'date': date,
                'timestamp_us': timestamp,
                'scan_path': SCAN_ROOT / date / row['scan']['path'],
                'scan_sha256': row['scan']['sha256'],
                'T_BC': t_bc,
                'T_WB': None,
            }
            ground_truth = row.get('ground_truth')
            if isinstance(ground_truth, dict):
                gt_path = Path(data_root) / date / ground_truth['path']
                t_wc = np.asarray(np.loadtxt(str(gt_path), dtype=np.float64),
                                  dtype=np.float64).reshape(4, 4)
                record['T_WB'] = (t_wc @ t_cb).astype(np.float32)
            frames.append(record)
    return frames


def train_frames():
    paths = [TRAIN_SMALL / date / 'manifest.json' for date in TRAIN_DATES]
    frames = load_manifest_frames(paths, CANONICAL_DATA_ROOT)
    if any(frame['T_WB'] is None for frame in frames):
        raise ValueError('training frames must all carry ground-truth poses')
    return frames


def dev_frames(dates=DEV_DATES):
    paths = [OFFICIAL_300 / date / 'manifest.json' for date in dates]
    frames = load_manifest_frames(paths, CANONICAL_DATA_ROOT)
    truth = {}
    for date in dates:
        payload = json.loads((EVALUATOR_TRUTH / (date + '.json')).read_text(encoding='utf-8'))
        truth[date] = payload
    for frame in frames:
        entry = truth[frame['date']].get(str(frame['timestamp_us']))
        if entry is None:
            raise ValueError('no evaluator ground truth for {}'.format(frame['frame_key']))
        frame['T_WB'] = np.asarray(entry['T_WB'], dtype=np.float32)
    return frames


def prepare_frame(frame):
    """Exact copy of the canonical loader's preprocessing."""
    points, labels = get_velo(str(frame['scan_path']))
    points = np.asarray(points, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32).reshape(-1, 1)
    ranges = np.linalg.norm(points, axis=1)
    keep = (ranges > MIN_RANGE) & (ranges < MAX_RANGE)
    points = np.ascontiguousarray(points[keep])
    labels = np.ascontiguousarray(labels[keep])
    if not len(points):
        raise ValueError('scan contains no points within the frozen range: {}'.format(frame['frame_key']))
    polar = cartesian_to_polar_expansion(points, VOXEL_SIZE * HORIZONTAL_RES)
    feats = np.column_stack((polar[:, 2], polar[:, 1], labels[:, 0])).astype(np.float32)
    coords, feats = ME.utils.sparse_quantize(coordinates=polar, features=feats,
                                             quantization_size=VOXEL_SIZE)
    return np.asarray(coords, dtype=np.int32), np.asarray(feats, dtype=np.float32)


def prepare_all(frames, progress=None):
    """Voxelise every frame once and hold the result in memory."""
    store = []
    for index, frame in enumerate(frames):
        coords, feats = prepare_frame(frame)
        store.append((torch.from_numpy(coords), torch.from_numpy(feats)))
        if progress and (index + 1) % progress == 0:
            print('  voxelised %d/%d frames' % (index + 1, len(frames)), flush=True)
    return store


def collate(batch):
    """batch: list of ((coords, feats), T_WB).  Mirrors collate_leader_frames."""
    coords = [item[0][0] for item in batch]
    feats = [item[0][1] for item in batch]
    poses = [item[1] for item in batch]
    return {
        'coords': ME.utils.batched_coordinates([c.numpy() for c in coords]),
        'feats': torch.cat(feats, dim=0),
        'T': torch.from_numpy(np.stack([np.asarray(p, dtype=np.float32) for p in poses])),
    }

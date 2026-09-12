import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp


TRAIN_DATES = ('2012-01-22', '2012-02-02', '2012-02-18', '2012-05-11')
TEST_DATES = ('2012-02-12', '2012-02-19', '2012-03-31', '2012-05-26')


def camera_rows(root, camera_number):
    with (Path(root) / 'all_images.csv').open() as handle:
        rows = [r for r in csv.DictReader(handle) if r['camera'] == f'Cam{camera_number}']
    for row in rows:
        row['timestamp_us'] = int(row['original_image_timestamp'])
    return sorted(rows, key=lambda r: r['timestamp_us'])


def validate_dates(train_dates, test_dates):
    if set(train_dates) & set(test_dates):
        raise ValueError('Training and test dates overlap')
    if not set(train_dates) <= set(TRAIN_DATES) or not set(test_dates) <= set(TEST_DATES):
        raise ValueError('Dates must respect the LEADER NCLT train/test split')


def stored_intrinsics(K_raw, row, path):
    with Image.open(path) as im:
        width, height = im.size
    raw_width, raw_height = int(row['original_width']), int(row['original_height'])
    if min(raw_width, raw_height, width, height) <= 0:
        raise ValueError('Invalid image dimensions')
    K = np.asarray(K_raw, dtype=float).copy()
    K[0] *= width / raw_width
    K[1] *= height / raw_height
    return K, (height, width)


def preprocess_image(path, K_stored, image_resolution):
    if image_resolution <= 0:
        raise ValueError('image_resolution must be positive')
    with Image.open(path) as im:
        width, height = im.size
        if width < height:
            raise ValueError('Expected landscape NCLT images for CamLocDataset short-edge resize')
        scale = image_resolution / height
        im = im.convert('RGB').resize((round(width * scale), image_resolution), Image.BILINEAR)
        gray = np.asarray(im.convert('L'), dtype=np.float32) / 255.0
    K = np.asarray(K_stored, dtype=float).copy()
    K[:2] *= scale
    return gray, K


class NCLTTrajectory:
    def __init__(self, path):
        data = np.loadtxt(path, delimiter=',', ndmin=2)
        data = data[np.isfinite(data[:, :7]).all(axis=1), :7]
        data = data[np.argsort(data[:, 0], kind='stable')]
        _, unique = np.unique(data[:, 0], return_index=True)
        data = data[unique]
        if len(data) < 2:
            raise ValueError(f'Insufficient valid GT poses: {path}')
        self.timestamps = data[:, 0].astype(np.int64)
        self.origin = int(self.timestamps[0])
        self.seconds = (self.timestamps - self.origin) / 1e6
        self.xyz = data[:, 1:4]
        self.rotations = Slerp(self.seconds, Rotation.from_euler('xyz', data[:, 4:7]))

    def at(self, timestamps):
        ts = np.asarray(timestamps, dtype=np.int64).reshape(-1)
        if np.any(ts < self.timestamps[0]) or np.any(ts > self.timestamps[-1]):
            raise ValueError('Camera/scan timestamp outside GT trajectory; extrapolation is forbidden')
        seconds = (ts - self.origin) / 1e6
        poses = np.tile(np.eye(4), (len(ts), 1, 1))
        poses[:, :3, :3] = self.rotations(seconds).as_matrix()
        for axis in range(3):
            poses[:, axis, 3] = np.interp(seconds, self.seconds, self.xyz[:, axis])
        return poses


def trajectory_path(dataset_folder, sequence):
    return Path(dataset_folder) / 'NCLT' / sequence / f'groundtruth_{sequence}.csv'

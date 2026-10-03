import numpy as np
import torch

from data.leader_nclt_manifest import IMAGE_SIZE_HW, CAM5_INTRINSICS
from data.robotcar_sdk.python.velodyne import get_velo


def backend_sample(dataset, index):
    record = dataset.records[int(index)]
    dataset._verify_file(record.scan_path, record.scan_sha256)
    dataset._verify_file(record.image_path, record.image_sha256)
    points, _ = get_velo(str(record.scan_path))
    points = np.asarray(points, dtype=np.float32)
    ranges = np.linalg.norm(points, axis=1)
    points = np.ascontiguousarray(points[(ranges > dataset.min_range) & (ranges < dataset.max_range)])
    if not len(points):
        raise ValueError('empty backend scan')
    sample = dict(frame_key=record.frame_key, date=record.date, timestamp_us=record.timestamp_us,
                  points=points, T_corr=np.eye(4, dtype=np.float32),
                  T_BC=np.asarray(record.T_BC, dtype=np.float32),
                  manifest_sha256=record.manifest_sha256, scan_sha256=record.scan_sha256,
                  image_sha256=record.image_sha256)
    if dataset.split == 'train':
        sample['T_WB'] = np.asarray(record.T_WB, dtype=np.float32)
    if dataset.sam_store is not None:
        embedding, valid_mask, metadata = dataset.sam_store.load(record.frame_key)
        height, width = IMAGE_SIZE_HW
        resized_height, resized_width = map(int, metadata['resized_size_hw'])
        intrinsic = CAM5_INTRINSICS.astype(np.float32).copy()
        intrinsic[0, :] *= resized_width / width
        intrinsic[1, :] *= resized_height / height
        sample.update(sam_features=embedding, intrinsics=torch.from_numpy(intrinsic),
                      camera_from_lidar=torch.from_numpy(np.linalg.inv(record.T_BC).astype(np.float32)),
                      image_bounds=torch.tensor([resized_width, resized_height], dtype=torch.float32),
                      image_valid_mask=valid_mask, sam_metadata=metadata)
    return sample


class BackendView:
    def __init__(self, dataset):
        self.base = dataset
        self.records = dataset.records

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        return backend_sample(self.base, index)


def enable(model, chunk=512):
    from field_runtime import enable as vectorize
    vectorize(model)
    model.point_chunk = chunk
    return model

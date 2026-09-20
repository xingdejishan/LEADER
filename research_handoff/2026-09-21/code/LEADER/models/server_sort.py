from functools import lru_cache

import numpy as np


@lru_cache(maxsize=8)
def _network(width):
    threads = np.arange(width // 2)
    steps = []
    size = 2
    while size < width:
        direction = (threads & (size // 2)) != 0
        stride = size // 2
        while stride:
            left = 2 * threads - (threads & (stride - 1))
            steps.append((left, left + stride, direction))
            stride //= 2
        size *= 2
    stride = width // 2
    while stride:
        left = 2 * threads - (threads & (stride - 1))
        steps.append((left, left + stride, False))
        stride //= 2
    return steps


def server_argsort_numpy(values, axis=-1, descending=True):
    values = np.asarray(values)
    if values.dtype.kind != "f":
        raise TypeError("Server-compatible sorting requires floating-point scores")
    if not np.isfinite(values).all():
        raise ValueError("Server-compatible sorting requires finite scores")
    moved = np.moveaxis(values, axis, -1)
    count = moved.shape[-1]
    if count <= 1:
        return np.zeros(values.shape, dtype=np.int64)
    if count > 2048:
        keys = -moved if descending else moved
        return np.moveaxis(np.argsort(keys, axis=-1, kind="stable"), -1, axis)
    # These padded widths and equal-key swaps reproduce PyTorch 1.12 CUDA,
    # rather than imposing a new stable-sort baseline on the server.
    width = 32 if count <= 32 else 128 if count <= 128 else 1024 if count <= 1024 else 2048
    shape = moved.shape[:-1] + (width,)
    keys = np.zeros(shape, dtype=values.dtype)
    keys[..., :count] = moved
    indices = np.broadcast_to(np.arange(width, dtype=np.int64), shape).copy()
    valid = np.broadcast_to(np.arange(width) < count, shape).copy()
    for left, right, direction in _network(width):
        a, b = keys[..., left], keys[..., right]
        va, vb = valid[..., left], valid[..., right]
        if descending:
            greater = (a > b) | (np.isnan(a) & ~np.isnan(b))
        else:
            greater = (a < b) | (np.isnan(b) & ~np.isnan(a))
        swap = ((greater & va) | ~vb) == direction
        for array in (keys, indices, valid):
            x, y = array[..., left], array[..., right]
            array[..., left] = np.where(swap, y, x)
            array[..., right] = np.where(swap, x, y)
    return np.moveaxis(indices[..., :count], -1, axis)


def server_argsort(scores, dim=-1, descending=True):
    import torch

    order = server_argsort_numpy(scores.detach().cpu().numpy(), axis=dim, descending=descending)
    return torch.as_tensor(np.ascontiguousarray(order), dtype=torch.long, device=scores.device)

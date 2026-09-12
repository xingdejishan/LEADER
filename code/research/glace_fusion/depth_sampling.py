import torch


def depth_sampling_weights(image_mask, lidar_valid, fraction=0.5):
    if not 0 < fraction < 1:
        raise ValueError('Depth fraction must be between zero and one')
    if image_mask.numel() != lidar_valid.numel():
        raise ValueError('Sampling mask dimensions differ')
    valid = image_mask.reshape(-1) > 0
    supported = valid & (lidar_valid.reshape(-1) > 0)
    if not valid.any():
        raise ValueError('Invalid sampling masks')
    remaining = valid & ~supported
    weights = valid.float()
    if supported.any() and remaining.any():
        weights = supported.float() * (fraction / supported.sum())
        weights += remaining.float() * ((1 - fraction) / remaining.sum())
    return weights.reshape(image_mask.shape)

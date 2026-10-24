"""Build head checkpoints from the closed-form PCA-64 basis, plain and Hadamard-rotated.

The image student predicts PCA coordinates, so it lives in the same 64-d coordinate frame.
Rotating that frame is a fixed 64x64 multiply applied to the code after the network, and the
basis is rotated the other way, so the exact reconstruction is unchanged and only the 4-bit
quantiser sees a different per-channel dynamic range.
"""
import sys
from pathlib import Path

import numpy as np
import torch

basis = dict(np.load('/home/zhangcy/projects/nclt-feat-bottleneck-20261023-v1/results/basis.npz'))
W = np.asarray(basis['pca64_int8'], dtype=np.float32)
mean = np.asarray(basis['mean'], dtype=np.float32).reshape(-1)
k = W.shape[1]

H = np.array([[1.0]])
while H.shape[0] < k:
    H = np.block([[H, H], [H, -H]])
H = H[:k, :k] / np.sqrt(k)
print('rotation orthogonality error %.3e' % np.abs(H @ H.T - np.eye(k)).max())

out = Path('/home/zhangcy/projects/nclt-compact-encoder-20261023-v1/pca_heads')
out.mkdir(parents=True, exist_ok=True)
torch.save(dict(weight=torch.from_numpy(W.copy()), mean=torch.from_numpy(mean.copy())),
           out / 'plain.pt')
torch.save(dict(weight=torch.from_numpy((W @ H).astype(np.float32)), mean=torch.from_numpy(mean.copy())),
           out / 'rot.pt')
np.save(out / 'rotation.npy', H)
print('saved', out)

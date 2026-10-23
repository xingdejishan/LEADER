"""Rotate the 64-d code by an orthogonal matrix before storage.

Quantising code c to 4 bits per channel spends levels on each channel's range.  The
subspace is unchanged by an orthogonal rotation Q (W' = W Q, c' = c Q keeps
c' W'^T = c W^T exactly), but the per-channel dynamic range changes, so the same bit
budget can be distributed better.  This tests the cheapest possible fix for the 4-bit
gap: no retraining, only a change of basis inside the same subspace.
"""
import numpy as np
import torch
from pathlib import Path
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
mode = sys.argv[3] if len(sys.argv) > 3 else 'hadamard'

state = torch.load(src, map_location='cpu')
W = state['weight'].numpy().astype(np.float64)
d, k = W.shape
if mode == 'hadamard':
    H = np.array([[1.0]])
    while H.shape[0] < k:
        H = np.block([[H, H], [H, -H]])
    H = H[:k, :k] / np.sqrt(k)
    Q = H
elif mode == 'random':
    rng = np.random.default_rng(20261023)
    A = rng.normal(size=(k, k))
    Q, _ = np.linalg.qr(A)
else:
    raise SystemExit('unknown mode')
W2 = W @ Q
err = np.abs(W2 @ Q.T - W).max()
print('orthogonality roundtrip error %.3e' % err)
state['weight'] = torch.as_tensor(W2).float()
state['rotation'] = mode
torch.save(state, dst)
print('wrote', dst)

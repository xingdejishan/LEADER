"""Decompose the hard-frame rotation error into body-frame roll / pitch / yaw.

The rotation error is R_pred^T R_gt, i.e. expressed in the body frame, and its
axis-angle vector is reported per axis. For small errors the three components are
the roll / pitch / yaw errors; for the few very large ones they mix, so the >8 deg
band is a magnitude reading, not a strict Euler decomposition.

    python rot_decomp.py [prediction.npz]

With no argument it reports the deployed baseline only.
"""
import importlib.util
import sys

import numpy as np
from scipy.spatial.transform import Rotation

import config


def load_scorer():
    spec = importlib.util.spec_from_file_location('score_unified', config.SCORER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decompose(pred, gt, name):
    Rp, Rg = pred[:, :3, :3], gt[:, :3, :3]
    Rd = np.swapaxes(Rp, -1, -2) @ Rg
    rv = Rotation.from_matrix(Rd).as_rotvec()
    ang = np.degrees(np.linalg.norm(rv, axis=1))
    deg = np.degrees(rv)
    print('\n== %s ==' % name)
    print('mean angle      : %.4f deg   (median %.4f, p90 %.4f)'
          % (ang.mean(), np.median(ang), np.percentile(ang, 90)))
    print('mean |x| |y| |z|: %.4f  %.4f  %.4f   (body roll / pitch / yaw)'
          % (np.abs(deg[:, 0]).mean(), np.abs(deg[:, 1]).mean(), np.abs(deg[:, 2]).mean()))
    tot = (deg ** 2).sum(1).mean()
    print('share of squared error: x %.1f%%  y %.1f%%  z %.1f%%'
          % (100 * (deg[:, 0] ** 2).mean() / tot, 100 * (deg[:, 1] ** 2).mean() / tot,
             100 * (deg[:, 2] ** 2).mean() / tot))
    for lo, hi, tag in ((0, 2, '<=2 deg'), (2, 8, '2-8 deg'), (8, 1e9, '>8 deg')):
        m = (ang > lo) & (ang <= hi)
        if m.sum() < 5:
            continue
        print('  band %-9s n=%4d  mean angle %6.3f  |roll| %5.3f  |pitch| %5.3f  |yaw| %5.3f'
              % (tag, int(m.sum()), ang[m].mean(), np.abs(deg[m, 0]).mean(),
                 np.abs(deg[m, 1]).mean(), np.abs(deg[m, 2]).mean()))
    return ang, deg


def main():
    scorer = load_scorer()
    w = scorer.load_world()
    keys = w['hard_keys']
    gt = np.asarray(w['hard_gt'], float)
    base = np.asarray(w['hard_base'], float)
    print('frames %d' % len(keys))
    decompose(base, gt, 'deployed baseline')
    if len(sys.argv) > 1:
        z = np.load(sys.argv[1], allow_pickle=True)
        lookup = {str(k): p for k, p in zip(z['keys'], z['poses'])}
        pred = np.stack([lookup[k] for k in keys])
        decompose(pred, gt, sys.argv[1])


if __name__ == '__main__':
    main()

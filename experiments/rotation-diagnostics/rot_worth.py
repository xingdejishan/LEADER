"""What each rotation axis is worth: recompute the mean angle with one axis zeroed.

This is an upper bound, not an achievable gain: it assumes the named component is
estimated perfectly. It answers "if we could fix only roll, only pitch, only yaw,
how much MOE would that buy", which is a different question from "how much of the
squared error is that axis" -- the two orderings differ, because the mean of the
angle is not the mean of the squared angle.
"""
import importlib.util

import numpy as np
from scipy.spatial.transform import Rotation

import config


def main():
    spec = importlib.util.spec_from_file_location('score_unified', config.SCORER)
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    gt = np.asarray(w['hard_gt'], float)
    base = np.asarray(w['hard_base'], float)

    Rd = np.swapaxes(base[:, :3, :3], -1, -2) @ gt[:, :3, :3]
    rv = np.degrees(Rotation.from_matrix(Rd).as_rotvec())
    full = np.linalg.norm(rv, axis=1)

    def moe(drop):
        keep = [i for i in range(3) if i not in drop]
        return float(np.linalg.norm(rv[:, keep], axis=1).mean())

    print('baseline MOE                       : %.4f deg' % full.mean())
    for name, drop in (('yaw', {2}), ('roll+pitch', {0, 1}), ('roll', {0}), ('pitch', {1})):
        v = moe(drop)
        print('if the %-10s error were removed : %.4f deg  (-%.4f)' % (name, v, full.mean() - v))

    big = full > 8
    print('\ntail (>8 deg, n=%d): mean %.1f deg, of which yaw %.1f, roll %.1f, pitch %.1f'
          % (big.sum(), full[big].mean(), np.abs(rv[big, 2]).mean(),
             np.abs(rv[big, 0]).mean(), np.abs(rv[big, 1]).mean()))
    print('those frames carry %.1f%% of the summed angle over all frames'
          % (100 * full[big].sum() / full.sum()))

    if config.DIAG.is_file():
        import json
        ratio = np.array([json.loads(l)['ratio'] for l in open(config.DIAG)])
        g = np.isfinite(ratio) & (ratio < 0.21)
        print('gated frames n=%d  mean angle %.2f deg  %.1f%% of the summed angle'
              % (g.sum(), full[g].mean(), 100 * full[g].sum() / full.sum()))


if __name__ == '__main__':
    main()

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .io import load_arrays
from .mapping import sha256
from .resources import ResourceBudget


def fit_noise(frames, labels):
    errors, nonrigid = [], []
    for path in frames:
        frame, meta = load_arrays(path, online=True)
        key = meta['frame_key']
        if key.split('/')[0] not in ('2012-01-22','2012-02-02','2012-02-18','2012-05-11'):
            raise ValueError('noise fitting may only use training dates')
        truth = np.asarray(labels[key]['pose'])
        base = frame['T0'].numpy()
        relative = np.linalg.inv(base) @ truth
        error = np.r_[relative[:3,3], Rotation.from_matrix(relative[:3,:3]).as_rotvec()]
        errors.append(error)
        source, target, weights = [frame[k].numpy() for k in ('source','target','weights')]
        weights = weights / max(weights.sum(),1e-12)
        sp, tp = weights @ source, weights @ target
        u, _, vt = np.linalg.svd((source-sp).T @ ((target-tp)*weights[:,None]))
        fix = np.diag([1.,1.,np.linalg.det(vt.T@u.T)])
        rotation = vt.T @ fix @ u.T
        residual = (source-sp) @ rotation.T - (target-tp)
        nonrigid.append(np.sum(weights[:,None]*residual**2)/3)
    if len(errors) < 12:
        raise ValueError('at least 12 training frames required to fit the six-dimensional shared error')
    errors = np.asarray(errors)
    covariance = errors.T @ errors / len(errors)
    if np.linalg.eigvalsh(covariance)[0] <= 1e-12:
        raise ValueError('training errors do not identify a full covariance')
    return dict(schema='leader-correlated-point-noise-v1', independent_sigma=float(np.sqrt(np.mean(nonrigid))),
                shared_covariance=covariance.tolist(), training_frames=len(errors), calibrated_on='training errors only; no generalization claim',
                approximation='shared local rigid error plus independent nonrigid point residual')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', nargs='+', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    ResourceBudget(args.output, 120)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    result = fit_noise(args.frames, json.loads(Path(args.labels).read_text()))
    result['input_sha256'] = [sha256(p) for p in args.frames]
    output.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

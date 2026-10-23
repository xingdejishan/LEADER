"""Read-only sanity check of the merged output: shapes, finiteness, key order, and
that every frame's diagnostic carries the expected fields."""
import json

import numpy as np

P = '/home/zhangcy/projects/e2e-moe-20261020-v1'
z = np.load(P + '/results/three_route_merged.npz', allow_pickle=True)
keys = [str(k) for k in z['keys']]
poses = np.asarray(z['poses'], float)
ini = np.load(P + '/inputs/initial.npz', allow_pickle=True)
ikeys = [str(k) for k in ini['keys']]
print('frames=%d  poses=%s  dtype=%s' % (len(keys), poses.shape, poses.dtype))
print('keys match inputs/initial.npz:', keys == ikeys)
print('all finite:', bool(np.isfinite(poses).all()))
rot = poses[:, :3, :3]
orth = np.abs(rot @ np.swapaxes(rot, 1, 2) - np.eye(3)).max()
det = np.linalg.det(rot)
print('max |R R^T - I| = %.2e   det range [%.6f, %.6f]' % (orth, det.min(), det.max()))
print('bottom row all [0,0,0,1]:',
      bool(np.allclose(poses[:, 3, :], np.tile([0, 0, 0, 1.0], (len(poses), 1)))))

diag = [json.loads(l) for l in open(P + '/results/three_route_merged_diag.jsonl')]
need = ('key', 'gain', 'ran_fine', 'rotated', 'ratio', 'c_route3', 'c_visual', 'dcost',
        'visual', 'chosen', 'gate')
missing = [d['key'] for d in diag if any(f not in d for f in need)]
print('diag lines=%d  missing fields in %d lines' % (len(diag), len(missing)))
print('fine ran=%d  rotated=%d  visual=%d  severe(ratio<0.21)=%d'
      % (sum(d['ran_fine'] for d in diag), sum(d['rotated'] for d in diag),
         sum(d['visual'] for d in diag),
         sum(1 for d in diag if np.isfinite(d['ratio']) and d['ratio'] < 0.21)))
print('diag keys match pose table:', [d['key'] for d in diag] == keys)

import numpy as np

P = '/home/zhangcy/projects/e2e-moe-20261020-v1'
for name in ('results/online-shard0.npz', 'results/online19-shard0.npz', 'inputs/initial.npz',
             'results/three_route_merged.npz'):
    z = np.load(P + '/' + name, allow_pickle=True)
    poses = np.asarray(z['poses'], float)
    R = poses[:, :3, :3]
    dev = np.abs(np.einsum('nij,nkj->nik', R, R) - np.eye(3)).max()
    det = np.abs(np.linalg.det(R) - 1.0).max()
    bottom = np.abs(poses[:, 3, :] - np.array([0, 0, 0, 1.0])).max()
    print('%-32s n=%3d  |RR^T-I|max=%.3e  |det-1|max=%.3e  |last row err|=%.3e'
          % (name, len(poses), dev, det, bottom))

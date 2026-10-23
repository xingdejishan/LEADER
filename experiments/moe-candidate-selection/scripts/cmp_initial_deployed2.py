import numpy as np
a = np.load('/home/zhangcy/projects/e2e-moe-20261020-v1/inputs/initial.npz', allow_pickle=True)
b = np.load('/home/zhangcy/projects/r-unified-eval-20261018-v1/deployed_baseline.npz', allow_pickle=True)
pa, pb = a['poses'], b['refined_V7']
print('max abs diff initial vs refined_V7', float(np.abs(pa - pb).max()))
tr = np.linalg.norm(pa[:, :3, 3] - pb[:, :3, 3], axis=1)
print('world trans diff max', float(tr.max()))
print('replaced count', int(b['replaced'].sum()), 'of', len(b['replaced']))
po = b['original_V7']
d = np.linalg.norm(po[:, :3, 3] - pb[:, :3, 3], axis=1)
print('refined vs original_V7 trans diff mean/max', float(d.mean()), float(d.max()))

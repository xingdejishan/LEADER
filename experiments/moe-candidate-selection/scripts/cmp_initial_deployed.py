import numpy as np

a = np.load('/home/zhangcy/projects/e2e-moe-20261020-v1/inputs/initial.npz', allow_pickle=True)
b = np.load('/home/zhangcy/projects/r-unified-eval-20261018-v1/deployed_baseline.npz', allow_pickle=True)
print('initial.npz fields', a.files, a['poses'].shape)
print('deployed fields', b.files, {k: b[k].shape for k in b.files})
ka = [str(x) for x in a['keys']]
kb = [str(x) for x in b['keys']] if 'keys' in b.files else None
if kb is not None:
    print('keys equal', ka == kb)
    pa, pb = a['poses'], b['poses']
    if pa.shape == pb.shape:
        d = np.abs(pa - pb)
        print('max abs diff poses', float(d.max()))
        tr = np.linalg.norm(pa[:, :3, 3] - pb[:, :3, 3], axis=1)
        print('translation diff mean/max (m)', float(tr.mean()), float(tr.max()))

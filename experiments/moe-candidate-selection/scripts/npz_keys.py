import numpy as np

for name in ('sweep_full.npz', 'sweep_rot.npz', 'sweep-summary.npz'):
    path = '/home/zhangcy/.dsh_scratch/' + name
    try:
        z = np.load(path, allow_pickle=True)
    except IOError:
        path = '/home/zhangcy/projects/e2e-moe-20261020-v1/results/' + name
        z = np.load(path, allow_pickle=True)
    print('==', name, path)
    for k in z.keys():
        a = np.asarray(z[k])
        print('   %-16s %-18s %s' % (k, str(a.shape), a.dtype))

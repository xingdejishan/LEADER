import numpy as np, os, json
RES = '/home/zhangcy/projects/e2e-moe-20261020-v1/results'
for n in ['three_route_merged.npz', 'verify_route3.npz', 'final.npz']:
    p = os.path.join(RES, n)
    if not os.path.exists(p):
        print(n, 'MISSING'); continue
    z = np.load(p, allow_pickle=True)
    print(n, z.files)
    k = [str(x) for x in z['keys']]
    print('  n=%d head=%s tail=%s' % (len(k), k[:3], k[-2:]))
    print('  poses', np.asarray(z['poses']).shape)
z = np.load('/data2/zhangcy/rpge-crossvote-hard1502-20261012-v1/scored.npz', allow_pickle=True)
print('scored keys head', [str(x) for x in z['keys'][:3]])
z2 = np.load('/home/zhangcy/projects/e2e-moe-20261020-v1/inputs/initial.npz', allow_pickle=True)
print('initial', z2.files, [str(x) for x in z2['keys'][:3]])
fr = json.load(open('/home/zhangcy/projects/e2e-moe-20261020-v1/inputs/frames.json'))
print('frames', len(fr), fr[0].get('key'), sorted(fr[0].keys()))
print('frames400 key', fr[400].get('key'))

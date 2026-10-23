import os, json
import numpy as np
P = '/home/zhangcy/projects/e2e-moe-20261020-v1'
print(open(P + '/stages/fast_search/run_frames.py').read())
print('=' * 40)
z = np.load(P + '/results/full-online.npz', allow_pickle=True)
print('npz keys:', list(z.keys()))
keys = [str(k) for k in z['keys']]
t0 = np.asarray(z['t0'], np.float64)
fin = np.asarray(z['poses'], np.float64)
print('n =', len(keys), t0.shape, fin.shape)
TRUTH = '/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth'
truth = {}
for f in os.listdir(TRUTH):
    dd = f.replace('.json', '')
    for ts, v in json.load(open(os.path.join(TRUTH, f))).items():
        truth[(dd, str(ts))] = np.asarray(v['T_WB'], np.float64)
print('truth entries', len(truth))
e0 = np.array([np.linalg.norm(t0[i][:3, 3] - truth[tuple(keys[i].split('/'))][:3, 3])
               for i in range(len(keys))])
e1 = np.array([np.linalg.norm(fin[i][:3, 3] - truth[tuple(keys[i].split('/'))][:3, 3])
               for i in range(len(keys))])
for lo, hi in ((5, 8), (5, 8.0001)):
    m = (e0 >= lo) & (e0 < hi)
    print('band', lo, hi, 'n =', int(m.sum()), 'inside 1m after:', int((e1[m] <= 1).sum()))
m = (e0 > 5) & (e0 <= 8)
print('inside 1m (e0>5,<=8):', int((e1[m] <= 1).sum()), 'of', int(m.sum()))
print('e0 percentiles', np.percentile(e0, [50, 90, 99]), 'max', e0.max())
idx = np.where((e0 >= 5) & (e0 <= 8))[0]
print('first idx', idx[:10])
print('mean e0 in band', e0[idx].mean(), 'mean e1', e1[idx].mean())
print('of band, e1<=1:', int((e1[idx] <= 1).sum()))
print('ordinary (<1m e0) count:', int((e0 <= 1).sum()), ' (<0.5):', int((e0 < 0.5).sum()))
refs = json.load(open(P + '/inputs/refs.json'))
print('refs type', type(refs), len(refs), list(refs[0].keys()))
print('ref0', {k: (np.asarray(v).shape if hasattr(v, '__len__') else v) for k, v in refs[0].items()})
print('scan exists', os.path.exists(refs[0]['scan_path']), refs[0]['scan_path'])

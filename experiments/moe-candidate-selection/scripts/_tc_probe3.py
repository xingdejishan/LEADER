import json, os, re
import numpy as np
P = '/home/zhangcy/projects/e2e-moe-20261020-v1'
FILES = ['full-shard0.jsonl', 'fullb-shard1.jsonl', 'fullb-shard3.jsonl']
rows = {}
for name in FILES:
    with open(P + '/results/' + name) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get('key') and r.get('seconds'):
                rows[r['key']] = r
z = np.load(P + '/results/full-online.npz', allow_pickle=True)
keys = [str(x) for x in z['keys']]
t0 = np.asarray(z['t0'], np.float64)
fin = np.asarray(z['poses'], np.float64)
TRUTH = '/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth'
truth = {}
for f in os.listdir(TRUTH):
    dd = f.replace('.json', '')
    for ts, v in json.load(open(os.path.join(TRUTH, f))).items():
        truth[(dd, str(ts))] = np.asarray(v['T_WB'], np.float64)
e0 = np.array([np.linalg.norm(t0[i][:3, 3] - truth[tuple(keys[i].split('/'))][:3, 3])
               for i in range(len(keys))])
band = np.where((e0 >= 5) & (e0 <= 8))[0]
S = [rows[keys[i]] for i in band]
sv = np.array([r['s'] for r in S]); dv = np.array([r['d'] for r in S])
print('band n=%d  |s|<1e-9: %d   s range %.2f..%.2f   d range %.2f..%.2f'
      % (len(S), int((np.abs(sv) < 1e-9).sum()), sv.min(), sv.max(), dv.min(), dv.max()))
print('s values:', np.round(sv, 2).tolist())
print('expanded:', [r.get('expanded') for r in S])
print('map_points:', [r.get('map_points') for r in S][:5], 'scan_points:', [r.get('scan_points') for r in S][:5])
print('ran_fine:', sum(1 for r in S if r.get('ran_fine')), 'rotated:', sum(1 for r in S if r.get('rotated')))
e1 = np.array([np.linalg.norm(fin[i][:3, 3] - truth[tuple(keys[i].split('/'))][:3, 3]) for i in band])
print('e1 in band: <=1m %d, <1m %d, median %.2f' % (int((e1 <= 1).sum()), int((e1 < 1).sum()),
                                                   np.median(e1)))
src = open(P + '/e2e_online.py').read()
for m in re.finditer(r'Config\([^)]*\)', src):
    print('CFG:', m.group(0))
for m in re.finditer(r'(coarse_points|fine_points|topk|solver)\s*=\s*[^,\n)]+', src):
    print('  ', m.group(0))

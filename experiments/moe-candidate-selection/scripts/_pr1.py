import json
import os
import numpy as np

ROOT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
VERIFY = ('/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/'
          'verify-frozen_nopruneoptREF/%06d.json')

z = np.load(os.path.join(ROOT, 'inputs/initial.npz'), allow_pickle=True)
keys = [str(k) for k in z['keys']]
print('initial keys', len(keys), keys[:5])
print('poses', z['poses'].shape, z['poses'].dtype)
frames = json.load(open(os.path.join(ROOT, 'inputs/frames.json')))
print('frames', len(frames), frames[0].keys(), frames[0]['key'])
refs = json.load(open(os.path.join(ROOT, 'inputs/refs.json')))
print('refs', len(refs), refs[0].keys())

for name in ('three_route.npz', 'three_route_unified.npz', 'three_route_merged.npz'):
    p = os.path.join(ROOT, 'results', name)
    if not os.path.exists(p):
        print(name, 'MISSING'); continue
    zz = np.load(p, allow_pickle=True)
    print(name, zz.files, len(zz['poses']), [str(k) for k in zz['keys']][:2])

for name in ('three_route_merged_diag.jsonl', 'three_route_route3_diag.jsonl',
             'three_route_unified_diag.jsonl'):
    p = os.path.join(ROOT, 'results', name)
    rows = [json.loads(l) for l in open(p)]
    print(name, len(rows), 'visual=', sum(1 for r in rows if r.get('visual')),
          'fine=', sum(1 for r in rows if r.get('ran_fine')))
    print('  first row', rows[0])

# check the frozen visual table for the first 5 frames
for i in range(5):
    rec = json.load(open(VERIFY % i))['poses']
    best = None
    for nm, row in rec.items():
        if not nm.startswith('candidate_'):
            continue
        sc = (row.get('fair') or {}).get('geometry_score')
        if sc is None:
            continue
        if best is None or sc > best[0]:
            best = (float(sc), np.asarray(row['pose'], float))
    print(i, keys[i], 'visual?', best is not None,
          'score', None if best is None else round(best[0], 4),
          'pos', None if best is None else np.round(best[1][:3, 3], 3))

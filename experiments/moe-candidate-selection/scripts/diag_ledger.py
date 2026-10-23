"""Compare my per-frame map size with the online ledger's map_points (same map?)."""
import json
import os

import numpy as np

P = '/home/zhangcy/projects/e2e-moe-20261020-v1'
R = P + '/results/'
row = {}
for name in ('full-shard0.jsonl', 'fullb-shard1.jsonl', 'fullb-shard3.jsonl'):
    p = R + name
    if not os.path.exists(p):
        print('missing', name)
        continue
    for line in open(p):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        row[r['key']] = r
print('ledger rows', len(row))
k = list(row)[0]
print('row fields', sorted(row[k].keys()))
k2 = '2012-05-26/1338075245561106'
print(json.dumps({a: b for a, b in row[k2].items() if a not in ('t0', 'pose')}, ensure_ascii=False)[:800])

z = np.load(P + '/results/full-online.npz', allow_pickle=True)
keys = [str(x) for x in z['keys']]
import subprocess
print(subprocess.run(['stat', '-c', '%y %s %n', P + '/inputs/refs.json'], capture_output=True,
                     text=True).stdout)
for name in ('inputs/refs.json', 'inputs/frames.json', 'inputs/initial.npz'):
    print(subprocess.run(['md5sum', P + '/' + name], capture_output=True, text=True).stdout.strip())

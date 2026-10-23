import hashlib
import json
import os

import numpy as np

PROJ = '/home/zhangcy/projects/e2e-moe-20261020-v1'


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


print('driver project   ', sha(os.path.join(PROJ, 'e2e_online.py')))
print('driver scratch   ', sha('/home/zhangcy/.dsh_scratch/e2e_online.py'))
print('same             ', sha(os.path.join(PROJ, 'e2e_online.py')) == sha('/home/zhangcy/.dsh_scratch/e2e_online.py'))

for name in ('online-shard0.npz', 'online-shard0.jsonl', 'online19-shard0.npz', 'online19-shard0.jsonl'):
    p = os.path.join(PROJ, 'results', name)
    print('%-22s %8d bytes' % (name, os.path.getsize(p)))

z = np.load(os.path.join(PROJ, 'results/online-shard0.npz'), allow_pickle=True)
print('npz keys', z['keys'].shape, z['keys'][:2], 'poses', z['poses'].shape, z['poses'].dtype)
z19 = np.load(os.path.join(PROJ, 'results/online19-shard0.npz'), allow_pickle=True)
print('npz19 keys', z19['keys'].shape, z19['keys'][:2], 'poses', z19['poses'].shape)
print('fin', np.isfinite(z['poses']).all(), 'orthonormal',
      bool(np.allclose(np.einsum('nij,nkj->nik', z['poses'][:, :3, :3], z['poses'][:, :3, :3]),
                       np.eye(3), atol=1e-9)) if z['keys'].size else None)

rows = [json.loads(l) for l in open(os.path.join(PROJ, 'results/online-shard0.jsonl'))]
print('jsonl fields', sorted(rows[0].keys()))
print('jsonl line sample', json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                      for k, v in rows[0].items() if k not in ('pose', 't0', 'seconds')})[:300])

# files touched in the project since the driver was written
start = os.path.getmtime(os.path.join(PROJ, 'e2e_online.py'))
touched = []
for root, dirs, files in os.walk(PROJ):
    dirs[:] = [d for d in dirs if d != '__pycache__']
    for name in files:
        p = os.path.join(root, name)
        if os.path.getmtime(p) >= start - 1:
            touched.append(os.path.relpath(p, PROJ))
print('project files at/after driver mtime:', sorted(touched))

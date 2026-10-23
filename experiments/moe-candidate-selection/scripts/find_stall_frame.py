"""Locate the frame that stalls shard 1 (read-only)."""
import glob
import json
import os

ROOT = '/data2/dataset/NCLT'
dates = sorted(os.listdir(ROOT))
keys = []
per_date = {}
for d in dates:
    scan_dir = os.path.join(ROOT, d, 'velodyne_sync')
    if not os.path.isdir(scan_dir):
        continue
    names = sorted(os.path.basename(p)[:-4] for p in glob.glob(scan_dir + '/*.bin'))
    per_date[d] = names
    keys.extend('%s/%s' % (d, n) for n in names)
print('dates:', {d: len(v) for d, v in per_date.items()}, 'total', len(keys))

jpath = '/home/zhangcy/projects/e2e-moe-20261020-v1/results/full-shard1.jsonl'
last = None
with open(jpath) as fh:
    for line in fh:
        if line.strip():
            last = json.loads(line)
print('last written key:', last['key'], 'rows so far')

# shard 1 of 2 takes keys[1::2]; the frame after the last written one is the suspect
shard = keys[1::2]
idx = shard.index(last['key']) if last['key'] in shard else -1
print('index in shard:', idx)
for j in range(idx, min(idx + 4, len(shard))):
    k = shard[j]
    d, n = k.split('/')
    b = os.path.join(ROOT, d, 'velodyne_sync', n + '.bin')
    jp = os.path.join(ROOT, d, 'cam5', n + '.jpg')
    print('  %d %s  bin=%.1f MB jpg=%s'
          % (j, k, os.path.getsize(b) / 2. ** 20 if os.path.exists(b) else -1,
             ('%.2f MB' % (os.path.getsize(jp) / 2. ** 20)) if os.path.exists(jp) else 'MISSING'))

"""Find the frame after the last written one on the same date, and let the run pass it.

The online driver's jsonl is the resume ledger: a row counts as done. Appending one
explicitly-marked 'stalled' row lets the shard move on; the row is tagged so the merge
step can drop it (a stall means no pose was produced for that frame).
"""
import glob
import json
import os

ROOT = '/data2/dataset/NCLT'
J = '/home/zhangcy/projects/e2e-moe-20261020-v1/results/full-shard1.jsonl'

last = None
rows = 0
with open(J) as fh:
    for line in fh:
        if line.strip():
            last = json.loads(line)
            rows += 1
print('rows=%d last key=%s' % (rows, last['key']))

date, stamp = last['key'].split('/')
names = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ROOT, date, 'velodyne_sync', '*.bin')))
later = [n for n in names if n > stamp]
print('date=%s  frames after last on this date: %d  next=%s' % (date, len(later), later[0] if later else '-'))

if later:
    suspect = '%s/%s' % (date, later[0])
    b = os.path.join(ROOT, date, 'velodyne_sync', later[0] + '.bin')
    jp = os.path.join(ROOT, date, 'cam5', later[0] + '.jpg')
    print('suspect=%s  bin=%.2f MB  jpg=%s'
          % (suspect, os.path.getsize(b) / 2. ** 20 if os.path.exists(b) else -1,
             os.path.exists(jp)))
    with open(J, 'a') as fh:
        fh.write(json.dumps(dict(key=suspect, pose=[0.0] * 16, t0=[0.0] * 16,
                                 seconds={}, total=0.0, skipped='driver_stall',
                                 note='frame stalled the pipeline twice; no pose produced')) + '\n')
    print('appended skip row for', suspect)

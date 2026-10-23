"""Per-hour throughput of the two full-run shards (read-only)."""
import json
import time

import numpy as np

P = '/home/zhangcy/projects/e2e-moe-20261020-v1/results/'
for shard in (0, 1):
    rows = []
    try:
        with open(P + 'full-shard%d.jsonl' % shard) as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except IOError:
        continue
    if not rows:
        continue
    t = np.array([float(r['t_end']) for r in rows])
    tot = np.array([float(r['total']) for r in rows])
    hours = ((t - t.min()) // 3600).astype(int)
    print('shard%d  frames=%d  span=%.2f h  overall=%.2f fps  median_frame=%.3f s'
          % (shard, len(rows), (t.max() - t.min()) / 3600.0, len(rows) / (t.max() - t.min()),
             float(np.median(tot))))
    for h in range(hours.max() + 1):
        m = hours == h
        if m.sum():
            print('   hour %2d: %6d frames  %5.2f fps  median %.3f s'
                  % (h, int(m.sum()), m.sum() / 3600.0, float(np.median(tot[m]))))

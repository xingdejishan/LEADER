"""Current rate of the two pipelined shards (read-only)."""
import json

import numpy as np

P = '/home/zhangcy/projects/e2e-moe-20261020-v1/results/'
total = 0
for shard in (1, 3):
    rows = []
    try:
        with open(P + 'fullb-shard%d.jsonl' % shard) as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except IOError:
        continue
    if not rows:
        continue
    t = np.array([float(r.get('t_end', 0)) for r in rows])
    t = t[t > 0]
    if len(t) < 2:
        continue
    recent = t[-200:]
    span = recent[-1] - recent[0]
    fps = (len(recent) - 1) / span if span > 0 else float('nan')
    print('shard%d rows=%d  last-200 rate=%.2f fps (%.0f ms/frame)  last t_end in file=%s'
          % (shard, len(rows), fps, 1000.0 / fps if fps else -1, 'ok'))
    total += fps
print('aggregate recent rate = %.2f fps -> %.0f ms/frame' % (total, 1000.0 / total if total else -1))
remaining = 52366 * 2 - sum(1 for _ in open(P + 'fullb-shard1.jsonl')) - sum(1 for _ in open(P + 'fullb-shard3.jsonl'))
print('remaining frames in this split = %d -> %.1f h at this rate'
      % (remaining, remaining / total / 3600.0 if total else -1))

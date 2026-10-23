import json
import math
import os
from collections import Counter

D = '/home/zhangcy/projects/e2e-moe-20261020-v1/results'
rows = []
for line in open(os.path.join(D, 'three_route_merged_diag.jsonl')):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    if isinstance(r.get('ratio'), (int, float)) and math.isfinite(r['ratio']):
        rows.append(r)
print('total', len(rows))
zero = [r for r in rows if r['ratio'] == 0.0]
low = [r for r in rows if 0.0 < r['ratio'] < 0.21]
high = [r for r in rows if r['ratio'] >= 0.21]
print('ratio==0', len(zero), '0<ratio<0.21', len(low), '>=0.21', len(high))
print('dates of zero', Counter(r['key'].split('/')[0] for r in zero))
print('dates of low', Counter(r['key'].split('/')[0] for r in low))
spread = [low[i] for i in range(0, len(low), max(1, len(low) // 8))][:8]
picked = spread + zero[:2]
seen = set()
out = []
for r in picked:
    if r['key'] in seen:
        continue
    seen.add(r['key'])
    out.append(r)
loose = [high[i] for i in range(0, len(high), max(1, len(high) // 10))][:10]
keys = [r['key'] for r in out] + [r['key'] for r in loose]
print('gated picked:')
for r in out:
    print('  %.4f %s' % (r['ratio'], r['key']))
print('ungated picked:')
for r in loose:
    print('  %.4f %s' % (r['ratio'], r['key']))
json.dump(keys, open('/home/zhangcy/.dsh_scratch/vb/v2_keys.json', 'w'), indent=1)
print('wrote', len(keys))

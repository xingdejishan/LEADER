import json
import os

D = '/home/zhangcy/projects/e2e-moe-20261020-v1/results'
rows = []
for line in open(os.path.join(D, 'three_route_merged_diag.jsonl')):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    if r.get('ratio') is None:
        continue
    rows.append(r)
print('frames with ratio', len(rows))
import math
finite = [r for r in rows if isinstance(r['ratio'], float) and math.isfinite(r['ratio'])]
gated = sorted(finite, key=lambda r: r['ratio'])[:10]
loose = sorted(finite, key=lambda r: -r['ratio'])[:10]
print('gated (lowest ratio):')
for r in gated:
    print('  %.4f %s' % (r['ratio'], r['key']))
print('not gated (highest ratio):')
for r in loose[:10]:
    print('  %.4f %s' % (r['ratio'], r['key']))
keys = [r['key'] for r in gated] + [r['key'] for r in loose]
json.dump(keys, open('/home/zhangcy/.dsh_scratch/vb/v2_keys.json', 'w'), indent=1)
print('wrote', len(keys), 'keys')
print('gated count', sum(1 for r in gated))

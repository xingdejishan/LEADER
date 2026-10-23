import json
from pathlib import Path
import numpy as np

VERIFY = Path('/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/verify-frozen_nopruneoptREF')
rows = []
for i in range(1502):
    p = VERIFY / ('%06d.json' % i)
    if not p.is_file():
        continue
    rec = json.loads(p.read_text())
    poses = rec['poses']
    base = (poses.get('baseline_V7') or {})
    bs = ((base.get('fair') or {}).get('geometry_score'))
    best = None
    for name, row in poses.items():
        if not name.startswith('candidate_'):
            continue
        s = ((row.get('fair') or {}).get('geometry_score'))
        if s is None:
            continue
        if best is None or s > best[0]:
            best = (float(s), name, row)
    ncand = sum(1 for n in poses if n.startswith('candidate_'))
    rows.append(dict(i=i, key=rec['key'], base=bs, best=None if best is None else best[0],
                     best_name=None if best is None else best[1], ncand=ncand))
have = [r for r in rows if r['best'] is not None]
beat = [r for r in have if r['base'] is not None and r['best'] > r['base'] + 0.10]
print('frames', len(rows), 'with candidates', len(have), 'beat base by >0.10', len(beat))
strict = [r for r in have if r['base'] is not None and r['best'] > r['base']]
print('beat base (any)', len(strict))
beat.sort(key=lambda r: -(r['best'] - r['base']))
print('top margins:')
for r in beat[:15]:
    print('  %4d %s ncand=%d base=%.4f best=%.4f d=%+.4f %s'
          % (r['i'], r['key'], r['ncand'], r['base'], r['best'], r['best'] - r['base'], r['best_name']))
json.dump([dict(i=r['i'], key=r['key'], base=r['base'], best=r['best']) for r in beat[:40]],
          open('/home/zhangcy/.dsh_scratch/vb/v1_candidates.json', 'w'), indent=1)
# also a spread sample: every 300th frame with candidates
spread = [r for r in have if r['i'] % 300 == 0][:6]
for r in spread:
    print('spread %4d %s ncand=%d base=%s best=%.4f' % (r['i'], r['key'], r['ncand'], r['base'], r['best']))

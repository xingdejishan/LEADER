"""R4 descriptive counts used in the write-up (no scoring)."""
import glob
import json
from collections import Counter

import numpy as np

OUT = '/data2/zhangcy/r4-full-20261019-v1'
rows = []
for line in open(OUT + '/diagnostics_main.jsonl'):
    if line.strip():
        rows.append(json.loads(line))
n = len(rows)
print('frames', n)
print('reference-set size:', Counter(len(r['ref_ids']) for r in rows))
refdates = Counter()
for r in rows:
    refdates[tuple(sorted({k.split('/')[0] for k in r['ref_keys']}))] += 1
print('reference date combos:', dict(sorted(refdates.items())))
print('query points: min=%d ; frames with 4000: %d (%.1f%%) ; median=%d'
      % (min(r['query_points'] for r in rows),
         sum(r['query_points'] == 4000 for r in rows),
         100.0 * np.mean([r['query_points'] == 4000 for r in rows]),
         int(np.median([r['query_points'] for r in rows]))))
ident = sum(1 for r in rows if r['s'] == 0.0 and r['d'] == 0.0)
print('selected (s,d) == (0,0): %d (%.1f%%)' % (ident, 100.0 * ident / n))
print('u source x date:')
c = Counter((r['date'], r['u_source']) for r in rows)
for d in sorted({r['date'] for r in rows}):
    print('  %s deploy_quad=%4d ref_quad=%4d' % (d, c[(d, 'deploy_quad')], c[(d, 'ref_quad')]))
print('abstain flag any:', sum(1 for r in rows if r['abstain']))
print('window chord percentiles (deploy_quad):',
      np.round(np.percentile([r['u_meta']['chord'] for r in rows
                              if r['u_source'] == 'deploy_quad'], [5, 50, 95]), 1))
print('refine: updates>0 frames %d/%d ; mean moved %.3f m'
      % (sum(1 for r in rows if (r.get('refine_evidence') or {}).get('updates', 0) > 0),
         n, np.mean([r.get('refine_moved') or 0.0 for r in rows])))
print('refine reasons:', Counter((r.get('refine_evidence') or {}).get('reason') for r in rows))

import glob, json, sys
import numpy as np
sys.path.insert(0, '/home/zhangcy/projects/r-unified-eval-20261018-v1')
import score_unified as scorer

V = '/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/verify-frozen_nopruneoptREF'
world = scorer.load_world()
truth = dict(zip(world['hard_keys'], world['hard_gt']))

files = sorted(glob.glob(V + '/*.json'))
rows = []
for path in files:
    d = json.load(open(path))
    key = d['key']
    if key not in truth:
        continue
    gt = truth[key][None]
    cands = []
    for name, value in d['poses'].items():
        if not name.startswith('candidate_'):
            continue
        pose = value.get('pose')
        if pose is None:
            continue
        p = np.asarray(pose)[None]
        t = float(scorer.pos_err(p, gt)[0])
        r = float(scorer.rot_deg(p[:, :3, :3], gt[:, :3, :3])[0])
        score = (value.get('fair') or {}).get('geometry_score')
        cands.append(dict(name=name, t=t, r=r, ok=(t <= 1 and r <= 2), near=(t <= 10),
                          score=score if score is not None else np.nan))
    rows.append(dict(key=key, cands=cands, nn=d.get('nearest_neighbours')))

n = len(rows)
with_cand = [r for r in rows if r['cands']]
print('frames', n, 'with candidates', len(with_cand))
counts = [len(r['cands']) for r in rows]
print('candidates per frame: mean %.2f median %d min %d max %d' % (
    np.mean(counts), int(np.median(counts)), min(counts), max(counts)))

any_ok = sum(1 for r in rows if any(c['ok'] for c in r['cands']))
any_near = sum(1 for r in rows if any(c['near'] for c in r['cands']))
print('frames with >=1 good candidate (<=1m,<=2deg): %d (%.1f%%)' % (any_ok, 100.0 * any_ok / n))
print('frames with >=1 near candidate (<=10m):       %d (%.1f%%)' % (any_near, 100.0 * any_near / n))
print('frames with no candidate at all: %d' % sum(1 for r in rows if not r['cands']))

# does the geometry score pick the good candidate?
top1_hit = 0
top1_near = 0
gaps = []
for r in rows:
    scored = [c for c in r['cands'] if not np.isnan(c['score'])]
    if not scored or not any(c['ok'] for c in scored):
        continue
    best = max(scored, key=lambda c: c['score'])
    top1_hit += int(best['ok'])
    top1_near += int(best['near'])
    good = sorted([c['score'] for c in scored if c['ok']], reverse=True)[0]
    bad = [c['score'] for c in scored if not c['ok']]
    if bad:
        gaps.append(good - max(bad))
usable = [r for r in rows if r['cands'] and any(c['ok'] for c in r['cands'])]
print('of the %d frames whose pool contains a good candidate, the top-scoring one is good in %d (%.1f%%)'
      % (len(usable), top1_hit, 100.0 * top1_hit / max(1, len(usable))))
gaps = np.asarray(gaps)
if len(gaps):
    print('score gap (best good - best bad): median %.4f  p10 %.4f  p90 %.4f  negative %d/%d' % (
        np.median(gaps), np.percentile(gaps, 10), np.percentile(gaps, 90),
        int((gaps < 0).sum()), len(gaps)))

# separability: can the best geometry score tell "pool contains a good candidate" apart?
pos = [max([c['score'] for c in r['cands'] if not np.isnan(c['score'])], default=np.nan)
       for r in rows if any(c['ok'] for c in r['cands'])]
neg = [max([c['score'] for c in r['cands'] if not np.isnan(c['score'])], default=np.nan)
       for r in rows if not any(c['ok'] for c in r['cands'])]
pos = np.asarray([v for v in pos if not np.isnan(v)])
neg = np.asarray([v for v in neg if not np.isnan(v)])
if len(pos) and len(neg):
    auc = np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg])
    print('AUC of best geometry score for "pool has a good candidate": %.3f  (pos n=%d, neg n=%d)'
          % (auc, len(pos), len(neg)))
    print('  pos median %.4f  neg median %.4f' % (np.median(pos), np.median(neg)))

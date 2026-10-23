"""Which stage actually determined the final pose, over the produced frames (read-only)."""
import collections
import json

P = '/home/zhangcy/projects/e2e-moe-20261020-v1/results/'
FILES = ['full-shard0.jsonl', 'fullb-shard1.jsonl', 'fullb-shard3.jsonl']

rows = []
for name in FILES:
    try:
        with open(P + name) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get('skipped') or not r.get('seconds'):
                    continue
                rows.append(r)
    except IOError:
        pass
n = len(rows)
print('frames produced: %d' % n)

rounds = collections.Counter(int(r.get('rounds', 0)) for r in rows)
status = collections.Counter(str(r.get('status')) for r in rows)
print('refine rounds histogram:', dict(sorted(rounds.items())))
print('refine status:', dict(status))

zero_rounds = sum(1 for r in rows if int(r.get('rounds', 0)) == 0)
print('\n纯 LEADER（精修 0 轮，T 未改）: %d (%.1f%%)'
      % (zero_rounds, 100.0 * zero_rounds / n))

moved = [r for r in rows if int(r.get('rounds', 0)) > 0]
print('走 V7 精修（至少一轮）: %d (%.1f%%)' % (len(moved), 100.0 * len(moved) / n))

def moved_t(r):
    s, d = r.get('s'), r.get('d')
    return (s not in (None, 0.0)) or (d not in (None, 0.0))

only_rot = sum(1 for r in rows if r.get('rotated') and not moved_t(r))
trans = sum(1 for r in rows if moved_t(r) and not r.get('rotated'))
both = sum(1 for r in rows if moved_t(r) and r.get('rotated'))
neither = sum(1 for r in rows if (not r.get('rotated')) and (not moved_t(r)))
print('\n几何段是否改动:')
print('  平移与旋转都动 : %6d (%.1f%%)' % (both, 100.0 * both / n))
print('  只动平移       : %6d (%.1f%%)' % (trans, 100.0 * trans / n))
print('  只动旋转       : %6d (%.1f%%)' % (only_rot, 100.0 * only_rot / n))
print('  完全没动       : %6d (%.1f%%)' % (neither, 100.0 * neither / n))
print('\n最终位姿来源:')
print('  纯 LEADER      : %6d (%.1f%%)' % (zero_rounds, 100.0 * zero_rounds / n))
print('  V7 精修        : %6d (%.1f%%)' % (len(moved), 100.0 * len(moved) / n))
print('  其中被几何段再改动: %6d (%.1f%%)'
      % (both + trans + only_rot, 100.0 * (both + trans + only_rot) / n))
print('  几何段原样保留 V7 : %6d (%.1f%%)' % (neither, 100.0 * neither / n))

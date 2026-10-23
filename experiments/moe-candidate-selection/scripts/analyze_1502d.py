import json, os, collections
import numpy as np

z = np.load('/home/zhangcy/projects/e2e-moe-20261020-v1/inputs/initial.npz', allow_pickle=True)
keys = [str(x) for x in z['keys']]
bydate = collections.defaultdict(list)
for k in keys:
    d, ts = k.split('/')
    bydate[d].append(ts)

BASE = '/data2/zhangcy/leader-v2-full-trajectory-20261001/NCLT'
total_in = 0
for d in sorted(bydate):
    m = json.load(open(os.path.join(BASE, d, 'manifest.json')))
    frames = m['frames']
    have = set(str(x['timestamp_us']) for x in frames)
    want = set(bydate[d])
    hit = want & have
    total_in += len(hit)
    print(d, 'split', m.get('split'), 'frame_count', m.get('frame_count'), 'want', len(want), 'hit', len(hit))
    print('   rec0', repr(frames[0])[:260])
    if want - have:
        print('   miss_examples', sorted(want - have)[:3])
print('TOTAL_IN_MANIFEST', total_in, 'of', len(keys))

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
    print(d, 'split', m.get('split'), 'frame_count', m.get('frame_count'), 'frames_len', len(frames), 'rec0keys', list(frames[0].keys())[:12])
    # find timestamp field
    f0 = frames[0]
    tskey = 'timestamp' if 'timestamp' in f0 else ('frame_key' if 'frame_key' in f0 else None)
    print('   ts field', tskey, 'example', repr(f0.get('timestamp') or f0.get('frame_key'))[:80])
    if tskey == 'timestamp':
        have = set(str(x['timestamp']) for x in frames)
    else:
        have = set(str(x['frame_key']).split('/')[-1] for x in frames)
    want = set(bydate[d])
    hit = want & have
    total_in += len(hit)
    print('   want', len(want), 'hit', len(hit), 'miss', len(want) - len(hit))
    if want - have:
        print('   miss_examples', sorted(want - have)[:3])
print('TOTAL_IN_MANIFEST', total_in, 'of', len(keys))

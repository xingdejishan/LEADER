"""Shape of the tail: are the failures bursts or isolated frames? (read-only)"""
import json
import os

import numpy as np

TRUTH = '/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth'
z = np.load('/home/zhangcy/.dsh_scratch/sweep_full.npz', allow_pickle=True)
keys = [str(k) for k in z['keys']]
t0 = np.asarray(z['t0'], float)

truth = {}
for f in os.listdir(TRUTH):
    d = f.replace('.json', '')
    for ts, v in json.load(open(os.path.join(TRUTH, f))).items():
        truth[(d, str(ts))] = np.asarray(v['T_WB'], float)

order = sorted(range(len(keys)), key=lambda i: (keys[i].split('/')[0], int(keys[i].split('/')[1])))
err = np.array([np.linalg.norm(t0[i][:3, 3] - truth[tuple(keys[i].split('/'))][:3, 3]) for i in order])
dates = np.array([keys[i].split('/')[0] for i in order])
ts = np.array([int(keys[i].split('/')[1]) for i in order])

for thr in (1.0, 2.0, 0.5):
    mask = err > thr
    runs = []
    start = None
    for i in range(len(mask)):
        if mask[i] and start is None:
            start = i
        same_date = i + 1 < len(mask) and dates[i + 1] == dates[i]
        if start is not None and (not mask[i + 1] if i + 1 < len(mask) else True):
            pass
        if start is not None and (i + 1 >= len(mask) or not mask[i + 1] or dates[i + 1] != dates[i]):
            runs.append((start, i))
            start = None
    lens = np.array([b - a + 1 for a, b in runs])
    print('\nthreshold > %.1f m: %d frames in %d bursts' % (thr, int(mask.sum()), len(runs)))
    if len(lens):
        print('  burst length: median %d, p90 %d, max %d frames' % (np.median(lens), np.percentile(lens, 90), lens.max()))
        print('  share of failing frames inside bursts >=5 frames: %.0f%%'
              % (100.0 * lens[lens >= 5].sum() / max(lens.sum(), 1)))
        print('  frames per burst >=5: %d' % int((lens >= 5).sum()))
        by = {}
        for (a, b) in runs:
            if b - a + 1 >= 5:
                by[dates[a]] = by.get(dates[a], 0) + (b - a + 1)
        print('  long bursts by date:', {k: int(v) for k, v in sorted(by.items())})
        # duration of long bursts
        durs = [(ts[b] - ts[a]) / 1e6 for a, b in runs if b - a + 1 >= 5]
        if durs:
            print('  long-burst duration: median %.1f s, max %.1f s' % (np.median(durs), max(durs)))

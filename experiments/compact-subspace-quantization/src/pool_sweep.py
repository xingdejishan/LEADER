"""Sweep closed-form pooling descriptors for cross-date retrieval on the same folds.

Fixes the earlier artefact: no per-token channel centring.  Variants differ only in how
the 32x32x64 code grid is aggregated and how the descriptor is normalised.
"""
import json

import numpy as np

DATES = ('2012-01-22', '2012-02-02', '2012-02-18', '2012-05-11')
CACHE = '/home/zhangcy/projects/nclt-compact-encoder-20261023-v1/retr_cache'
RADIUS_M = 2.0
HEADING_DEG = 10.0


def positives(qpos, qyaw, rpos, ryaw):
    dpos = np.linalg.norm(rpos[None, :, :] - qpos[:, None, :], axis=2)
    dyaw = np.abs((ryaw[None, :] - qyaw[:, None] + np.pi / 2) % np.pi - np.pi / 2)
    return (dpos < RADIUS_M) & (np.degrees(dyaw) < HEADING_DEG)


def l2(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def gem(x, p):
    y = np.maximum(x, 0.0) ** p
    return (y.mean((1, 2)) + 1e-8) ** (1.0 / p)


def describe(feats, how):
    x = feats.astype(np.float32)
    if how == 'mean':
        d = x.mean((1, 2))
    elif how == 'max':
        d = x.max((1, 2))
    elif how == 'meanmax':
        d = np.concatenate([x.mean((1, 2)), x.max((1, 2))], axis=1)
    elif how.startswith('gem'):
        d = gem(x, float(how[3:]))
    else:
        raise SystemExit('unknown pooling ' + how)
    return d


def recall(mask, sim):
    order = np.argsort(-sim, axis=1)
    return {k: float(np.take_along_axis(mask, order[:, :k], axis=1).any(1).mean())
            for k in (1, 5, 10, 50)}


def main():
    data = {}
    for date in DATES:
        z = np.load(CACHE + '/' + date + '.npz')
        data[date] = dict(feats=z['feats'], pos=z['pos'], yaw=z['yaw'], keys=z['keys'])
    rng = np.random.default_rng(11)
    hows = ['mean', 'max', 'meanmax', 'gem2', 'gem3', 'gem5']
    folds = {}
    for held in DATES:
        train = [d for d in DATES if d != held]
        Q = data[held]
        sel = rng.choice(len(Q['keys']), size=min(300, len(Q['keys'])), replace=False)
        mask = positives(Q['pos'][sel], Q['yaw'][sel],
                         np.concatenate([data[d]['pos'] for d in train]),
                         np.concatenate([data[d]['yaw'] for d in train]))
        row = {}
        for how in hows:
            q = describe(Q['feats'][sel], how)
            r = describe(np.concatenate([data[d]['feats'] for d in train]), how)
            mu = r.mean(0, keepdims=True)
            row[how] = recall(mask, l2(q - mu) @ l2(r - mu).T)
        folds[held] = row
        print('held=%s ' % held + ' '.join('%s R@1=%.3f R@10=%.3f R@50=%.3f |' % (
            k, v[1], v[10], v[50]) for k, v in row.items()), flush=True)
    summary = {how: {k: float(np.mean([folds[d][how][k] for d in DATES])) for k in (1, 5, 10, 50)}
               for how in hows}
    print(json.dumps(summary, indent=1))
    json.dump(summary, open('/home/zhangcy/projects/nclt-compact-encoder-20261023-v1/retr/pool_sweep.json', 'w'), indent=2)


if __name__ == '__main__':
    main()

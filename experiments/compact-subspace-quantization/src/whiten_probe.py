"""Retrieval baselines on the same splits: token-space PCA-whitening, then pooling.

Whitening is applied per token (64x64 covariance, cheap and stable) before the spatial
mean, which is the standard way to fight hubness in cross-date retrieval.  The full
8192-dim flattened covariance is never formed.
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


def pool(feats, whiten=None):
    x = feats.astype(np.float32)
    x = x - x.mean(2, keepdims=True)
    if whiten is not None:
        x = x @ whiten
    return x.mean((1, 2))


def fit_whitening(arrays, dim, eps=1e-2):
    sample = []
    for a in arrays:
        flat = a.astype(np.float32).reshape(-1, a.shape[-1])
        idx = np.random.default_rng(0).choice(len(flat), size=min(20000, len(flat)), replace=False)
        sample.append(flat[idx])
    x = np.concatenate(sample) - np.concatenate(sample).mean(0)
    cov = np.cov(x.T) + eps * np.eye(x.shape[1])
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1][:dim]
    return vectors[:, order] / np.sqrt(values[order] + 1e-12)


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
    folds = {}
    for held in DATES:
        train = [d for d in DATES if d != held]
        Tfeats = np.concatenate([data[d]['feats'] for d in train])
        Tpos = np.concatenate([data[d]['pos'] for d in train])
        Tyaw = np.concatenate([data[d]['yaw'] for d in train])
        Q = data[held]
        sel = rng.choice(len(Q['keys']), size=min(300, len(Q['keys'])), replace=False)
        Qfeats = Q['feats'][sel]
        mask = positives(Q['pos'][sel], Q['yaw'][sel], Tpos, Tyaw)
        ceiling = float(mask.any(1).mean())
        result = {}
        base = l2(pool(Qfeats))
        ref = l2(pool(Tfeats))
        result['mean'] = recall(mask, base @ ref.T)
        for dim in (16, 32, 64):
            W = fit_whitening([data[d]['feats'] for d in train], dim)
            q = l2(pool(Qfeats, W))
            r = l2(pool(Tfeats, W))
            result['white%d' % dim] = recall(mask, q @ r.T)
        folds[held] = dict(ceiling=ceiling, **result)
        print('held=%s ceiling=%.3f ' % (held, ceiling) + ' '.join(
            '%s R@1=%.3f R@10=%.3f R@50=%.3f |' % (k, v[1], v[10], v[50]) for k, v in result.items()),
            flush=True)
    summary = {'ceiling': float(np.mean([folds[d]['ceiling'] for d in DATES]))}
    for key in ('mean', 'white16', 'white32', 'white64'):
        summary[key] = {k: float(np.mean([folds[d][key][k] for d in DATES])) for k in (1, 5, 10, 50)}
    print(json.dumps(summary, indent=1))
    json.dump(dict(folds=folds, summary=summary),
              open('/home/zhangcy/projects/nclt-compact-encoder-20261023-v1/retr/whiten_report.json', 'w'),
              indent=2)


if __name__ == '__main__':
    main()

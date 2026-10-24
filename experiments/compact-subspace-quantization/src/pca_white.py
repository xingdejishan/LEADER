"""PCA-whitened mean-pooled descriptor via the Gram trick (no 8192x8192 covariance)."""
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


def describe(feats):
    d = feats.astype(np.float32).mean((1, 2))
    return d - d.mean(1, keepdims=True)


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
        Xr = describe(np.concatenate([data[d]['feats'] for d in train]))
        Tpos = np.concatenate([data[d]['pos'] for d in train])
        Tyaw = np.concatenate([data[d]['yaw'] for d in train])
        Q = data[held]
        sel = rng.choice(len(Q['keys']), size=min(300, len(Q['keys'])), replace=False)
        Xq = describe(Q['feats'][sel])
        mask = positives(Q['pos'][sel], Q['yaw'][sel], Tpos, Tyaw)
        row = {}
        row['plain'] = recall(mask, l2(Xq) @ l2(Xr).T)
        mu = Xr.mean(0)
        Xc = Xr - mu
        G = Xc @ Xc.T
        vals, vecs = np.linalg.eigh(G)
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        for k in (128, 256, 512):
            kk = min(k, len(vals))
            scale = 1.0 / np.sqrt(np.maximum(vals[:kk], 1e-8))
            proj_r = (vecs[:, :kk] * scale)          # N x kk
            emb_r = l2(Xr @ (Xc.T @ proj_r))         # whitened coordinates of refs
            emb_q = l2(Xq @ (Xc.T @ proj_r))
            row['white%d' % kk] = recall(mask, emb_q @ emb_r.T)
        folds[held] = row
        print('held=%s ' % held + ' '.join('%s R@1=%.3f R@10=%.3f R@50=%.3f |' % (
            k, v[1], v[10], v[50]) for k, v in row.items()), flush=True)
    keys = list(folds[DATES[0]])
    summary = {k: {m: float(np.mean([folds[d][k][m] for d in DATES])) for m in (1, 5, 10, 50)}
               for k in keys}
    print(json.dumps(summary, indent=1))
    json.dump(summary, open('/home/zhangcy/projects/nclt-compact-encoder-20261023-v1/retr/pca_white.json', 'w'), indent=2)


if __name__ == '__main__':
    main()

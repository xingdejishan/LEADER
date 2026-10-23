"""Cross-date retrieval probe with three descriptor variants and pool-mean centering."""
import json
import os

import numpy as np

DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001'
SAM = os.path.join(DATA, 'sam')
DATES = ('2012-01-22', '2012-02-02', '2012-02-18', '2012-05-11')
PER_DATE = 1200
PER_QUERY = 300
RADIUS_M = 2.0
HEADING_DEG = 10.0


def main():
    basis = dict(np.load('/home/zhangcy/projects/nclt-feat-bottleneck-20261023-v1/results/basis.npz'))
    mean = basis['mean'].astype(np.float32)
    W = basis['pca64_int8'].astype(np.float32)
    rng = np.random.default_rng(20261023)

    cache = {}
    for date in DATES:
        truth = json.load(open(os.path.join(DATA, 'train-truth', date + '.json')))
        keys = sorted(truth, key=lambda k: int(k))
        manifest = json.load(open(os.path.join(SAM, date, 'cache_manifest.json')))['frames']
        pick = np.sort(rng.choice(len(keys), size=min(PER_DATE, len(keys)), replace=False))
        sam_mean, pca_mean, pca_norm = [], [], []
        pos, yaw, kept = [], [], []
        for i in pick:
            entry = manifest.get(date + '/' + keys[i])
            path = os.path.join(SAM, date, entry['cache_file']) if entry else None
            if path is None or not os.path.exists(path):
                continue
            x = np.load(path).astype(np.float32).reshape(-1, 256)
            norms = np.linalg.norm(x, axis=1)
            keep = norms > 0.05 * np.median(norms)
            xn = x[keep]
            a = xn.mean(0); a = a / (np.linalg.norm(a) + 1e-8)
            sam_mean.append(a)
            code = (xn - mean) @ W
            b = code.mean(0); b = b / (np.linalg.norm(b) + 1e-8)
            pca_mean.append(b)
            cn = code - code.mean(0)
            cn = cn / (np.linalg.norm(cn, axis=1, keepdims=True) + 1e-8)
            c = cn.mean(0); c = c / (np.linalg.norm(c) + 1e-8)
            pca_norm.append(c)
            T = np.asarray(truth[keys[i]]['T_WB'])
            pos.append(T[:3, 3]); yaw.append(np.arctan2(T[1, 0], T[0, 0]))
            kept.append(keys[i])
        cache[date] = dict(sam=np.stack(sam_mean), pca=np.stack(pca_mean),
                           pca_norm=np.stack(pca_norm), pos=np.stack(pos),
                           yaw=np.array(yaw), keys=kept)
        print('cached', date, len(kept), flush=True)

    for held in DATES:
        refs = [d for d in DATES if d != held]
        RP = np.concatenate([cache[d]['pos'] for d in refs])
        RY = np.concatenate([cache[d]['yaw'] for d in refs])
        Q = cache[held]
        n = min(PER_QUERY, len(Q['keys']))
        sel = rng.choice(len(Q['keys']), size=n, replace=False)
        qpos, qyaw = Q['pos'][sel], Q['yaw'][sel]
        dpos = np.linalg.norm(RP[None, :, :] - qpos[:, None, :], axis=2)
        dyaw = np.abs((RY[None, :] - qyaw[:, None] + np.pi / 2) % np.pi - np.pi / 2)
        pos_mask = (dpos < RADIUS_M) & (np.degrees(dyaw) < HEADING_DEG)
        ceiling = float(pos_mask.any(1).mean())
        for kind in ('sam', 'pca', 'pca_norm'):
            QF = Q[kind][sel]
            RF = np.concatenate([cache[d][kind] for d in refs])
            mu = np.concatenate([QF, RF]).mean(0)
            q = QF - mu; q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
            r = RF - mu; r = r / (np.linalg.norm(r, axis=1, keepdims=True) + 1e-8)
            sim = q @ r.T
            order = np.argsort(-sim, axis=1)
            hits = {k: float(np.take_along_axis(pos_mask, order[:, :k], axis=1).any(1).mean())
                    for k in (1, 5, 10, 50)}
            print('%-9s held=%s n=%d ceiling=%.3f R@1=%.3f R@5=%.3f R@10=%.3f R@50=%.3f  mean_sim=%.3f'
                  % (kind, held, n, ceiling, hits[1], hits[5], hits[10], hits[50], float(sim.mean())),
                  flush=True)


if __name__ == '__main__':
    main()

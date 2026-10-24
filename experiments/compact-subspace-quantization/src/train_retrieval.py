"""Train a dedicated cross-date retrieval descriptor on top of the 64-channel code.

Baseline (previous probe): mean-pooled code gives R@10 = 0.205, R@50 = 0.299 against a
0.664 ceiling.  Here a small head turns the 32x32 subsampled code grid into a 32-d L2
embedding, trained with InfoNCE where positives are frames of another date within 2 m and
10 deg.  Leave-one-date-out: train on three dates, evaluate on the fourth, exactly as the
pooled baseline was measured.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001'
SAM = os.path.join(DATA, 'sam')
DATES = ('2012-01-22', '2012-02-02', '2012-02-18', '2012-05-11')
RADIUS_M = 2.0
HEADING_DEG = 10.0
STRIDE = 2


def load_poses(date):
    truth = json.load(open(os.path.join(DATA, 'train-truth', date + '.json')))
    keys = sorted(truth, key=lambda k: int(k))
    pos = np.array([np.asarray(truth[k]['T_WB'])[:3, 3] for k in keys])
    yaw = np.array([np.arctan2(np.asarray(truth[k]['T_WB'])[1, 0],
                               np.asarray(truth[k]['T_WB'])[0, 0]) for k in keys])
    return keys, pos, yaw


def build_cache(cache_dir, per_date):
    basis = dict(np.load('/home/zhangcy/projects/nclt-feat-bottleneck-20261023-v1/results/basis.npz'))
    mean = basis['mean'].astype(np.float32)
    W = basis['pca64_int8'].astype(np.float32)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261023)
    for date in DATES:
        out = cache_dir / (date + '.npz')
        if out.exists():
            continue
        keys, pos, yaw = load_poses(date)
        manifest = json.load(open(os.path.join(SAM, date, 'cache_manifest.json')))['frames']
        pick = np.sort(rng.choice(len(keys), size=min(per_date, len(keys)), replace=False))
        feats, fe_pos, fe_yaw, kept = [], [], [], []
        for i in pick:
            key = date + '/' + keys[i]
            entry = manifest.get(key)
            path = os.path.join(SAM, date, entry['cache_file']) if entry else None
            if path is None or not os.path.exists(path):
                continue
            x = np.load(path).astype(np.float32).reshape(-1, 256)
            code = ((x - mean) @ W).reshape(64, 64, -1)
            code = code[::STRIDE, ::STRIDE]
            feats.append(code.astype(np.float16))
            fe_pos.append(pos[i])
            fe_yaw.append(yaw[i])
            kept.append(key)
        np.savez(out, feats=np.stack(feats), pos=np.stack(fe_pos), yaw=np.array(fe_yaw),
                 keys=np.array(kept))
        print('cached', date, len(kept), flush=True)


def positives(qpos, qyaw, rpos, ryaw):
    dpos = np.linalg.norm(rpos[None, :, :] - qpos[:, None, :], axis=2)
    dyaw = np.abs((ryaw[None, :] - qyaw[:, None] + np.pi / 2) % np.pi - np.pi / 2)
    return (dpos < RADIUS_M) & (np.degrees(dyaw) < HEADING_DEG)


def make_head(torch, channels, embed=32):
    import torch.nn as nn
    import torch.nn.functional as F

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.tower = nn.Sequential(
                nn.Conv2d(channels, 128, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1, groups=16), nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
            self.head = nn.Conv2d(128, embed, 1)

        def forward(self, x):
            if x.dim() == 4 and x.shape[-1] == 64:
                x = x.permute(0, 3, 1, 2)
            z = self.tower(x)
            z = self.head(z).flatten(2)
            z = z.mean(-1)
            return F.normalize(z, dim=-1)
    return Head()


def evaluate(torch, head, device, qf, qpos, qyaw, rf, rpos, ryaw, ks=(1, 5, 10, 50)):
    with torch.no_grad():
        qe = head(torch.from_numpy(qf.astype(np.float32)).to(device)).cpu().numpy()
        re = head(torch.from_numpy(rf.astype(np.float32)).to(device)).cpu().numpy()
    sim = qe @ re.T
    order = np.argsort(-sim, axis=1)
    mask = positives(qpos, qyaw, rpos, ryaw)
    ceiling = float(mask.any(1).mean())
    out = {k: float(np.take_along_axis(mask, order[:, :k], axis=1).any(1).mean()) for k in ks}
    return ceiling, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--per-date', type=int, default=1500)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--batch', type=int, default=96)
    parser.add_argument('--gpu', default='3')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import torch
    torch.set_num_threads(4)
    device = 'cuda'

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = '/home/zhangcy/projects/nclt-compact-encoder-20261023-v1/retr_cache'
    build_cache(cache_dir, args.per_date)

    data = {}
    for date in DATES:
        z = np.load(os.path.join(cache_dir, date + '.npz'))
        data[date] = dict(feats=z['feats'], pos=z['pos'], yaw=z['yaw'], keys=z['keys'])
        print(date, data[date]['feats'].shape, flush=True)

    rng = np.random.default_rng(7)
    results = {}
    for held in DATES:
        train_dates = [d for d in DATES if d != held]
        # pooled baseline on the same split
        Q = data[held]
        Rf = np.concatenate([data[d]['feats'] for d in train_dates])
        Rpos = np.concatenate([data[d]['pos'] for d in train_dates])
        Ryaw = np.concatenate([data[d]['yaw'] for d in train_dates])
        sel = rng.choice(len(Q['keys']), size=min(300, len(Q['keys'])), replace=False)
        qf, qpos, qyaw = Q['feats'][sel], Q['pos'][sel], Q['yaw'][sel]
        base_q = qf.astype(np.float32).reshape(len(qf), -1)
        base_r = Rf.astype(np.float32).reshape(len(Rf), -1)
        base_q = base_q - base_q.mean(1, keepdims=True)
        base_r = base_r - base_r.mean(1, keepdims=True)
        base_q /= np.linalg.norm(base_q, axis=1, keepdims=True) + 1e-8
        base_r /= np.linalg.norm(base_r, axis=1, keepdims=True) + 1e-8
        mask = positives(qpos, qyaw, Rpos, Ryaw)
        order = np.argsort(-(base_q @ base_r.T), axis=1)
        base = {k: float(np.take_along_axis(mask, order[:, :k], axis=1).any(1).mean())
                for k in (1, 5, 10, 50)}

        head = make_head(torch, data[held]['feats'].shape[-1]).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
        # training pool: all frames of the three training dates
        Tf = np.concatenate([data[d]['feats'] for d in train_dates])
        Tpos = np.concatenate([data[d]['pos'] for d in train_dates])
        Tyaw = np.concatenate([data[d]['yaw'] for d in train_dates])
        Tdate = np.concatenate([[i] * len(data[d]['keys']) for i, d in enumerate(train_dates)])
        pair_q, pair_p = [], []
        for i in range(len(Tf)):
            d = np.linalg.norm(Tpos - Tpos[i], axis=1)
            dy = np.abs((Tyaw - Tyaw[i] + np.pi / 2) % np.pi - np.pi / 2)
            ok = (d < RADIUS_M) & (np.degrees(dy) < HEADING_DEG) & (Tdate != Tdate[i])
            cand = np.flatnonzero(ok)
            if len(cand):
                pair_q.append(i)
                pair_p.append(int(rng.choice(cand)))
        pair_q = np.array(pair_q); pair_p = np.array(pair_p)
        print('fold %s pairs %d' % (held, len(pair_q)), flush=True)
        idx_all = torch.from_numpy(Tf.astype(np.float32))
        for epoch in range(args.epochs):
            perm = rng.permutation(len(pair_q))
            for step in range(args.steps):
                if step * args.batch >= len(perm):
                    break
                take = perm[step * args.batch:(step + 1) * args.batch]
                qi, pi = pair_q[take], pair_p[take]
                batch = np.concatenate([qi, pi])
                labels = torch.arange(len(qi)).to(device)
                x = idx_all[batch].to(device)
                emb = head(x)
                logits = emb[:len(qi)] @ emb[len(qi):].T / 0.07
                loss = torch.nn.functional.cross_entropy(logits, labels) + \
                    torch.nn.functional.cross_entropy(logits.T, labels)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        ceiling, rec = evaluate(torch, head, device, qf, qpos, qyaw, Rf, Rpos, Ryaw)
        results[held] = dict(ceiling=ceiling, base=base, head=rec)
        print('held=%s ceiling=%.3f  base R@1=%.3f R@10=%.3f R@50=%.3f | head R@1=%.3f R@10=%.3f R@50=%.3f'
              % (held, ceiling, base[1], base[10], base[50], rec[1], rec[10], rec[50]), flush=True)
        torch.save(dict(state=head.state_dict(), stride=STRIDE), out / ('head_%s.pt' % held))

    summary = {}
    for key in ('base', 'head'):
        summary[key] = {k: float(np.mean([results[d][key][k] for d in DATES])) for k in (1, 5, 10, 50)}
    summary['ceiling'] = float(np.mean([results[d]['ceiling'] for d in DATES]))
    json.dump(dict(folds=results, summary=summary), open(out / 'retrieval_report.json', 'w'), indent=2)
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()

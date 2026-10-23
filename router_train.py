"""Fit the three-way router from the per-frame feature table.

Input: /tmp/rt/router-dump.npz + router-dump_diag.jsonl, produced by router_dump.py
(three arm poses per frame plus the 43 features).  Protocol: leave-one-date-out over
the four dates, standardisation and the epoch count fitted on the training dates only
(inner chronological 80/20 split), three seeds averaged, argmax at inference.

    --cw 0 --stakes 1 --t0margin 0.25      round 2, the delivered weights
    --cw 1 --stakes 0 --t0margin 0        round 1, inverse-frequency class weights

Writes results/router-weights.npz (deployment: all four dates) and the held-out
products results/router-heldout-<tag>.npz / _diag.jsonl.
""""""Fit the 3-way router: leave-one-date-out, discrete argmax, honest hold-out report.

Round 1 (--cw on  --stakes off --t0margin 0)   inverse-frequency class weights, plain CE.
Round 2 (--cw off --stakes on  --t0margin .25) no class re-weighting (the CE argmax is the
    posterior mode, and the posteriors are already reasonably balanced); each frame is
    weighted by the stakes of the decision, w = clip(max_a L_a - min_a L_a, 0.05, 1), so
    near-tie frames (label noise) contribute little and the frames where the choice is
    worth hundreds of metres dominate; and T0 is only labelled when it beats both experts
    by a margin, matching "significantly worse than T0" instead of "worse by 1 mm".
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, '/tmp/rt')
sys.path.insert(0, '/home/zhangcy/projects/e2e-hypothesis-arbitration-20261023-v1')
import rt_eval as E
import router as R

OUT = '/tmp/rt'
RES = '/home/zhangcy/projects/e2e-hypothesis-arbitration-20261023-v1/results'


def load_dump():
    z = np.load(os.path.join(OUT, 'router-dump.npz'), allow_pickle=True)
    keys = [str(k) for k in z['keys']]
    rows = [json.loads(l) for l in open(os.path.join(OUT, 'router-dump_diag.jsonl'))]
    by = {str(r['key']): r for r in rows}
    rows = [by[k] for k in keys]
    P = dict(t0=np.asarray(z['t0'], float), geo=np.asarray(z['geo'], float),
             vis=np.asarray(z['vis'], float),
             vis_ok=np.asarray(z['vis_ok']).astype(bool),
             chosen=np.asarray(z['poses'], float))
    return keys, rows, P


def matrix(rows, names, log):
    X = np.empty((len(rows), len(names)), float)
    for i, r in enumerate(rows):
        f = R.extract(r)
        for j, n in enumerate(names):
            v = f.get(n, np.nan)
            if n in log and np.isfinite(v):
                v = np.log1p(max(v, 0.0))
            X[i, j] = v
    return X


def fit_scaler(X, mask):
    A = np.where(mask[:, None], X, np.nan)
    with np.errstate(all='ignore'):
        med = np.nanmedian(A, axis=0)
        q1 = np.nanpercentile(A, 25, axis=0)
        q3 = np.nanpercentile(A, 75, axis=0)
    scale = np.where(np.isfinite(q3 - q1) & (np.abs(q3 - q1) > 1e-9), q3 - q1, 1.0)
    return np.where(np.isfinite(med), med, 0.0), scale


def standardize(X, med, scale):
    Z = (X - med) / scale
    Z = np.where(np.isfinite(Z), Z, 0.0)
    return np.clip(Z, -6.0, 6.0)


def make_labels(L0, Lg, Lv, vis_ok, t0margin=0.0):
    """argmin over the arms that exist; T0 only wins when it beats both by t0margin."""
    y = np.zeros(len(L0), np.int64)
    for i in range(len(L0)):
        le = [Lg[i]]
        if vis_ok[i]:
            le.append(Lv[i])
        be = min(le)
        y[i] = 0 if (be - L0[i]) > t0margin else (1 if be == Lg[i] else 2)
    return y


def stakes(L0, Lg, Lv, vis_ok, lo=0.05, hi=1.0):
    out = np.zeros(len(L0))
    for i in range(len(L0)):
        a = [L0[i], Lg[i]]
        if vis_ok[i]:
            a.append(Lv[i])
        out[i] = np.clip(max(a) - min(a), lo, hi)
    return out


def train(Xtr, ytr, Xva, yva, seed, dims, epochs, lr, wd, cw, wsample=None):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    layers = []
    d = Xtr.shape[1]
    for h in dims:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers += [nn.Linear(d, 3)]
    net = nn.Sequential(*layers)
    nparam = int(sum(p.numel() for p in net.parameters()))
    if cw:
        cnt = np.bincount(ytr, minlength=3).astype(float)
        w = np.where(cnt > 0, cnt.sum() / (3.0 * np.maximum(cnt, 1.0)), 1.0)
    else:
        w = np.ones(3)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32),
                               reduction='mean' if wsample is None else 'none')
    sw = None if wsample is None else torch.tensor(wsample, dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    xv = torch.tensor(Xva, dtype=torch.float32)
    yv = torch.tensor(yva, dtype=torch.long)
    best, best_ce, best_ep = None, 1e18, 0
    for ep in range(1, epochs + 1):
        net.train()
        opt.zero_grad()
        out = lossf(net(xt), yt)
        if sw is not None:
            out = (out * sw).sum() / sw.sum()
        out.backward()
        opt.step()
        if ep % 20 == 0:
            net.eval()
            with torch.no_grad():
                ce = float((net(xv).argmax(1) == yv).float().mean()) if False else None
                ce = float(nn.functional.cross_entropy(net(xv), yv,
                                                       weight=torch.tensor(w, dtype=torch.float32)))
            if ce < best_ce - 1e-7:
                best = [p.detach().clone() for p in net.parameters()]
                best_ce, best_ep = ce, ep
    if best is not None:
        for p, s in zip(net.parameters(), best):
            p.data.copy_(s)
    net.eval()
    return net, nparam, best_ep


def logits(net, X):
    import torch
    with torch.no_grad():
        return net(torch.tensor(X, dtype=torch.float32)).numpy().astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--dims', default='64,64')
    ap.add_argument('--epochs', type=int, default=600)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--cw', type=int, default=0)
    ap.add_argument('--stakes', type=int, default=1)
    ap.add_argument('--t0margin', type=float, default=0.25)
    ap.add_argument('--tag', default='v2')
    ap.add_argument('--importance', type=int, default=0)
    a = ap.parse_args()
    dims = tuple(int(x) for x in a.dims.split(','))
    seeds = [int(s) for s in a.seeds.split(',')]

    keys, rows, P = load_dump()
    T = E.Truth()
    assert keys == list(T.keys)
    gt, dates, replaced = T.gt, T.dates, T.replaced
    ok = P['vis_ok']
    e0 = E.pos_err(P['t0'], gt); r0 = E.rot_deg(P['t0'][:, :3, :3], gt[:, :3, :3])
    eg = E.pos_err(P['geo'], gt); rg = E.rot_deg(P['geo'][:, :3, :3], gt[:, :3, :3])
    ev = np.full(len(gt), 1e6); rv = np.full(len(gt), 1e6)
    ev[ok] = E.pos_err(P['vis'][ok], gt[ok])
    rv[ok] = E.rot_deg(P['vis'][ok][:, :3, :3], gt[ok][:, :3, :3])
    gm, _ = E.gate_mask(keys)
    udates = sorted(set(dates.tolist()))

    res = dict(n=int(len(gt)), vis_ok=int(ok.sum()), gate_n=int(gm.sum()), dates=udates,
               date_counts={d: int((dates == d).sum()) for d in udates},
               config=dict(cw=a.cw, stakes=a.stakes, t0margin=a.t0margin, seeds=seeds,
                           dims=list(dims), epochs=a.epochs, lr=a.lr, wd=a.wd,
                           n_features=len(R.FEATURES)))
    arms = {}
    for name, pose in (('T0', P['t0']), ('T_geo', P['geo']), ('old_ksigma', P['chosen'])):
        rs, rm, _ = E.rescue(pose, gt, gm, keys)
        arms[name] = dict(all=E.metrics(pose, gt), gate=E.metrics(pose, gt, gm),
                          rescue_sum=rs, rescue_mean=rm, excluded=E.metrics(pose, gt, ~replaced))
    res['arms'] = arms

    Ls = {'both': (e0 + r0, eg + rg, ev + rv), 'trans': (e0, eg, ev), 'rot': (r0, rg, rv)}
    res['labels'] = {}
    for nm, (L0, Lg, Lv) in Ls.items():
        y = make_labels(L0, Lg, Lv, ok, a.t0margin)
        res['labels'][nm] = dict(counts=[int((y == c).sum()) for c in range(3)],
                                 frac=[float((y == c).mean()) for c in range(3)])
    L0, Lg, Lv = Ls['both']
    y = make_labels(L0, Lg, Lv, ok, a.t0margin)
    sw = stakes(L0, Lg, Lv, ok) if a.stakes else None
    pose_o = P['t0'].copy(); pose_o[y == 1] = P['geo'][y == 1]; pose_o[y == 2] = P['vis'][y == 2]
    rs_o, rm_o, _ = E.rescue(pose_o, gt, gm, keys)
    res['arms']['oracle3'] = dict(all=E.metrics(pose_o, gt), gate=E.metrics(pose_o, gt, gm),
                                  rescue_sum=rs_o, rescue_mean=rm_o,
                                  excluded=E.metrics(pose_o, gt, ~replaced))
    res['spread'] = dict(L0_mean=float(L0.mean()), Lg_mean=float(Lg.mean()),
                         Lv_mean=float(Lv[ok].mean()),
                         stakes_mean=float(np.mean(stakes(L0, Lg, Lv, ok))))

    X = matrix(rows, R.FEATURES, R.LOG_FEATURES)
    X3 = np.stack([np.log1p(np.clip([r['c_geo'] for r in rows], 0, None)),
                   np.log1p(np.clip([r.get('c_vis_ev', np.nan) for r in rows], 0, None)),
                   np.log1p(np.clip([r['arb_sigma'] for r in rows], 0, None))], 1)
    n3 = ['c_geo', 'c_vis', 'sigma_c']

    out = dict(meta=res, folds={}, control={})
    old = np.where([bool(r.get('visual')) for r in rows], 2, 1)

    def assemble(pred):
        pose = P['t0'].copy()
        pose[pred == 1] = P['geo'][pred == 1]
        pose[pred == 2] = P['vis'][pred == 2]
        return pose

    def splits(te_date):
        te = dates == te_date
        inner = np.zeros(len(gt), bool)
        for dd in udates:
            if dd == te_date:
                continue
            m = np.flatnonzero(dates == dd)
            inner[m[int(0.8 * len(m)):]] = True
        tr = ~te & ~inner
        return te, tr, inner

    def loo(X, names, log, tag):
        pred = np.zeros(len(gt), np.int64)
        conf = np.zeros(len(gt))
        folds = []
        npar = 0
        for d in udates:
            te, tr, inner = splits(d)
            med, scale = fit_scaler(X, tr)
            Xz = standardize(X, med, scale)
            vote = np.zeros((int(te.sum()), 3))
            for s in seeds:
                net, npar, bep = train(Xz[tr], y[tr], Xz[inner], y[inner], s, dims,
                                       a.epochs, a.lr, a.wd, a.cw,
                                       None if sw is None else sw[tr])
                vote += logits(net, Xz[te])
            vote /= len(seeds)
            vote[~ok[te], 2] = -1e9
            pred[te] = vote.argmax(1)
            e = np.exp(vote - vote.max(1, keepdims=True))
            conf[te] = (e / e.sum(1, keepdims=True)).max(1)
            pose_te = assemble(pred)
            folds.append(dict(date=d, n=int(te.sum()),
                              metrics=E.metrics(pose_te[te], gt[te]),
                              gate=E.metrics(pose_te, gt, gm & te),
                              acc=float((pred[te] == y[te]).mean()),
                              agree_old=float((pred[te] == old[te]).mean()),
                              dist=[int((pred[te] == c).sum()) for c in range(3)]))
        pose = assemble(pred)
        rs, rm, _ = E.rescue(pose, gt, gm, keys)
        out[tag] = dict(folds=folds, all=E.metrics(pose, gt), gate=E.metrics(pose, gt, gm),
                        excluded=E.metrics(pose, gt, ~replaced), rescue_sum=rs, rescue_mean=rm,
                        dist=[int((pred == c).sum()) for c in range(3)],
                        agree_old=float((pred == old).mean()),
                        acc_label=float((pred == y).mean()),
                        mean_conf=float(conf.mean()), nparam=int(npar))
        return pose, pred, conf

    pose_f, pred_f, conf_f = loo(X, R.FEATURES, R.LOG_FEATURES, 'folds')
    pose_3, pred_3, conf_3 = loo(X3, n3, [], 'control')
    out['cross'] = dict(agree_full_old=float((pred_f == old).mean()),
                        agree_full_three=float((pred_f == pred_3).mean()),
                        agree_three_old=float((pred_3 == old).mean()))
    # arm-by-arm outcome of the router's own choices: how bad is a wrong T_vis / T0
    Lchosen = np.where(pred_f == 0, L0, np.where(pred_f == 1, Lg, Lv))
    out['chosen_loss'] = dict(mean=float(Lchosen.mean()),
                              by_arm={R.ARMS[c]: float(Lchosen[pred_f == c].mean())
                                      for c in range(3) if (pred_f == c).any()},
                              label_hit={R.ARMS[c]: float((y[pred_f == c] == c).mean())
                                         for c in range(3) if (pred_f == c).any()})

    base = float(Lchosen.mean())
    if a.importance:
        rng = np.random.RandomState(0)
        imp = {}

        def perm_one(Xp):
            ch = np.zeros(len(gt), np.int64)
            for dd in udates:
                te, tr, inner = splits(dd)
                med, scale = fit_scaler(X, tr)
                Xz = standardize(Xp, med, scale)
                vote = np.zeros((int(te.sum()), 3))
                for s in seeds[:1]:
                    net, _, _ = train(Xz[tr], y[tr], Xz[inner], y[inner], s, dims,
                                      a.epochs, a.lr, a.wd, a.cw,
                                      None if sw is None else sw[tr])
                    vote += logits(net, Xz[te])
                vote[~ok[te], 2] = -1e9
                ch[te] = vote.argmax(1)
            return float(np.mean(np.where(ch == 0, L0, np.where(ch == 1, Lg, Lv))))
        for j, n in enumerate(R.FEATURES):
            Xp = X.copy()
            Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
            imp[n] = perm_one(Xp) - base
        out['perm_importance'] = dict(base_loss=base,
                                      ranked=sorted(imp.items(), key=lambda kv: -kv[1]))

    # deployment weights: all four dates, inner split on the last 20% of every date
    inner_all = np.zeros(len(gt), bool)
    for dd in udates:
        m = np.flatnonzero(dates == dd)
        inner_all[m[int(0.8 * len(m)):]] = True
    med, scale = fit_scaler(X, ~inner_all)
    Xz = standardize(X, med, scale)
    npar = 0
    acc = None
    for s in seeds:
        net, npar, bep = train(Xz[~inner_all], y[~inner_all], Xz[inner_all], y[inner_all],
                               s, dims, a.epochs, a.lr, a.wd, a.cw,
                               None if sw is None else sw[~inner_all])
        ps = [p.detach().numpy().copy() for p in net.parameters()]
        acc = ps if acc is None else [u + v for u, v in zip(acc, ps)]
    acc = [u / float(len(seeds)) for u in acc]
    wb = [(acc[i], acc[i + 1]) for i in range(0, len(acc), 2)]
    R.save_weights(os.path.join(RES, 'router-weights.npz'), R.FEATURES, R.LOG_FEATURES,
                   med, scale, wb)
    out['deploy'] = dict(path=os.path.join(RES, 'router-weights.npz'), nparam=int(npar),
                         dims=list(dims), seeds=seeds, epochs=a.epochs, lr=a.lr, wd=a.wd,
                         n_features=len(R.FEATURES), n_train=int((~inner_all).sum()))
    np.savez(os.path.join(RES, 'router-heldout-%s.npz' % a.tag), keys=np.array(keys), poses=pose_f)
    np.savez(os.path.join(RES, 'router-heldout-%s-three.npz' % a.tag), keys=np.array(keys),
             poses=pose_3)
    with open(os.path.join(RES, 'router-heldout-%s_diag.jsonl' % a.tag), 'w') as fh:
        for i, k in enumerate(keys):
            fh.write(json.dumps(dict(key=k, arm=R.ARMS[int(pred_f[i])], prob=float(conf_f[i]),
                                     arm3=R.ARMS[int(pred_3[i])],
                                     old_arm=R.ARMS[int(old[i])],
                                     label_arm=R.ARMS[int(y[i])],
                                     stakes=float(sw[i]) if sw is not None else 1.0,
                                     gate=bool(gm[i]))) + '\n')
    json.dump(out, open(os.path.join(OUT, 'router_%s.json' % a.tag), 'w'), indent=1)
    print(json.dumps({k: dict(MPE=v['all']['MPE'], MOE=v['all']['MOE'],
                              ok=v['all']['ok12'], over5=v['all']['over5'],
                              over10=v['all']['over10'],
                              gateMPE=v['gate']['MPE'], gateMOE=v['gate']['MOE'],
                              rescue=v['rescue_sum'], dist=v['dist'],
                              agree_old=v['agree_old'], acc=v['acc_label'])
                      for k, v in (('folds', out['folds']), ('control', out['control']))}, indent=1))
    print('CROSS', json.dumps(out['cross']))
    print('LABELS', json.dumps(out['labels']))
    print('CHOSEN', json.dumps(out['chosen_loss']))
    print('DEPLOY', json.dumps(out['deploy']))
    if a.importance:
        print('IMP', json.dumps(out['perm_importance']['ranked'][:15]))


if __name__ == '__main__':
    main()

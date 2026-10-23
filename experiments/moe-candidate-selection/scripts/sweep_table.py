"""Offline table over the geometric branch's decision points, full 104,728-frame set.

Objective (set by the project owner): minimise orientation error (MOE) subject to
position error not exceeding the front end's (MPE <= t0's MPE). ok(1m,2deg) and the
>1m / >5m counts are secondary.

Two npz inputs, both produced by the sweeps in this directory:
  sweep_full.npz  per frame: gain, t0, search pose, the rotation block on the search
                  pose at 5 margins, the rotation block on t0 at 1e-6
  sweep_rot.npz   per frame: the rotation block on t0 at 7 margins (translation never
                  moves, so MPE is the front end's by construction)

    python sweep_table.py                 # -> results/sweep.md + results/sweep-summary.npz
"""
import argparse
import json
import os
import sys

import numpy as np

PROJECT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
TRUTH = '/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth'
ONLINE = os.path.join(PROJECT, 'results', 'full-online.npz')
HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, 'sweep_full.npz')
ROT = os.path.join(HERE, 'sweep_rot.npz')
MD = os.path.join(PROJECT, 'results', 'sweep.md')
GATES = (None, 0.005, 0.01, 0.02, 0.03, 0.05)
EPS = (1e-6, 0.005, 0.01, 0.02, 0.05)
EPS2 = (1e-6, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1)


def load_truth():
    out = {}
    for f in os.listdir(TRUTH):
        date = f.replace('.json', '')
        for ts, v in json.load(open(os.path.join(TRUTH, f))).items():
            out[(date, str(ts))] = np.asarray(v['T_WB'], np.float64)
    return out


def rot_deg(a, b):
    r = np.swapaxes(a, -1, -2) @ b
    c = np.clip((np.trace(r, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(c))


def metrics(e, r):
    return dict(mpe=float(e.mean()), moe=float(r.mean()), med=float(np.median(e)),
                ok=100.0 * float(((e <= 1) & (r <= 2)).mean()),
                gt1=int((e > 1).sum()), gt5=int((e > 5).sum()), n=int(len(e)))


def md_row(name, m, touched, meets):
    return ('| %-26s | %6.1f | %6d | %.4f | %.4f | %.3f | %5.1f | %4d | %4d | %s |'
            % (name, 100.0 * touched, m['n'], m['mpe'], m['moe'], m['med'], m['ok'],
               m['gt1'], m['gt5'], 'yes' if meets else 'no'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default=NPZ)
    ap.add_argument('--rot-npz', default=ROT)
    ap.add_argument('--md', default=MD)
    ap.add_argument('--summary', default=os.path.join(PROJECT, 'results', 'sweep-summary.npz'))
    ap.add_argument('--no-write', action='store_true')
    a = ap.parse_args()

    z = np.load(a.npz, allow_pickle=True)
    keys = [str(k) for k in z['keys']]
    n = len(keys)
    gain = np.asarray(z['gain'], np.float64)
    t0 = np.asarray(z['t0'], np.float64)
    rot = np.asarray(z['P_rot'], np.float64)
    rot_t0 = np.asarray(z['P_rot_t0'], np.float64)

    truth = load_truth()
    g = np.full((n, 4, 4), np.nan)
    have = np.zeros(n, bool)
    for i, k in enumerate(keys):
        t = truth.get(tuple(k.split('/')))
        if t is not None:
            g[i] = t
            have[i] = True
    missing = int((~have).sum())

    ok = have & np.isfinite(t0[:, :3, 3]).all(1)
    e_t0 = np.linalg.norm(t0[:, :3, 3] - g[:, :3, 3], axis=1)
    r_t0 = rot_deg(t0[:, :3, :3], g[:, :3, :3])
    base = metrics(e_t0[ok], r_t0[ok])
    LIMIT = base['mpe']                      # the front end is the position budget
    print('frames %d  truth %d  missing %d  front end MPE %.6f  MOE %.4f'
          % (n, int(have.sum()), missing, base['mpe'], base['moe']))

    rows = []

    def add(name, P, touched, tag):
        m = ok & np.isfinite(P[:, :3, 3]).all(1)
        e = np.linalg.norm(P[m][:, :3, 3] - g[m][:, :3, 3], axis=1)
        r = rot_deg(P[m][:, :3, :3], g[m][:, :3, :3])
        d = metrics(e, r)
        d.update(name=name, touched=touched, tag=tag,
                 meets=bool(d['mpe'] <= LIMIT + 1e-12),
                 dmoe=d['moe'] - base['moe'], dmpe=d['mpe'] - base['mpe'])
        rows.append(d)
        return d

    # ---- pure rotation at the front-end pose: MPE untouched by construction
    rot2 = None
    if os.path.exists(a.rot_npz):
        z2 = np.load(a.rot_npz, allow_pickle=True)
        k2 = [str(x) for x in z2['keys']]
        rot2 = np.asarray(z2['rot'], np.float64)
        moved2 = np.asarray(z2['moved'], bool)
        steps2 = np.asarray(z2['steps'], np.uint8)
        if k2 != keys:
            sys.exit('rotation npz keys do not line up with the sweep npz')
        both = np.isfinite(rot_t0[:, 0, 0]) & np.isfinite(rot2[:, 0, 0, 0])
        print('pass 1 vs pass 2 rotation at t0, eps=1e-6: max |diff| = %.3e over %d frames'
              % (np.abs(rot_t0[both] - rot2[both, 0]).max() if both.any() else float('nan'),
                 int(both.sum())))
        for m_idx, eps in enumerate(EPS2):
            add('rot@t0 eps=%g' % eps, rot2[:, m_idx], float(moved2[:, m_idx][ok].mean()),
                'rotonly')
    else:
        print('missing', a.rot_npz)

    # ---- rotation on the search pose, gated or not (translation moves)
    # "rot@skip=y" uses the rotation block on t0 at the *same* margin, so the two
    # stages are compared at one eps rather than against pass 1's fixed 1e-6.
    def skip_pose_at(m_idx):
        if rot2 is not None:
            return rot2[:, list(EPS2).index(EPS[m_idx])], moved2[
                :, list(EPS2).index(EPS[m_idx])]
        return rot_t0, None

    for gate in GATES:
        run = np.isfinite(gain) & (True if gate is None else gain > gate)
        share = float(run[ok].mean())
        for rot_skip in (False, True):
            for m_idx, eps in enumerate(EPS):
                skip_pose, skip_moved = skip_pose_at(m_idx) if rot_skip else (t0, None)
                P = np.where(run[:, None, None], rot[:, m_idx], skip_pose)
                if rot_skip:
                    moved = np.where(run, True, skip_moved)
                else:
                    moved = run
                add('search%s eps=%g rot@skip=%s'
                    % ('@off' if gate is None else '@%.3f' % gate, eps,
                       'y' if rot_skip else 'n'),
                    P, float(moved[ok].mean()), 'search')

    # ---- oracle: per-frame better of t0 and the production branch
    e_p = np.linalg.norm(rot[:, 0][:, :3, 3] - g[:, :3, 3], axis=1)
    take = np.isfinite(e_p) & (e_p < e_t0)
    P_or = np.where(take[:, None, None], rot[:, 0], t0)
    mo = ok & np.isfinite(P_or[:, :3, 3]).all(1)
    oracle = metrics(np.linalg.norm(P_or[mo][:, :3, 3] - g[mo][:, :3, 3], axis=1),
                     rot_deg(P_or[mo][:, :3, :3], g[mo][:, :3, :3]))

    # ---- the online run that motivated this sweep, scored the same way
    prod_ref = None
    if os.path.exists(ONLINE):
        zo = np.load(ONLINE, allow_pickle=True)
        po = np.asarray(zo['poses'], np.float64)
        ko = [str(k) for k in zo['keys']]
        eo = np.full(len(ko), np.nan)
        ro = np.full(len(ko), np.nan)
        for i, k in enumerate(ko):
            t = truth.get(tuple(k.split('/')))
            if t is not None:
                eo[i] = np.linalg.norm(po[i][:3, 3] - t[:3, 3])
                ro[i] = rot_deg(po[i][:3, :3], t[:3, :3])
        m = np.isfinite(eo) & np.isfinite(ro)
        prod_ref = metrics(eo[m], ro[m])

    # ---- ranking under the constraint
    good = [r for r in rows if r['meets']]
    good.sort(key=lambda r: (r['moe'], abs(r['dmpe'])))
    best = good[0] if good else None
    print('\nrows meeting MPE <= %.4f: %d / %d' % (LIMIT, len(good), len(rows)))
    print('best MOE under the constraint: %s  MOE %.4f (%+.4f)  MPE %.4f (%+.4f)'
          % (best['name'], best['moe'], best['dmoe'], best['mpe'], best['dmpe']) if best else 'none')

    lines = []
    lines.append('# Geometric branch: two decision points, full 104,728-frame set')
    lines.append('')
    lines.append('Objective: minimise MOE subject to MPE <= %.4f (the front-end pose).' % LIMIT)
    lines.append('')
    lines.append('Front end (t0): MPE %.4f  MOE %.4f  median %.3f  ok %.1f%%  >1m %d  >5m %d'
                 % (base['mpe'], base['moe'], base['med'], base['ok'], base['gt1'], base['gt5']))
    if prod_ref is not None:
        lines.append('Online run (gate off, eps 1e-6, rotation on every frame): '
                     'MPE %.4f  MOE %.4f  median %.3f  ok %.1f%%  >1m %d  >5m %d'
                     % (prod_ref['mpe'], prod_ref['moe'], prod_ref['med'], prod_ref['ok'],
                        prod_ref['gt1'], prod_ref['gt5']))
    lines.append('Oracle (per-frame better of t0 and the branch): MPE %.4f  MOE %.4f  '
                 'ok %.1f%%  >1m %d  >5m %d'
                 % (oracle['mpe'], oracle['moe'], oracle['ok'], oracle['gt1'], oracle['gt5']))
    lines.append('')
    lines.append('## 1. Rotation block at the front-end pose (translation never moves)')
    lines.append('')
    lines.append('| config | touched% | n | MPE | MOE | median | ok% | >1m | >5m | MPE ok |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        if r['tag'] == 'rotonly':
            lines.append(md_row(r['name'], r, r['touched'], r['meets']))
    lines.append('')
    lines.append('## 2. Rotation on the search pose, by compute gate (translation moves)')
    lines.append('')
    lines.append('| config | touched% | n | MPE | MOE | median | ok% | >1m | >5m | MPE ok |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        if r['tag'] == 'search':
            lines.append(md_row(r['name'], r, r['touched'], r['meets']))
    lines.append('')

    # ---- orientation change by front-end rotation band, for the best pure rotation
    if rot2 is not None:
        lines.append('## 3. Where the rotation block helps or hurts (rotation at t0)')
        lines.append('')
        lines.append('| eps | front-end rot | frames | touched | MOE before | MOE after | delta '
                     '| helped | hurt |')
        lines.append('|---|---|---|---|---|---|---|---|---|')
        for eps_t in (1e-6, 0.002, 0.01, 0.05):
            m_idx = list(EPS2).index(eps_t)
            rP = rot_deg(rot2[:, m_idx, :3, :3], g[:, :3, :3])
            for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 91)):
                b = ok & (r_t0 >= lo) & (r_t0 < hi) & np.isfinite(rP)
                if b.sum() == 0:
                    continue
                dt = rP[b] - r_t0[b]
                lines.append('| %g | %.0f-%.0f deg | %d | %.1f%% | %.3f | %.3f | %+.3f | %d | %d |'
                             % (eps_t, lo, hi, int(b.sum()),
                                100.0 * moved2[b, m_idx].mean(), r_t0[b].mean(),
                                rP[b].mean(), dt.mean(), int((dt < -0.1).sum()),
                                int((dt > 0.1).sum())))
        lines.append('')

    lines.append('| summary | value |')
    lines.append('|---|---|')
    lines.append('| frames | %d |' % n)
    lines.append('| frames without truth | %d |' % missing)
    lines.append('| front-end MPE budget | %.4f |' % LIMIT)
    if best is not None:
        lines.append('| best MOE under the constraint | %s |' % best['name'])
        lines.append('| its MOE / MPE | %.4f / %.4f |' % (best['moe'], best['mpe']))
    lines.append('')

    if not a.no_write:
        with open(a.md, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
        print('wrote', a.md)

    print()
    print('\n'.join(lines[4:]))

    # ---- compact summary for the project
    if not a.no_write:
        out = dict(keys=np.array(keys), gain=np.asarray(gain, np.float32),
                   eps_rot_t0=np.array(EPS2), eps_search=np.array(EPS),
                   gates=np.array([-1.0 if x is None else x for x in GATES]),
                   names=np.array([r['name'] for r in rows]),
                   row_mpe=np.array([r['mpe'] for r in rows]),
                   row_moe=np.array([r['moe'] for r in rows]),
                   row_median=np.array([r['med'] for r in rows]),
                   row_ok=np.array([r['ok'] for r in rows]),
                   row_gt1=np.array([r['gt1'] for r in rows]),
                   row_gt5=np.array([r['gt5'] for r in rows]),
                   row_touched=np.array([r['touched'] for r in rows]),
                   row_meets=np.array([r['meets'] for r in rows]),
                   t0_mpe=base['mpe'], t0_moe=base['moe'], t0_median=base['med'],
                   t0_ok=base['ok'], t0_gt1=base['gt1'], t0_gt5=base['gt5'],
                   oracle_mpe=oracle['mpe'], oracle_moe=oracle['moe'],
                   oracle_ok=oracle['ok'], oracle_gt1=oracle['gt1'], oracle_gt5=oracle['gt5'],
                   limit_mpe=LIMIT, n_frames=n, n_missing_truth=missing,
                   recommendation=best['name'] if best else '')
        np.savez(a.summary, **out)
        print('wrote', a.summary)
    return 0


if __name__ == '__main__':
    sys.exit(main())

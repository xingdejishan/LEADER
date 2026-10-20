"""R4 report: gates, baseline comparison, per-date, damage, oracle, abstention cost, u split.

    python r4_report.py --tag main --truth truth --old old

Reads the collected npz + diagnostics and prints every number used in the write-up;
writes summary_<tag>.json.
"""
import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

OUT = Path('/data2/zhangcy/r4-full-20261019-v1')
sys.path.insert(0, '/home/zhangcy/projects/r-unified-eval-20261018-v1')
import score_unified as SU  # noqa: E402

GATE_OK = 1242      # 2 x 621 baseline frames within 1 m / 2 deg
GATE_10M = 8        # baseline count of frames with position error > 10 m


def load_tag(tag):
    z = np.load(OUT / ('poses_%s.npz' % tag), allow_pickle=True)
    zr = np.load(OUT / ('poses_%s_refined.npz' % tag), allow_pickle=True)
    diag = {}
    for p in glob.glob(str(OUT / ('diagnostics_%s.jsonl' % tag))):
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                diag[r['key']] = r
    return ([str(k) for k in z['keys']], np.asarray(z['poses'], float),
            np.asarray(zr['poses'], float), np.asarray(zr['has_refined'], bool), diag)


def metrics(w, poses, idx=None):
    gt = w['hard_gt'] if idx is None else w['hard_gt'][idx]
    base = w['hard_base'] if idx is None else w['hard_base'][idx]
    repl = w['hard_replaced'] if idx is None else w['hard_replaced'][idx]
    t = SU.pos_err(poses, gt)
    r = SU.rot_deg(poses[:, :3, :3], gt[:, :3, :3])
    bt = SU.pos_err(base, gt)
    keep = ~repl
    return dict(
        n=int(len(t)), MPE=float(t.mean()), MOE=float(r.mean()),
        medT=float(np.median(t)), medR=float(np.median(r)),
        ok12=int(((t <= 1) & (r <= 2)).sum()),
        ok38=int(((t <= 3) & (r <= 8)).sum()),
        over10=int((t > 10).sum()), over100=int((t > 100).sum()),
        improved=int((t < bt - 1e-9).sum()), worsened=int((t > bt + 1e-9).sum()),
        MPE_keep=float(t[keep].mean()) if keep.any() else float('nan'),
        ok12_keep=int(((t <= 1) & (r <= 2) & keep).sum()),
        over10_keep=int((t[keep] > 10).sum()),
        damage05=int(((t > bt + 0.5) & keep).sum()),
        damage05_all=int((t > bt + 0.5).sum()),
        keep_n=int(keep.sum()),
        d_pos=t, d_rot=r, base_t=bt)


def flatm(m):
    out = {}
    for k, v in m.items():
        if isinstance(v, np.ndarray):
            continue
        out[k] = float(v) if hasattr(v, 'item') else v
    return out


def row(name, m):
    print('%-22s n=%4d MPE=%7.4f MOE=%7.4f medT=%6.3f ok12=%4d ok38=%4d '
          '>10m=%3d >100m=%3d impr=%4d worse=%4d dmg>0.5=%3d'
          % (name, m['n'], m['MPE'], m['MOE'], m['medT'], m['ok12'], m['ok38'],
             m['over10'], m['over100'], m['improved'], m['worsened'], m['damage05_all']))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='main')
    ap.add_argument('--truth', default='truth')
    ap.add_argument('--old', default='old')
    A = ap.parse_args()

    w = SU.load_world()
    keys = w['hard_keys']
    dates = w['dates_hard']
    keep = ~w['hard_replaced']
    summary = {}

    # ---------------- 1. baseline and main arms
    print('=== 1. overall (canonical keys, GT = scored.npz[reference]) ===')
    base = metrics(w, w['hard_base'])
    row('deployed refined_V7', base)
    base_orig = metrics(w, w['hard_base_ref'])
    row('original V7', base_orig)

    k, P, PR, has_ref, diag = load_tag(A.tag)
    lookup = {kk: i for i, kk in enumerate(k)}
    order = [lookup[kk] for kk in keys]
    P = P[order]
    PR = PR[order]
    has_ref = has_ref[order]
    m_main = row('R4 main (sweep)', metrics(w, P))
    summary['main'] = flatm(m_main)
    print('   refined poses: %d/%d' % (has_ref.sum(), len(has_ref)))
    m_ref = row('R4 main + refine', metrics(w, PR))
    summary['main_refined'] = flatm(m_ref)

    print('\n--- on the 1266 frames outside the suspect window ---')
    ik = np.flatnonzero(keep)
    row('base (keep)', metrics(w, w['hard_base'][ik], ik))
    row('main (keep)', metrics(w, P[ik], ik))
    row('main+refine (keep)', metrics(w, PR[ik], ik))

    print('\n--- per date (suspect window excluded) ---')
    per_date = {}
    for d in sorted(set(dates)):
        idx = np.flatnonzero((dates == d) & keep)
        mb = metrics(w, w['hard_base'][idx], idx)
        mm = metrics(w, P[idx], idx)
        mr = metrics(w, PR[idx], idx)
        per_date[d] = dict(base=mb['MPE'], main=mm['MPE'], refined=mr['MPE'],
                           base_ok12=mb['ok12'], main_ok12=mm['ok12'],
                           refined_ok12=mr['ok12'], base_over10=mb['over10'],
                           main_over10=mm['over10'], n=len(idx))
        print('%-12s n=%4d  MPE base %7.4f -> main %7.4f -> refined %7.4f | '
              'ok12 %3d -> %3d -> %3d | >10m %2d -> %2d'
              % (d, len(idx), mb['MPE'], mm['MPE'], mr['MPE'], mb['ok12'],
                 mm['ok12'], mr['ok12'], mb['over10'], mm['over10']))
    summary['per_date'] = per_date

    # ---------------- 2. gate
    g1 = m_main['ok12'] >= GATE_OK
    g2 = m_main['over10'] <= GATE_10M
    print('\n=== 2. pre-registered gate (main arm, all 1502) ===')
    print('G1 ok12 %d >= %d : %s   (%.2fx baseline 621)'
          % (m_main['ok12'], GATE_OK, g1, m_main['ok12'] / 621.0))
    print('G2 >10m %d <= %d : %s' % (m_main['over10'], GATE_10M, g2))
    ref_g1 = m_ref['ok12'] >= GATE_OK
    ref_g2 = m_ref['over10'] <= GATE_10M
    print('   (main+refine: G1 %s, G2 %s)' % (ref_g1, ref_g2))
    summary['gate'] = dict(main_g1=bool(g1), main_g2=bool(g2),
                           refined_g1=bool(ref_g1), refined_g2=bool(ref_g2),
                           target_ok12=GATE_OK, target_over10=GATE_10M)

    # paired comparison at the 1 m threshold, which is where G1 lived
    tb = m_main['base_t']
    tm = m_main['d_pos']
    print('   paired at 1 m: rescued (base>1, main<=1) %d ; killed (base<=1, main>1) %d'
          % (int(((tb > 1) & (tm <= 1)).sum()), int(((tb <= 1) & (tm > 1)).sum())))
    summary['paired_1m'] = dict(rescued=int(((tb > 1) & (tm <= 1)).sum()),
                                killed=int(((tb <= 1) & (tm > 1)).sum()))

    # ---------------- 3. oracle over the same candidate set
    z = np.load('/data2/zhangcy/v7-raw-geometry-hard1502-20261013-v1/scored.npz',
                allow_pickle=True)
    S = np.round(np.arange(-10.0, 10.0001, 0.5), 3)
    D = np.round(np.arange(-2.0, 2.0001, 0.05), 4)
    orc = np.zeros((len(keys), 4, 4))
    orc_sel = np.zeros((len(keys), 4, 4))
    n_out = 0
    for i, key in enumerate(keys):
        r = diag[key]
        T0 = w['hard_base'][i]
        u = np.asarray(r['u'], float)
        e = np.asarray(r['lateral'], float)
        if 's_star' in r:
            ia = int(np.argmin(np.abs(S - r['s_star'])))
            ib = int(np.argmin(np.abs(D - r['d_star'])))
            if abs(r['s_star']) > 10.0 or abs(r['d_star']) > 2.0:
                n_out += 1
        else:
            ia, ib = int(np.argmin(np.abs(S))), int(np.argmin(np.abs(D)))
            n_out += 1
        orc[i] = T0
        orc[i, :3, 3] = T0[:3, 3] + S[ia] * u + D[ib] * e
        orc_sel[i] = T0
        orc_sel[i, :3, 3] = T0[:3, 3] + r['s'] * u + r['d'] * e
    m_orc = row('ORACLE best-of-grid', metrics(w, orc))
    print('   frames whose truth lies outside the hypothesis set: %d' % n_out)
    print('   mean |oracle - main| translation = %.3f m'
          % np.linalg.norm(orc[:, :3, 3] - orc_sel[:, :3, 3], axis=1).mean())
    print('   oracle on kept: MPE=%.4f ok12=%d'
          % (metrics(w, orc[ik], ik)['MPE'], metrics(w, orc[ik], ik)['ok12']))
    summary['oracle'] = flatm(m_orc)
    summary['oracle_outside'] = n_out

    # the frozen-rotation ceiling on the joint 1 m + 2 deg criterion
    r_deg = m_main['d_rot']
    print('\n--- rotation ceiling (rotation is FROZEN at the deployment pose) ---')
    print('   rotation err: med=%.2f p90=%.2f deg; frames with rot<=2 deg: %d (%.1f%%)'
          % (np.median(r_deg), np.percentile(r_deg, 90), int((r_deg <= 2).sum()),
             100.0 * (r_deg <= 2).mean()))
    print('   frames with pos<=1 m: base %d -> main %d -> oracle %d'
          % (int((m_main['base_t'] <= 1).sum()), int((m_main['d_pos'] <= 1).sum()),
             int((m_orc['d_pos'] <= 1).sum())))
    print('   => the joint ok12 count can never exceed %d/1502 (%.1f%%) with frozen rotation,'
          ' while G1 asked for %d' % (int((r_deg <= 2).sum()),
                                      100.0 * (r_deg <= 2).mean(), GATE_OK))
    print('   of the %d frames with rot<=2, oracle gets pos<=1 on %d'
          % (int((r_deg <= 2).sum()), int(((r_deg <= 2) & (m_orc['d_pos'] <= 1)).sum())))
    summary['rotation_ceiling'] = dict(
        rot_le2=int((r_deg <= 2).sum()), rot_med=float(np.median(r_deg)),
        oracle_pos1=int((m_orc['d_pos'] <= 1).sum()),
        oracle_pos1_and_rot2=int(((r_deg <= 2) & (m_orc['d_pos'] <= 1)).sum()),
        gate_target=GATE_OK)
    print('   translation gain capture: base MPE %.4f -> main %.4f -> oracle %.4f  '
          '(main takes %.1f%% of the oracle gain)'
          % (m_main['base_t'].mean(), m_main['d_pos'].mean(), m_orc['MPE'],
             100.0 * (m_main['base_t'].mean() - m_main['d_pos'].mean())
             / max(m_main['base_t'].mean() - m_orc['d_pos'].mean(), 1e-12)))
    summary['capture_pct'] = float(
        100.0 * (m_main['base_t'].mean() - m_main['d_pos'].mean())
        / max(m_main['base_t'].mean() - m_orc['d_pos'].mean(), 1e-12))

    # ---------------- 4. u source split + s/d behaviour
    print('\n=== 3. along-track direction: source split and sweep behaviour ===')
    src = Counter(diag[key]['u_source'] for key in keys)
    print('u source counts:', dict(src))
    reach = {}
    for name in sorted(src):
        sel = np.array([diag[key]['u_source'] == name for key in keys])
        mm = metrics(w, P[sel], np.flatnonzero(sel))
        mb = metrics(w, w['hard_base'][sel], np.flatnonzero(sel))
        reach[name] = dict(n=int(sel.sum()), base_MPE=mb['MPE'], main_MPE=mm['MPE'],
                           base_ok12=mb['ok12'], main_ok12=mm['ok12'],
                           base_over10=mb['over10'], main_over10=mm['over10'])
        print('  %-14s n=%4d  MPE %.4f -> %.4f | ok12 %3d -> %3d | >10m %2d -> %2d'
              % (name, sel.sum(), mb['MPE'], mm['MPE'], mb['ok12'], mm['ok12'],
                 mb['over10'], mm['over10']))
    summary['u_source'] = reach
    chord = np.array([diag[key].get('u_meta', {}).get('chord', np.nan) for key in keys])
    print('  window chord (deploy_quad frames): med=%.1f p10=%.1f p90=%.1f'
          % (np.nanmedian(chord), np.nanpercentile(chord, 10), np.nanpercentile(chord, 90)))

    s = np.array([diag[key]['s'] for key in keys])
    d = np.array([diag[key]['d'] for key in keys])
    gain = np.array([diag[key]['gain'] if diag[key]['gain'] is not None else 0.0
                     for key in keys])
    nbetter = np.array([diag[key].get('n_better_than_deploy', 0) for key in keys])
    print('  |s|: med=%.1f p90=%.1f max=%.1f   s==0: %d (%.1f%%)   |s|>=5: %d'
          % (np.median(np.abs(s)), np.percentile(np.abs(s), 90), np.abs(s).max(),
             int((s == 0).sum()), 100.0 * (s == 0).mean(), int((np.abs(s) >= 5).sum())))
    print('  |d|: med=%.2f p90=%.2f max=%.2f   |d|>=1: %d   |d|>=1.9: %d'
          % (np.median(np.abs(d)), np.percentile(np.abs(d), 90), np.abs(d).max(),
             int((np.abs(d) >= 1).sum()), int((np.abs(d) >= 1.9).sum())))
    print('  cost gain (deploy cost - selected cost): med=%.4f p90=%.4f max=%.4f  '
          'exactly zero: %d' % (np.median(gain), np.percentile(gain, 90), gain.max(),
                                int((gain <= 1e-9).sum())))
    summary['sweep'] = dict(s_abs_med=float(np.median(np.abs(s))),
                            s_zero=int((s == 0).sum()),
                            d_abs_med=float(np.median(np.abs(d))),
                            d_big=int((np.abs(d) >= 1.9).sum()),
                            gain_med=float(np.median(gain)))

    # ---------------- 5. map points / abstention cost
    print('\n=== 4. local map size and the price of the discarded abstention rule ===')
    mp = np.array([diag[key]['map_points'] for key in keys])
    qp = np.array([diag[key]['query_points'] for key in keys])
    print('map points: med=%d p10=%d p90=%d max=%d ; query points med=%d'
          % (np.median(mp), np.percentile(mp, 10), np.percentile(mp, 90), mp.max(),
             np.median(qp)))
    t_main = SU.pos_err(P, w['hard_gt'])
    t_base = SU.pos_err(w['hard_base'], w['hard_gt'])
    pear = float(np.corrcoef(mp, t_base)[0, 1])
    spear = float(np.corrcoef(np.argsort(np.argsort(mp)),
                              np.argsort(np.argsort(t_base)))[0, 1])
    print('  pearson(map_points, deploy err)=%.3f  spearman=%.3f' % (pear, spear))
    would = np.array([bool(diag[key].get('would_abstain')) for key in keys])
    alt = P.copy()
    alt[would] = w['hard_base'][would]
    m_alt = metrics(w, alt)
    good = int(((t_base <= 1.0) & (t_main <= 1.0) & would).sum())
    print('  would-abstain (>20000): %d frames (%.1f%%); of those deploy err<=1m: %d'
          % (would.sum(), 100.0 * would.mean(), int(((t_base <= 1.0) & would).sum())))
    print('  if reverted to deployment on those frames: MPE %.4f -> %.4f, ok12 %d -> %d,'
          ' >10m %d -> %d' % (m_main['MPE'], m_alt['MPE'], m_main['ok12'], m_alt['ok12'],
                              m_main['over10'], m_alt['over10']))
    print('  frames the abstention rule would have thrown away while main fixed them: %d'
          % int(((t_main <= 1.0) & (t_base > 1.0) & would).sum()))
    summary['map_points'] = dict(med=float(np.median(mp)), p90=float(np.percentile(mp, 90)),
                                 max=int(mp.max()), pearson=pear, spearman=spear,
                                 would_abstain=int(would.sum()),
                                 abstain_ok12=int(m_alt['ok12']),
                                 abstain_MPE=float(m_alt['MPE']))

    # ---------------- 6. damage
    print('\n=== 5. damage outside the suspect window ===')
    delta = t_main - t_base
    dm = (delta > 0.5) & keep
    print('frames worse by >0.5 m (kept): %d (%.2f%%)   worse by >1 m: %d   worst +%.2f m'
          % (dm.sum(), 100.0 * dm.mean(), int(((delta > 1.0) & keep).sum()),
             delta[keep].max()))
    print('frames better by >0.5 m (kept): %d   better by >1 m: %d   best %.2f m'
          % (int(((delta < -0.5) & keep).sum()), int(((delta < -1.0) & keep).sum()),
             delta[keep].min()))
    worst = np.argsort(-delta)[:15]
    print('  %-28s %8s %8s %7s %6s %6s %6s %8s' %
          ('key', 'deploy', 'main', 'delta', 's', 'd', 'u', 'map'))
    for i in worst:
        r = diag[keys[i]]
        print('  %-28s %8.2f %8.2f %+7.2f %+6.1f %+6.2f %6s %8d%s'
              % (keys[i], t_base[i], t_main[i], delta[i], r['s'], r['d'],
                 r['u_source'][:6], r['map_points'],
                 '  [suspect]' if w['hard_replaced'][i] else ''))
    summary['damage'] = dict(gt05=int(dm.sum()), gt10=int(((delta > 1.0) & keep).sum()),
                             better05=int(((delta < -0.5) & keep).sum()),
                             better10=int(((delta < -1.0) & keep).sum()),
                             worst_delta=float(delta[keep].max()))

    # ---------------- 7. truth-tangent upper-bound arm
    tfiles = glob.glob(str(OUT / ('diagnostics_%s.jsonl' % A.truth)))
    if tfiles:
        print('\n=== 6. upper-bound arm: truth-trajectory tangent (fixed 100-frame subset) ===')
        kt, Pt, PRt, hrt, diagt = load_tag(A.truth)
        sub_keys = sorted(diagt)
        look = {kk: i for i, kk in enumerate(keys)}
        idx = np.array([look[kk] for kk in sub_keys])
        mt = row('truth-u arm (subset)', metrics(w, Pt[idx], idx))
        mm = row('main arm, same frames', metrics(w, P[idx], idx))
        mb = row('baseline, same frames', metrics(w, w['hard_base'][idx], idx))
        print('  truth u sources:', dict(Counter(diagt[kk]['u_source'] for kk in sub_keys)))
        dP = np.linalg.norm(Pt[idx, :3, 3] - P[idx, :3, 3], axis=1)
        print('  |pose(truth u) - pose(deployable u)|: med=%.2f p90=%.2f m'
              % (np.median(dP), np.percentile(dP, 90)))
        t_t = SU.pos_err(Pt[idx], w["hard_gt"][idx])
        t_m = SU.pos_err(P[idx], w['hard_gt'][idx])
        print('  truth-u better on %d frames, worse on %d, equal on %d'
              % (int((t_t < t_m - 1e-9).sum()), int((t_t > t_m + 1e-9).sum()),
                 int((np.abs(t_t - t_m) <= 1e-9).sum())))
        summary['truth_arm'] = dict(subset=int(len(idx)), truth=dict(
            MPE=mt['MPE'], ok12=mt['ok12'], over10=mt['over10']),
            main=dict(MPE=mm['MPE'], ok12=mm['ok12'], over10=mm['over10']),
            base=dict(MPE=mb['MPE'], ok12=mb['ok12']))
    else:
        print('\n(truth-u arm not collected yet)')

    # ---------------- 8. old-u-estimator arm (engineering comparison)
    ofiles = glob.glob(str(OUT / ('diagnostics_%s.jsonl' % A.old)))
    if ofiles:
        print('\n=== 7. the pre-revision u estimator, same frames (engineering only) ===')
        ko, Po, PRo, hro, diago = load_tag(A.old)
        sub_keys = sorted(diago)
        look = {kk: i for i, kk in enumerate(keys)}
        idx = np.array([look[kk] for kk in sub_keys])
        mo = row('old endpoint-diff u', metrics(w, Po[idx], idx))
        mm = row('revised u, same frames', metrics(w, P[idx], idx))
        summary['old_arm'] = dict(subset=int(len(idx)),
                                  old=dict(MPE=mo['MPE'], ok12=mo['ok12'], over10=mo['over10']),
                                  new=dict(MPE=mm['MPE'], ok12=mm['ok12'], over10=mm['over10']))

    json.dump(summary, open(OUT / ('summary_%s.json' % A.tag), 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print('\nwrote %s' % (OUT / ('summary_%s.json' % A.tag)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

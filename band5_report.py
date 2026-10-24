"""Score the three variants of the >5 m pose-free study.  The only place truth is read.

    python band5_report.py --subset results/band5-nopose-subset.npz \
        --out /tmp/band5/out --dest results

Definitions
    PnP pose          the expert produced any pose at all
    gated              >=15 unique inliers, >=6 of 16 cells, >=20% inlier ratio
    qualified/solved   gated AND >=2 references agreeing within 5 m / 10 deg
    adopted            the pose the deployed rule takes: highest fair score among the
                       8 cheapest scored hypotheses (budget 8), regardless of gating
    forced adoption    take `adopted` whenever it exists, else keep t0
"""
import argparse, collections, glob, json, os
import numpy as np

DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001'
TRUTH = DATA + '/evaluator-only-truth/%s.json'
LIBRARY = '/data2/zhangcy/sp-lightglue-hard1502-20261013-v1/library.json'
VARIANTS = ['A5', 'B20', 'C5']

ap = argparse.ArgumentParser()
ap.add_argument('--subset', default='results/band5-nopose-subset.npz')
ap.add_argument('--out', default='/tmp/band5/out')
ap.add_argument('--dest', default='results')
ap.add_argument('--quiet', action='store_true')
a = ap.parse_args()

sub = np.load(a.subset, allow_pickle=True)
keys = [str(k) for k in sub['keys']]
t0 = np.asarray(sub['t0'], float)
err0 = np.asarray(sub['err_t0'], float)
rot0 = np.asarray(sub['rot_t0'], float)
n = len(keys)
truth = {d: json.load(open(TRUTH % d)) for d in sorted({k.split('/')[0] for k in keys})}
gt = np.stack([np.asarray(truth[k.split('/')[0]][k.split('/')[1]]['T_WB'], float) for k in keys])
library = json.load(open(LIBRARY))
refpos = np.asarray([r['body_position_world'] for r in library['references']], float)

rows = collections.defaultdict(dict)
for path in glob.glob(os.path.join(a.out, '*_s*.jsonl')):
    variant = os.path.basename(path).split('_')[0]
    for line in open(path):
        row = json.loads(line)
        rows[variant][row['key']] = row
for variant in VARIANTS:
    missing = [k for k in keys if k not in rows[variant]]
    assert not missing, '%s missing %d rows' % (variant, len(missing))

bands = [('5-8', 5, 8), ('8-20', 8, 20), ('>20', 20, 1e9)]
band_mask = {b: (err0 >= lo) & (err0 < hi) for b, lo, hi in bands}


def err(pose, g):
    return float(np.linalg.norm(np.asarray(pose, float).reshape(4, 4)[:3, 3] - g[:3, 3]))


def rot(pose, g):
    R = np.asarray(pose, float).reshape(4, 4)[:3, :3]
    return float(np.degrees(np.arccos(np.clip((np.trace(R.T @ g[:3, :3]) - 1) / 2, -1, 1))))


def fmt(x):
    return '%.4f / %.4f' % (np.mean(x), np.median(x))


def say(*values):
    if not a.quiet:
        print(*values)


summary = dict(frames=n, t0_mpe=[float(err0.mean()), float(np.median(err0))],
               t0_moe=[float(rot0.mean()), float(np.median(rot0))],
               t0_total=float(err0.sum()),
               bands={b: int(m.sum()) for b, m in band_mask.items()})
say('frames %d  t0 MPE %s  t0 MOE %s  total %.2f m  bands %s'
    % (n, fmt(err0), fmt(rot0), err0.sum(), summary['bands']))

arrays = dict(keys=np.array(keys), t0=t0, err_t0=err0, rot_t0=rot0)
for variant in VARIANTS:
    has_pose = np.zeros(n, bool); gate = np.zeros(n, bool); solved = np.zeros(n, bool)
    adopted = np.zeros(n, bool)
    q_err = np.full(n, np.inf); q_rot = np.full(n, np.inf)
    a_err = np.full(n, np.nan); a_rot = np.full(n, np.nan)
    score = np.full(n, np.nan)
    pose = np.full((n, 4, 4), np.nan)
    hit5 = np.zeros(n, bool); nref = np.zeros(n, int); seconds = np.zeros(n, float)
    errors = 0
    for i, key in enumerate(keys):
        row = rows[variant][key]
        errors += 'error' in row
        seconds[i] = row['seconds']
        hyps = row['hypotheses']
        has_pose[i] = row.get('pose') is not None
        if hyps:
            gate[i] = any(h['qualified'] for h in hyps)
            q = [h for h in hyps if h['qualified'] and h['consensus'] >= 2]
            solved[i] = bool(q)
            if q:
                q_err[i] = min(err(h['pose'], gt[i]) for h in q)
                q_rot[i] = min(rot(h['pose'], gt[i]) for h in q)
        if row.get('pose') is not None:
            adopted[i] = True
            a_err[i] = err(row['pose'], gt[i])
            a_rot[i] = rot(row['pose'], gt[i])
            score[i] = row['score']
            pose[i] = np.asarray(row['pose'], float).reshape(4, 4)
        refs = sorted(set((row['retrieval']['visual'] or []) + (row['retrieval']['position'] or [])))
        nref[i] = len(refs)
        if refs:
            hit5[i] = bool(np.min(np.linalg.norm(refpos[refs] - gt[i][:3, 3], axis=1)) <= 5.0)
    final = np.where(adopted, a_err, err0)
    final_rot = np.where(adopted, a_rot, rot0)
    saved = err0 - final
    arrays['%s_T_adopted' % variant] = pose
    arrays['%s_score' % variant] = score
    arrays['%s_has_pose' % variant] = has_pose
    arrays['%s_gated' % variant] = gate
    arrays['%s_has_qualified' % variant] = solved
    arrays['%s_qual_err' % variant] = q_err
    arrays['%s_qual_rot_deg' % variant] = q_rot
    arrays['%s_adopted_err' % variant] = a_err
    arrays['%s_adopted_rot_deg' % variant] = a_rot
    arrays['%s_final_err' % variant] = final
    arrays['%s_final_rot_deg' % variant] = final_rot
    arrays['%s_saved' % variant] = saved
    arrays['%s_hit5' % variant] = hit5
    arrays['%s_n_refs' % variant] = nref
    say('')
    say('%s: pose %d  gated %d  solved %d  (errors %d)' % (variant, has_pose.sum(), gate.sum(),
                                                           solved.sum(), errors))
    say('   solved -> hypothesis <5 m %d ; <1 m %d ; <5 m and rot<10 deg %d'
        % ((q_err[solved] < 5).sum(), (q_err[solved] < 1).sum(),
           ((q_err[solved] < 5) & (q_rot[solved] < 10)).sum()))
    say('   final MPE %s  MOE %s' % (fmt(final), fmt(final_rot)))
    say('   <1 m %d  <5 m %d  >=5 m %d ; total %.2f m ; saved %+.2f m'
        % ((final < 1).sum(), (final < 5).sum(), (final >= 5).sum(), final.sum(), saved.sum()))
    say('   adopted err>20 m %d ; >100 m %d ; rot>150 deg %d'
        % ((a_err > 20).sum(), (a_err > 100).sum(), (a_rot > 150).sum()))
    for b, _, _ in bands:
        m = band_mask[b]
        say('   %-5s n=%3d solved %3d  MPE %.3f->%.3f med %.3f->%.3f  <5m 0->%d  saved %+.1f m'
            % (b, m.sum(), solved[m].sum(), err0[m].mean(), final[m].mean(),
               np.median(err0[m]), np.median(final[m]), (final[m] < 5).sum(), saved[m].sum()))
    say('   retrieval: hit<=5 m %d ; hit but not solved %d ; solved without hit %d'
        % (hit5.sum(), (hit5 & ~solved).sum(), (solved & ~hit5).sum()))
    say('   solved but adopted >=5 m %d ; solved but best hypothesis >=5 m %d ; '
        'time mean %.2f s total %.1f min'
        % ((solved & (a_err >= 5)).sum(), (solved & (q_err >= 5)).sum(),
           seconds.mean(), seconds.sum() / 60))
    summary[variant] = dict(
        has_pose=int(has_pose.sum()), gated=int(gate.sum()), solved=int(solved.sum()),
        solved_hyp5=int((q_err[solved] < 5).sum()), solved_hyp1=int((q_err[solved] < 1).sum()),
        final_mpe=[float(final.mean()), float(np.median(final))],
        final_moe=[float(final_rot.mean()), float(np.median(final_rot))],
        lt1=int((final < 1).sum()), lt5=int((final < 5).sum()), ge5=int((final >= 5).sum()),
        total=float(final.sum()), saved=float(saved.sum()), hit5=int(hit5.sum()),
        hit5_not_solved=int((hit5 & ~solved).sum()),
        solved_not_hit=int((solved & ~hit5).sum()),
        solved_adopted_ge5=int((solved & (a_err >= 5)).sum()),
        solved_best_ge5=int((solved & (q_err >= 5)).sum()),
        adopted_gt20=int((a_err > 20).sum()), adopted_gt100=int((a_err > 100).sum()),
        adopted_rot150=int((a_rot > 150).sum()), errors=int(errors),
        bands={b: dict(n=int(band_mask[b].sum()), solved=int(solved[band_mask[b]].sum()),
                       mpe0=float(err0[band_mask[b]].mean()),
                       mpe1=float(final[band_mask[b]].mean()),
                       lt5=int((final[band_mask[b]] < 5).sum()),
                       saved=float(saved[band_mask[b]].sum())) for b, _, _ in bands})

gross = np.where(band_mask['>20'])[0]
if len(gross):
    say('')
    say('the >20 m frames')
    for i in gross:
        say('  %s t0 %.2f m / %.1f deg' % (keys[i], err0[i], rot0[i]))
        for variant in VARIANTS:
            row = rows[variant][keys[i]]
            q = [h for h in row['hypotheses'] if h['qualified'] and h['consensus'] >= 2]
            say('    %-4s solved=%-5s adopted=%s best_qualified=%s score=%.3f'
                % (variant, bool(q),
                   '%.2f m' % err(row['pose'], gt[i]) if row.get('pose') else '-',
                   '%.2f m' % min(err(h['pose'], gt[i]) for h in q) if q else '-',
                   row.get('score', float('nan'))))

os.makedirs(a.dest, exist_ok=True)
json.dump(summary, open(os.path.join(a.dest, 'band5-nopose-summary.json'), 'w'), indent=1)
np.savez_compressed(os.path.join(a.dest, 'band5-nopose-perframe.npz'), **arrays)
say('')
say('wrote %s/band5-nopose-summary.json and band5-nopose-perframe.npz' % a.dest)

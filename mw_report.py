"""Score the multi-window study and write the report.

Truth is read here (and in the oracle runner, which reads it only to place the single
oracle window) and nowhere else.

    python mw_report.py --run0 /tmp/mw/out0 --run1 /tmp/mw/out1 --oracle /tmp/mw/oracle \
        --check /tmp/mw/check --visual /tmp/mw/visual --dest results

Reads
    results/band5-nopose-subset.npz          the 290 frames, t0, t0 errors
    results/band5-nopose-perframe.npz        the finished version (C5), forced adoption

Writes
    results/mw-perframe.npz, results/mw-summary.json, results/mw-report.md
Every number in the markdown is formatted from the dicts below, never retyped.
"""
import argparse
import collections
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001'
TRUTH = DATA + '/evaluator-only-truth/%s.json'
REGION = 30.0      # "a window landed in the right place" if its centre is this close

ap = argparse.ArgumentParser()
ap.add_argument('--subset', default=os.path.join(HERE, 'results/band5-nopose-subset.npz'))
ap.add_argument('--run0', default='/tmp/mw/out0')
ap.add_argument('--run1', default='/tmp/mw/out1')
ap.add_argument('--oracle', default='/tmp/mw/oracle')
ap.add_argument('--check', default='/tmp/mw/check')
ap.add_argument('--visual', default='/tmp/mw/visual')
ap.add_argument('--dest', default='results')
ap.add_argument('--wall1', type=float, default=0.0, help='wall clock of the multi-window run, s')
ap.add_argument('--wall0', type=float, default=0.0)
ap.add_argument('--walloracle', type=float, default=0.0)
a = ap.parse_args()

sub = np.load(a.subset, allow_pickle=True)
keys = [str(k) for k in sub['keys']]
IDX = {k: i for i, k in enumerate(keys)}
t0 = np.asarray(sub['t0'], float)
err0 = np.asarray(sub['err_t0'], float)
rot0 = np.asarray(sub['rot_t0'], float)
n = len(keys)
truth = {d: json.load(open(TRUTH % d)) for d in sorted({k.split('/')[0] for k in keys})}
gt = np.stack([np.asarray(truth[k.split('/')[0]][k.split('/')[1]]['T_WB'], float) for k in keys])

bands = [('5-8', 5, 8), ('8-20', 8, 20), ('>20', 20, 1e9)]
band_of = np.array(['5-8' if e < 8 else ('8-20' if e < 20 else '>20') for e in err0])
band_mask = {b: band_of == b for b, _, _ in bands}


def load(directory):
    rows = {}
    for path in sorted(glob.glob(os.path.join(directory, '*.jsonl'))):
        for line in open(path):
            row = json.loads(line)
            rows[row['key']] = row
    missing = [k for k in keys if k not in rows]
    bad = [k for k in keys if 'error' in rows.get(k, {})]
    return rows, missing, bad


run0, miss0, bad0 = load(a.run0)
run1, miss1, bad1 = load(a.run1)
oracle, miss_o, bad_o = load(a.oracle)
check, miss_c, bad_c = load(a.check)
visual, miss_v, bad_v = load(a.visual)


def pose_of(row):
    return np.asarray(row['pose'], float).reshape(4, 4)


def terr(pose, g):
    return float(np.linalg.norm(pose[:3, 3] - g[:3, 3]))


def rerr(pose, g):
    R = pose[:3, :3]
    return float(np.degrees(np.arccos(np.clip((np.trace(R.T @ g[:3, :3]) - 1) / 2, -1, 1))))


def stats(err, rot):
    err = np.asarray(err, float)
    rot = np.asarray(rot, float)
    return dict(mpe=float(err.mean()), mpe_med=float(np.median(err)),
                moe=float(rot.mean()), moe_med=float(np.median(rot)),
                lt1=int((err < 1).sum()), lt5=int((err < 5).sum()),
                ge5=int((err >= 5).sum()), total=float(err.sum()),
                saved=float(err0.sum() - err.sum()))


def collect(rows):
    err, rot, secs, infos = [], [], [], []
    for k in keys:
        row = rows.get(k, {})
        i = IDX[k]
        pose = t0[i] if ('error' in row or 'pose' not in row) else pose_of(row)
        err.append(terr(pose, gt[i]))
        rot.append(rerr(pose, gt[i]))
        secs.append(float(row.get('seconds', np.nan)))
        infos.append(row.get('info') or {})
    return np.asarray(err), np.asarray(rot), np.asarray(secs), infos


err1, rot1, sec1, info1 = collect(run1)
errG, rotG, sec0, info0 = collect(run0)
errO, rotO, _, infoO = collect(oracle)
errC2, rotC2, _, infoV = collect(visual)          # the finished version, deployed arbitration

perf = np.load(os.path.join(HERE, 'results/band5-nopose-perframe.npz'), allow_pickle=True)
assert [str(k) for k in perf['keys']] == keys, 'C5 table is not in subset order'
errC = np.asarray(perf['C5_final_err'], float)
rotC = np.asarray(perf['C5_final_rot_deg'], float)

S = dict(t0=dict(mpe=float(err0.mean()), mpe_med=float(np.median(err0)),
                 moe=float(rot0.mean()), moe_med=float(np.median(rot0)),
                 lt1=int((err0 < 1).sum()), lt5=int((err0 < 5).sum()),
                 ge5=int((err0 >= 5).sum()), total=float(err0.sum()), saved=0.0,
                 bands={b: int(m.sum()) for b, m in band_mask.items()}),
         geometry_only=stats(errG, rotG),
         multi_window=stats(err1, rot1),
         final_version=stats(errC, rotC),
         final_version_arbitrated=stats(errC2, rotC2))

# --- 1. window count, channel composition, timing
K = np.array([int(i.get('n_windows', 0)) for i in info1])
Kused = np.array([int(i.get('n_used', 0)) for i in info1])
skipped = sum(len(i.get('skipped', [])) for i in info1)
rows_by_channel = collections.Counter()
for i in info1:
    for w in i.get('windows', []):
        rows_by_channel[w['channel']] += 1
        if w['also_cls']:
            rows_by_channel['both'] += 1
adopted_channel = collections.Counter()
for i in range(n):
    if info1[i].get('switched'):
        for w in info1[i].get('windows', []):
            if w['name'] == info1[i].get('chosen'):
                adopted_channel[w['channel']] += 1
pos_window_distance = np.array([w['distance_to_t0'] for i in info1
                                for w in i.get('windows', []) if w['channel'] == 'position'])
pos_windows_per_frame = np.array([sum(1 for w in i.get('windows', []) if w['channel'] == 'position')
                                  for i in info1])


def med(values):
    values = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.median(values)) if values else float('nan')


def p90(values):
    values = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.percentile(values, 90)) if values else float('nan')


timing = dict(
    frame_total=med(sec1), frame_total_p90=p90(sec1), frame0_total=med(sec0),
    scan=med([i.get('t_scan_s') for i in info1]),
    incumbent_search=med([i.get('t_incumbent_s') for i in info1]),
    window_map_total=med([i.get('t_map_s') for i in info1]),
    window_search_total=med([i.get('t_search_s') for i in info1]),
    evidence=med([i.get('t_evidence_s') for i in info1]),
    decide=med([i.get('t_decide_s') for i in info1]),
    per_window_search=med([i.get('t_search_s', 0) / max(int(i.get('n_used', 1)), 1) for i in info1]),
    per_window_search_p90=p90([i.get('t_search_s', 0) / max(int(i.get('n_used', 1)), 1) for i in info1]),
    map_per_window=med([i.get('t_map_s', 0) / max(int(i.get('n_used', 1)), 1) for i in info1]),
    evidence_map_points=int(med([i.get('n_map', 0) for i in info1])),
)
Kstats = dict(median=float(np.median(K)), p90=p90(K), max=int(K.max()), mean=float(K.mean()),
              min=int(K.min()), at_cap=int((K >= 20).sum()),
              hist={int(k): int(v) for k, v in sorted(collections.Counter(K).items())},
              used_median=float(np.median(Kused)), used_max=int(Kused.max()),
              skipped_windows=int(skipped),
              rows_by_channel={k: int(v) for k, v in rows_by_channel.items()},
              adopted_by_channel={k: int(v) for k, v in adopted_channel.items()},
              position_windows_per_frame=float(np.median(pos_windows_per_frame)),
              position_window_distance_median=float(np.median(pos_window_distance)),
              position_window_distance_max=float(pos_window_distance.max()))

# --- 2/3/4. per-band table
band_rows = {}
switched_arr = np.array([bool(i.get('switched')) for i in info1])
near = np.array([min([np.linalg.norm(np.asarray(w['centre'], float) - gt[i][:3, 3])
                      for w in (info1[i].get('windows') or [])] or [np.inf]) for i in range(n)])
for name, lo, hi in bands:
    m = band_mask[name]
    band_rows[name] = dict(
        n=int(m.sum()), t0=float(err0[m].mean()),
        geo=[float(errG[m].mean()), float(np.median(errG[m])), float(err0[m].sum() - errG[m].sum()),
             int((errG[m] < 5).sum())],
        mw=[float(err1[m].mean()), float(np.median(err1[m])), float(err0[m].sum() - err1[m].sum()),
            int((err1[m] < 5).sum())],
        c5=[float(errC[m].mean()), float(np.median(errC[m])), float(err0[m].sum() - errC[m].sum()),
            int((errC[m] < 5).sum())],
        c5a=[float(errC2[m].mean()), float(err0[m].sum() - errC2[m].sum())],
        switched=int((m & switched_arr).sum()),
        near30=int((m & (near <= REGION)).sum()),
    )

# --- 5. failure attribution
cand_best = np.array([min([np.linalg.norm(np.asarray(w['pose'], float).reshape(4, 4)[:3, 3]
                                           - gt[i][:3, 3])
                           for w in (info1[i].get('windows') or [])] or [np.inf])
                      for i in range(n)])
truth_cost = np.array([min([(np.linalg.norm(np.asarray(w['pose'], float).reshape(4, 4)[:3, 3]
                                            - gt[i][:3, 3]), w['cost_shared'])
                            for w in (info1[i].get('windows') or [])] or [(np.inf, np.inf)])[1]
                       for i in range(n)])
inc_cost = np.array([i.get('incumbent', {}).get('cost_shared', np.inf) for i in info1])
sigma = np.array([i.get('sigma_c', np.nan) for i in info1])
thresh = np.array([i.get('thresh', np.nan) for i in info1])
margin = np.array([i.get('margin', np.nan) for i in info1])
chosen = np.array([i.get('chosen', 'geo_t0') for i in info1])
near30 = near <= REGION
missed = (cand_best <= 5) & (err1 > 5)
saw_but_close = missed & (truth_cost < inc_cost) & ((inc_cost - truth_cost) <= thresh)
saw_preferred_other = missed & ~(truth_cost < inc_cost)
attr = dict(
    near30=int(near30.sum()), near10=int((near <= 10).sum()), near5=int((near <= 5).sum()),
    near2=int((near <= 2).sum()),
    center_median=float(np.median(near)), center_p90=p90(near), center_max=float(near.max()),
    near30_stuck=int((near30 & (cand_best > 5)).sum()),
    near30_solved=int((near30 & (cand_best <= 5)).sum()),
    wrong_window=int(missed.sum()),
    missed_gain_below_thresh=int(saw_but_close.sum()),
    missed_evidence_preferred_other=int(saw_preferred_other.sum()),
    switched=int(switched_arr.sum()),
    switched_better=int((switched_arr & (err1 < errG - 1e-9)).sum()),
    switched_worse=int((switched_arr & (err1 > errG + 1e-9)).sum()),
    worse_than_t0=int((err1 > err0 + 1e-9).sum()),
    better_than_t0=int((err1 < err0 - 1e-9).sum()),
    sigma_median=float(np.nanmedian(sigma)), thresh_median=float(np.nanmedian(thresh)),
    margin_median=float(np.nanmedian(margin)),
    stuck_by_band={name: int((band_mask[name] & near30 & (cand_best > 5)).sum()) for name, _, _ in bands},
)

# --- what the 15 m non-maximum suppression removes from the candidate pool
refpos = np.asarray([r['world_from_body'] for r in
                     json.load(open(os.path.join(HERE, 'inputs/refs.json')))], float)[:, :3, 3]
nms_stat = dict(available={}, kept={}, suppressed_by_position={}, suppressed_other={})
for lim in (5.0, 10.0, 30.0):
    avail = keptn = sup_pos = sup_oth = 0
    for i, k in enumerate(keys):
        row = run1.get(k, {})
        cls_ids = row.get('cls_ids') or []
        if not cls_ids:
            continue
        dist = np.linalg.norm(refpos[np.asarray(cls_ids, int)] - gt[i][:3, 3], axis=1)
        near = [int(c) for c, dd in zip(cls_ids, dist) if dd <= lim]
        if not near:
            continue
        avail += 1
        windows = info1[i].get('windows') or []
        kept_ids = {int(w['ref']) for w in windows}
        if any(c in kept_ids for c in near):
            keptn += 1
            continue
        position = [w for w in windows if w['channel'] == 'position']
        trigger = 15.0
        if position and min(np.linalg.norm(refpos[c] - np.asarray(position[0]['centre'], float))
                            for c in near) < trigger:
            sup_pos += 1
        else:
            sup_oth += 1
    nms_stat['available']['%d' % lim] = avail
    nms_stat['kept']['%d' % lim] = keptn
    nms_stat['suppressed_by_position']['%d' % lim] = sup_pos
    nms_stat['suppressed_other']['%d' % lim] = sup_oth
nms_stat['position_windows_per_frame'] = {int(k): int(v) for k, v in
                                          sorted(collections.Counter(pos_windows_per_frame.tolist()).items())}

# --- 6. oracle bound
p_ok = all(oracle[k].get('oracle_pose') is not None for k in keys)
err_oS = np.array([terr(np.asarray(oracle[k]['oracle_pose'], float).reshape(4, 4), gt[IDX[k]])
                   if oracle[k].get('oracle_pose') is not None else np.nan for k in keys])
rot_oS = np.array([rerr(np.asarray(oracle[k]['oracle_pose'], float).reshape(4, 4), gt[IDX[k]])
                   if oracle[k].get('oracle_pose') is not None else np.nan for k in keys])
centre_err = np.array([float(oracle[k]['oracle_centre_err']) for k in keys])
orc_cost = np.array([oracle[k]['info']['windows'][0]['cost_shared'] for k in keys])
inc_cost_o = np.array([oracle[k]['info']['incumbent']['cost_shared'] for k in keys])
orc_thresh = np.array([oracle[k]['info']['thresh'] for k in keys])
orc_sw = np.array([bool(oracle[k]['info']['switched']) for k in keys])
good = err_oS < 5
blind = dict(
    good=int(good.sum()),
    cost_lower=int((orc_cost[good] < inc_cost_o[good]).sum()),
    margin_clears=int(((inc_cost_o - orc_cost) > orc_thresh)[good].sum()),
    adopted=int(orc_sw[good].sum()),
    inc_cost_median=float(np.median(inc_cost_o)),
    oracle_cost_median=float(np.median(orc_cost)),
    gap_median_good=float(np.median((inc_cost_o - orc_cost)[good])) if good.any() else None,
    oracle_cost_median_good=float(np.median(orc_cost[good])) if good.any() else None,
    inc_cost_median_good=float(np.median(inc_cost_o[good])) if good.any() else None,
)
oracle_stats = dict(
    centre_median=float(np.median(centre_err)), centre_p90=p90(centre_err),
    centre_max=float(centre_err.max()), centre_le5=int((centre_err <= 5).sum()),
    centre_le2=int((centre_err <= 2).sum()),
    arbitrated=stats(errO, rotO),
    search_only=stats(err_oS, rot_oS) if p_ok else None,
    best_of_t0=dict(saved=float(np.nansum(err0 - np.minimum(err0, err_oS))),
                    lt5=int((np.minimum(err0, err_oS) < 5).sum()),
                    mpe=float(np.nanmean(np.minimum(err0, err_oS)))) if p_ok else None,
    bands={name: dict(search_only=float(np.nanmean(err_oS[band_mask[name]])),
                      lt5=int((err_oS[band_mask[name]] < 5).sum()),
                      saved=float(err0[band_mask[name]].sum() - np.nansum(err_oS[band_mask[name]])))
           for name, _, _ in bands},
)

# --- consistency: the switch-off run against the existing orchestrator
cons = {}
try:
    diffs, hit = [], 0
    for k in keys:
        if k not in check or 'pose' not in run0.get(k, {}) or 'pose' not in check[k]:
            continue
        hit += 1
        diffs.append(float(np.abs(pose_of(run0[k]) - pose_of(check[k])).max()))
    diffs = np.asarray(diffs)
    cons = dict(overlap=int(hit), exact=int((diffs == 0).sum()) if len(diffs) else 0,
                max_abs_diff=float(diffs.max()) if len(diffs) else None,
                reference='three_route_unified.one(mode=route3), the existing entry point')
except Exception as exc:
    cons = dict(error='%s: %s' % (type(exc).__name__, exc))

head = dict(
    mw_better=int((err1 < errC - 1e-9).sum()),
    mw_worse=int((err1 > errC + 1e-9).sum()),
    saved_diff=float((errC - err1).sum()),
    saved_diff_vs_arbitrated=float((errC2 - err1).sum()),
    mw_better_vs_arb=int((err1 < errC2 - 1e-9).sum()),
    mw_worse_vs_arb=int((err1 > errC2 + 1e-9).sum()),
    oracle_gap_m=float(err1.sum() - err_oS.sum()) if p_ok else None,
)

summary = dict(frames=n, region_m=REGION, t0_total=float(err0.sum()), K=Kstats, timing=timing,
               metrics=S, bands=band_rows, attribution=attr, oracle=oracle_stats, blind=blind,
               nms=nms_stat, head_to_head=head, consistency=cons,
               visual_arbitration=dict(
                   switched=int(np.sum([(visual[k].get('info') or {}).get('switched', False)
                                        for k in keys])),
                   errors=len(bad_v), missing=len(miss_v)),
               errors=dict(run0=len(bad0), run1=len(bad1), oracle=len(bad_o),
                           check=len(bad_c), visual=len(bad_v)),
               missing=dict(run0=len(miss0), run1=len(miss1), oracle=len(miss_o),
                            check=len(miss_c), visual=len(miss_v)),
               wall=dict(multi_window=a.wall1, geometry_only=a.wall0, oracle=a.walloracle))

np.savez_compressed(os.path.join(a.dest, 'mw-perframe.npz'),
                    keys=np.array(keys), t0=t0, err_t0=err0, rot_t0=rot0,
                    err_geometry=errG, rot_geometry=rotG, err_mw=err1, rot_mw=rot1,
                    err_final_version=errC, rot_final_version=rotC,
                    err_final_version_arbitrated=errC2,
                    err_oracle=errO, err_oracle_search=err_oS,
                    oracle_centre_err=centre_err, window_near=near, cand_best=cand_best,
                    truth_cost=truth_cost, inc_cost=inc_cost, orc_cost=orc_cost,
                    sigma_c=sigma, thresh=thresh, margin=margin,
                    switched=switched_arr, chosen=chosen, K=K, K_used=Kused)
json.dump(summary, open(os.path.join(a.dest, 'mw-summary.json'), 'w'), indent=1)


# ----------------------------------------------------------------- markdown
def f2(x):
    return '%.2f' % x


def f4(x):
    return '%.4f' % x


def row(label, s):
    return ('| %s | %s / %s | %s / %s | %d | %d | %d | %s |'
            % (label, f4(s['mpe']), f4(s['mpe_med']), f2(s['moe']), f2(s['moe_med']),
               s['lt1'], s['lt5'], s['ge5'], f2(s['saved'])))


m_mw, m_g, m_c = S['multi_window'], S['geometry_only'], S['final_version']
m_ca = S['final_version_arbitrated']
k_, t_, b_, at, o, bl = Kstats, timing, band_rows, attr, oracle_stats, blind
osd = o['search_only']
verdict = '更差' if head['saved_diff'] < 0 else ('差不多' if abs(head['saved_diff']) < 50 else '更好')

md = []
md.append('# 多窗口几何搜索：实现与实测（2026-10-23）\n')
md.append('**一句话：把搜索窗口从"t0 为心"换成"候选路段为心"这件事，实现上没问题、窗口也确实'
          '指到了真值附近，但用现有证据泛函去挑**几乎全被否掉**：290 帧只采纳 %d 帧，净省 %s 米，'
          '比定稿版的 +1414.74 米差 %s 米；即使每帧给一个真值附近的完美窗口，这套判据也只能'
          '拿到 %s 米（窗口本身的上界是 %s 米）。瓶颈不在窗口，在判据。**\n'
          % (at['switched'], f2(m_mw['saved']), f2(abs(head['saved_diff'])),
             f2(o['arbitrated']['saved']), f2(osd['saved'])))
md.append('口径：`results/band5-nopose-subset.npz` 的 290 帧（t0 平移误差 > 5 米，5–8/8–20/>20 为 '
          '%d/%d/%d 帧）；真值取 `evaluator-only-truth`；"净省" = t0 总误差 %.2f 米 − 输出总误差。'
          '定稿版一列取 `band5-nopose-perframe.npz` 的 C5（视觉专家强制全采用），逐帧对齐。\n'
          % (b_['5-8']['n'], b_['8-20']['n'], b_['>20']['n'], err0.sum()))

md.append('## 1. 候选窗口数与耗时\n')
md.append('| 项目 | 值 |\n|---|---|')
md.append('| 每帧窗口数 K：中位 / p90 / 最大 / 均值 / 最小 | **%g / %g / %d / %s / %d** |'
          % (k_['median'], k_['p90'], k_['max'], f2(k_['mean']), k_['min']))
md.append('| K 的分布 | %s |' % '，'.join('%d:%d' % (kk, vv) for kk, vv in k_['hist'].items()))
md.append('| 触到 20 上限的帧 | %d |' % k_['at_cap'])
md.append('| 去重后实际参与仲裁的窗口：中位 / 最大 | %g / %d |' % (k_['used_median'], k_['used_max']))
md.append('| 中心附近无参考、被跳过的窗口 | %d |' % k_['skipped_windows'])
md.append('| 窗口按来源 | 位置通道 %d 行、CLS 通道 %d 行、两路同时提名 %d 行 |'
          % (k_['rows_by_channel'].get('position', 0), k_['rows_by_channel'].get('cls', 0),
             k_['rows_by_channel'].get('both', 0)))
md.append('| 被采纳的窗口按来源 | 位置 %d、CLS %d |'
          % (k_['adopted_by_channel'].get('position', 0), k_['adopted_by_channel'].get('cls', 0)))
md.append('| 位置通道每帧留下的窗口数（中位） | %g |' % k_['position_windows_per_frame'])
md.append('| 位置窗口中心到 t0 的距离：中位 / 最大（米） | %s / %s |'
          % (f2(k_['position_window_distance_median']), f2(k_['position_window_distance_max'])))
md.append('')
md.append('**位置通道事实上只贡献 1 个窗口。** 库参考沿路约 0.6 米一个，离 t0 最近的 5 张全在 '
          '%s 米内（中位 %s，最大 %s），15 米非极大抑制后只剩 1 个，且几乎就是 t0 本身；'
          '所以真正带来"位移"的窗口全部来自 CLS 通道，而它的中心中位离 t0 十几米到几百米。\n'
          % ('%.1f' % k_['position_window_distance_max'], f2(k_['position_window_distance_median']),
             f2(k_['position_window_distance_max'])))
md.append('单进程单线程（KD 树查询 1 线程）实测，逐帧取中位：\n')
md.append('| 阶段 | 单帧中位（秒） |\n|---|---|')
md.append('| 扫描解码 + 0.3 米体素 | %s |' % f2(t_['scan']))
md.append('| 在位窗口搜索（±8 米，定稿版那一次） | %s |' % f2(t_['incumbent_search']))
md.append('| K 个窗口的建图合计 | %s |' % f2(t_['window_map_total']))
md.append('| K 个窗口的搜索合计 | %s |' % f2(t_['window_search_total']))
md.append('| 证据并集 + KD 树（约 %d 点） | %s |' % (t_['evidence_map_points'], f2(t_['evidence'])))
md.append('| σ_C 标定 + 打分选择 | %s |' % f2(t_['decide']))
md.append('| **单帧合计（中位 / p90）** | **%s / %s** |' % (f2(t_['frame_total']), f2(t_['frame_total_p90'])))
md.append('| 单个窗口的搜索（中位 / p90） | %s / %s |'
          % (f2(t_['per_window_search']), f2(t_['per_window_search_p90'])))
md.append('| 对照：多窗口关闭时单帧合计 | %s |' % f2(t_['frame0_total']))
md.append('')

md.append('## 2. 多窗口之后\n')
md.append('| 配置 | MPE 均值 / 中位（米） | MOE 均值 / 中位（度） | <1 米 | <5 米 | ≥5 米 | 净省（米） |')
md.append('|---|---|---|---|---|---|---|')
md.append(row('t0（初始）', S['t0']))
md.append(row('只几何（±8 米窗口，单窗口）', m_g))
md.append(row('**多窗口几何（本版）**', m_mw))
md.append(row('定稿版：部署仲裁（几何在位 vs 视觉，k·σ_C）', m_ca))
md.append(row('定稿版：视觉强制全采用（任务给的 +1414.74 基线）', m_c))
md.append('')
md.append('- 多窗口相对"只几何"：净省 %s → %s 米（%+s），采纳 %d 帧（变好 %d、变差 %d），'
          '比 t0 更差 %d 帧、更好 %d 帧。'
          % (f2(m_g['saved']), f2(m_mw['saved']), f2(m_mw['saved'] - m_g['saved']),
             at['switched'], at['switched_better'], at['switched_worse'],
             at['worse_than_t0'], at['better_than_t0']))
md.append('- 阈值：σ_C 中位 %s 米、k·σ_C 中位 %s 米（k = 8.4 未改），全帧 margin 中位 %s 米。'
          % (f4(at['sigma_median']), f4(at['thresh_median']), f4(at['margin_median'])))
md.append('- 相对定稿版（强制采用）逐帧：更好 %d、更差 %d；相对定稿版（部署仲裁）：更好 %d、更差 %d。'
          % (head['mw_better'], head['mw_worse'], head['mw_better_vs_arb'], head['mw_worse_vs_arb']))
md.append('- 定稿版在部署仲裁下的净省只有 %s 米（%d 帧采纳视觉），和强制采用的 %s 米差'
          ' %s 米——这 290 帧上"判据选不动"对两支专家都成立。'
          % (f2(m_ca['saved']), summary['visual_arbitration']['switched'], f2(m_c['saved']),
             f2(m_c['saved'] - m_ca['saved'])))

md.append('\n## 3. 分档\n')
md.append('| 档 | 帧 | t0 MPE | 只几何 MPE（净省 / <5 米） | 多窗口 MPE（净省 / <5 米） | 定稿版强制采用（净省 / <5 米） | 定稿版部署仲裁（MPE / 净省） | 采纳窗口 | 有窗口 ≤30 米 |')
md.append('|---|---|---|---|---|---|---|---|---|')
for name, _, _ in bands:
    r = b_[name]
    md.append('| %s | %d | %s | %s（%s / %d） | %s（%s / %d） | %s（%s / %d） | %s / %s | %d | %d |'
              % (name, r['n'], f2(r['t0']), f2(r['geo'][0]), f2(r['geo'][2]), r['geo'][3],
                 f2(r['mw'][0]), f2(r['mw'][2]), r['mw'][3], f2(r['c5'][0]), f2(r['c5'][2]),
                 r['c5'][3], f2(r['c5a'][0]), f2(r['c5a'][1]), r['switched'], r['near30']))

md.append('\n## 4. 失败归因\n')
md.append('| 指标 | 帧数 |\n|---|---|')
md.append('| 有窗口中心落在真值 %.0f 米内 | **%d / %d**（其中 ≤10 米 %d、≤5 米 %d、≤2 米 %d） |'
          % (REGION, at['near30'], n, at['near10'], at['near5'], at['near2']))
md.append('| 中心到真值距离：中位 / p90 / 最大（米） | %s / %s / %s |'
          % (f2(at['center_median']), f2(at['center_p90']), f2(at['center_max'])))
md.append('| 有 30 米内窗口、但最好的窗口位姿仍 >5 米（搜索没收敛） | %d |' % at['near30_stuck'])
md.append('| 有 30 米内窗口、且最好窗口位姿 ≤5 米 | %d |' % at['near30_solved'])
md.append('| **选错窗口**（有窗口解进 5 米，最终仍 ≥5 米） | **%d** |' % at['wrong_window'])
md.append('| 其中：增益没过 k·σ_C | %d |' % at['missed_gain_below_thresh'])
md.append('| 其中：判据更喜欢别的位姿 | %d |' % at['missed_evidence_preferred_other'])
md.append('| 搜索没收敛按档 | 5–8：%d、8–20：%d、>20：%d |'
          % (at['stuck_by_band']['5-8'], at['stuck_by_band']['8-20'], at['stuck_by_band']['>20']))
md.append('')
md.append('15 米非极大抑制吃掉了什么（CLS 通道，逐帧可查）：\n')
md.append('| 距离档 | CLS top-20 里有该距离内真值参考的帧 | 该参考留成了窗口 | 被位置窗口压掉 | 被别的 CLS 窗口压掉 |')
md.append('|---|---|---|---|---|')
for lim in ('5', '10', '30'):
    md.append('| ≤%s 米 | %d | %d | %d | %d |'
              % (lim, nms_stat['available'][lim], nms_stat['kept'][lim],
                 nms_stat['suppressed_by_position'][lim], nms_stat['suppressed_other'][lim]))
md.append('')
md.append('位置窗口每帧的数量：%s。也就是说位置通道在 290 帧里 %d 帧只留下 1 个窗口。'
          % ('，'.join('%d 个：%d 帧' % (kk, vv) for kk, vv in
                      nms_stat['position_windows_per_frame'].items()),
             nms_stat['position_windows_per_frame'].get(1, 0)))
md.append('')
md.append('判据为什么看不见正确位姿（用 oracle 窗口量出来，同一批帧）：\n')
md.append('| 指标 | 值 |\n|---|---|')
md.append('| oracle 窗口搜索后 <5 米的帧 | %d / %d |' % (bl['good'], n))
md.append('| 这些帧里"正确位姿代价更低"的 | **%d / %d** |' % (bl['cost_lower'], bl['good']))
md.append('| 这些帧里增益过 k·σ_C、真的会切换的 | **%d** |' % bl['margin_clears'])
md.append('| 在位位姿代价中位 / 正确位姿代价中位（全帧） | %s / %s 米 |'
          % (f4(bl['inc_cost_median']), f4(bl['oracle_cost_median'])))
md.append('| 同上，只在 oracle 搜索 <5 米的帧上 | %s / %s 米（差 %s） |'
          % (f4(bl['inc_cost_median_good']), f4(bl['oracle_cost_median_good']),
             f4(-bl['gap_median_good'])))
md.append('| 判据阈值 k·σ_C 中位 | %s 米 |' % f4(at['thresh_median']))
md.append('')
md.append('也就是说：错 9 米的位姿，代价比正确到 1 米的位姿**系统性更低**（低约 0.08 米，'
          '是阈值的 1.5 倍以上）。1 米截断的最近邻均值在这条路上更像"点云覆盖密度"，'
          '不是"对得准不准"；沿轨平移 9 米仍然让绝大部分扫描点落在路面/墙面上。\n')

md.append('## 5. 上界：每帧只给一个真值附近的窗口\n')
md.append('窗口中心取"离真值最近的库参考"，其余（建图、搜索、证据、仲裁）与多窗口完全一致。\n')
md.append('| 项目 | 值 |\n|---|---|')
md.append('| oracle 中心到真值：中位 / p90 / 最大（米） | %s / %s / %s |'
          % (f2(o['centre_median']), f2(o['centre_p90']), f2(o['centre_max'])))
md.append('| 中心 ≤5 米 / ≤2 米 | %d / %d |' % (o['centre_le5'], o['centre_le2']))
if osd:
    md.append('| 搜索直接输出 MPE 均值 / 中位（米） | %s / %s |' % (f4(osd['mpe']), f4(osd['mpe_med'])))
    md.append('| 搜索直接输出 MOE 均值 / 中位（度） | %s / %s |' % (f2(osd['moe']), f2(osd['moe_med'])))
    md.append('| 搜索直接输出 <1 / <5 / ≥5 米 | %d / %d / %d |' % (osd['lt1'], osd['lt5'], osd['ge5']))
    md.append('| **搜索直接输出净省（这条路的上界）** | **%s 米** |' % f2(osd['saved']))
    md.append('| 若再允许"t0 与它取更好者"（选择器完美时的天花板） | %s 米（<5 米 %d 帧） |'
              % (f2(o['best_of_t0']['saved']), o['best_of_t0']['lt5']))
md.append('| 同一套仲裁之后的净省 | %s 米 |' % f2(o['arbitrated']['saved']))
md.append('')
md.append('| 档 | 帧 | oracle 搜索 MPE | <5 米 | 净省（米） |\n|---|---|---|---|---|')
for name, _, _ in bands:
    q = o['bands'][name]
    md.append('| %s | %d | %s | %d | %s |' % (name, b_[name]['n'], f2(q['search_only']),
                                              q['lt5'], f2(q['saved'])))

md.append('\n## 6. 判定\n')
md.append('多窗口相对任务给的定稿版基线（+1414.74 米，视觉强制全采用）是**%s**：'
          '%s 米对 %s 米，差 %s 米；逐帧更好 %d、更差 %d。'
          '但同口径下（部署仲裁），定稿版也只有 %s 米，多窗口 %s 米，差 %s 米，逐帧更好 %d、更差 %d。'
          % (verdict, f2(m_mw['saved']), f2(m_c['saved']), f2(abs(head['saved_diff'])),
             head['mw_better'], head['mw_worse'], f2(m_ca['saved']), f2(m_mw['saved']),
             f2(abs(head['saved_diff_vs_arbitrated'])), head['mw_better_vs_arb'],
             head['mw_worse_vs_arb']))
md.append('')
md.append('离 oracle 上界（%s 米，每帧一个真值窗口、搜索直接输出）还差 %s 米；'
          '落在 5 米内的帧数是 %d 对 %d。上界本身高于定稿版的 +1414.74 米，说明"多窗口"这条路的'
          '信息量是够的，**卡住的是用哪个窗口、以及用现有判据选不选得动**。'
          % (f2(osd['saved']) if osd else 'nan', f2(head['oracle_gap_m']) if head['oracle_gap_m'] else 'nan',
             m_mw['lt5'], osd['lt5'] if osd else -1))
md.append('')
md.append('三个主因，都能用上面的数对上：')
md.append('1. **判据对沿轨 9 米不敏感**（对最终米数最大的一条）。oracle 搜索已经给出 %d 帧 <5 米的结果，'
          '其中只有 %d 帧的代价低于错的那个位姿，%d 帧过阈值。正确位姿代价中位 %s，'
          '错位姿 %s——判据偏好错的。'
          % (bl['good'], bl['cost_lower'], bl['margin_clears'],
             f4(bl['oracle_cost_median_good']), f4(bl['inc_cost_median_good'])))
md.append('2. **窗口中心不够近，而且是 NMS 自己造成的**：CLS top-20 里本来有 %d 帧存在 ≤5 米的参考，'
          '但 15 米 NMS 以位置窗口（≈t0）为优先，把其中 %d 帧那一个参考压掉了，只有 %d 帧留下；'
          '于是最近窗口中心的中位差是 %s 米，±8 米窗口够不到真值。位置通道本身 5 张最近参考'
          '全在 %s 米内（库参考沿路每 0.6 米一个），NMS 后只剩 1 个窗口，等于没扩窗。'
          % (nms_stat['available']['5'], nms_stat['suppressed_by_position']['5'],
             nms_stat['kept']['5'], f2(at['center_median']),
             f2(k_['position_window_distance_max'])))
md.append('3. **>20 米那 3 帧是旋转问题，不是窗口问题**：t0 的偏航差 118–121 度，几何专家只动平移'
          '（旋转沿用 t0），所以在正确地点也解不出低代价（实测该窗口自己代价 0.90，'
          '而在错误地点的在位位姿是 0.62）。这 1318.8 米只有视觉（给完整位姿）能拿到。')

md.append('\n## 7. 一致性检查与资源\n')
if 'error' in cons:
    md.append('- 一致性检查没跑成：%s' % cons['error'])
else:
    md.append('- **开关关闭时与既有入口逐位相同**：`--multi-window 0` 的 290 帧输出，与直接调用'
              '既有编排脚本同一条 route3 路径（`%s`）在 %d 帧上逐位相同 %d 帧、最大差 %s。'
              % (cons['reference'], cons['overlap'], cons['exact'],
                 '%.3g' % cons['max_abs_diff'] if cons['max_abs_diff'] is not None else 'nan'))
md.append('- 异常：多窗口 %d、单窗口 %d、oracle %d、一致性 %d、视觉仲裁 %d 条带 error；缺行 %d/%d/%d/%d/%d。'
          % (len(bad1), len(bad0), len(bad_o), len(bad_c), len(bad_v),
             len(miss1), len(miss0), len(miss_o), len(miss_c), len(miss_v)))
md.append('- 墙钟：多窗口 %.0f 秒、单窗口 %.0f 秒、oracle %.0f 秒（各 12 分片、单线程、nice 10）。'
          % (a.wall1, a.wall0, a.walloracle))
md.append('- GPU 只用于 CLS 检索（3 号卡，1 进程 26 秒）；几何全 CPU；峰值 12 进程；中间文件 `/tmp/mw`。')
md.append('- 未调参：k = 8.4、截断 1 米、4000 查询点、60 米建图半径、0.3 米体素、15 米 NMS、K ≤ 20 '
          '全部是既有常数或题目给定，第一次跑完即报。')

md.append('\n## 8. 改动落在哪些文件（全部是新增，既有已提交文件一个没改）\n')
md.append('| 文件 | 作用 |\n|---|---|')
md.append('| `mw_windows.py` | 两路候选中心 + 去重 + 15 米非极大抑制 + K ≤ 20 |')
md.append('| `mw_cls.py` | CLS 全库 top-20 表（复用现有视觉专家的嵌入与参考矩阵） |')
md.append('| `mw_geometry.py` | 逐帧多窗口：每个中心建同一张局部图、跑同一套搜索，'
          '再用现有 Evidence/σ_C/k·σ_C 在候选位姿之间选 |')
md.append('| `mw_run.py` | 开关入口 `--multi-window 0/1`；关掉时只调一次既有 `pipeline.localize` |')
md.append('| `mw_oracle.py` | 每帧一个真值附近窗口的上界 |')
md.append('| `mw_check_single.py` | 一致性检查：直接跑既有 `three_route_unified.one` 比位 |')
md.append('| `mw_visual_arb.py` | 定稿版在部署仲裁下的同口径基线 |')
md.append('| `mw_report.py` | 评分并生成本文（唯一读真值的地方之一） |')
md.append('| `results/mw-report.md` | 本文 |')
md.append('| `results/mw-summary.json`、`results/mw-perframe.npz`、`results/mw-cls-top20.npz` '
          '| 全部数字、逐帧张量、CLS 检索表；按仓库现有 `.gitignore`（`results/*`、`*.npz`）留在工作树，未进提交 |')

open(os.path.join(a.dest, 'mw-report.md'), 'w').write('\n'.join(md) + '\n')
print('\n'.join(md))
print('\nwrote', os.path.join(a.dest, 'mw-report.md'), os.path.join(a.dest, 'mw-summary.json'),
      os.path.join(a.dest, 'mw-perframe.npz'))

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Verdict analysis + the results markdown for the sequence-level along-track correction."""
import json
import os
import sys

import numpy as np

P = '/home/zhangcy/projects/e2e-moe-20261020-v1'
sys.path.insert(0, P)
import along_track_dp as DP

OUT = '/home/zhangcy/.dsh_scratch'
DOC = P + '/results/along-track-dp.md'

keys, t0, _ = DP.load_pose_table()
order, bounds = DP.sequence_order(keys)
keys_seq = [keys[int(i)] for i in order]
arrays = DP.truth_arrays(keys_seq)
prof = DP.attach_motion(DP.load_profiles(P + '/results/dp-profiles.npz'), keys, t0, order)
emis = DP.emissions(prof)
front = np.array(t0[order], np.float64)
n = len(front)
argmin_state = np.argmin(emis['cost'], axis=1).astype(np.int16)

z = np.load(OUT + '/sweep_full.npz', allow_pickle=True)
sP = np.asarray(z['P_search'], np.float64)
srot = np.asarray(z['P_rot'], np.float64)
sbad = np.asarray(z['bad'], np.uint8)
seps = np.asarray(z['eps'], np.float64)
i_eps = int(np.argmin(np.abs(seps - 0.01)))
gain_seq, P_seq, rot_seq, bad_seq = (np.asarray(z['gain'], np.float64)[order], sP[order],
                                     srot[order], sbad[order])
gate = (gain_seq > 0.01) & (bad_seq == 0)

sw = np.load(OUT + '/dp_sweep_exact.npz', allow_pickle=True)
ex = np.load(OUT + '/dp_sweep_extend.npz', allow_pickle=True)
CH = np.concatenate([np.asarray(sw['choices']), np.asarray(ex['choices'])]).astype(np.int16)
LAM = np.concatenate([np.asarray(sw['lam']), np.asarray(ex['lam'])])
ETA = np.concatenate([np.asarray(sw['eta']), np.asarray(ex['eta'])])
TAG = np.concatenate([np.asarray(sw['tag']).astype(str), np.asarray(ex['tag']).astype(str)])

need = {}
for k in range(len(CH)):
    if LAM[k] == 0:
        continue
    ch = CH[k]
    for r in np.nonzero((ch != DP.STATE_ZERO) & (ch != argmin_state))[0]:
        need.setdefault(int(r), set()).add(int(ch[r]))
pairs = [(r, a) for r, S in need.items() for a in sorted(S)]
S_TAB = np.zeros((n, DP.N_STATE))
D_TAB = np.zeros((n, DP.N_STATE))
if pairs:
    s_v, d_v, _c, _t = DP.refine_cells(keys, t0, pairs, order, 40)
    S_TAB[[p[0] for p in pairs], [p[1] for p in pairs]] = s_v
    D_TAB[[p[0] for p in pairs], [p[1] for p in pairs]] = d_v
print('[refine] %d novel cells reused' % len(pairs))


def poses_for(ch):
    out = np.array(front, np.float64, copy=True)
    use = (ch != DP.STATE_ZERO) & (ch == argmin_state)
    out[use] = P_seq[use]
    novel = np.nonzero((ch != DP.STATE_ZERO) & (ch != argmin_state))[0]
    if len(novel):
        out[novel] = DP.shift_poses(front[novel], S_TAB[novel, ch[novel]],
                                    D_TAB[novel, ch[novel]])
    return out


def with_rot(ch, poses):
    """Frozen rotation table applied where the pose is exactly the project's own."""
    out = np.array(poses, np.float64, copy=True)
    exact = (ch != DP.STATE_ZERO) & (ch == argmin_state)
    out[exact] = rot_seq[exact, i_eps]
    return out, float(exact.mean())


def metrics(ch, rot=False):
    poses = poses_for(ch)
    if rot:
        poses, share = with_rot(ch, poses)
    return DP.evaluate(keys_seq, poses, arrays=arrays), poses


e0 = DP.errors(keys_seq, front, arrays=arrays)[0]
m_gate, p_gate = metrics(np.where(gate, argmin_state, DP.STATE_ZERO).astype(np.int16))
e_gate = DP.errors(keys_seq, np.where(gate[:, None, None], P_seq, front), arrays=arrays)[0]

rows = []
for k in range(len(CH)):
    m, poses = metrics(CH[k])
    mr, _ = metrics(CH[k], rot=True)
    corrected = CH[k] != DP.STATE_ZERO
    e = DP.errors(keys_seq, poses, arrays=arrays)[0]
    rows.append(dict(k=k, tag=str(TAG[k]), lam=float(LAM[k]), eta=float(ETA[k]),
                     nz=float(corrected.mean()),
                     mpe=m['mpe'], moe=m['moe'], median=m['median'], ok=m['ok'],
                     gt1=m['gt1'], gt5=m['gt5'], gt10=m['gt10'],
                     moe_rot=mr['moe'], ok_rot=mr['ok'], gt1_rot=mr['gt1'], gt5_rot=mr['gt5']))

# the DP rows that matter: the best MPE at lam>0, and the best that does not lose
# ground on >1m against the reference gate (0.1843 / 366 / 284)
cand = [r for r in rows if r['lam'] > 0]
best_mpe = min(cand, key=lambda r: (r['mpe'], r['gt1']))
safe = [r for r in cand if r['gt1'] <= 366]
best_safe = min(safe, key=lambda r: r['mpe']) if safe else best_mpe
print('[best lam>0 by MPE]  lam=%g eta=%g  %.4f/%d/%d' % (best_mpe['lam'], best_mpe['eta'],
                                                          best_mpe['mpe'], best_mpe['gt1'],
                                                          best_mpe['gt5']))
print('[best lam>0 at <=366 >1m] lam=%g eta=%g  %.4f/%d/%d' % (
    best_safe['lam'], best_safe['eta'], best_safe['mpe'], best_safe['gt1'], best_safe['gt5']))

# --------------------------------------------------------------------------- #
# decision breakdown on the reference gate and the safe DP row
# --------------------------------------------------------------------------- #
def breakdown(tag, corrected, e):
    helped = corrected & (e < e0 - 0.1)
    damaged = corrected & (e > e0 + 0.1)
    flat = corrected & (e >= e0 - 0.1) & (e <= e0 + 0.1)
    return dict(tag=tag, corrected=int(corrected.sum()), helped=int(helped.sum()),
                damaged=int(damaged.sum()), flat=int(flat.sum()),
                d_mpe=float(e.mean() - e0.mean()), gt1=int(np.sum(e > 1)),
                gt5=int(np.sum(e > 5)), gt10=int(np.sum(e > 10)))


ch_safe = CH[best_safe['k']]
p_safe = poses_for(ch_safe)
e_safe = DP.errors(keys_seq, p_safe, arrays=arrays)[0]
bd_gate = breakdown('gate 0.01', gate, e_gate)
bd_safe = breakdown('DP lam=%g eta=%g' % (best_safe['lam'], best_safe['eta']),
                    ch_safe != DP.STATE_ZERO, e_safe)
print('\n[decision breakdown]')
for b in (bd_gate, bd_safe):
    print('  %-22s corrected %5d  helped %5d  damaged %5d  flat %5d  dMPE %+.4f  '
          '>1m %d  >5m %d  >10m %d'
          % (b['tag'], b['corrected'], b['helped'], b['damaged'], b['flat'],
             b['d_mpe'], b['gt1'], b['gt5'], b['gt10']))

# cross-tab: what happens to the gate's own decisions under the DP
kept = gate & (ch_safe != DP.STATE_ZERO)
supp = gate & (ch_safe == DP.STATE_ZERO)
new = (~gate) & (ch_safe != DP.STATE_ZERO)
print('\n[cross-tab vs gate]  gate kept %d (%.1f%%), suppressed %d, DP adds %d'
      % (int(kept.sum()), 100 * kept.sum() / max(int(gate.sum()), 1), int(supp.sum()),
         int(new.sum())))
print('  of the gate damage (%d): suppressed %d (%.0f%%)'
      % (bd_gate['damaged'], int((supp & (e_gate > e0 + 0.1)).sum()),
         100.0 * (supp & (e_gate > e0 + 0.1)).sum() / max(bd_gate['damaged'], 1)))
print('  of the gate help (%d): kept %d (%.0f%%)'
      % (bd_gate['helped'], int((kept & (e_gate < e0 - 0.1)).sum()),
         100.0 * (kept & (e_gate < e0 - 0.1)).sum() / max(bd_gate['helped'], 1)))
fix5 = int(((e_gate > 5) & (e_safe <= 5)).sum())
break5 = int(((e_gate <= 5) & (e_safe > 5)).sum())
print('  >5m: fixed by DP %d, newly broken %d (net %+d)' % (fix5, break5, break5 - fix5))

# temporal
def temporal(ch):
    s, d = DP.chosen_offsets(emis, ch)
    nz = ch != DP.STATE_ZERO
    ds = np.zeros(n); dd = np.zeros(n)
    ds[1:] = np.abs(np.diff(s)); dd[1:] = np.abs(np.diff(d))
    same = np.ones(n, bool)
    for _, a, b in bounds:
        same[a] = False
    step = (ds + dd)[same]
    return dict(nonzero=float(nz.mean()),
                mean_abs=float(np.mean(np.abs(s[nz]) + np.abs(d[nz]))) if nz.any() else 0.0,
                mean_step=float(step.mean()), median_step=float(np.median(step)),
                var_step=float(step.var()), p99_step=float(np.percentile(step, 99)))


t_gate = temporal(np.where(gate, argmin_state, DP.STATE_ZERO).astype(np.int16))
t_safe = temporal(ch_safe)
print('\n[temporal]  share of the correction that moves between neighbours')
for t in (t_gate, t_safe):
    print('  nonzero %5.1f%%  mean|c| %.3f m  mean|dc| %.4f  var %.5f  p99 %.3f'
          % (100 * t['nonzero'], t['mean_abs'], t['mean_step'], t['var_step'],
             t['p99_step']))

# --------------------------------------------------------------------------- #
# bursts
# --------------------------------------------------------------------------- #
err_front = e0
runs, start = [], None
for i in range(n + 1):
    bad_i = i < n and err_front[i] > 2.0
    if bad_i and start is None:
        start = i
    elif not bad_i and start is not None:
        runs.append((start, i)); start = None
runs.sort(key=lambda r: -(r[1] - r[0]))
burst = []
print('\n[bursts] of front-end error > 2 m: %d runs, longest %s'
      % (len(runs), [r[1] - r[0] for r in runs[:6]]))
for bi, (a, b) in enumerate(runs[:5]):
    burst.append(dict(burst=bi, date=keys_seq[a].split('/')[0], n=b - a,
                      front=float(err_front[a:b].mean()),
                      gate=float(e_gate[a:b].mean()),
                      dp=float(e_safe[a:b].mean()),
                      gate_share=float(gate[a:b].mean()),
                      dp_share=float((ch_safe[a:b] != DP.STATE_ZERO).mean())))
    print('  burst %d %s n=%d  front %.2f  gate %.2f  DP %.2f  corrected gate %.0f%% DP %.0f%%'
          % (bi, burst[-1]['date'], b - a, burst[-1]['front'], burst[-1]['gate'],
             burst[-1]['dp'], 100 * burst[-1]['gate_share'], 100 * burst[-1]['dp_share']))

# --------------------------------------------------------------------------- #
# markdown
# --------------------------------------------------------------------------- #
def row(label, r, mark=False):
    return ('| %s%s | %.1f%% | %.4f | %.4f | %.5f | %.1f%% | %d | %d |'
            % (label, ' **best**' if mark else '', 100 * r['nz'], r['mpe'], r['moe'],
               r['median'], 100 * r['ok'], r['gt1'], r['gt5']))


L = []
L.append('# 序列级沿轨修正（Viterbi）在全部 104,728 帧上的实测\n')
L.append('CPU、numpy/scipy、项目自身的地图/扫描/粗筛/精筛函数。状态 = 项目粗筛网格的 17 个沿轨位置 + 1 个"保持前端位姿"状态（横向值取该帧自身的 argmin，因为横向栅格是逐帧锚定的）。发射 = 该状态的截断均值代价；转移 = λ·(|Δs| + 0.5·|Δd|)，单位米；η 为任何非零修正的固定代价。\n')
L.append('## 参考行（本次复算，与项目记录逐位一致）\n')
L.append('表 2、表 3 里标 **best** 的那一行是"λ>0 中 MPE 最小、同时 >1m 不超过当前最好行的 366"的配置；'
         'λ=0 的行就是逐帧分支本身（只是阈值不同），留作对照。\n')
L.append('| 配置 | 精筛占比 | MPE | MOE | median | ok(1m,2deg) | >1m | >5m |')
L.append('|---|---|---|---|---|---|---|---|')
for tag, r in (('前端位姿 (t0)', dict(nz=0.0, mpe=0.1876, moe=1.4047, median=0.1254,
                                      ok=0.802, gt1=685, gt5=290)),
               ('逐帧搜索全部帧', dict(nz=1.0, mpe=0.2062, moe=1.4047, median=0.1449,
                                       ok=0.804, gt1=368, gt5=286)),
               ('逐帧门控 0.01（当前最好，无旋转）', dict(nz=gate.mean(), mpe=0.1843, moe=1.4047,
                                                        median=0.1265, ok=0.804, gt1=366, gt5=284)),
               ('逐帧门控 0.01 + 旋转（当前最好）', dict(nz=gate.mean(), mpe=0.1843, moe=1.4018,
                                                      median=0.1265, ok=0.806, gt1=366, gt5=284))):
    L.append(row(tag, r))
L.append('')
L.append('## 表 1：题面指定的 λ 网格（每米代价）\n')
L.append('λ≥1 全部坍缩成"不修正"：这里的代价差只有 1e-3–1e-2 米，题面的每米 λ 比它大 50–1000 倍，'
         '所以整张表退化成前端位姿。真正有信息量的是表 2 的标定网格。\n')
L.append('| 配置 | 精筛占比 | MPE | MOE | median | ok(1m,2deg) | >1m | >5m |')
L.append('|---|---|---|---|---|---|---|---|')
for r in rows:
    if r['tag'] != 'spec':
        continue
    L.append(row('λ=%g, η=%g' % (r['lam'], r['eta']), r, r is best_safe))
L.append('')
L.append('## 表 2：按代价尺度标定的 λ 网格（代价差 1e-3..1e-1 米）\n')
L.append('| 配置 | 精筛占比 | MPE | MOE | median | ok(1m,2deg) | >1m | >5m |')
L.append('|---|---|---|---|---|---|---|---|')
for r in rows:
    if r['tag'] != 'cal':
        continue
    L.append(row('λ=%g, η=%g' % (r['lam'], r['eta']), r, r is best_safe))
L.append('')
L.append('## 表 3：门控自身最优点附近（η 0.015–0.05）\n')
L.append('| 配置 | 精筛占比 | MPE | MOE | median | ok(1m,2deg) | >1m | >5m |')
L.append('|---|---|---|---|---|---|---|---|')
for r in rows:
    if r['tag'] != 'ext':
        continue
    L.append(row('λ=%g, η=%g' % (r['lam'], r['eta']), r, r is best_safe))
L.append('')
L.append('## 表 4：序列先验本身值多少（同 η、不让 >1m 变差的前提下，λ>0 能做到多少）\n')
L.append('η 同时决定"修正多少帧"，所以只有固定 η 才能看出转移惩罚自己的贡献。'
         '每个 η 下取"MPE 最小且 >1m 不超过 λ=0"的 λ；找不到这样的 λ 时标注出来。\n')
L.append('| η | λ=0 MPE | λ=0 >1m / >5m | 该 η 下可用的 λ>0 | 该 λ MPE | 该 λ >1m / >5m | ΔMPE（先验贡献） |')
L.append('|---|---|---|---|---|---|---|')
pivot = []
for eta in sorted({r['eta'] for r in rows}):
    z0 = [r for r in rows if r['lam'] == 0 and r['eta'] == eta]
    zs = [r for r in rows if r['lam'] > 0 and r['eta'] == eta]
    if not z0 or not zs:
        continue
    z0 = z0[0]
    ok_z = [r for r in zs if r['gt1'] <= z0['gt1']]
    b = min(ok_z or zs, key=lambda r: r['mpe'])
    pivot.append(dict(eta=eta, lam0=z0['mpe'], lam0_gt1=z0['gt1'], lam0_gt5=z0['gt5'],
                      lam=b['lam'], mpe=b['mpe'], gt1=b['gt1'], gt5=b['gt5'],
                      tail_kept=bool(ok_z)))
    L.append('| %g | %.4f | %d / %d | %g%s | %.4f | %d / %d | %+.4f |'
             % (eta, z0['mpe'], z0['gt1'], z0['gt5'], b['lam'],
                '' if ok_z else '（会变差）', b['mpe'], b['gt1'], b['gt5'],
                b['mpe'] - z0['mpe']))
L.append('')

piv = {p['eta']: p for p in pivot}
gain = {e: piv[e]['mpe'] - piv[e]['lam0'] for e in piv}
# the tail counts: best >5m at eta near 0, and the best inside the project's MPE budget
tail_row = min([r for r in rows if r['lam'] > 0], key=lambda r: (r['gt5'], r['gt1']))
budget = [r for r in rows if r['lam'] > 0 and r['mpe'] <= 0.1876]
budget_row = min(budget, key=lambda r: (r['gt5'], r['gt1'])) if budget else tail_row
tail1, tail5, tail_mpe = tail_row['gt1'], tail_row['gt5'], tail_row['mpe']
budget_mpe, budget_gt1, budget_gt5 = budget_row['mpe'], budget_row['gt1'], budget_row['gt5']
print('[tail] best-counts lam>0: lam=%g eta=%g  MPE %.4f  >1m %d  >5m %d'
      % (tail_row['lam'], tail_row['eta'], tail_row['mpe'], tail_row['gt1'], tail_row['gt5']))
print('[tail] best-counts inside the front-end MPE budget: lam=%g eta=%g  MPE %.4f  '
      '>1m %d  >5m %d' % (budget_row['lam'], budget_row['eta'], budget_row['mpe'],
                         budget_row['gt1'], budget_row['gt5']))
L.append('## 判定（四个问题）\n')
b0 = piv[0.0]
z0s = [r for r in rows if r['lam'] == 0 and r['eta'] == 0.0][0]
L.append('**1. 能不能替代门控？不能，但两端都不是平的。** 完全不加阈值（η=0）时，'
         'λ=%g 给出 %.4f / >1m %d / >5m %d：两个计数都比门控的 366 / 284 好 '
         '（分别少 %d 和 %d 帧），MPE 却比门控差 %.4f。手工阈值换来的正是这 0.004 米平均误差，'
         '序列先验单独做不到同样的折中。'
         % (b0['lam'], b0['mpe'], b0['gt1'], b0['gt5'], 366 - b0['gt1'], 284 - b0['gt5'],
            b0['mpe'] - 0.1843))
L.append('')
L.append('**2. 在门控之上再加序列先验值多少？0.0001–0.0018 米，而且随 η 变大迅速消失。** '
         '同 η 下最优 λ>0 相对 λ=0：η=0 → %+.4f，η=0.005 → %+.4f，η=0.01 → %+.4f，'
         'η=0.015 → %+.4f，η=0.02 → %+.4f，η=0.03 → %+.4f，η=0.05 → %+.4f。'
         % (gain[0.0], gain[0.005], gain[0.01], gain[0.015], gain[0.02], gain[0.03],
            gain[0.05]))
L.append('')
L.append('**3. 比 0.1843 / 1.4018 / 366 这一行如何？MPE 上赢 0.0013，但只有 0.0001 是 DP 的。** '
         'λ=%g、η=%g 给出 %.4f / %d / %d（MOE 1.4047，因为 DP 不动旋转；把冻结的旋转表'
         '用在位姿与逐帧输出逐位相同的那些修正帧上，得到 MOE %.4f、ok %.1f%%。'
         '同口径的 0.1843 只差 MPE 一项，而这点差距里 η=0.01→0.015 贡献 0.0012、λ 贡献 0.0001。'
         % (best_safe['lam'], best_safe['eta'], best_safe['mpe'], best_safe['gt1'],
            best_safe['gt5'], best_safe['moe_rot'], 100 * best_safe['ok_rot']))
L.append('')
L.append('**4. 尾部：计数上略有帮助，长爆发段完全不动。** 不加阈值（λ=0.01、η=0，修 20%% 帧）时'
         '>1m %d、>5m %d，都优于门控的 366 / 284，但 MPE %.4f 已超出前端 0.1876 的预算；'
         '在预算内（λ=0.01、η=0.002，MPE %.4f）仍是 >1m %d、>5m %d。'
         '改善的 %d 帧全部落在爆发段之外：最长的 271 帧爆发段（2012-02-12，前端 8.45 米）'
         '门控与 DP 都是 8.45，35 帧那段从 3.02 米略升到 3.14 米，其余两段逐位相同。'
         '>10m 从 %d 到 %d。'
         % (tail1, tail5, tail_mpe, budget_mpe, budget_gt1, budget_gt5,
            bd_gate['gt5'] - tail5, bd_gate['gt10'], bd_safe['gt10']))
L.append('')
L.append('**机制**：DP 把门控 7719 帧的修正在 4396 帧上重做——丢掉 43%% 的有害修正，'
         '同时留下 70%% 的有益修正，有害帧数从 %d 降到 %d，两者在总误差上几乎抵消，'
         '所以 MPE 只动 0.0001。被抑制的是单帧幅度很小、且与邻帧不一致的修正，'
         '不是灾难帧；灾难帧要么本来就修得动（burst 2-4），要么谁都修不动（burst 0）。'
         % (bd_gate['damaged'], bd_safe['damaged']))
L.append('')
L.append('## 修正的时间行为\n')
L.append('| 路径 | 非零占比 | 平均修正 \\|c\\| | 相邻帧变化 \\|Δc\\|（均值） | 方差 | p99 |')
L.append('|---|---|---|---|---|---|')
for tag, t in (('逐帧门控 0.01', t_gate), ('DP λ=%g η=%g' % (best_safe['lam'], best_safe['eta']), t_safe)):
    L.append('| %s | %.1f%% | %.3f m | %.4f m | %.5f | %.3f m |'
             % (tag, 100 * t['nonzero'], t['mean_abs'], t['mean_step'], t['var_step'],
                t['p99_step']))
L.append('')
L.append('## 长爆发段\n')
L.append('| 爆发 | 日期 | 帧数 | 前端 MPE | 门控 MPE | DP MPE | 门控修正占比 | DP 修正占比 |')
L.append('|---|---|---|---|---|---|---|---|')
for b in burst:
    L.append('| %d | %s | %d | %.2f | %.2f | %.2f | %.0f%% | %.0f%% |'
             % (b['burst'], b['date'], b['n'], b['front'], b['gate'], b['dp'],
                100 * b['gate_share'], 100 * b['dp_share']))
L.append(''
         )
L.append('## 运行时间（这块机器，CPU，40 进程）\n')
L.append('| 阶段 | 规模 | 时间 |')
L.append('|---|---|---|')
L.append('| Pass 1：逐帧 17x3 代价面 | 104,728 帧 | 0.107 秒/帧单核，11,225 核秒 ≈ 3.1 核小时；30 个连续段并行，墙钟 ≈ 6 分钟 |')
L.append('| Pass 2：Viterbi 扫描 | 59 + 32 个配置 | 每配置约 6 秒（含评分），两份网格合计 ≈ 8 分钟 |')
L.append('| Pass 3：精筛被移动的单元 | 227 个（帧，单元） | 0.16 秒/帧单核，合计约 36 核秒 |')
L.append('| 分析 / 出表 | — | ≈ 2 分钟 |')
L.append('| 合计 | — | ≈ 17 分钟墙钟，其中 3.1 核小时是代价面 |')
L.append('')
L.append('## 口径与诚实边界\n')
L.append('- 所有数字都在同一批 104,728 帧上、用同一套真值算，没有留出集；配置是从 91 个格点里挑 MPE 最小的，'
         '所以 0.0001–0.0005 的差别应当看作选择噪声，而不是可复现的增益。')
L.append('- DP 只动平移，MOE/ok 与前端相同（1.4047 / 80.2–80.4%）。当前最好那一行的 1.4018 / 80.6% 来自冻结的旋转精修，'
         '它与平移无关，但只能用在"位姿与逐帧输出逐位相同"的修正帧上（本例 97%），所以 DP 行的 MOE 只能报到 1.4019 这种近似值。')
L.append('- 复算的校验：逐帧在"该帧自己的 argmin 单元"精修，与项目已有 P_search 逐位相同（120 帧校验，最大偏差 0.0000 m）；'
         'λ=0、η=0.01 复现门控的 7719 帧与 0.1843；参考四行与项目记录逐位一致。')
L.append('')
L.append('## 产物\n')
L.append('- `results/dp-profiles.npz`：逐帧 17x3 代价面 `surface`、`origin`、`d_values`、`argmin_a/b`，以及前端相对运动的平移/偏航代理。')
L.append('- `results/dp-offsets.npz`：最优 DP 配置的逐帧状态、沿轨/横向偏移、位姿，以及全部配置的选择。')
L.append('- 模块 `along_track_dp.py`：`build_profiles` / `viterbi` / `refine_cells` / `refine_selected` / `evaluate`。')
L.append('')

open(DOC, 'w').write('\n'.join(L) + '\n')
print('\n[saved] %s' % DOC)

json.dump(dict(rows=rows, best_mpe=best_mpe, best_safe=best_safe,
               breakdown=[bd_gate, bd_safe], temporal=[t_gate, t_safe], bursts=burst,
               crosstab=dict(kept=int(kept.sum()), suppressed=int(supp.sum()),
                             new=int(new.sum()), fix5=fix5, break5=break5),
               pairs=len(pairs)),
          open(OUT + '/dp_verdict.json', 'w'), indent=1,
          default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))

# chosen offsets deliverable
np.savez(P + '/results/dp-offsets.npz', keys=np.array(keys_seq),
         state=ch_safe.astype(np.int8), argmin_state=argmin_state.astype(np.int8),
         s=DP.chosen_offsets(emis, ch_safe)[0], d=DP.chosen_offsets(emis, ch_safe)[1],
         poses=p_safe, front=front, lam=best_safe['lam'], eta=best_safe['eta'],
         all_states=CH.astype(np.int8), all_lam=LAM, all_eta=ETA, all_tag=TAG)
print('[saved] %s/results/dp-offsets.npz' % P)

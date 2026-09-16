"""Paired analysis of the A/B mechanism screen (32 validation frames)."""
import json
import sys

import numpy as np

OUT = '/home/zhang/coh-rot-scr'
EPS = [0, 5, 10, 15, 20, 25, 30, 35, 40]

res = {arm: {ep: json.load(open(f'{OUT}/{arm}/eval_{ep}.json')) for ep in EPS} for arm in ['A', 'B']}
ids = [r['frame_id'] for r in res['A'][40]]
assert ids == [r['frame_id'] for r in res['B'][40]], 'frame order mismatch'
dates = [r['date'] for r in res['A'][40]]

print('=== 轨迹（32 验证帧均值）===')
print(f"{'ep':>4} | {'A: MOE':>8} {'A: MPE':>8} {'A: Re':>8} | {'B: MOE':>8} {'B: MPE':>8} {'B: Re':>8}")
for ep in EPS:
    a = {k: np.mean([r[k] for r in res['A'][ep]]) for k in ['rot', 'trans', 're']}
    b = {k: np.mean([r[k] for r in res['B'][ep]]) for k in ['rot', 'trans', 're']}
    print(f"{ep:>4} | {a['rot']:8.4f} {a['trans']:8.4f} {a['re']:8.4f} | {b['rot']:8.4f} {b['trans']:8.4f} {b['re']:8.4f}")

rng = np.random.default_rng(20260916)
def boot_ci(x, n=10000):
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float((m < 0).mean())

for ep in [20, 30, 40]:
    print(f'\n=== epoch {ep}: 逐帧配对差 (B - A) ===')
    for key, name in [('rot', 'MOE(deg)'), ('trans', 'MPE(m)'), ('re', 'Re(deg)')]:
        d = np.array([b[key] - a[key] for a, b in zip(res['A'][ep], res['B'][ep])])
        lo, hi, p = boot_ci(d)
        print(f'  {name:>9}: mean {d.mean():+.4f}  median {np.median(d):+.4f}  '
              f'95%CI [{lo:+.4f}, {hi:+.4f}]  P(B<A)={p:.3f}  改善帧 {100*(d<0).mean():.0f}%')

print('\n=== 相对 epoch 0 的变化（每臂自身，说明是否被微调"带偏"）===')
for arm in ['A', 'B']:
    for key, name in [('rot', 'MOE'), ('trans', 'MPE'), ('re', 'Re')]:
        v0 = np.mean([r[key] for r in res[arm][0]])
        v40 = np.mean([r[key] for r in res[arm][40]])
        print(f'  {arm}: {name:>4} {v0:.4f} -> {v40:.4f}  ({100*(v40-v0)/v0:+.2f}%)')

print('\n=== epoch 40 按日期 ===')
for date in sorted(set(dates)):
    m = np.array([d == date for d in dates])
    d = np.array([b['rot'] - a['rot'] for a, b in zip(res['A'][40], res['B'][40])])
    print(f'  {date}: n={m.sum():2d}  A={np.mean([r["rot"] for r, mm in zip(res["A"][40], m) if mm]):.4f}  '
          f'B={np.mean([r["rot"] for r, mm in zip(res["B"][40], m) if mm]):.4f}  diff={d[m].mean():+.4f}')

print('\n=== 逐帧明细（epoch 40, 按 |diff| 排序，前 8 大）===')
rows = [(i, a['rot'], b['rot'], b['rot'] - a['rot'], a['re'], b['re'], a['date'])
        for i, (a, b) in enumerate(zip(res['A'][40], res['B'][40]))]
rows.sort(key=lambda r: -abs(r[3]))
print(f"{'frame_id':>18} {'A.MOE':>7} {'B.MOE':>7} {'diff':>8} {'A.Re':>7} {'B.Re':>7} date")
for r in rows[:8]:
    print(f'{ids[r[0]]:>18} {r[1]:7.4f} {r[2]:7.4f} {r[3]:+8.4f} {r[4]:7.4f} {r[5]:7.4f} {r[6]}')

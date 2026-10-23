"""Did V7 refinement improve or damage LEADER's pose?

A. hard 1502: both arms are already stored (original_V7 = LEADER, refined_V7 = after
   V7) in deployed_baseline.npz, so this part is exact and free.
B. ordinary frames: the online ledger stores t0 (V7 output) but not T_leader, so a
   sample is re-run through the LEADER forward only and compared against t0 + truth.
"""
import importlib.util
import json
import os
import sys

import numpy as np

PROJ = '/home/zhangcy/projects/e2e-moe-20261020-v1'
TRUTH = '/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth'
DEPLOYED = '/home/zhangcy/projects/r-unified-eval-20261018-v1/deployed_baseline.npz'
P = PROJ + '/results/'


def rot_deg(a, b):
    r = np.swapaxes(a, -1, -2) @ b
    c = np.clip((np.trace(r, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(c))


# ---------- A: hard 1502, both arms stored
z = np.load(DEPLOYED, allow_pickle=True)
gt = z['gt']
lead = z['original_V7']
ref = z['refined_V7']
e_lead = np.linalg.norm(lead[:, :3, 3] - gt[:, :3, 3], axis=1)
e_ref = np.linalg.norm(ref[:, :3, 3] - gt[:, :3, 3], axis=1)
r_lead = rot_deg(lead[:, :3, :3], gt[:, :3, :3])
r_ref = rot_deg(ref[:, :3, :3], gt[:, :3, :3])
d = e_ref - e_lead
print('=== A. hard1502 (LEADER -> V7) ===')
print('LEADER  MPE=%.4f  MOE=%.4f' % (e_lead.mean(), r_lead.mean()))
print('V7      MPE=%.4f  MOE=%.4f' % (e_ref.mean(), r_ref.mean()))
print('V7 helped %d (%.1f%%), damaged %d (%.1f%%), flat(|d|<=0.1 m) %d'
      % (int((d < -0.1).sum()), 100.0 * (d < -0.1).mean(),
         int((d > 0.1).sum()), 100.0 * (d > 0.1).mean(), int((np.abs(d) <= 0.1).sum())))
print('damage >1 m: %d, >5 m: %d, >10 m: %d' % (int((d > 1).sum()), int((d > 5).sum()), int((d > 10).sum())))

# ---------- B: ordinary sample, LEADER forward re-run
truth = {}
for f in os.listdir(TRUTH):
    date = f.replace('.json', '')
    for ts, v in json.load(open(os.path.join(TRUTH, f))).items():
        truth[(date, str(ts))] = np.asarray(v['T_WB'], np.float64)

rows = []
for name in ('full-shard0.jsonl',):
    with open(P + name) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get('skipped') or not r.get('seconds'):
                continue
            rows.append(r)
by_date = {}
for r in rows:
    by_date.setdefault(r['key'].split('/')[0], []).append(r)
sample = []
for date, items in sorted(by_date.items()):
    step = max(1, len(items) // 50)
    sample.extend(items[::step][:50])
print('\n=== B. ordinary sample: %d frames ===' % len(sample))

sys.path.insert(0, PROJ)
import e2e_online as EO  # noqa: E402
boot = EO.boot_online()
torch = boot['torch']
runtime = boot['runtime']
F, Pg, A, rot_only = EO.load_geometry()
record_index = {rec[2]: i for i, rec in enumerate(runtime.records)}
center = boot['center'] if 'center' in boot else runtime.center.double().cpu().numpy()

e_l, e_t0, dates = [], [], []
with torch.no_grad():
    for row in sample:
        key = row['key']
        if key not in record_index:
            continue
        rec = runtime.records[record_index[key]]
        loaded = runtime.load(rec)
        initial = runtime.initial(loaded)
        geometry = runtime.geometry(loaded, initial[1].detach().cpu())
        frame = runtime.frame(loaded, *initial, geometry)
        T_leader = np.asarray(frame['initial'].detach().cpu(), np.float64).reshape(4, 4)
        T_leader[:3, 3] += center
        g = truth.get((key.split('/')[0], key.split('/')[1]))
        if g is None:
            continue
        T0 = np.asarray(row['t0'], np.float64).reshape(4, 4)
        e_l.append(float(np.linalg.norm(T_leader[:3, 3] - g[:3, 3])))
        e_t0.append(float(np.linalg.norm(T0[:3, 3] - g[:3, 3])))
        dates.append(key.split('/')[0])
        del loaded, initial, geometry, frame

e_l = np.array(e_l); e_t0 = np.array(e_t0)
dd = e_t0 - e_l
print('LEADER  MPE=%.4f  median=%.4f' % (e_l.mean(), np.median(e_l)))
print('V7      MPE=%.4f  median=%.4f' % (e_t0.mean(), np.median(e_t0)))
print('V7 helped %d (%.1f%%), damaged %d (%.1f%%), flat(|d|<=0.1 m) %d'
      % (int((dd < -0.1).sum()), 100.0 * (dd < -0.1).mean(),
         int((dd > 0.1).sum()), 100.0 * (dd > 0.1).mean(), int((np.abs(dd) <= 0.1).sum())))
print('damage >1 m: %d, >5 m: %d' % (int((dd > 1).sum()), int((dd > 5).sum())))
for date in sorted(set(dates)):
    m = np.array(dates) == date
    print('  %-12s n=%3d  LEADER %.3f -> V7 %.3f  helped %d / damaged %d'
          % (date, int(m.sum()), e_l[m].mean(), e_t0[m].mean(),
             int((dd[m] < -0.1).sum()), int((dd[m] > 0.1).sum())))



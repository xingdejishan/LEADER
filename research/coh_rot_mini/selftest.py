"""Synthetic self-tests for coh_rot. No data, no GPU required (runs on CPU too).

Checks:
  1  pure whole-frame translation -> L = 0
  2  known rigid rotation -> SVD angle recovered, L monotone in angle
  3  per-frame independence (batching two frames == computing separately)
  4  finite-difference gradient check on the linearised-omega path
  5  gradient direction actually reduces the rotation
  6  planar point set is NOT treated as degenerate
  7  collinear point set IS skipped
  8  angle gate (>20 deg) skips the frame
  9  weight concentration guard
 10  all outputs finite
"""
import math
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from coh_rot import coherent_rotation_loss, linear_omega  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
ok = []


def check(name, cond):
    ok.append((name, bool(cond)))
    print(("  PASS  " if cond else "  FAIL  ") + name)


def rot(axis, deg):
    a = np.asarray(axis, float); a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(math.radians(deg)) * K + (1 - math.cos(math.radians(deg))) * K @ K


n = 40
rng = np.random.default_rng(0)
P = rng.normal(size=(n, 3)) * 8.0                      # generic 3D cloud
w = torch.ones(n)
bidx = torch.zeros(n, dtype=torch.long)

# 1 pure translation
tgt = torch.tensor(P, dtype=torch.float64)
pred = tgt + torch.tensor([3.0, -7.0, 2.5], dtype=torch.float64)
L, st = coherent_rotation_loss(pred, tgt, w, bidx)
check("1 pure translation -> L=0", abs(float(L)) < 1e-12 and st["valid"] == 1)

# 2 known rotation
for deg in [0.5, 1.0, 2.0, 5.0]:
    R = torch.tensor(rot([0.3, -0.7, 0.2], deg), dtype=torch.float64)
    pred = tgt @ R.T
    L, st = coherent_rotation_loss(pred, tgt, w, bidx)
    check(f"2 rotation {deg} deg detected (angle={st['angle_mean']:.3f})",
          abs(st["angle_mean"] - deg) < 1e-6)
L1, _ = coherent_rotation_loss(tgt @ torch.tensor(rot([1, 0, 0], 1.0), dtype=torch.float64).T, tgt, w, bidx)
L5, _ = coherent_rotation_loss(tgt @ torch.tensor(rot([1, 0, 0], 5.0), dtype=torch.float64).T, tgt, w, bidx)
check("2b L monotone in angle", float(L5) > float(L1) > 0)

# 3 per-frame independence
P2 = rng.normal(size=(25, 3)) * 8.0
tgt2 = torch.tensor(P2, dtype=torch.float64)
pred2 = tgt2 @ torch.tensor(rot([0, 1, 0], 3.0), dtype=torch.float64).T
bidx2 = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(25, dtype=torch.long)])
L_both, st_both = coherent_rotation_loss(torch.cat([pred, pred2]), torch.cat([tgt, tgt2]),
                                         torch.cat([w, torch.ones(25)]), bidx2)
La, _ = coherent_rotation_loss(pred, tgt, w, bidx)
Lb, _ = coherent_rotation_loss(pred2, tgt2, torch.ones(25), torch.zeros(25, dtype=torch.long))
check("3 batching == separate", abs(float(L_both) - (float(La) + float(Lb)) / 2) < 1e-12)

# 4+5 finite-difference gradient + descent direction
p = (tgt @ torch.tensor(rot([0, 0, 1], 2.0), dtype=torch.float64).T).clone().requires_grad_(True)
L, _ = coherent_rotation_loss(p, tgt, w, bidx)
L.backward()
g = p.grad.clone()
# finite difference on a few coordinates
eps = 1e-6
max_err = 0.0
for i in [0, 5, 11, 23]:
    for j in [0, 1, 2]:
        pp = p.detach().clone(); pp[i, j] += eps
        pm = p.detach().clone(); pm[i, j] -= eps
        Lp, _ = coherent_rotation_loss(pp, tgt, w, bidx)
        Lm, _ = coherent_rotation_loss(pm, tgt, w, bidx)
        fd = (float(Lp) - float(Lm)) / (2 * eps)
        max_err = max(max_err, abs(fd - float(g[i, j])))
check(f"4 finite-difference gradient (max err {max_err:.2e})", max_err < 1e-5)

with torch.no_grad():
    p2 = p.detach() - 0.05 * g
L_after, _ = coherent_rotation_loss(p2, tgt, w, bidx)
check(f"5 gradient step reduces L ({float(L):.6f} -> {float(L_after):.6f})", float(L_after) < float(L))

# 6 planar cloud (z = 0 plane) is fine
Pl = np.stack([rng.normal(size=60) * 8, rng.normal(size=60) * 8, np.zeros(60)], 1)
tgt_l = torch.tensor(Pl, dtype=torch.float64)
pred_l = tgt_l @ torch.tensor(rot([0, 0, 1], 1.5), dtype=torch.float64).T
L, st = coherent_rotation_loss(pred_l, tgt_l, torch.ones(60), torch.zeros(60, dtype=torch.long))
check(f"6 planar set valid (angle={st['angle_mean']:.3f})", st["valid"] == 1 and abs(st["angle_mean"] - 1.5) < 1e-6)

# 7 collinear cloud is skipped
t = np.linspace(-10, 10, 40)[:, None] * np.array([1.0, 0.3, -0.2])
tgt_c = torch.tensor(t, dtype=torch.float64)
pred_c = tgt_c @ torch.tensor(rot([0, 0, 1], 2.0), dtype=torch.float64).T
L, st = coherent_rotation_loss(pred_c, tgt_c, torch.ones(40), torch.zeros(40, dtype=torch.long))
check(f"7 collinear skipped (deg={st['skipped_degenerate']})", st["valid"] == 0 and st["skipped_degenerate"] == 1)

# 8 gate
pred_g = tgt @ torch.tensor(rot([0, 1, 0], 30.0), dtype=torch.float64).T
L, st = coherent_rotation_loss(pred_g, tgt, w, bidx)
check(f"8 30deg gated (gate={st['skipped_gate']})", st["valid"] == 0 and st["skipped_gate"] == 1)

# 9 weight concentration guard
w9 = torch.zeros(n, dtype=torch.float64); w9[:3] = 1.0
pred9 = tgt @ torch.tensor(rot([0, 1, 0], 2.0), dtype=torch.float64).T
L, st = coherent_rotation_loss(pred9, tgt, w9, bidx)
check(f"9 concentrated weights handled (valid={st['valid']}, degen={st['skipped_degenerate']})",
      st["valid"] == 0 and st["skipped_degenerate"] == 1)

# 10 finiteness of everything
vals = [float(L1), float(L5), float(L_both), float(L_after)]
check("10 all outputs finite", all(np.isfinite(v) for v in vals))

print()
n_pass = sum(c for _, c in ok)
print(f"{n_pass}/{len(ok)} PASSED")
sys.exit(0 if n_pass == len(ok) else 1)

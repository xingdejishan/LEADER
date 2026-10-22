"""Is the bulk rotation error a *timing* offset rather than a geometric failure?

If the estimate is the true pose at t+tau, then in the body frame
    log(R_gt^T R_pred) ~= tau * omega_body      (omega from the truth trajectory)
    R_gt^T (p_pred - p_gt) ~= tau * v_body
so a single scalar tau explains both, and the residual is the real error.
"""
import json
import os

import numpy as np

R = '/home/zhangcy/projects/e2e-moe-20261020-v1/'
TRUTH = '/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth'


def log_so3(Rm):
    tr = np.clip((np.trace(Rm, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    ang = np.arccos(tr)
    w = np.stack([Rm[..., 2, 1] - Rm[..., 1, 2],
                  Rm[..., 0, 2] - Rm[..., 2, 0],
                  Rm[..., 1, 0] - Rm[..., 0, 1]], axis=-1)
    s = np.sin(ang)
    small = s < 1e-8
    scale = np.where(small, 0.5, ang / (2.0 * np.where(small, 1.0, s)))
    return w * scale[..., None]


def exp_so3(w):
    th = np.linalg.norm(w, axis=-1)
    k = w / np.where(th > 1e-12, th, 1.0)[..., None]
    K = np.zeros(w.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    th2 = th[..., None, None]
    I = np.eye(3)
    return I + np.sin(th2) * K + (1 - np.cos(th2)) * (K @ K)


def rot_angle(Rm):
    tr = np.clip((np.trace(Rm, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(tr))


# ---- load truth per date, sorted, and build neighbour velocities
tracks = {}
for f in os.listdir(TRUTH):
    d = f.replace('.json', '')
    raw = json.load(open(os.path.join(TRUTH, f)))
    ts = sorted(raw.keys(), key=lambda s: int(s))
    T = np.array([np.asarray(raw[s]['T_WB'], float) for s in ts])
    tracks[d] = (ts, T)
    print(d, 'truth entries', len(ts), 'span %.1f s' % ((int(ts[-1]) - int(ts[0])) / 1e6))

Z = np.load(R + 'results/full-online-final.npz', allow_pickle=True)
keys = np.array([str(k) for k in Z['keys']])
sets = {'t0': Z['t0'], 'now': Z['poses']}

dates = np.array([k.split('/')[0] for k in keys])
idx_in = {}
for d, (ts, T) in tracks.items():
    m = {s: i for i, s in enumerate(ts)}
    idx_in[d] = m

n = len(keys)
omega = np.zeros((n, 3))
vel = np.zeros((n, 3))
dtv = np.zeros(n)
valid = np.zeros(n, bool)
for i, k in enumerate(keys):
    d, s = k.split('/')
    ts, T = tracks[d]
    j = idx_in[d].get(s)
    if j is None or j == 0 or j == len(ts) - 1:
        continue
    t0_, t1_ = int(ts[j - 1]), int(ts[j + 1])
    dt = (t1_ - t0_) / 1e6
    if dt <= 0:
        continue
    Ra, Rb = T[j - 1][:3, :3], T[j + 1][:3, :3]
    w_world = log_so3(Ra.T @ Rb) / dt          # world-frame angular velocity
    omega[i] = Ra.T @ w_world                   # body frame
    vel[i] = Ra.T @ ((T[j + 1][:3, 3] - T[j - 1][:3, 3]) / dt)
    dtv[i] = dt
    valid[i] = True
print('frames with velocity: %d ; median neighbour dt %.3f s ; |omega| median %.2f deg/s ; |v| median %.2f m/s'
      % (valid.sum(), np.median(dtv[valid]), np.degrees(np.linalg.norm(omega[valid], axis=1)).mean(),
         np.linalg.norm(vel[valid], axis=1).mean()))

for tag in ('t0', 'now'):
    P = sets[tag]
    gR = np.zeros((n, 3, 3))
    gp = np.zeros((n, 3))
    for i, k in enumerate(keys):
        d, s = k.split('/')
        T = tracks[d][1][idx_in[d][s]]
        gR[i], gp[i] = T[:3, :3], T[:3, 3]
    dR = log_so3(np.swapaxes(gR, -1, -2) @ P[:, :3, :3])        # body-frame rotation error
    dp = np.einsum('nij,nj->ni', np.swapaxes(gR, -1, -2), P[:, :3, 3] - gp)  # body-frame position error
    m = valid
    # least-squares tau, in samples with enough motion to be informative
    for name, mask in (('all', m), ('|omega|>5', m & (np.degrees(np.linalg.norm(omega, axis=1)) > 5)),
                       ('|omega|>15', m & (np.degrees(np.linalg.norm(omega, axis=1)) > 15))):
        num = (dR[mask] * omega[mask]).sum()
        den = (omega[mask] * omega[mask]).sum()
        tau_r = num / den if den else 0.0
        num2 = (dp[mask] * vel[mask]).sum()
        den2 = (vel[mask] * vel[mask]).sum()
        tau_t = num2 / den2 if den2 else 0.0
        ang = rot_angle(np.swapaxes(gR, -1, -2) @ P[:, :3, :3])
        corr = np.einsum('nij,nj->ni', np.swapaxes(gR, -1, -2), P[:, :3, 3] - gp) - tau_r * omega
        print('  %-4s %-10s tau_rot %+7.4f s  tau_trans %+7.4f s   (n=%d)'
              % (tag, name, tau_r, tau_t, int(mask.sum())))
        # MOE after removing tau*omega
        P2 = P.copy()
        P2[:, :3, :3] = P[:, :3, :3] @ exp_so3(-tau_r * omega)
        a2 = rot_angle(np.swapaxes(gR, -1, -2) @ P2[:, :3, :3])
        print('        MOE %.4f -> %.4f   (bulk MPE<1: %.4f -> %.4f)'
              % (ang[m].mean(), a2[m].mean(), ang[m & (np.linalg.norm(dp, axis=1) < 1)].mean(),
                 a2[m & (np.linalg.norm(dp, axis=1) < 1)].mean()))
    # per-axis correlation
    wmag = np.degrees(np.linalg.norm(omega[m], axis=1))
    emag = np.degrees(np.linalg.norm(dR[m], axis=1))
    cc = np.corrcoef(wmag, emag)[0, 1]
    print('  %-4s corr(|dR|,|omega|) = %.3f ;  median |dR| %.3f deg at |omega|<1 deg/s, %.3f deg at |>10'
          % (tag, cc, np.median(emag[wmag < 1]), np.median(emag[wmag > 10]) if (wmag > 10).any() else -1))
    # per-frame tau, robust
    good = m & (wmag > 8)
    ti = (dR[good] * omega[good]).sum(1) / (omega[good] * omega[good]).sum(1)
    print('  %-4s per-frame tau (s) p10 %+.4f  median %+.4f  p90 %+.4f'
          % (tag, np.percentile(ti, 10), np.median(ti), np.percentile(ti, 90)))

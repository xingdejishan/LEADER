"""Re-calibrate lambda for L_coh-rot on a real batch.

The earlier calibration used four frames (`coherent_rotation_stats.valid == 4`) and
reported that the reference lambda=0.1 gives the new term only ~3 % of the gradient.
That is too thin to set a training coefficient on.  This repeats the measurement on
hundreds of training frames, and additionally reports the per-date spread of the
coherent rotation angle, because a loss aimed at a systematic term needs that term to
be present consistently.

Read-only except for the JSON it writes.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '3')

import numpy as np
import torch

HERE = Path('/home/zhangcy/projects/coh-rot-led-20261010')
sys.path.insert(0, '/home/zhangcy/projects/LEADER-magic-909c81d')
sys.path.insert(0, str(HERE))

from models.model_mink import LEADER                      # noqa: E402

import leader_nclt                                       # noqa: E402
from coh_rot import coherent_rotation_loss, weights_from_scores   # noqa: E402
from train_cohrot_ab import BASE_CHECKPOINT, forward, trr_value, repo_trr  # noqa: E402

OUT = '/home/zhangcy/.dsh_scratch/moe_rot_20261022/cohrot_recalibration.json'
BATCH = 8
N_FRAMES = int(os.environ.get('N', '256'))
N_GRAD_FRAMES = int(os.environ.get('NG', '32'))
PER_DATE = int(os.environ.get('PD', '64'))


def sample_frames():
    """Even sample across the four training dates (train_frames() is grouped by date)."""
    all_frames = leader_nclt.train_frames()
    by_date = {}
    for i, f in enumerate(all_frames):
        by_date.setdefault(f['date'], []).append(i)
    picked = []
    for d in sorted(by_date):
        idx = by_date[d]
        take = min(PER_DATE, len(idx))
        step = max(1, len(idx) // take)
        picked.extend(idx[::step][:take])
    return all_frames, sorted(picked)


def main():
    device = 'cuda:0'
    torch.set_num_threads(2)
    t0 = time.time()
    frames = leader_nclt.train_frames()
    _all, pick = sample_frames()
    frames = [frames[i] for i in pick]
    print('frames %d over %s' % (len(frames),
                                 sorted(set(f['date'] for f in frames))), flush=True)
    store = leader_nclt.prepare_all(frames)
    centre = torch.tensor(leader_nclt.CANONICAL_CENTER_T, dtype=torch.float32, device=device)

    state = torch.load(BASE_CHECKPOINT, map_location='cpu')
    model = LEADER(in_channels=3, out_channels=4, feat_channels=512, width=1024, magic=False)
    model.load_state_dict(state['model'], strict=False)
    model = model.to(device).eval()
    loss_fn, source = repo_trr(10.0)
    print('TRR source: %s ; prepare %.1f s' % (source, time.time() - t0), flush=True)

    rows = []
    for start in range(0, len(frames), BATCH):
        chunk = list(range(start, min(start + BATCH, len(frames))))
        batch = leader_nclt.collate([(store[i], frames[i]['T_WB']) for i in chunk])
        with torch.no_grad():
            prediction, targets, batch_index = forward(model, batch, centre, device)
            weights = weights_from_scores(prediction[:, 3])
            for b, i in enumerate(chunk):
                m = batch_index == b
                if int(m.sum()) < 32:
                    continue
                _, stats = coherent_rotation_loss(prediction[m, :3], targets[m],
                                                  weights[m], torch.zeros(int(m.sum()),
                                                                          dtype=torch.long,
                                                                          device=device))
                if stats.get('valid', 0) < 1:
                    continue
                rows.append(dict(i=i, date=frames[i]['date'],
                                 angle=float(stats['angle_mean']),
                                 omega=float(stats['omega_mean_deg'])))
        if (start // BATCH) % 8 == 0:
            print('  %d/%d  %.0f s' % (start + len(chunk), len(frames), time.time() - t0),
                  flush=True)

    a = np.array([r['angle'] for r in rows])
    w = np.array([r['omega'] for r in rows])
    print('\n## coherent rotation of the predicted field, %d frames' % len(rows))
    print('   mean %.3f  median %.3f  p75 %.3f  p90 %.3f  max %.3f deg'
          % (a.mean(), np.median(a), np.percentile(a, 75), np.percentile(a, 90), a.max()))
    print('   omega (translation-free part) mean %.3f deg' % w.mean())
    dates = sorted(set(r['date'] for r in rows))
    print('   per date:')
    for d in dates:
        m = np.array([r['date'] == d for r in rows])
        print('     %-12s n=%3d  mean %.3f  median %.3f' % (d, int(m.sum()), a[m].mean(),
                                                            np.median(a[m])))
    frac_over2 = float((a > 2.0).mean())
    print('   fraction of frames with a coherent rotation above 2 deg: %.1f%%'
          % (100 * frac_over2))

    # ---- gradient share on a subset, same procedure as the earlier calibration
    model.train()
    grad = []
    skipped = 0
    for start in range(0, min(N_GRAD_FRAMES, len(rows)), BATCH):
        chunk = [r['i'] for r in rows[start:start + BATCH]]
        if len(chunk) < 2:
            continue
        batch = leader_nclt.collate([(store[i], frames[i]['T_WB']) for i in chunk])
        prediction, targets, batch_index = forward(model, batch, centre, device)
        trr, _ = trr_value(loss_fn, targets, prediction, batch_index)
        weights = weights_from_scores(prediction[:, 3].detach())
        rotation, stats = coherent_rotation_loss(prediction[:, :3], targets, weights, batch_index)
        if stats.get('valid', 0) < 1 or not rotation.requires_grad:
            model.zero_grad(set_to_none=True)
            skipped += 1
            continue
        parameters = [p for p in model.parameters() if p.requires_grad]
        g_trr = torch.autograd.grad(trr, parameters, retain_graph=True, allow_unused=True)
        g_coh = torch.autograd.grad(rotation, parameters, retain_graph=True, allow_unused=True)

        def norm(gs):
            return float(np.sqrt(sum(float((g.double() ** 2).sum().item())
                                     for g in gs if g is not None)))
        n_trr, n_coh = norm(g_trr), norm(g_coh)
        grad.append(dict(n_trr=n_trr, n_coh=n_coh, l_trr=float(trr), l_coh=float(rotation)))
        model.zero_grad(set_to_none=True)

    if grad:
        n_trr = np.mean([g['n_trr'] for g in grad])
        n_coh = np.mean([g['n_coh'] for g in grad])
        rep = dict(lam_for_10pct=0.10 * n_trr / n_coh, lam_for_15pct=0.15 * n_trr / n_coh,
                   lam_for_30pct=0.30 * n_trr / n_coh,
                   lam_for_50pct=0.50 * n_trr / n_coh,
                   grad_ratio=n_trr / n_coh,
                   share_at_ref_lam=0.1 * n_coh / (n_trr + 0.1 * n_coh),
                   n_trr=n_trr, n_coh=n_coh, batches=len(grad))
        print('\n## gradient share (mean of %d batches, %d skipped)' % (len(grad), skipped))
        print('   |g_trr| %.4f  |g_coh| %.4f  ratio %.3f'
              % (n_trr, n_coh, n_trr / n_coh))
        for k in ('lam_for_10pct', 'lam_for_15pct', 'lam_for_30pct', 'lam_for_50pct'):
            print('   %-14s %.3f' % (k, rep[k]))
        print('   share at the previously used lam=0.1 : %.2f%%'
              % (100 * rep['share_at_ref_lam']))
    else:
        rep = {}

    payload = dict(frames=len(rows), angles=dict(
        mean=float(a.mean()), median=float(np.median(a)), p75=float(np.percentile(a, 75)),
        p90=float(np.percentile(a, 90)), max=float(a.max()),
        frac_over_2deg=frac_over2),
        omega_mean=float(w.mean()),
        per_date=dict((d, float(np.mean([r['angle'] for r in rows if r['date'] == d])))
                      for d in dates),
        gradient=rep,
        rows=[dict(i=r['i'], date=r['date'], angle=r['angle'], omega=r['omega']) for r in rows])
    Path(OUT).write_text(json.dumps(payload, indent=1), encoding='utf-8')
    print('\nwrote %s  total %.0f s' % (OUT, time.time() - t0))


if __name__ == '__main__':
    main()

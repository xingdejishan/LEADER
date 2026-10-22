"""Lambda calibration: how much gradient does L_coh-rot actually contribute?

The project's own method (`coh-rot-scr/calibrate_lambda.py`): measure
||g_trr|| and ||g_coh-rot|| separately on a small batch and report the coefficient that
would give the new term a chosen share of the total gradient.  The A/B itself keeps the
reference implementation's lam = 0.1; this run only documents what that means here.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = Path('/home/zhangcy/projects/LEADER-magic-909c81d')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from models.model_mink import LEADER  # noqa: E402

import leader_nclt  # noqa: E402
from coh_rot import coherent_rotation_loss, weights_from_scores  # noqa: E402
from train_cohrot_ab import BASE_CHECKPOINT, forward, trr_value, repo_trr  # noqa: E402


def main():
    device = 'cuda:0'
    frames = leader_nclt.train_frames()[:8]
    store = leader_nclt.prepare_all(frames)
    center = torch.tensor(leader_nclt.CANONICAL_CENTER_T, dtype=torch.float32, device=device)

    state = torch.load(BASE_CHECKPOINT, map_location='cpu')
    model = LEADER(in_channels=3, out_channels=4, feat_channels=512, width=1024, magic=False)
    model.load_state_dict(state['model'], strict=False)
    model = model.to(device).eval()
    loss_fn, source = repo_trr(10.0)
    print('TRR source: %s' % source, flush=True)

    batch = leader_nclt.collate([(store[i], frames[i]['T_WB']) for i in range(4)])
    prediction, targets, batch_index = forward(model, batch, center, device)
    trr, l2 = trr_value(loss_fn, targets, prediction, batch_index)
    weights = weights_from_scores(prediction[:, 3].detach())
    rotation, stats = coherent_rotation_loss(prediction[:, :3], targets, weights, batch_index)
    print('l_trr=%.6f  l_coh=%.6f  raw_l2=%.6f' % (trr.item(), rotation.item(), l2.item()), flush=True)
    print('coh stats: %s' % json.dumps({k: v for k, v in stats.items()}), flush=True)

    parameters = [p for p in model.parameters() if p.requires_grad]
    g_trr = torch.autograd.grad(trr, parameters, retain_graph=True, allow_unused=True)
    g_coh = torch.autograd.grad(rotation, parameters, retain_graph=True, allow_unused=True)

    def norm(gradients):
        total = 0.0
        for gradient in gradients:
            if gradient is not None:
                total += float((gradient.double() ** 2).sum().item())
        return float(np.sqrt(total))

    n_trr, n_coh = norm(g_trr), norm(g_coh)
    ratio = n_trr / n_coh if n_coh else float('inf')
    report = {
        'l_trr': float(trr.item()), 'l_coh': float(rotation.item()), 'l2_raw': float(l2.item()),
        'grad_trr_norm': n_trr, 'grad_coh_norm': n_coh, 'grad_ratio': ratio,
        'lam_for_15pct_grad_share': 0.15 * n_trr / n_coh if n_coh else None,
        'lam_for_50pct_grad_share': 0.50 * n_trr / n_coh if n_coh else None,
        'reference_lam': 0.1,
        'coh_grad_share_at_reference_lam': (0.1 * n_coh) / (n_trr + 0.1 * n_coh) if n_coh else None,
        'coherent_rotation_stats': stats,
    }
    print('CALIBRATION_JSON ' + json.dumps(report), flush=True)
    (HERE / 'calibration.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

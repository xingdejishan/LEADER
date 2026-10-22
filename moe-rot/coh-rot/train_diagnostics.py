"""Training-frame diagnostic for every checkpoint, on one consistent definition.

For each checkpoint, over all 1445 training frames, report
  * raw mean point error  ||pred - gt||            (metres)
  * weighted TRR                                    (the training objective)
  * weighted Kabsch field-rotation angle            (no gate: the true field rotation)
  * weighted first-order |omega|                    (what the loss differentiates)

The evaluation script only reports `field_rotation_deg` on the test dates, and arm A never
logged it during training, so this fills the train-side column for all arms at once.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = Path('/home/zhangcy/projects/LEADER-magic-909c81d')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from models.model_mink import LEADER  # noqa: E402

import leader_nclt  # noqa: E402
from coh_rot import kabsch_forward, linear_omega, weights_from_scores  # noqa: E402
from train_cohrot_ab import forward, trr_value, repo_trr  # noqa: E402

CHECKPOINTS = {
    'base': '/dev/shm/cohrot-20261010/base/official_l0.pt',
    'A': '/dev/shm/cohrot-20261010/run-A/armA.pt',
    'B': '/dev/shm/cohrot-20261010/run-B/armB.pt',
    'C': '/dev/shm/cohrot-20261010/run-C/armC.pt',
}


@torch.no_grad()
def diagnostics(prediction, targets, batch_index, center):
    angles, omegas, l2 = [], [], []
    frames = int(batch_index.max().item()) + 1
    # targets are already in the centred world frame; recover the absolute field rotation
    for f in range(frames):
        mask = batch_index == f
        x = prediction[mask, :3].detach()
        y = targets[mask].detach()
        weight = weights_from_scores(prediction[mask, 3].detach())
        weight = weight / weight.sum().clamp_min(1e-12)
        xc = x - (weight[:, None] * x).sum(0, keepdim=True)
        yc = y - (weight[:, None] * y).sum(0, keepdim=True)
        q, _ = kabsch_forward(xc, yc, weight)
        cos_q = ((q.diagonal().sum() - 1) / 2).clamp(-1, 1)
        angles.append(float(torch.rad2deg(torch.acos(cos_q)).item()))
        omegas.append(float(torch.rad2deg(torch.linalg.norm(linear_omega(xc, yc, weight))).item()))
        l2.append(float((x - y).norm(dim=-1).mean().item()))
    return angles, omegas, l2


def main():
    device = 'cuda:0'
    torch.set_num_threads(1)
    frames = leader_nclt.train_frames()
    store = leader_nclt.prepare_all(frames)
    center = torch.tensor(leader_nclt.CANONICAL_CENTER_T, dtype=torch.float32, device=device)
    loss_fn, _ = repo_trr(10.0)

    results = {}
    for name, path in CHECKPOINTS.items():
        state = torch.load(path, map_location='cpu')
        model = LEADER(in_channels=3, out_channels=4, feat_channels=512, width=1024, magic=False)
        model.load_state_dict(state['model'])
        model = model.to(device).eval()
        started = time.monotonic()
        all_angles, all_omegas, all_l2, all_trr = [], [], [], []
        batch_size = 4
        for start in range(0, len(store), batch_size):
            indices = list(range(start, min(start + batch_size, len(store))))
            batch = leader_nclt.collate([(store[i], frames[i]['T_WB']) for i in indices])
            prediction, targets, batch_index = forward(model, batch, center, device)
            trr, _ = trr_value(loss_fn, targets, prediction, batch_index)
            angles, omegas, l2 = diagnostics(prediction, targets, batch_index, center)
            all_angles.extend(angles)
            all_omegas.extend(omegas)
            all_l2.extend(l2)
            all_trr.append(float(trr.item()))
        results[name] = {
            'frames': len(all_angles),
            'point_error_m': float(np.mean(all_l2)),
            'trr': float(np.mean(all_trr)),
            'field_rotation_deg': float(np.mean(all_angles)),
            'field_rotation_median_deg': float(np.median(all_angles)),
            'field_omega_deg': float(np.mean(all_omegas)),
            'seconds': time.monotonic() - started,
        }
        print('%-5s point_err=%.4f m  trr=%.4f  field_rot=%.4f deg (median %.4f)  |omega|=%.4f deg'
              % (name, results[name]['point_error_m'], results[name]['trr'],
                 results[name]['field_rotation_deg'], results[name]['field_rotation_median_deg'],
                 results[name]['field_omega_deg']), flush=True)
        del model
        torch.cuda.empty_cache()

    (HERE / 'train-diagnostics.json').write_text(json.dumps(results, indent=2) + '\n',
                                                 encoding='utf-8')
    print('TRAIN_DIAG_JSON ' + json.dumps(results), flush=True)


if __name__ == '__main__':
    main()

import hashlib
import json
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent
run = root / 'phase-results/run'
freeze = json.loads((run / 'train-freeze.json').read_text())
fit = [v for v in freeze['records'] if v['index']%5 != 0 and v['points']]
cal = json.loads((run / 'calibration-e04.json').read_text())['rows']
grid = np.linspace(-4, 4, 21)
fit_angles = np.array([v['angle_deg'] for v in fit])
cal_angles = np.array([v['angle_deg'] for v in cal])
fit_label = np.abs(fit_angles[:, None]-grid).argmin(1)
fit_inside = np.abs(fit_angles)<=4
count = np.bincount(fit_label[fit_inside], minlength=21)
mode = int(count.argmax())
inside = [v for v in cal if v['in_grid']]
prior = (count+1)/(count.sum()+21)
prior_mean = float(prior@grid)
measured = np.array([v['mean_deg'] for v in cal])
map_angle = grid[np.array([v['argmax'] for v in cal])]
per_date = {}
for date in sorted(set(v['frame_key'].split('/')[0] for v in cal)):
    ids = [i for i,v in enumerate(cal) if v['frame_key'].startswith(date+'/')]
    per_date[date] = dict(frames=len(ids), zero_rmse_deg=float(np.sqrt(np.mean(cal_angles[ids]**2))),
        measured_rmse_deg=float(np.sqrt(np.mean((measured[ids]-cal_angles[ids])**2))))
components = np.load(run / 'baseline-rotation-components.npz')
vector = components['correction_rotvec_deg']
decomposition = json.loads((run / 'baseline-rotation-decomposition.json').read_text())
assert abs(float((vector[:,2]**2).sum()/(vector*vector).sum())-decomposition['yaw_squared_fraction'])<1e-12
result = dict(scope='coarse21 phase measurement, four fixed epochs; training-side heldout only, no new development pose evaluation',
    train_frames=len(freeze['records']), fit_frames=len(fit), fit_in_grid=int(fit_inside.sum()), calibration_frames=len(cal),
    uniform_top1=1/21, learned_top1=float(np.mean([v['argmax']==v['label'] for v in inside])),
    constant_zero_top1=float(np.mean([v['label']==10 for v in inside])),
    fit_majority_class=mode, fit_majority_angle_deg=float(grid[mode]), fit_majority_top1=float(np.mean([v['label']==mode for v in inside])),
    fit_class_histogram=count.tolist(), fit_class_prior_calibration_ce=float(np.mean([-np.log(prior[v['label']]) for v in inside])),
    fit_smoothed_prior_mean_deg=prior_mean, fit_prior_mean_calibration_rmse_deg=float(np.sqrt(np.mean((cal_angles-prior_mean)**2))),
    measured_rmse_deg=float(np.sqrt(np.mean((cal_angles-measured)**2))), zero_rmse_deg=float(np.sqrt(np.mean(cal_angles**2))),
    argmax_phase_rmse_deg=float(np.sqrt(np.mean((cal_angles-map_angle)**2))), per_date=per_date,
    original1200_final_pose_decomposition=decomposition,
    source_inputs_sha256={name:hashlib.sha256((run/name).read_bytes()).hexdigest() for name in ['train-freeze.json','calibration-e04.json','baseline-rotation-components.npz']})
(root / 'phase-result-analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))

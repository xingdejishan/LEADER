import hashlib
import json
import os
from pathlib import Path
import numpy as np
import cv2
import torch
from fusion import project
from run import WORKSPACE, read_scan

assets = WORKSPACE / 'research/projection_audit_assets'
root = Path('/home/zhang/leader-image-gate')
out = root / 'projection_audit'
out.mkdir(exist_ok=True)
scope = {}
source = (assets / 'project_vel_to_cam.py').read_text().split('def main(')[0]
exec(compile(source, str(assets / 'project_vel_to_cam.py'), 'exec'), scope)
extrinsic = scope['ssc_to_homo']([.035, .002, -1.23, -179.93, -.23, .50]) @ scope['ssc_to_homo'](np.loadtxt(assets / 'x_lb3_c5.csv', delimiter=','))
k = np.loadtxt(assets / 'K_cam5.csv', delimiter=',')
bundle = WORKSPACE / 'glace-local'
meta = json.loads((bundle / 'data/validation_scene/scene_meta.json').read_text())
report = dict(extrinsic_max_abs_error=float(np.max(np.abs(extrinsic - meta['T_BC_camera_to_body']))),
              intrinsics_max_abs_error=float(np.max(np.abs(k - meta['K_raw']))))
rows = json.loads((root / 'manifest.json').read_text())
parity = []
old = os.getcwd()
os.chdir(assets)
for row in [rows[0], rows[32], rows[64]]:
    scan, _ = read_scan(row['scan'])
    official = scope['project_vel_to_cam'](np.column_stack([scan, np.ones(len(scan))]).T, 5)
    uv_ref = (official[:2] / official[2:]).T * .5
    kk = torch.tensor(k, dtype=torch.float32)
    kk[:2] *= .5
    uv, depth, inside = project(torch.tensor(scan), torch.tensor(extrinsic, dtype=torch.float32), kk, (616, 808))
    parity.append(float(np.max(np.abs(uv.numpy()[inside] - uv_ref[inside]))))
os.chdir(old)
report['official_projection_pixel_max_abs_error'] = max(parity)
text = assets / 'U2D_Cam5_1616X1232.txt'
mapping = np.loadtxt(text, skiprows=1, dtype=np.float32)
mapu, mapv = np.zeros((1232, 1616), np.float32), np.zeros((1232, 1616), np.float32)
rr, cc = mapping[:, 0].astype(int), mapping[:, 1].astype(int)
mapu[rr, cc], mapv[rr, cc] = mapping[:, 3], mapping[:, 2]
valid = cv2.remap(np.ones((1232, 1616), np.float32), mapu, mapv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
valid = cv2.resize(valid, (808, 616), interpolation=cv2.INTER_AREA)
existing = np.load(bundle / 'data/valid_mask.npy')
report['official_undistortion_mask_max_abs_error'] = float(np.max(np.abs(valid - existing)))
report['official_undistortion_mask_exact'] = bool(np.array_equal(valid, existing))
report['official_undistortion_mask_mean_abs_error'] = float(np.mean(np.abs(valid - existing)))
report['official_undistortion_mask_full_valid_disagreement_pixels'] = int(((valid > .999) != (existing > .999)).sum())
np.save(out / 'official_mask.npy', valid)
photo_checks = []
for row in [rows[0], rows[32], rows[64]]:
    image = cv2.imread(row['image'])
    black = cv2.erode((valid == 0).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    photo_checks.append(dict(frame_id=row['frame_id'], black_region_mean_rgb=float(image[black].mean()),
                            black_region_nonblack_fraction=float((image[black].max(-1) > 8).mean())))
report['image_black_border_checks'] = photo_checks
report['source_hashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in assets.iterdir() if p.is_file()}
report['limitation'] = 'Matches official parameter chain and undistortion support; local raw TIFFs absent, so original-image-to-JPEG remapping and independent physical landmark residuals are not verified.'
(out / 'calibration.json').write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != 'source_hashes'}, indent=2))

import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'image_gate'))
import run

OUT = Path('/home/zhang/crossframe-visual-probe')
ARGS = SimpleNamespace(output=OUT, checkpoint=run.WORKSPACE / 'research/image_gate_checkpoint',
                       workspace=run.WORKSPACE, bundle=run.WORKSPACE / 'glace-local',
                       pca=Path('/home/zhang/rscore-l-local/data/proc/pcad3LB_128.pth'), projection_mapping=OUT / 'mapping')


def manifest():
    raw = json.loads((ARGS.bundle / 'data/train907_raw_scan_audit/paired_manifest.json').read_text())['train']
    rows = []
    for row in raw:
        if not row['paired']:
            continue
        r = {k: row[k] for k in ['frame_id', 'session_id', 'scan_sha256']}
        for key in ['image', 'scan', 'pose', 'calibration']:
            r[key] = row[key].replace('C:\\', '/mnt/c/').replace('\\', '/')
            assert Path(r[key]).is_file()
        assert run.digest(Path(r['scan'])) == r['scan_sha256']
        rows.append(r)
    rows.sort(key=lambda r: r['frame_id'])
    for date in sorted({r['session_id'] for r in rows}):
        subset = [r for r in rows if r['session_id'] == date]
        boundary = int(.8 * len(subset))
        for i, row in enumerate(subset):
            row['split'] = 'reference' if i < boundary else 'query'
    run.save_json(OUT / 'manifest.json', rows)
    protocol = dict(frames=len(rows), missing=[r['frame_id'] for r in raw if not r['paired']],
        split='Chronological first floor(80%) per date reference; remaining query. Only original train907; no old val/test.',
        candidates=16, similarity='L2-normalized cosine; reference contains image-valid voxels only; candidate selection by LiDAR only',
        positive_m=.5, negative_m=2., ambiguity='Top two LiDAR cosine gap <= 0.02 AND candidate set has a <=0.5m positive and >=2m negative',
        exclusions='Same frame; same date within 10 seconds; camera centers within 5m AND camera orientation difference below 15 degrees, including cross-date',
        labels='Raw Cartesian representative transformed by GT body pose; no GT candidate insertion',
        shuffles=[2089,2090,2091], shuffle='Independent fixed per-frame valid-descriptor permutation in reference and query; same candidate set',
        masks='Existing Cam5 calibration, 480-pixel resizing, 4px depth buffer, 0.5m depth tolerance, valid undistortion mask unchanged',
        viability='At least 200 ambiguous-positive queries across at least 20 query frames and two dates; otherwise inconclusive',
        success='Aligned top1 accuracy beats LiDAR and each shuffled seed on ambiguous-positive subset overall and per date with >=50 such points; paired frame bootstrap lower bound >0 versus LiDAR and mean shuffled',
        scope='Necessary-condition descriptor diagnostic, no training and no final pose improvement claim; existing PCA fitted on these training dates')
    run.save_json(OUT / 'protocol.json', protocol)
    print('manifest', len(rows), {s:sum(r['split']==s for r in rows) for s in ['reference','query']}, flush=True)


def lidar(rows):
    import MinkowskiEngine as ME
    from projection_audit import representative_points, stages
    from utils.pose_util import cartesian_to_polar_expansion, polar_expansion_to_cartesian
    from PIL import Image
    from torch.nn import functional as F
    model = run.load_leader(ARGS)
    extrinsic = np.asarray(json.loads((ARGS.bundle / 'data/validation_scene/scene_meta.json').read_text())['T_BC_camera_to_body'])
    mask = torch.tensor(np.load(ARGS.bundle / 'data/valid_mask.npy'), dtype=torch.float32)
    (OUT / 'lidar').mkdir(exist_ok=True)
    (OUT / 'mapping').mkdir(exist_ok=True)
    for i, row in enumerate(rows):
        dest = OUT / 'lidar' / (row['frame_id'] + '.npz')
        mapping = OUT / 'mapping' / dest.name
        if dest.exists() and mapping.exists():
            continue
        scan, intensity = run.read_scan(row['scan'])
        polar = cartesian_to_polar_expansion(scan, .2 * 1024)
        feat = np.column_stack([polar[:,2], polar[:,1], intensity]).astype(np.float32)
        coords, features, keep = ME.utils.sparse_quantize(coordinates=polar, features=feat, quantization_size=.2, return_index=True)
        sparse = ME.SparseTensor(torch.tensor(features, device='cuda'), ME.utils.batched_coordinates([coords]).cuda())
        with torch.inference_mode():
            enc = model.encoder(sparse)
            center = polar_expansion_to_cartesian((enc.C[:,1:].float() + torch.tensor(enc.tensor_stride, device='cuda') / 2) * .2, .2 * 1024).cpu().numpy()
        xyz, indices, supported = representative_points(scan, coords, keep, enc.C[:,1:].cpu().numpy(), enc.tensor_stride)
        assert np.array_equal(np.floor(cartesian_to_polar_expansion(xyz[supported], .2*1024)/.2).astype(np.int64)//enc.tensor_stride*enc.tensor_stride, enc.C[:,1:].cpu().numpy()[supported])
        gt = np.loadtxt(row['pose']) @ np.linalg.inv(extrinsic)
        w,h = Image.open(row['image']).size
        nh,nw = int(np.ceil(h*480/min(h,w)/8))*8, int(np.ceil(w*480/min(h,w)/8))*8
        k = torch.tensor(np.loadtxt(row['calibration']), dtype=torch.float32)
        k[0] *= nw/w
        k[1] *= nh/h
        m = F.interpolate(mask[None,None], size=(nh,nw), mode='nearest')[0,0]
        uv, valid, _ = stages(torch.tensor(xyz), torch.tensor(scan), torch.tensor(extrinsic,dtype=torch.float32), k, m, torch.tensor(supported))
        np.savez(mapping, projection_xyz=xyz, projection_supported=supported, localization_xyz=center, valid=valid.numpy(), raw_index=indices)
        np.savez(dest, features=enc.F.cpu().numpy(), source=center, representative_world=xyz@gt[:3,:3].T+gt[:3,3], GT=gt, camera_pose=np.loadtxt(row['pose']))
        if i%25==0 or i==len(rows)-1:
            print('lidar',i+1,len(rows),flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    stage = sys.argv[1]
    if stage == 'manifest':
        manifest()
    else:
        rows = json.loads((OUT/'manifest.json').read_text())
        if stage == 'lidar':
            lidar(rows)
        elif stage == 'visual':
            run.visual(ARGS, rows)

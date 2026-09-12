import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from .glace_adapter import GLACEAdapter, deit_global_feature_fn, pixel_grid_uv
from .nclt_camera import preprocess_image


def camera_evidence(xyz, uv, K, T_WC):
    camera = (xyz - T_WC[:3, 3]) @ T_WC[:3, :3]
    valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0)
    error = np.full(len(uv), np.inf)
    projection = camera[valid] @ K.T
    with np.errstate(divide='ignore', invalid='ignore'):
        error[valid] = np.linalg.norm(projection[:, :2] / projection[:, 2:] - uv[valid], axis=1)
    error[~np.isfinite(error)] = np.inf
    return {'S_C': float(np.mean(np.minimum((error / 10.) ** 2, 1.))),
            'q_C': float(np.mean(error < 10.)),
            'median_reprojection_px': float(np.median(error)),
            'positive_finite_depth_fraction': float(np.mean(valid))}


def wrong_pose(T_WC):
    result = T_WC.copy()
    result[:3, 3] += [2., 0., 0.]
    result[:3, :3] = Rotation.from_euler('z', 5., degrees=True).as_matrix() @ result[:3, :3]
    return result


def main():
    root = Path('/root/rivermind-data/glace_nclt_corrected_20260912')
    out = root / 'camera_separability_train64'
    out.mkdir(exist_ok=False)
    (out / 'coordinates').mkdir()
    quality = json.loads((root / 'quality_report.json').read_text())
    head = root / 'glace_head.pt'
    head_hash = hashlib.sha256(head.read_bytes()).hexdigest()
    if head_hash != quality['head_sha256']:
        raise ValueError('Head differs from failed-head quality report')
    records = quality['records']
    if len(records) != 64 or len({r['image'] for r in records}) != 64:
        raise ValueError('Expected exactly the original 64 quality-check training images')
    scene = root / 'scene/train'
    meta = json.loads((root / 'scene/scene_meta.json').read_text())
    E = np.asarray(meta['T_BC_camera_to_body'])
    vendor = root / 'vendor_corrected'
    torch.set_num_threads(4)
    torch.manual_seed(2089)
    fn = deit_global_feature_fn(vendor, '/root/rivermind-data/LEADER-v1-visual-glace/research/visual_glace/CVPR23_DeitS_Rerank.pth')
    adapter = GLACEAdapter(vendor, head, T_BC=E, global_feature_fn=fn)
    paths = {p.stem: p for p in (scene / 'rgb').iterdir()}
    results = []
    manifest = {'head_sha256': head_hash, 'selected_images': [r['image'] for r in records],
                'scope': 'exact 64 training images from failed-head quality report; 16 per training date',
                'score': 'mean(min((L2_reprojection_px/10)^2,1)) over all cells',
                'inlier_fraction': 'mean(L2_reprojection_px < 10) over all cells',
                'invalid_projection': 'nonpositive depth or nonfinite projection: score 1, outlier, denominator retained',
                'poses': 'all T_WC, camera to world; LEADER T_WB multiplied by fixed T_BC',
                'wrong_pose': 'camera center +[2,0,0] metres in world; R_wrong=Rz_world(+5deg)@R_GT',
                'pose_generation': 'no GLACE PnP/DSAC, no fusion, no optimization, no retraining',
                'leader_variant': 'original LEADER SC2-PCR top50%; no v1-two-stage refinement',
                'seed': 2089}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    for entry in records:
        stem = entry['image']
        K = np.loadtxt(scene / 'calibration' / (stem + '.txt'))
        gray, K = preprocess_image(paths[stem], K, 616)
        GT = np.loadtxt(scene / 'poses' / (stem + '.txt'))
        wrong = wrong_pose(GT)
        global_feature = torch.from_numpy(np.asarray(fn(gray),dtype=np.float32))[None].cuda()
        image = torch.from_numpy((gray-.4)/.25)[None,None].cuda()
        with torch.inference_mode(), torch.cuda.amp.autocast():
            coords = adapter.regressor(image, global_feature)
        coords = coords.float().cpu().numpy()[0]
        uv = pixel_grid_uv(8, *coords.shape[1:])
        xyz = coords.reshape(3,-1).T.astype(np.float64)
        np.savez_compressed(out / 'coordinates' / (stem+'.npz'), xyz=xyz, uv=uv, K=K, GT=GT, wrong=wrong)
        row = {'sequence':entry['sequence'], 'image':stem, 'N':len(uv),
               'GT':camera_evidence(xyz,uv,K,GT), 'wrong':camera_evidence(xyz,uv,K,wrong)}
        row['previous_gt_median_difference_px'] = row['GT']['median_reprojection_px'] - entry['median_reprojection_px']
        results.append(row)
        (out / 'camera_scores.json').write_text(json.dumps(results,indent=2))
        print(json.dumps({'stage':'fixed_camera_evidence','completed':len(results),'GT':row['GT'],'wrong':row['wrong']}),flush=True)
    del adapter, fn, global_feature, image, coords
    torch.cuda.empty_cache()

    import MinkowskiEngine as ME
    from safetensors.torch import load_file
    from torch.utils.data import DataLoader
    sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'tools'))
    from eval_clean_ablation import IndexedSubset, collate_samples
    from data.NCLTVelodyne_datagenerator_mink import NCLT_mink
    from models.model_mink import LEADER
    from models.sc2pcr import Matcher
    from utils.pose_util import polar_expansion_to_cartesian

    dataset = NCLT_mink('/root/rivermind-data/datasets',train=True,voxel_size=.2,horizontal_res=1024)
    lookup = {(Path(p).parent.parent.name,Path(p).stem):i for i,p in enumerate(dataset.pcs)}
    indices = [lookup[(r['sequence'],r['image'])] for r in results]
    by_index = dict(zip(indices,results))
    loader = DataLoader(IndexedSubset(dataset,indices),batch_size=4,shuffle=False,collate_fn=collate_samples,num_workers=4,pin_memory=True)
    checkpoint = Path('/root/rivermind-data/LEADER/checkpoints/checkpoint_epoch_49')
    model = LEADER(in_channels=3,out_channels=4,feat_channels=512,width=1024)
    model.load_state_dict(load_file(str(checkpoint/'model.safetensors')),strict=True)
    model.cuda().eval()
    center = torch.tensor(json.loads((checkpoint/'extra.json').read_text())['center_t'],device='cuda',dtype=torch.float32)
    matcher = Matcher(inlier_threshold=2.,d_thre=2,num_iterations=10,ratio=.15,nms_radius=.1,max_points=3000,k1=30)
    completed = 0
    with torch.inference_mode():
        for batch in loader:
            encoded = model.encoder(ME.SparseTensor(batch['feats'].cuda(),batch['coords'].cuda()))
            pred = model.decoder(encoded.F).float()
            stride = torch.tensor(encoded.tensor_stride,device='cuda',dtype=torch.float32)
            local = polar_expansion_to_cartesian((encoded.C[:,1:].float()+stride/2)*.2,204.8)
            for position,index in enumerate(batch['indices']):
                torch.manual_seed(2089+int(index))
                mask = encoded.C[:,0].long()==position
                source,target,reliability = local[mask],pred[mask,:3],pred[mask,3]
                keep = max(min(50,reliability.numel()),int(.5*reliability.numel()))
                top = torch.topk(reliability,keep).indices
                pose = matcher.estimator(source[top][None],target[top][None])[0]
                pose[:3,3] += center
                pose = (pose @ batch['T_corr'][position].cuda()).cpu().numpy().astype(np.float64) @ E
                row = by_index[index]
                p = out/'coordinates'/(row['image']+'.npz')
                with np.load(p) as saved:
                    data = {k:saved[k] for k in saved.files}
                row['LEADER'] = camera_evidence(data['xyz'],data['uv'],data['K'],pose)
                row['leader_error_at_camera_time'] = {
                    'translation_m':float(np.linalg.norm(pose[:3,3]-data['GT'][:3,3])),
                    'rotation_deg':float(Rotation.from_matrix(pose[:3,:3].T@data['GT'][:3,:3]).magnitude()*180/np.pi)}
                row['scan_timestamp_us'] = int(Path(dataset.pcs[index]).stem)
                row['sync_delta_us'] = row['scan_timestamp_us']-int(row['image'])
                np.savez_compressed(p,**data,LEADER=pose)
                completed += 1
                (out/'records.json').write_text(json.dumps(results,indent=2))
                print(json.dumps({'stage':'leader_evidence','completed':completed,'scores':{m:row[m]['S_C'] for m in ['GT','LEADER','wrong']}}),flush=True)
    manifest['leader_checkpoint_sha256'] = hashlib.sha256((checkpoint/'model.safetensors').read_bytes()).hexdigest()
    manifest['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'complete.json').write_text(json.dumps({'frames':completed,'complete':completed==64}))


if __name__ == '__main__':
    main()

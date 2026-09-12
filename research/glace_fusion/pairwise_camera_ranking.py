import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from .camera_separability import camera_evidence
from .glace_adapter import GLACEAdapter, deit_global_feature_fn, pixel_grid_uv
from .make_glace_scene import calibration_chain
from .nclt_camera import camera_rows, stored_intrinsics, preprocess_image, NCLTTrajectory, TRAIN_DATES


LEVELS = {'translation':[0.,.2,.5,1.,2.], 'rotation':[0.,.5,1.,2.,5.]}
TIE_EPS = 1e-8


def candidates(GT, mode):
    result = [{'level':0.,'direction':'origin','pose':GT.copy()}]
    for level in LEVELS[mode][1:]:
        for axis in range(3):
            for sign in [-1,1]:
                direction = np.eye(3)[axis]*sign
                pose = GT.copy()
                if mode == 'translation':
                    pose[:3,3] += GT[:3,:3] @ (level*direction)
                else:
                    pose[:3,:3] = GT[:3,:3] @ Rotation.from_rotvec(np.deg2rad(level)*direction).as_matrix()
                result.append({'level':level,'direction':'xyz'[axis]+('+' if sign>0 else '-'),'pose':pose})
    return result


def pair_counts(scored, same_direction=False, levels=None, direction=None):
    correct = wrong = ties = 0
    for low,high in itertools.combinations(scored,2):
        if low['level'] == high['level']:
            continue
        if low['level'] > high['level']:
            low,high = high,low
        if levels is not None and (low['level'],high['level']) != tuple(levels):
            continue
        if same_direction and low['level'] != 0 and low['direction'] != high['direction']:
            continue
        if direction is not None and any(c['level'] and c['direction']!=direction for c in [low,high]):
            continue
        delta = high['S_C']-low['S_C']
        if delta > TIE_EPS:
            correct += 1
        elif delta < -TIE_EPS:
            wrong += 1
        else:
            ties += 1
    total = correct+wrong+ties
    return {'correct':correct,'wrong':wrong,'ties':ties,'pairs':total,
            'accuracy':correct/total if total else None,
            'half_credit_accuracy':(correct+.5*ties)/total if total else None}


def legacy_main():
    root = Path('/root/rivermind-data/glace_nclt_corrected_20260912')
    out = root/'pairwise_ranking_test64'
    out.mkdir(exist_ok=False)
    (out/'coordinates').mkdir()
    original = [json.loads(line) for line in (root/'paired_diagnostic_64/records.jsonl').read_text().splitlines()]
    if len(original)!=64 or any(r['sequence'] in TRAIN_DATES for r in original):
        raise ValueError('Expected original 64 held-out test frames')
    selected = sorted(original,key=lambda r:(r['sequence'],r['scan_timestamp']))
    camera_root = Path('/root/rivermind-data/datasets/NCLT_camera_v1')
    rows = camera_rows(camera_root,5)
    by_key = {(r['sequence'],r['timestamp_us']):r for r in rows}
    samples = [by_key[(r['sequence'],r['scan_timestamp'])] for r in selected]
    training_manifest = json.loads((root/'camera_separability_train64/manifest.json').read_text())
    if set(training_manifest['selected_images']) & {str(r['timestamp_us']) for r in samples}:
        raise ValueError('Training-image overlap')
    K_raw,E = calibration_chain(camera_root,5,'0.035,0.002,-1.23,-179.93,-0.23,0.50',None)
    trajectories = {date:NCLTTrajectory(Path('/root/rivermind-data/datasets/NCLT')/date/f'groundtruth_{date}.csv') for date in {r['sequence'] for r in samples}}
    head = root/'glace_head.pt'
    head_hash = hashlib.sha256(head.read_bytes()).hexdigest()
    if head_hash != training_manifest['head_sha256']:
        raise ValueError('Head changed')
    manifest = {'scope':'fixed 64 previously selected held-out test frames, 2012-02-12 only; not full NCLT',
                'head_sha256':head_hash,'samples':[{'sequence':r['sequence'],'timestamp_us':r['timestamp_us']} for r in samples],
                'levels':LEVELS,'pose_convention':'camera-to-world T_WC',
                'translation':'camera-local +/-X,+/-Y,+/-Z; orientation fixed',
                'rotation':'camera-local +/-X,+/-Y,+/-Z; camera center fixed',
                'candidates_per_modality':25,'all_unequal_error_pairs_per_frame_modality':240,
                'same_direction_pairs_per_frame_modality':60,
                'tie_epsilon':TIE_EPS,'ties':'counted as incorrect in primary strict accuracy; separately reported',
                'primary':'all candidate pairs with unequal true error magnitudes, translation and rotation separately',
                'score':'mean(min((L2_reprojection_px/10)^2,1)); invalid projection=1, denominator retained',
                'global_features':'existing head-compatible gray replicated to 3 channels; cached once in batches of 16; not official RGB',
                'local_features':'same normalized grayscale input, height 616',
                'training_or_fusion_changes':False,'GT_usage':'construct synthetic diagnostic candidates only, not deployment',
                'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    vendor = root/'vendor_corrected'
    torch.set_num_threads(4)
    torch.manual_seed(2089)
    fn = deit_global_feature_fn(vendor,'/root/rivermind-data/LEADER-v1-visual-glace/research/visual_glace/CVPR23_DeitS_Rerank.pth')
    adapter = GLACEAdapter(vendor,head,T_BC=E,global_feature_fn=fn)
    gray_images,Ks,GTs = [],[],[]
    for row in samples:
        path = camera_root/row['saved_path']
        K,_ = stored_intrinsics(K_raw,row,path)
        gray,K = preprocess_image(path,K,616)
        gray_images.append(gray);Ks.append(K)
        GTs.append(trajectories[row['sequence']].at([row['timestamp_us']])[0]@E)
    features = np.concatenate([fn.batch(gray_images[i:i+16]) for i in range(0,len(samples),16)])
    np.save(out/'features.npy',features)
    results = []
    for index,(row,gray,K,GT) in enumerate(zip(samples,gray_images,Ks,GTs)):
        global_feature = torch.from_numpy(features[index].copy())[None].cuda()
        image = torch.from_numpy((gray-.4)/.25)[None,None].cuda()
        with torch.inference_mode(),torch.cuda.amp.autocast():
            coords = adapter.regressor(image,global_feature)
        coords = coords.float().cpu().numpy()[0]
        uv = pixel_grid_uv(8,*coords.shape[1:])
        xyz = coords.reshape(3,-1).T.astype(np.float64)
        np.savez_compressed(out/'coordinates'/(str(row['timestamp_us'])+'.npz'),xyz=xyz,uv=uv,K=K,GT=GT)
        result = {'sequence':row['sequence'],'timestamp_us':row['timestamp_us'],'modalities':{}}
        for mode in LEVELS:
            scored = []
            for candidate in candidates(GT,mode):
                score = camera_evidence(xyz,uv,K,candidate['pose'])
                scored.append({'level':candidate['level'],'direction':candidate['direction'],'pose':candidate['pose'].tolist(),**score})
            result['modalities'][mode] = {'candidates':scored,'all_pairs':pair_counts(scored),
                                           'same_direction':pair_counts(scored,same_direction=True)}
        results.append(result)
        (out/'records.json').write_text(json.dumps(results,indent=2))
        print(json.dumps({'frames':len(results),**{mode:result['modalities'][mode]['all_pairs'] for mode in LEVELS}}),flush=True)
    manifest['features_sha256'] = hashlib.sha256((out/'features.npy').read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'complete.json').write_text(json.dumps({'frames':len(results),'complete':len(results)==64}))


def main():
    import sys
    if '--legacy' in sys.argv:
        sys.argv.remove('--legacy')
        legacy_main()
    else:
        from .evaluate_scene import main as evaluate
        evaluate()


if __name__ == '__main__':
    main()

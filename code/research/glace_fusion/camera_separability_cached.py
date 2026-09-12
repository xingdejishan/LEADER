import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from .camera_separability import camera_evidence
from .glace_adapter import pixel_grid_uv
from .nclt_camera import preprocess_image


def main():
    root = Path('/root/rivermind-data/glace_nclt_corrected_20260912')
    previous = root / 'camera_separability_train64'
    out = root / 'camera_separability_cached_train64'
    out.mkdir(exist_ok=False)
    (out / 'coordinates').mkdir()
    previous_manifest = json.loads((previous / 'manifest.json').read_text())
    head = root / 'glace_head.pt'
    head_hash = hashlib.sha256(head.read_bytes()).hexdigest()
    if head_hash != previous_manifest['head_sha256']:
        raise ValueError('Head changed')
    records = json.loads((previous / 'records.json').read_text())
    scene = root / 'scene/train'
    rgb_paths = sorted((scene / 'rgb').iterdir())
    features_path = scene / 'features.npy'
    features = np.load(features_path, mmap_mode='r')
    if features.shape != (len(rgb_paths),256):
        raise ValueError('Feature row count or dimension mismatch')
    index_by_stem = {p.stem:i for i,p in enumerate(rgb_paths)}
    if len(index_by_stem) != len(rgb_paths):
        raise ValueError('Duplicate image stem')
    vendor = root / 'vendor_corrected'
    sys.path.insert(0,str(vendor))
    from ace_network import Regressor
    model = Regressor.create_from_split_state_dict(
        torch.load(vendor/'ace_encoder_pretrained.pt',map_location='cpu'),
        torch.load(head,map_location='cpu')).cuda().eval()
    torch.set_num_threads(4)
    torch.manual_seed(2089)
    output = []
    for old in records:
        stem = old['image']
        index = index_by_stem[stem]
        with np.load(previous/'coordinates'/(stem+'.npz')) as archive:
            fixed = {k:archive[k] for k in archive.files}
        K_stored = np.loadtxt(scene/'calibration'/(stem+'.txt'))
        gray,K = preprocess_image(rgb_paths[index],K_stored,616)
        np.testing.assert_array_equal(K,fixed['K'])
        image = torch.from_numpy((gray-.4)/.25)[None,None].cuda()
        global_feature = torch.from_numpy(np.asarray(features[index],dtype=np.float32).copy())[None].cuda()
        with torch.inference_mode(),torch.cuda.amp.autocast():
            coords = model(image,global_feature)
        coords = coords.float().cpu().numpy()[0]
        xyz = coords.reshape(3,-1).T.astype(np.float64)
        uv = pixel_grid_uv(8,*coords.shape[1:])
        np.testing.assert_array_equal(uv,fixed['uv'])
        row = {'sequence':old['sequence'],'image':stem,'feature_index':index,'N':len(uv),
               'xyz_max_difference_m':float(np.max(np.abs(xyz-fixed['xyz']))),
               'xyz_mean_difference_m':float(np.mean(np.abs(xyz-fixed['xyz'])))}
        for name in ['GT','LEADER','wrong']:
            row[name] = camera_evidence(xyz,uv,K,fixed[name])
            row[name+'_score_delta_from_online'] = row[name]['S_C']-old[name]['S_C']
            row[name+'_inlier_delta_from_online'] = row[name]['q_C']-old[name]['q_C']
        np.savez_compressed(out/'coordinates'/(stem+'.npz'),xyz=xyz,uv=uv,K=K,
                            GT=fixed['GT'],LEADER=fixed['LEADER'],wrong=fixed['wrong'])
        output.append(row)
        (out/'records.json').write_text(json.dumps(output,indent=2))
        print(json.dumps({'completed':len(output),'scores':{n:row[n]['S_C'] for n in ['GT','LEADER','wrong']}}),flush=True)
    summary = {'frames':len(output),'methods':{},
               'max_xyz_difference_m':max(r['xyz_max_difference_m'] for r in output)}
    for name in ['GT','LEADER','wrong']:
        summary['methods'][name] = {metric:float(np.mean([r[name][metric] for r in output])) for metric in ['S_C','q_C']}
        summary['methods'][name]['max_absolute_score_delta'] = max(abs(r[name+'_score_delta_from_online']) for r in output)
        summary['methods'][name]['max_absolute_q_delta'] = max(abs(r[name+'_inlier_delta_from_online']) for r in output)
    summary['GT_score_wins_vs_wrong'] = sum(r['GT']['S_C']<r['wrong']['S_C'] for r in output)
    summary['GT_q_wins_vs_wrong'] = sum(r['GT']['q_C']>r['wrong']['q_C'] for r in output)
    summary['LEADER_score_wins_vs_wrong'] = sum(r['LEADER']['S_C']<r['wrong']['S_C'] for r in output)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    original_config = json.loads((root.parent/'glace_nclt_full_20260912/config.json').read_text())
    manifest = {'head_sha256':head_hash,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'features_resolved_path':str(features_path.resolve()),
                'features_sha256':hashlib.sha256(features_path.read_bytes()).hexdigest(),
                'training_global_preprocessing':original_config['global_preprocessing'],
                'features_index':'sorted(scene/train/rgb), image stem lookup',
                'only_changed_input':'global features loaded from training features.npy rather than recomputed online',
                'fixed_inputs':'same head, gray local image, K, uv, saved GT/LEADER/wrong poses',
                'previous_run':str(previous),'complete':len(output)==64}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()

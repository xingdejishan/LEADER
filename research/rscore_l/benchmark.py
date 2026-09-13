import json
import pickle
import time

import numpy as np
import torch
from hloc import extractors
from hloc.utils.base_model import dynamic_load

from scrstudio.data.samplers import PQKNN
from scrstudio.encoders.dedode_encoder import DedodeEncoderConfig
from scrstudio.encoders.pca_encoder import PCAEncoderConfig
from scrstudio.scripts.retrieval_feat import HLocDatasetConfig

from .dataset import NCLTDatasetConfig
from .evaluate import load_model, predict, visual_pose
from .fusion import fuse
from .prepare import save_json


def benchmark(root, bundle, variant):
    data = root / 'data'
    model, _ = load_model(root, variant)
    encoder = PCAEncoderConfig(encoder=DedodeEncoderConfig(detector='L', descriptor='B', k=5000), pca_path='pcad3LB_128.pth').setup(data_path=data).cuda().eval()
    images = NCLTDatasetConfig(data=data, split='test').setup(preprocess=encoder.preprocess)
    retrieval_images = HLocDatasetConfig(root=data / 'test', conf={'resize_max': 1024}).setup()
    netvlad = dynamic_load(extractors, 'netvlad')({'name': 'netvlad'}).cuda().eval()
    encoding = 'lidar_n2c.pt' if variant == 'reliable' or variant.startswith('lidar') else 'pose_n2c.pt'
    global_features = torch.load(data / 'train' / encoding, weights_only=True)['model.embedding.weight'].cuda().float()
    with (data / 'train/netvlad_feats_pq.pkl').open('rb') as file:
        pq, codes = pickle.load(file)
    retriever = PQKNN(pq, codes, n_neighbors=10)
    rows = json.loads((data / 'manifest.json').read_text())['test']
    E = np.asarray(json.loads((data / 'scene_meta.json').read_text())['T_BC_camera_to_body'])
    reliable = None
    if variant == 'reliable':
        from .reliability import ReliabilityHead, reliability_inputs
        saved = torch.load(root / 'reliability/head.pt', weights_only=True)
        reliable = ReliabilityHead().cuda().eval()
        reliable.load_state_dict(saved['model'])
    database = np.load(data / 'train/netvlad_feats.npy').astype(np.float32)
    records = []
    torch.cuda.reset_peak_memory_stats()
    chosen = np.linspace(0, len(images)-1, 16).astype(int)
    for index in np.r_[chosen[0], chosen]:
        torch.cuda.synchronize()
        started = time.perf_counter()
        image = images[int(index)]
        retrieval_input = torch.from_numpy(retrieval_images[int(index)]['image'][None]).cuda()
        with torch.inference_mode(), torch.autocast('cuda'):
            local = encoder.keypoint_features({k: image[k][None].cuda() for k in ('image', 'mask')})
        with torch.inference_mode():
            descriptor = netvlad({'image': retrieval_input})['global_descriptor'][0].cpu().numpy()
        hypotheses = retriever.kneighbors(descriptor)
        coordinates, hidden = predict(model, local['descriptors'].float(), global_features[hypotheses])
        if reliable is None:
            probability = torch.full(coordinates.shape[:2], .5, device='cuda')
        else:
            with torch.inference_mode():
                similarity = torch.from_numpy(descriptor @ database[hypotheses.cpu().numpy()].T).cuda()
                probability = torch.sigmoid(reliable(reliability_inputs(hidden, coordinates, local['keypoint_scores'], similarity)) / saved['temperature'])
        correspondence = dict(uv=local['keypoints'].float().cpu().numpy(), xyz=coordinates[:, :, 1].cpu().numpy(),
            reliability=probability.cpu().numpy(), K=image['intrinsics'].numpy(), image_size_hw=np.array(image['image'].shape[-2:]))
        torch.cuda.synchronize()
        frontend_end = time.perf_counter()
        visual_pose(correspondence)
        pnp_end = time.perf_counter()
        with np.load(bundle / 'cache/lidar_pools' / (rows[int(index)]['frame_id'] + '.npz')) as stored:
            pool = {k: stored[k] for k in ('v1_two_stage', 'T_corr', 'c_local_all', 'c_pred_all', 'center_t', 'u_pred_all', 'candidate_T_WB')}
        fuse(pool, correspondence, E)
        end = time.perf_counter()
        records.append(dict(frame_id=rows[int(index)]['frame_id'], frontend_seconds=frontend_end-started,
            standalone_pnp_seconds=pnp_end-frontend_end, cached_leader_fusion_seconds=end-pnp_end,
            image_to_fused_pose_seconds=frontend_end-started+end-pnp_end))
    records = records[1:]
    save_json(root / 'evaluation' / variant / 'benchmark.json', dict(records=records,
        mean={key: float(np.mean([r[key] for r in records])) for key in records[0] if key != 'frame_id'},
        peak_memory_mb=torch.cuda.max_memory_allocated()/2**20, head_parameters=sum(p.numel() for p in model.parameters()),
        scope='Image reading and preprocessing, frozen DeDoDe/PCA, frozen NetVLAD, retrieval, head, optional reliability, fusion with cached LEADER inputs',
        excludes='LEADER LiDAR frontend runtime; amortized model load', gpu=torch.cuda.get_device_name()))

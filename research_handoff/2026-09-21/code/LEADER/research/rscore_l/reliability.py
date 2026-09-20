import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scrstudio.data.samplers import PQKNN

from .evaluate import load_model, predict
from .fusion import spatial_budget
from .prepare import save_json


class ReliabilityHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(torch.nn.LayerNorm(771), torch.nn.Linear(771, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))

    def forward(self, features):
        return self.network(features).squeeze(-1)


def reliability_inputs(hidden, coordinates, detector, similarity):
    correction = (coordinates[:, :, 1] - coordinates[:, :, 0]).norm(dim=-1, keepdim=True).log1p()
    detector = detector.clamp_min(1e-12).log()[None, :, None].expand(len(hidden), -1, -1)
    similarity = similarity[:, None, None].expand(-1, hidden.shape[1], -1)
    return torch.cat([hidden, correction, detector, similarity], dim=-1)


def differentiable_camera_score(probability, xyz, uv, K, T_WB, T_BC, size, sigma=4.):
    transform = torch.linalg.inv(T_WB @ T_BC)
    camera = xyz @ transform[:3, :3].T + transform[:3, 3]
    projected = camera @ K.T
    pixel = projected[..., :2] / projected[..., 2:].clamp_min(1e-8)
    h, w = size
    valid = (camera[..., 2] > 0) & torch.isfinite(pixel).all(-1) & (pixel >= 0).all(-1)
    valid &= (pixel[..., 0] < w) & (pixel[..., 1] < h)
    squared = ((pixel - uv) ** 2).sum(-1)
    gaussian = torch.exp(-(squared / (2*sigma**2)).clamp_max(80)) / (2*np.pi*sigma**2)
    gaussian = torch.where(valid, gaussian, torch.zeros_like(gaussian))
    p = probability.clamp(0, 1-1e-6)
    cost = -(p*gaussian + (1-p)/(h*w)).clamp_min(1e-15).log()
    cells = (uv / uv.new_tensor([w, h]) * 4).long().clamp(0, 3)
    blocks = cells[:, 0] + 4*cells[:, 1]
    selected = (cells.sum(1) % 2) == 0
    values = [cost[:, (blocks == b) & selected].mean(1) for b in range(16) if ((blocks == b) & selected).any()]
    if not values:
        return probability.sum() * 0 + np.log(h*w)
    return torch.stack(values).mean(0).min()


def ranking_loss(model, pair):
    probability = torch.sigmoid(model(pair['features'].float().cuda()))
    arguments = {k: pair[k].cuda() if torch.is_tensor(pair[k]) else pair[k] for k in ('xyz', 'uv', 'K', 'T_BC', 'size')}
    positive = differentiable_camera_score(probability, T_WB=pair['positive'].cuda(), **arguments)
    negative = differentiable_camera_score(probability, T_WB=pair['negative'].cuda(), **arguments)
    return F.relu(positive - negative + .1)


def train_reliability(root):
    torch.manual_seed(2089)
    data = root / 'data'
    rows = json.loads((data / 'manifest.json').read_text())['train']
    poses = np.load(data / 'train/poses.npy')
    sessions = sorted(set(row['session_id'] for row in rows))
    retrieval = np.load(data / 'train/netvlad_feats.npy').astype(np.float32)
    global_features = torch.load(data / 'train/lidar_n2c.pt', weights_only=True)['model.embedding.weight'].cuda().float()
    with (data / 'train/netvlad_feats_pq.pkl').open('rb') as file:
        pq, retrieval_codes = pickle.load(file)
    destination = root / 'reliability'
    destination.mkdir(exist_ok=True)
    inputs, targets, calibration_masks = [], [], []
    rank_pairs = []
    E = np.asarray(json.loads((data / 'scene_meta.json').read_text())['T_BC_camera_to_body'])
    for fold, session in enumerate(sessions):
        model, checkpoint = load_model(root, 'lidar-fold-' + session)
        fitted = np.array([i for i, row in enumerate(rows) if row['session_id'] != session])
        retriever = PQKNN(pq, retrieval_codes[fitted], n_neighbors=10)
        heldout = [i for i, row in enumerate(rows) if row['session_id'] == session]
        for position, i in enumerate(tqdm(heldout, desc='Cross-fit reliability ' + session)):
            feature = dict(np.load(data / 'proc/features_train' / (rows[i]['frame_id'] + '.npz')))
            geometry = dict(np.load(data / 'proc/geometry_features_train' / (rows[i]['frame_id'] + '.npz')))
            selected = spatial_budget(feature['uv'], feature['image_size_hw'], 256)
            if len(selected) == 0:
                continue
            indices = fitted[retriever.kneighbors(retrieval[i]).cpu().numpy()]
            coordinates, hidden = predict(model, torch.from_numpy(feature['features'][selected].astype(np.float32)).cuda(), global_features[indices])
            vector = reliability_inputs(hidden, coordinates, torch.from_numpy(feature['scores'][selected]).cuda(),
                torch.from_numpy(retrieval[i] @ retrieval[indices].T).cuda())
            prediction = coordinates[:, :, 1].cpu().numpy()
            camera = (prediction - poses[i, :3, 3]) @ poses[i, :3, :3]
            ground = (geometry['xyz_target_world'][selected] - poses[i, :3, 3]) @ poses[i, :3, :3]
            rays = np.c_[feature['uv'][selected], np.ones(len(selected))] @ np.linalg.inv(feature['K']).T
            rays /= np.linalg.norm(rays, axis=1, keepdims=True)
            along = (camera * rays[None]).sum(-1)
            range_error = np.abs(along - (ground * rays).sum(-1)[None])
            projection = camera @ feature['K'].T
            pixel_error = np.linalg.norm(projection[:, :, :2] / np.maximum(projection[:, :, 2:], .1) - feature['uv'][selected][None], axis=-1)
            label = (range_error < 1.) & (pixel_error < 10.) & (camera[:, :, 2] > 0)
            labeled = geometry['geometry_valid'][selected]
            if labeled.any():
                inputs.append(vector[:, labeled].detach().cpu().half().reshape(-1, 771))
                targets.append(torch.from_numpy(label[:, labeled].astype(np.float32).reshape(-1)))
                calibration_masks.append(torch.full((int(labeled.sum()) * len(label),), session == sessions[-1], dtype=torch.bool))
            pool_path = data / 'train_candidate_pools' / (rows[i]['frame_id'] + '.npz')
            if pool_path.exists() and session != sessions[-1]:
                from .evaluate import pose_error
                from .fusion import lidar_score
                pool = dict(np.load(pool_path))
                candidates = np.concatenate([pool['v1_two_stage'][None], pool['candidate_T_WB']])
                candidates = candidates[np.isfinite(candidates).all(axis=(1, 2))]
                error = np.asarray([pose_error(T, poses[i] @ np.linalg.inv(E)) for T in candidates])
                cost = error[:, 0] / .5 + error[:, 1] / 2
                positive = int(np.argmin(cost))
                negatives = np.flatnonzero(cost > cost[positive] + .5)
                if len(negatives):
                    Q = pool['T_corr']
                    body = (pool['c_local_all'] - Q[:3, 3]) @ Q[:3, :3]
                    world = pool['c_pred_all'] + pool['center_t']
                    weights = np.exp(np.log(10)/np.pi*np.arctan(np.clip(pool['u_pred_all'].reshape(-1), -10*np.pi, 10*np.pi)))
                    weights /= weights.sum()
                    negative = min(negatives, key=lambda j: lidar_score(candidates[j], body, world, weights))
                    rank_pairs.append(dict(features=vector.detach().cpu().half(), xyz=coordinates[:, :, 1].detach().cpu(),
                        uv=torch.from_numpy(feature['uv'][selected]).float(), K=torch.from_numpy(feature['K']).float(),
                        T_BC=torch.from_numpy(E).float(), size=feature['image_size_hw'].tolist(),
                        positive=torch.from_numpy(candidates[positive]).float(), negative=torch.from_numpy(candidates[negative]).float()))
        del model
        torch.cuda.empty_cache()
    if not inputs:
        raise RuntimeError('No trusted cross-fit geometry labels; reliability cannot be fitted')
    x, y, calibration = torch.cat(inputs), torch.cat(targets), torch.cat(calibration_masks)
    train_indices = torch.nonzero(~calibration).flatten()
    calibration_indices = torch.nonzero(calibration).flatten()
    if not len(calibration_indices) or not len(train_indices):
        raise RuntimeError('Cross-fit calibration split is empty')
    model = ReliabilityHead().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
    for step in range(5000):
        indices = train_indices[torch.randint(len(train_indices), (4096,))]
        logits = model(x[indices].float().cuda())
        loss = F.binary_cross_entropy_with_logits(logits, y[indices].cuda())
        if rank_pairs:
            loss = loss + .1 * ranking_loss(model, rank_pairs[step % len(rank_pairs)])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 500 == 0:
            print(json.dumps(dict(stage='reliability', step=step, loss=float(loss))), flush=True)
    model.eval()
    with torch.inference_mode():
        logits = torch.cat([model(x[indices].float().cuda()).cpu() for indices in calibration_indices.split(4096)])
    labels = y[calibration_indices]
    temperatures = torch.logspace(-1, 1, 101)
    losses = torch.stack([F.binary_cross_entropy_with_logits(logits / t, labels) for t in temperatures])
    temperature = float(temperatures[losses.argmin()])
    probability = torch.sigmoid(logits / temperature)
    bins = torch.linspace(0, 1, 11)
    ece = 0.
    calibration_table = []
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (probability >= low) & (probability < high)
        if mask.any():
            confidence, frequency = float(probability[mask].mean()), float(labels[mask].mean())
            ece += float(mask.float().mean()) * abs(confidence - frequency)
            calibration_table.append(dict(low=float(low), high=float(high), count=int(mask.sum()), probability=confidence, frequency=frequency))
    torch.save(dict(model=model.cpu().state_dict(), temperature=temperature), destination / 'head.pt')
    save_json(destination / 'report.json', dict(samples=len(y), training_samples=len(train_indices), calibration_samples=len(calibration_indices),
        positive_fraction=float(y.mean()), brier=float(((probability-labels)**2).mean()), ece=ece, calibration=calibration_table,
        queries='Leave-one-training-session-out coordinate models; retrieved codes only from other sessions',
        calibration_session=sessions[-1],
        labels='Weak checked surface reference, <10px and <1m along ray', temperature=temperature,
        candidate_ranking_trained=bool(rank_pairs), candidate_ranking_pairs=len(rank_pairs),
        candidate_ranking_reason='Training-side real pools only' if rank_pairs else 'No real LEADER candidate pools for training-side heldout queries are present in the local bundle',
        validation_or_test_used_for_fitting=False))

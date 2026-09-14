import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORKSPACE = REPO.parents[1]
sys.path.insert(0, str(REPO))
from fusion import ImageGate, sample_visible


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix('.pending').write_text(json.dumps(value, indent=2))
    path.with_suffix('.pending').replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def make_manifest(args):
    scene = args.bundle / 'data/validation_scene/train'
    scans = args.bundle / 'data/scans/2012-02-18/velodyne_sync'
    names = sorted(p.stem for p in (scene / 'rgb').glob('*.jpg') if (scans / (p.stem + '.bin')).exists())
    if len(names) != 301:
        raise ValueError(f'Expected 301 local paired scans, found {len(names)}')
    rows = []
    for split, pool, count in [('train', names[:201], 64), ('val', names[221:], 32)]:
        for index in np.linspace(0, len(pool) - 1, count).round().astype(int):
            name = pool[index]
            rows.append(dict(frame_id=name, split=split, sequence='2012-02-18',
                             image=str(scene / 'rgb' / (name + '.jpg')),
                             scan=str(scans / (name + '.bin')),
                             calibration=str(scene / 'calibration' / (name + '.txt')),
                             pose=str(scene / 'poses' / (name + '.txt')),
                             scan_sha256=digest(scans / (name + '.bin'))))
    protocol = dict(seed=2089, steps=600, learning_rate=.0001, points_per_step=1024,
                    train_frames=64, val_frames=32, embargo_frames=20,
                    train_pool=201, val_pool=80, local_pairs=301,
                    frozen=['LEADER.encoder', 'LEADER.decoder', 'DeDoDe-B', 'PCA'],
                    arms=['baseline', 'aligned', 'shuffled'], image_smaller_side=480,
                    visibility=dict(cell_px=4, depth_tolerance_m=.5),
                    pass_rule='aligned rescues > damages at 1m/5deg; mean translation improves >=5%; mean rotation and p95 translation do not worsen >5%; aligned beats shuffled mean translation',
                    stop_rule='Stop after fixed 600 steps if validation pass rule fails; no test-based tuning',
                    status='local development kill-test; not independent blind test or full NCLT evaluation',
                    gt_usage='camera pose converted to body pose for training targets/evaluation only; never for image projection',
                    pca='Reuse PCA fit on 907 images from other training dates, no refit',
                    checkpoint_sha256=digest(args.checkpoint / 'model.safetensors'),
                    source_commit='f84b5f1', baseline='upstream/main 4a1bde8, no v1 residual head')
    save_json(args.output / 'manifest.json', rows)
    save_json(args.output / 'protocol.json', protocol)
    return rows


def load_leader(args):
    from models.model_mink import LEADER
    from safetensors.torch import load_file
    model = LEADER(in_channels=3, out_channels=4, feat_channels=512, width=1024).cuda().eval()
    model.load_state_dict(load_file(str(args.checkpoint / 'model.safetensors')), strict=True)
    model.requires_grad_(False)
    return model


def read_scan(path):
    data = np.fromfile(path, dtype=np.dtype([('x', '<u2'), ('y', '<u2'), ('z', '<u2'), ('i', 'u1'), ('l', 'u1')]))
    points = np.column_stack([data[k] for k in ('x', 'y', 'z')]).astype(np.float32) * .005 - 100
    keep = (np.linalg.norm(points, axis=-1) > 1) & (np.linalg.norm(points, axis=-1) < 100)
    return points[keep], data['i'][keep]


def lidar(args, rows):
    import MinkowskiEngine as ME
    from utils.pose_util import cartesian_to_polar_expansion, polar_expansion_to_cartesian
    model = load_leader(args)
    center = np.asarray(json.loads((args.checkpoint / 'extra.json').read_text())['center_t'], np.float32)
    extrinsic = np.asarray(json.loads((args.bundle / 'data/validation_scene/scene_meta.json').read_text())['T_BC_camera_to_body'])
    (args.output / 'lidar').mkdir(exist_ok=True)
    for i, row in enumerate(rows):
        target = args.output / 'lidar' / (row['frame_id'] + '.npz')
        if target.exists():
            continue
        start = time.perf_counter()
        scan, intensity = read_scan(row['scan'])
        polar = cartesian_to_polar_expansion(scan, .2 * 1024)
        features = np.column_stack([polar[:, 2], polar[:, 1], intensity]).astype(np.float32)
        coords, features = ME.utils.sparse_quantize(coordinates=polar, features=features, quantization_size=.2)
        sparse = ME.SparseTensor(torch.as_tensor(features, device='cuda'), ME.utils.batched_coordinates([coords]).cuda())
        with torch.inference_mode():
            encoded = model.encoder(sparse)
            stride = torch.tensor(encoded.tensor_stride, device='cuda')
            points = polar_expansion_to_cartesian((encoded.C[:, 1:].float() + stride / 2) * .2, .2 * 1024)
            prediction = model.decoder(encoded.F)
        gt = np.loadtxt(row['pose']) @ np.linalg.inv(extrinsic)
        source = points.cpu().numpy()
        torch.cuda.synchronize()
        np.savez(target, features=encoded.F.cpu().numpy(), source=source,
                 prediction=prediction.cpu().numpy(), target=source @ gt[:3, :3].T + gt[:3, 3] - center,
                 GT=gt, center=center, seconds=time.perf_counter() - start)
        print(f'lidar {i + 1}/{len(rows)} voxels={len(source)} seconds={time.perf_counter()-start:.2f}', flush=True)


def visual(args, rows):
    from PIL import Image
    from kornia.feature.dedode.dedode_models import get_descriptor
    from torch.nn import functional as F
    descriptor = get_descriptor('B').cuda().eval()
    descriptor.load_state_dict(torch.load(args.workspace / 'rscore-assets/dedode_descriptor_B.pth', map_location='cpu', weights_only=True))
    descriptor.requires_grad_(False)
    pca = torch.load(args.pca, map_location='cuda', weights_only=True)
    extrinsic = torch.tensor(json.loads((args.bundle / 'data/validation_scene/scene_meta.json').read_text())['T_BC_camera_to_body'], device='cuda', dtype=torch.float32)
    mask = torch.tensor(np.load(args.bundle / 'data/valid_mask.npy'), device='cuda', dtype=torch.float32)
    mean = torch.tensor([.485, .456, .406], device='cuda')[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device='cuda')[None, :, None, None]
    visual_folder = args.output / ('visual_raw' if args.projection_mapping is not None else 'visual')
    visual_folder.mkdir(exist_ok=True)
    for i, row in enumerate(rows):
        target = visual_folder / (row['frame_id'] + '.npz')
        if target.exists():
            continue
        start = time.perf_counter()
        image = Image.open(row['image']).convert('RGB')
        w, h = image.size
        ratio = 480 / min(h, w)
        nh, nw = int(np.ceil(h * ratio / 8)) * 8, int(np.ceil(w * ratio / 8)) * 8
        image = image.resize((nw, nh), Image.Resampling.BILINEAR)
        image = torch.tensor(np.array(image), device='cuda').permute(2, 0, 1)[None].float() / 255
        intrinsics = torch.tensor(np.loadtxt(row['calibration']), device='cuda', dtype=torch.float32)
        intrinsics[0] *= nw / w
        intrinsics[1] *= nh / h
        resized_mask = F.interpolate(mask[None, None], size=(nh, nw), mode='nearest')[0, 0]
        with np.load(args.output / 'lidar' / (row['frame_id'] + '.npz')) as cached:
            points = torch.tensor(cached['source'], device='cuda')
        supported = torch.ones(len(points), dtype=torch.bool, device='cuda')
        mapping_hash = ''
        if args.projection_mapping is not None:
            mapping_path = args.projection_mapping / (row['frame_id'] + '.npz')
            mapping_hash = digest(mapping_path)
            with np.load(mapping_path) as mapping:
                if not np.array_equal(points.cpu().numpy(), mapping['localization_xyz']):
                    raise ValueError('Projection mapping does not match cached LEADER output rows')
                points = torch.tensor(mapping['projection_xyz'], device='cuda')
                supported = torch.tensor(mapping['projection_supported'], device='cuda')
        scan, _ = read_scan(row['scan'])
        with torch.inference_mode(), torch.autocast('cuda'):
            dense = descriptor((image - mean) / std)
        with torch.inference_mode():
            compressed = F.conv2d(dense.float(), pca['weight'].float(), pca['bias'].float())
            sampled, valid = sample_visible(compressed, points, extrinsic, intrinsics, resized_mask,
                                            torch.tensor(scan, device='cuda'))
            valid &= supported
            sampled = torch.where(valid[:, None], sampled, torch.zeros_like(sampled))
            if args.projection_mapping is not None:
                with np.load(mapping_path) as mapping:
                    if not np.array_equal(valid.cpu().numpy(), mapping['valid']):
                        raise ValueError('Projection audit and actual feature sampler masks differ')
        torch.cuda.synchronize()
        np.savez(target, image=sampled.cpu().numpy(), valid=valid.cpu().numpy(), seconds=time.perf_counter() - start,
                 projection_mapping_sha256=mapping_hash)
        print(f'visual {i+1}/{len(rows)} visible={valid.float().mean():.4f}', flush=True)


def official_trr():
    tree = ast.parse((REPO / 'run_mink.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TRR')
    scope = dict(np=np, torch=torch, Tensor=torch.Tensor)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(REPO / 'run_mink.py'), 'exec'), scope)
    return scope['TRR'](scale=10.)


def frame(args, row):
    with np.load(args.output / 'lidar' / (row['frame_id'] + '.npz')) as data:
        result = {k: torch.as_tensor(data[k], device='cuda', dtype=torch.float32) for k in ['features', 'source', 'prediction', 'target', 'GT', 'center']}
    with np.load(args.output / 'visual' / (row['frame_id'] + '.npz')) as data:
        result['image'] = torch.tensor(data['image'], device='cuda')
        result['valid'] = torch.tensor(data['valid'], device='cuda')
    return result


def train(args, rows):
    torch.manual_seed(2089)
    model = load_leader(args)
    trr = official_trr()
    training = [r for r in rows if r['split'] == 'train']
    generator = np.random.default_rng(2089)
    schedule = generator.integers(len(training), size=600)
    for arm in ['aligned', 'shuffled']:
        torch.manual_seed(2089)
        gate = ImageGate().cuda()
        optimizer = torch.optim.Adam(gate.parameters(), lr=.0001)
        logs = []
        for step, index in enumerate(schedule):
            item = frame(args, training[index])
            indices = torch.randperm(len(item['features']), device='cuda')[:1024]
            feature, image, valid, target = [item[k][indices] for k in ['features', 'image', 'valid', 'target']]
            if arm == 'shuffled':
                image = image.clone()
                v = torch.where(valid)[0]
                image[v] = image[v.roll(1)]
            optimizer.zero_grad(set_to_none=True)
            pred = model.decoder(gate(feature, image, valid))
            loss = trr(target, pred[:, :3], pred[:, 3], torch.zeros(len(pred), device='cuda', dtype=torch.long))[0].mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{arm} step {step} nonfinite loss')
            loss.backward()
            optimizer.step()
            if step % 50 == 0 or step == 599:
                logs.append(dict(step=step, loss=float(loss.detach())))
                save_json(args.output / (arm + '_training.json'), logs)
                print(f'{arm} {step + 1}/600 loss={loss.item():.4f}', flush=True)
        torch.save(gate.cpu().state_dict(), args.output / (arm + '.pt'))


def metrics(errors):
    values = np.asarray(errors)
    return dict(mean=values.mean(0).tolist(), median=np.median(values, axis=0).tolist(),
                p95=np.percentile(values, 95, axis=0).tolist(),
                successes=int(((values[:, 0] < 1) & (values[:, 1] < 5)).sum()), count=len(values))


def evaluate(args, rows):
    from models.sc2pcr import Matcher
    model = load_leader(args)
    matcher = Matcher(inlier_threshold=2., d_thre=2, num_iterations=10, ratio=.15, nms_radius=.1, max_points=3000, k1=30)
    gates = {}
    for arm in ['aligned', 'shuffled']:
        gate = ImageGate().cuda().eval()
        gate.load_state_dict(torch.load(args.output / (arm + '.pt'), map_location='cuda'))
        gates[arm] = gate
    errors = {k: [] for k in ['baseline', 'aligned', 'shuffled', 'aligned_wrong', 'aligned_missing']}
    records = []
    for row in [r for r in rows if r['split'] == 'val']:
        item = frame(args, row)
        with torch.inference_mode():
            predictions = {'baseline': item['prediction']}
            image, valid = item['image'], item['valid']
            wrong = image.clone()
            v = torch.where(valid)[0]
            wrong[v] = image[v.roll(1)]
            for name, arm, visual, mask in [('aligned', 'aligned', image, valid), ('shuffled', 'shuffled', wrong, valid),
                                           ('aligned_wrong', 'aligned', wrong, valid), ('aligned_missing', 'aligned', image, torch.zeros_like(valid))]:
                predictions[name] = model.decoder(gates[arm](item['features'], visual, mask))
            parity = (predictions['aligned_missing'] - predictions['baseline']).abs().max().item()
            if parity > 1e-4:
                raise ValueError(f'Missing-image baseline parity failed: {parity}')
            for arm, prediction in predictions.items():
                torch.manual_seed(2089)
                n = len(prediction)
                selected = prediction[:, 3].topk(max(min(50, n), int(.5 * n))).indices
                pose = matcher.estimator(item['source'][selected][None], prediction[selected, :3][None])[0]
                pose[:3, 3] += item['center']
                trans = (pose[:3, 3] - item['GT'][:3, 3]).norm().item()
                cosine = ((pose[:3, :3].T @ item['GT'][:3, :3]).trace() - 1) / 2
                rotation = torch.rad2deg(cosine.clamp(-1, 1).acos()).item()
                errors[arm].append([trans, rotation])
        records.append(dict(frame_id=row['frame_id'], visible_fraction=float(valid.float().mean()), errors={k: v[-1] for k, v in errors.items()}, missing_prediction_max_error=parity))
        print(f'validation {len(records)} baseline={errors["baseline"][-1]} aligned={errors["aligned"][-1]}', flush=True)
        save_json(args.output / 'validation_frames.json', records)
    summary = {k: metrics(v) for k, v in errors.items()}
    base, aligned, shuffled = [summary[k] for k in ['baseline', 'aligned', 'shuffled']]
    a, b = np.asarray(errors['aligned']), np.asarray(errors['baseline'])
    sa, sb = (a[:, 0] < 1) & (a[:, 1] < 5), (b[:, 0] < 1) & (b[:, 1] < 5)
    rescue, damage = int((sa & ~sb).sum()), int((~sa & sb).sum())
    passed = rescue > damage and aligned['mean'][0] <= .95 * base['mean'][0]
    passed &= aligned['mean'][1] <= 1.05 * base['mean'][1] and aligned['p95'][0] <= 1.05 * base['p95'][0]
    passed &= aligned['mean'][0] < shuffled['mean'][0]
    save_json(args.output / 'result.json', dict(metrics=summary, rescue=rescue, damage=damage, passed=bool(passed),
              next_action='expand frozen development evaluation' if passed else 'stop: local kill-test failed',
              scope='32 local development frames; frozen encoder and decoder; one seed, not full NCLT'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['manifest', 'lidar', 'visual', 'train', 'evaluate'])
    parser.add_argument('--workspace', type=Path, default=WORKSPACE)
    parser.add_argument('--output', type=Path, default=Path('/home/zhang/leader-image-gate'))
    parser.add_argument('--pca', type=Path, default=Path('/home/zhang/rscore-l-local/data/proc/pcad3LB_128.pth'))
    parser.add_argument('--projection-mapping', type=Path)
    args = parser.parse_args()
    args.bundle = args.workspace / 'glace-local'
    args.checkpoint = args.workspace / 'research/image_gate_checkpoint'
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if args.stage == 'manifest':
        make_manifest(args)
        return
    rows = json.loads((args.output / 'manifest.json').read_text())
    globals()[args.stage](args, rows)


if __name__ == '__main__':
    main()

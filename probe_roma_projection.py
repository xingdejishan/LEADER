import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import e2e_online as E


SOURCE = Path(__file__).resolve()
ROMA = Path('/home/zhangcy/projects/roma-visual3d-b-20261013-v1')
FIXED = Path('/data2/zhangcy/e2e-projection-probe-1792833470/results/evidence.jsonl')
LIBRARY = Path('/data2/zhangcy/sp-lightglue-hard1502-20261013-v1/library.json')
COMMON = Path('/home/zhangcy/projects/sp-lightglue-hard1502-20261013-v1/common.py')


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def raw_to_rotated(uv):
    import numpy as np

    return np.column_stack((615 - uv[:, 1], uv[:, 0]))


def pixel_to_normalized(uv, width=616, height=808):
    import numpy as np

    return 2 * (np.asarray(uv) + .5) / np.array([width, height]) - 1


def normalized_to_pixel(uv, width=616, height=808):
    import numpy as np

    return (np.asarray(uv) + 1) * np.array([width, height]) / 2 - .5


def score_projection(world, matched_uv, confidence, pose, T_BC, K):
    import numpy as np

    if len(world) < 32:
        return dict(projection_score=None, agreement_8px=None, negative_median_px=None,
                    points=len(world), predicted_visible=0)
    transform = np.linalg.inv(np.asarray(pose).reshape(4, 4) @ T_BC)
    camera = world @ transform[:3, :3].T + transform[:3, 3]
    uv = camera @ K.T
    uv = uv[:, :2] / np.maximum(camera[:, 2:3], 1e-8)
    visible = ((camera[:, 2] > .5) & (camera[:, 2] < 120)
               & (uv[:, 0] >= 0) & (uv[:, 0] < 808)
               & (uv[:, 1] >= 0) & (uv[:, 1] < 616))
    residual = np.linalg.norm(raw_to_rotated(uv) - matched_uv, axis=1)
    residual[~visible] = 1024
    weights = np.asarray(confidence, float)
    return dict(projection_score=float(1 - np.average(np.minimum(residual / 32, 1), weights=weights)),
                agreement_8px=float(np.average(residual <= 8, weights=weights)),
                negative_median_px=-float(np.median(residual)),
                points=len(world), predicted_visible=int(visible.sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scratch-root', type=Path, required=True)
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--limit', type=int, choices=(2, 22), default=22)
    parser.add_argument('--validation-reference-mode', choices=('frozen', 'view-compatible'), default='frozen')
    args = parser.parse_args()
    parent = Path('/data2/zhangcy').resolve(strict=True)
    target = args.scratch_root
    if (not target.is_absolute() or target.parent.resolve(strict=True) != parent
            or not target.name.startswith('e2e-roma-probe-') or target.exists()
            or target.is_symlink() or parent.stat().st_uid != os.getuid()):
        raise ValueError('new owned isolated scratch directory required')
    target.mkdir(mode=0o700)
    (target / 'results').mkdir(mode=0o700)
    (target / 'results' / 'pairs').mkdir(mode=0o700)
    rows = [json.loads(line) for line in FIXED.read_text().splitlines()][:args.limit]
    if len(rows) != args.limit:
        raise ValueError('fixed sample mismatch')
    keys_file = target / 'results' / 'keys.json'
    keys_file.write_text(json.dumps([row['key'] for row in rows]))
    config = E.parser().parse_args(['--keys-file', str(keys_file), '--gpu', args.gpu,
                                   '--gpu-fraction', '.30', '--threads', '1'])
    config.shard_index, config.shard_count = 0, 1
    specs = {spec['key']: spec for spec in E.select_inputs(config)}
    E.PROJECT = target
    E.configure(config)
    guard = E.ResourceGuard(config)
    denied = []

    def no_truth(event, values):
        if event == 'open' and values and isinstance(values[0], (str, bytes, os.PathLike)):
            path = os.fsdecode(values[0])
            if 'evaluator-only-truth' in path or path.endswith('protocol.npz'):
                denied.append(path)
                raise PermissionError('labels prohibited during fixed-pose evidence generation')

    sys.addaudithook(no_truth)
    try:
        before = subprocess.check_output(['nvidia-smi', '-i', args.gpu,
                 '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if before:
            raise RuntimeError('selected GPU already has compute processes')
        free = int(subprocess.check_output(['nvidia-smi', '-i', args.gpu,
                   '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
        if free < 20 * 1024:
            raise MemoryError('insufficient initial GPU headroom')
        import numpy as np
        import torch
        from torch.nn import functional as F
        from PIL import Image
        from collections import OrderedDict

        torch.set_num_threads(1)
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(.30)
        started = time.monotonic()
        sys.path[:0] = [str(ROMA / 'vendor/deps'), str(ROMA / 'vendor/compat-RoMa')]
        compatibility = import_file('roma_probe_torch_compat', ROMA / 'torch_compat.py')
        compatibility.install()
        common = import_file('roma_probe_camera_common', COMMON)
        geometry = import_file('roma_probe_geometry', ROMA / 'geometry.py')
        from romatch import roma_outdoor

        assets = json.loads((ROMA / 'weights-freeze.json').read_text())
        for name, record in assets.items():
            if E.sha_file(ROMA / 'weights' / name) != record['sha256']:
                raise ValueError('existing RoMa weights changed')
        weights = torch.load(ROMA / 'weights/roma_outdoor.pth', map_location='cpu')
        dino = torch.load(ROMA / 'weights/dinov2_vitl14_pretrain.pth', map_location='cpu')
        model = roma_outdoor(device='cuda', weights=weights, dinov2_weights=dino,
                coarse_res=560, upsample_res=864, amp_dtype=torch.float16,
                symmetric=True, use_custom_corr=False, upsample_preds=True)
        model.eval().requires_grad_(False)
        model.encoder.dinov2_vitl14[0].eval().requires_grad_(False)
        del weights, dino
        library = json.loads(LIBRARY.read_text())
        references = library['references']
        T_BC = np.asarray(library['T_BC'])
        positions = np.asarray([r['body_position_world'] for r in references])
        forwards = np.asarray([np.asarray(r['T_WC'])[:3, 2] for r in references])
        rotated_K = np.array([[0, -1, 615], [1, 0, 0], [0, 0, 1]]) @ common.K
        cache = OrderedDict()

        def check():
            guard.check()
            if torch.cuda.mem_get_info()[0] < 8 * 2**30:
                raise MemoryError('GPU free memory below 8 GiB safety floor')
            if time.monotonic() - started > 1200:
                raise TimeoutError('finite probe time limit')

        def image(path, expected):
            if E.sha_file(path) != expected:
                raise ValueError('image hash mismatch')
            with Image.open(path) as loaded:
                if loaded.size != (808, 616):
                    raise ValueError('unexpected calibrated image size')
                return loaded.convert('RGB').transpose(Image.Transpose.ROTATE_270)

        def reference(index):
            if index in cache:
                value = cache.pop(index)
                cache[index] = value
                return value
            row = references[index]
            date, stamp = row['key'].split('/')
            path = E.DATA / 'NCLT' / date / 'velodyne_sync' / (stamp + '.bin')
            if E.sha_file(path) != row['scan_sha256']:
                raise ValueError('reference scan hash mismatch')
            body, _, _ = E.read_scan(path)
            visible = geometry.visible_returns(body, np.linalg.inv(T_BC), rotated_K, 616, 808)
            uv = visible['uv']
            cells = np.floor(uv / 16).astype(np.int64)
            ids = cells[:, 1] * 64 + cells[:, 0]
            distance = np.linalg.norm(uv - (cells + .5) * 16, axis=1)
            order = np.lexsort((distance, ids))
            first = order[np.r_[True, ids[order][1:] != ids[order][:-1]]] if len(order) else order
            if len(first) > 512:
                first = first[np.linspace(0, len(first) - 1, 512, dtype=np.int64)]
            world_from_body = np.asarray(row['T_WC']) @ np.linalg.inv(T_BC)
            value = dict(image=image(row['image_path'], row['image_sha256']), uv=uv[first],
                world=common.transform(visible['source'][first], world_from_body))
            cache[index] = value
            while len(cache) > 12:
                cache.popitem(last=False)
            return value

        def sample(field, uv):
            grid = torch.as_tensor(pixel_to_normalized(uv), device='cuda', dtype=torch.float32)[None, None]
            return F.grid_sample(field[None].permute(0, 3, 1, 2).float(), grid,
                                 align_corners=False, mode='bilinear')[0, :, 0].T

        def match(ref, query_image):
            check()
            tick = time.monotonic()
            with torch.no_grad():
                warp, certainty = model.match(ref['image'], query_image, device='cuda')
                if warp.ndim != 4 or len(warp) != 1:
                    raise ValueError('unexpected RoMa output contract')
                warp, certainty = warp[0], certainty[0]
                half = warp.shape[1] // 2
                forward = sample(warp[:, :half, 2:], ref['uv']).cpu().numpy()
                cf = sample(certainty[:, :half, None], ref['uv'])[:, 0].cpu().numpy()
                target_uv = normalized_to_pixel(forward)
                backward = sample(warp[:, half:, :2], target_uv).cpu().numpy()
                cb = sample(certainty[:, half:, None], target_uv)[:, 0].cpu().numpy()
                cycle = np.linalg.norm(normalized_to_pixel(backward) - ref['uv'], axis=1)
                valid = (np.isfinite(target_uv).all(1) & (cf >= .5) & (cb >= .5) & (cycle <= 2)
                         & (target_uv[:, 0] >= 0) & (target_uv[:, 0] < 616)
                         & (target_uv[:, 1] >= 0) & (target_uv[:, 1] < 808))
                world, uv, confidence = ref['world'][valid], target_uv[valid], np.minimum(cf, cb)[valid]
                torch.cuda.synchronize()
                elapsed = (time.monotonic() - tick) * 1000
                del warp, certainty
            return world, uv, confidence, elapsed

        warm = reference(0)
        sw, su, sc, warm_ms = match(warm, warm['image'])
        self_pose = np.asarray(references[0]['T_WC']) @ np.linalg.inv(T_BC)
        self_check = score_projection(sw, su, sc, self_pose, T_BC, common.K)
        if self_check['negative_median_px'] is None or self_check['negative_median_px'] < -3:
            raise ValueError('RoMa self-match projection check failed: ' + str(self_check))
        initialization = time.monotonic() - started
        torch.cuda.reset_peak_memory_stats()
        output_rows, pair_ms = [], []
        with (target / 'results/evidence.jsonl').open('x') as output:
            for index, row in enumerate(rows):
                check()
                tick = time.monotonic()
                spec = specs[row['key']]
                query_image = image(spec['image_path'], spec['image_sha256'])
                parts = []
                pairs = []
                validation_references = row['validation_references']
                if args.validation_reference_mode == 'view-compatible':
                    excluded = set(row['excluded_references'])
                    eligible_union = {}
                    for hypothesis in row['hypotheses']:
                        pose = np.asarray(hypothesis['original']['pose']).reshape(4, 4)
                        forward = (pose @ T_BC)[:3, 2]
                        distances = np.linalg.norm(positions - pose[:3, 3], axis=1)
                        angle_ok = forwards @ forward >= .5
                        eligible = [int(i) for i in np.argsort(distances)
                                    if distances[i] <= 30 and angle_ok[i] and int(i) not in excluded]
                        for rid in eligible[:2]:
                            eligible_union[rid] = min(eligible_union.get(rid, float('inf')), float(distances[rid]))
                    validation_references = sorted(eligible_union, key=lambda i: (eligible_union[i], i))[:12]
                if set(validation_references) & set(row['excluded_references']):
                    raise ValueError('validation references overlap with generation references')
                for rid in validation_references:
                    ref = reference(rid)
                    world, uv, confidence, elapsed = match(ref, query_image)
                    pair_ms.append(elapsed)
                    np.savez_compressed(target / 'results/pairs' / ('%02d-%04d.npz' % (index, rid)),
                        world=world, matched_uv=uv, confidence=confidence)
                    pairs.append(dict(reference=rid, anchors=len(ref['uv']), matches=len(world), milliseconds=elapsed))
                    if len(world):
                        parts.append((world, uv, confidence))
                world = np.concatenate([p[0] for p in parts]) if parts else np.empty((0, 3))
                uv = np.concatenate([p[1] for p in parts]) if parts else np.empty((0, 2))
                confidence = np.concatenate([p[2] for p in parts]) if parts else np.empty(0)
                hypotheses = []
                for h in row['hypotheses']:
                    scoring_started = time.monotonic()
                    evidence = score_projection(world, uv, confidence, h['original']['pose'], T_BC, common.K)
                    evidence['milliseconds'] = (time.monotonic() - scoring_started) * 1000
                    hypotheses.append(dict(original=h['original'], evidence=evidence))
                result = dict(key=row['key'], source_pose=row['source_pose'], hypotheses=hypotheses,
                    validation_references=validation_references, excluded_references=row['excluded_references'],
                    pairs=pairs, total_milliseconds=(time.monotonic() - tick) * 1000)
                output.write(json.dumps(result, allow_nan=False) + '\n')
                output.flush()
                output_rows.append(result)
                print(json.dumps(dict(key=row['key'], points=len(world), pairs=len(pairs),
                                      milliseconds=result['total_milliseconds'])), flush=True)
        summary = dict(frames=len(output_rows), candidates=sum(len(row['hypotheses']) for row in output_rows),
            initialization_seconds=initialization, self_check=self_check, self_match_ms=warm_ms,
            GT_attempts=len(denied), gpu=args.gpu, pair_ms=pair_ms,
            fixed_evidence_sha256=E.sha_file(FIXED), source_sha256=E.sha_file(SOURCE), weights=assets,
            validation_reference_mode=args.validation_reference_mode,
            method='existing frozen RoMa outdoor; held-out reference matching; fixed T; symmetric confidence>=0.5, cycle<=2px; 512 anchors/reference; 32-point evidence floor; no pose fitting',
            peak_torch_bytes=torch.cuda.max_memory_allocated(), elapsed_seconds=time.monotonic() - started)
        (target / 'results/summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps({k:v for k,v in summary.items() if k != 'pair_ms'}), flush=True)
    finally:
        guard.close()


if __name__ == '__main__':
    main()

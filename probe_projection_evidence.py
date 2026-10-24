import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tarfile
import time
from collections import OrderedDict
from pathlib import Path

import e2e_online as E


SOURCE = Path(__file__).resolve()
ARCHIVE = Path('/home/zhangcy/dsh-work/vpr/out/evidence/salad-pnp-jsonl.tar.gz')
LIBRARY = Path('/data2/zhangcy/sp-lightglue-hard1502-20261013-v1/library.json')
COMMON = Path('/home/zhangcy/projects/sp-lightglue-hard1502-20261013-v1/common.py')


def select_rows():
    rows = []
    with tarfile.open(ARCHIVE) as archive:
        for member in archive.getmembers():
            if member.name.startswith('S5_') and member.name.endswith('.jsonl'):
                if member.size > 2**20:
                    raise ValueError('oversized candidate record')
                rows.extend(json.loads(line) for line in archive.extractfile(member))
    if len(rows) != 290 or len({row['key'] for row in rows}) != 290:
        raise ValueError('unexpected candidate pool')
    selected = []
    for date in sorted({row['key'].split('/')[0] for row in rows}):
        group = sorted((row for row in rows if row['key'].startswith(date + '/')), key=lambda row: row['key'])
        count = min(6, len(group))
        selected.extend(group[round(i * (len(group) - 1) / (count - 1))] for i in range(count))
    return selected


def rotated_uv(uv):
    import numpy as np

    return np.column_stack(((616 - .5 - uv[:, 1]) * 322 / 616 - .5,
                            (uv[:, 0] + .5) * 322 / 808 - .5))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scratch-root', type=Path, required=True)
    parser.add_argument('--gpu', default='1')
    args = parser.parse_args()
    parent = Path('/data2/zhangcy').resolve(strict=True)
    target = args.scratch_root
    if (not target.is_absolute() or target.parent.resolve(strict=True) != parent
            or not target.name.startswith('e2e-projection-probe-') or target.exists()
            or target.is_symlink() or parent.stat().st_uid != os.getuid()):
        raise ValueError('new owned isolated scratch directory required')
    target.mkdir(mode=0o700)
    (target / 'results').mkdir(mode=0o700)
    selected = select_rows()
    keys_file = target / 'results' / 'keys.json'
    keys_file.write_text(json.dumps([row['key'] for row in selected]))
    config = E.parser().parse_args(['--keys-file', str(keys_file), '--gpu', args.gpu,
                                   '--gpu-fraction', '.08', '--threads', '1'])
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
                raise PermissionError('evaluation labels prohibited in evidence generation')

    sys.addaudithook(no_truth)
    try:
        before = subprocess.check_output(['nvidia-smi', '-i', args.gpu,
                    '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if before:
            raise RuntimeError('selected GPU already has compute processes')
        import numpy as np
        import torch
        import torch.nn.functional as F
        from scipy.spatial import cKDTree
        from salad_retrieval import build_model, model_image, MEAN, STD, SALAD_WEIGHTS

        torch.set_num_threads(1)
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(.08)
        started = time.monotonic()
        spec = importlib.util.spec_from_file_location('projection_probe_common', COMMON)
        common = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(common)
        library = json.loads(LIBRARY.read_text())
        references = library['references']
        T_BC = np.asarray(library['T_BC'])
        positions = np.asarray([row['body_position_world'] for row in references])
        model = build_model('cuda')
        capture = {}

        def hook(module, inputs, output):
            capture['patch'] = output[:, 1:].transpose(1, 2).reshape(1, 768, 23, 23).detach()

        handle = model.backbone.model.norm.register_forward_hook(hook)
        cache = OrderedDict()
        image_ms = []
        raw_ms = []

        def image_features(path, expected):
            if E.sha_file(path) != expected:
                raise ValueError('image hash mismatch')
            tick = time.monotonic()
            image = torch.from_numpy(model_image(path)).permute(2, 0, 1).float() / 255
            image = ((image - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None]).cuda()[None]
            with torch.inference_mode():
                model(image)
            patch = F.normalize(capture.pop('patch'), dim=1)
            torch.cuda.synchronize()
            image_ms.append((time.monotonic() - tick) * 1000)
            return patch

        def scan_projection(path, expected):
            tick = time.monotonic()
            if E.sha_file(path) != expected:
                raise ValueError('scan hash mismatch')
            body, _, _ = E.read_scan(path)
            camera = common.transform(body, np.linalg.inv(T_BC))
            mask = np.isfinite(camera).all(1) & (camera[:, 2] > .5) & (camera[:, 2] < 80)
            camera = camera[mask]
            uv = camera @ common.K.T
            uv = uv[:, :2] / uv[:, 2:3]
            mask = (uv[:, 0] >= 0) & (uv[:, 0] < 808) & (uv[:, 1] >= 0) & (uv[:, 1] < 616)
            camera, uv = camera[mask], uv[mask]
            pixel = np.floor(uv).astype(np.int64)
            cells = pixel[:, 1] * 808 + pixel[:, 0]
            order = np.lexsort((camera[:, 2], cells))
            first = order[np.r_[True, cells[order][1:] != cells[order][:-1]]]
            camera, uv = camera[first], uv[first]
            raw_ms.append((time.monotonic() - tick) * 1000)
            return camera, uv, cKDTree(uv)

        def sample_features(patch, uv):
            grid = torch.as_tensor(rotated_uv(uv), device='cuda', dtype=torch.float32)
            grid = (2 * (grid + .5) / 322 - 1).reshape(1, 1, -1, 2)
            sampled = F.grid_sample(patch, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            return F.normalize(sampled[0, :, 0].T, dim=1)

        def get_reference(index):
            if index in cache:
                value = cache.pop(index)
                cache[index] = value
                return value
            row = references[index]
            date, stamp = row['key'].split('/')
            scan = E.DATA / 'NCLT' / date / 'velodyne_sync' / (stamp + '.bin')
            camera, uv, tree = scan_projection(scan, row['scan_sha256'])
            model_uv = rotated_uv(uv)
            cells = np.floor((model_uv + .5) / 14).astype(np.int64).clip(0, 22)
            cell_id = cells[:, 1] * 23 + cells[:, 0]
            centre = (cells + .5) * 14 - .5
            distance = np.linalg.norm(model_uv - centre, axis=1)
            order = np.lexsort((distance, cell_id))
            first = order[np.r_[True, cell_id[order][1:] != cell_id[order][:-1]]]
            first = first[distance[first] <= 5]
            patch = image_features(row['image_path'], row['image_sha256'])
            value = dict(world=common.transform(camera[first], np.asarray(row['T_WC'])),
                         features=sample_features(patch, uv[first]),
                         camera=camera, uv=uv, tree=tree, patch=patch)
            cache[index] = value
            while len(cache) > 12:
                cache.popitem(last=False)
            return value

        def evidence(pose, query, indices):
            tick = time.monotonic()
            T_CW = np.linalg.inv(np.asarray(pose).reshape(4, 4) @ T_BC)
            sums = dict(projected=0, observed=0, occluded=0, testable=0, consistent=0, contradictions=0)
            cosines = []
            for index in indices:
                ref = cache[index]
                camera = common.transform(ref['world'], T_CW)
                uv = camera @ common.K.T
                uv = uv[:, :2] / np.maximum(camera[:, 2:3], 1e-8)
                valid = ((camera[:, 2] > .5) & (camera[:, 2] < 80)
                    & (uv[:, 0] >= 0) & (uv[:, 0] < 808) & (uv[:, 1] >= 0) & (uv[:, 1] < 616))
                ids = np.flatnonzero(valid)
                sums['projected'] += len(ids)
                if not len(ids):
                    continue
                distance, nearest = query['tree'].query(uv[ids])
                ids, nearest = ids[distance <= 3.5], nearest[distance <= 3.5]
                sums['observed'] += len(ids)
                observed = query['camera'][nearest, 2]
                tolerance = np.maximum(.5, .03 * observed)
                delta = camera[ids, 2] - observed
                occluded = delta > tolerance
                contradiction = delta < -tolerance
                consistent = ~occluded & ~contradiction
                sums['occluded'] += int(occluded.sum())
                sums['contradictions'] += int(contradiction.sum())
                sums['consistent'] += int(consistent.sum())
                sums['testable'] += int((~occluded).sum())
                kept = ids[consistent]
                if len(kept):
                    with torch.inference_mode():
                        features = sample_features(query['patch'], uv[kept])
                        reference_features = ref['features'][torch.as_tensor(kept, device='cuda')]
                        cosines.extend((features * reference_features).sum(1).cpu().tolist())
            feature = float(np.mean(cosines)) if len(cosines) >= 32 else None
            geometric = sums['consistent'] / sums['testable'] if sums['testable'] >= 32 else None
            composite = geometric * (feature + 1) / 2 if feature is not None and geometric is not None else None
            torch.cuda.synchronize()
            return dict(**sums, feature_cosine=feature, depth_agreement=geometric,
                        composite=composite, milliseconds=(time.monotonic() - tick) * 1000)

        first = references[0]
        reference = get_reference(0)
        check = evidence(np.asarray(first['T_WC']) @ np.linalg.inv(T_BC), reference, [0])
        if check['composite'] is None or check['composite'] < .999:
            raise ValueError('self-projection consistency failed: ' + str(check))
        torch.cuda.reset_peak_memory_stats()
        rows = []
        with (target / 'results' / 'evidence.jsonl').open('x') as output:
            for row in selected:
                guard.check()
                tick = time.monotonic()
                spec = specs[row['key']]
                camera, uv, tree = scan_projection(spec['scan_path'], spec['scan_sha256'])
                query = dict(camera=camera, uv=uv, tree=tree,
                    patch=image_features(spec['image_path'], spec['image_sha256']))
                hypotheses = [h for h in row['hypotheses'] if h.get('score') is not None][:8]
                excluded = set(row['retrieval']['visual'])
                excluded.update(index for h in row['hypotheses'] for index in h['refs'])
                indices = set()
                for hypothesis in hypotheses:
                    pose = E.valid_pose(np.asarray(hypothesis['pose']).reshape(4, 4))
                    distances = np.linalg.norm(positions - pose[:3, 3], axis=1)
                    eligible = [int(i) for i in np.argsort(distances)
                                if int(i) not in excluded and distances[i] <= 30]
                    indices.update(eligible[:2])
                indices = sorted(indices)
                if len(indices) > 12:
                    indices = indices[:12]
                for index in indices:
                    get_reference(index)
                evidence_rows = [dict(original=h, evidence=evidence(h['pose'], query, indices))
                                 for h in hypotheses]
                result = dict(key=row['key'], validation_references=indices,
                    excluded_references=sorted(excluded), source_pose=row['pose'],
                    hypotheses=evidence_rows, total_milliseconds=(time.monotonic() - tick) * 1000)
                output.write(json.dumps(result, allow_nan=False) + '\n')
                output.flush()
                rows.append(result)
                print(json.dumps(dict(key=row['key'], candidates=len(hypotheses),
                    references=len(indices), milliseconds=result['total_milliseconds'])), flush=True)
        handle.remove()
        summary = dict(frames=len(rows), candidate_count=sum(len(r['hypotheses']) for r in rows),
            scope='fixed S5 candidates; uniform time sampling per date without labels; held-out references excluded from all top5/generator references; sparse frozen SALAD patch reprojection with depth',
            self_projection=check, GT_attempts=len(denied), gpu=args.gpu,
            script_sha256=E.sha_file(SOURCE), archive_sha256=E.sha_file(ARCHIVE),
            library_sha256=E.sha_file(LIBRARY), weights_sha256=E.sha_file(SALAD_WEIGHTS),
            elapsed_seconds=time.monotonic() - started,
            frame_ms=[r['total_milliseconds'] for r in rows], image_ms=image_ms,
            scan_projection_ms=raw_ms, peak_torch_bytes=torch.cuda.max_memory_allocated())
        (target / 'results' / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps({k:v for k,v in summary.items() if not isinstance(v, list)}), flush=True)
    finally:
        guard.close()


if __name__ == '__main__':
    main()

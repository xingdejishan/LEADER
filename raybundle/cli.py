import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .io import save_arrays, load_arrays, load_model, candidate_indices
from .mapping import read_records, build_map, sha256
from .resources import ResourceBudget


def train_step(model, optimizer, frame, ray_map, candidates, truth, associations=None):
    model.train()
    optimizer.zero_grad()
    result = model(frame, ray_map, candidates, hard=False)
    translation = (result['pose'][:3, 3] - truth[:3, 3]).square().sum()
    rotation = (result['pose'][:3, :3] - truth[:3, :3]).square().sum()
    loss = translation + 25 * rotation
    if associations is not None and torch.any(associations >= 0):
        loss = loss + F.cross_entropy(result['scores'], associations)
    if not loss.requires_grad:
        loss = loss + sum(p.sum() * 0 for p in model.parameters())
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite training loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
    if not torch.isfinite(norm):
        raise FloatingPointError('nonfinite trainable gradient')
    optimizer.step()
    return float(loss.detach()), float(norm)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('build-map','prepare','train','predict'):
        p = sub.add_parser(command)
        p.add_argument('--output', required=True)
        p.add_argument('--seconds', type=int, default=600)
        if command in ('build-map','prepare'):
            p.add_argument('--data-root', required=True)
            p.add_argument('--manifests', nargs='+', required=True)
            p.add_argument('--start', type=int, default=0)
            p.add_argument('--count', type=int, default=8)
        if command == 'prepare':
            p.add_argument('--checkpoint', required=True)
            p.add_argument('--noise-model', required=True)
            p.add_argument('--split', choices=('train','dev'), required=True)
        if command in ('train','predict'):
            p.add_argument('--map', required=True)
            p.add_argument('--frames', nargs='+', required=True)
            p.add_argument('--model')
            p.add_argument('--noise-model')
        if command == 'train':
            p.add_argument('--labels', required=True)
            p.add_argument('--steps', type=int, required=True)
    args = parser.parse_args()
    budget = ResourceBudget(args.output, args.seconds)
    torch.set_num_threads(1)
    torch.manual_seed(20)
    if args.command == 'build-map':
        records = read_records(args.data_root, args.manifests, mapping=True)[args.start:args.start+args.count]
        data, metadata = build_map(records)
        budget.check()
        save_arrays(args.output, data, metadata)
        print(json.dumps(dict(map=args.output, tracks=len(data['mean']))))
    elif args.command == 'prepare':
        from data.leader_nclt_manifest import ManifestLEADERDataset
        from .leader import load_leader, encode_frame, L0_SHA
        dataset = ManifestLEADERDataset(args.manifests, args.data_root, args.split, strict_frozen=False)
        model, center = load_leader(args.checkpoint)
        noise = json.loads(Path(args.noise_model).read_text())
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=False)
        labels = {}
        for index in range(args.start, min(len(dataset), args.start+args.count)):
            budget.check()
            sample, record = dataset[index], dataset.records[index]
            frame = encode_frame(sample, record, model, center, noise)
            filename = str(index).zfill(5) + '.npz'
            save_arrays(out / filename, frame, dict(schema='leader-ray-query-v1', frame_key=record.frame_key,
                        image_sha256=record.image_sha256, scan_sha256=record.scan_sha256,
                        checkpoint_sha256=L0_SHA, noise_model_sha256=sha256(args.noise_model)))
            if args.split == 'train':
                labels[record.frame_key] = dict(pose=np.asarray(sample['T_WB']).tolist())
        if labels:
            (out/'training_labels.json').write_text(json.dumps(labels, indent=2))
        print(json.dumps(dict(prepared=min(args.count,len(dataset)-args.start), directory=str(out))))
    else:
        ray_map, map_meta = load_arrays(args.map)
        model = load_model(args.model, map_meta['sha256'])
        noise = json.loads(Path(args.noise_model).read_text()) if args.noise_model else None
        def read_frame(path):
            frame, metadata = load_arrays(path, online=True)
            if noise is not None:
                frame['independent_sigma'] = torch.tensor(noise['independent_sigma'], dtype=torch.float64)
                frame['shared_covariance'] = torch.tensor(noise['shared_covariance'], dtype=torch.float64)
            return frame, metadata
        out = Path(args.output)
        if out.exists():
            raise FileExistsError(out)
        out.mkdir(parents=True)
        if args.command == 'train':
            if args.steps <= 0:
                raise ValueError('positive training step count required')
            labels = json.loads(Path(args.labels).read_text())
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            history = []
            for step in range(args.steps):
                budget.check()
                path = args.frames[step % len(args.frames)]
                frame, metadata = read_frame(path)
                if metadata['frame_key'].split('/')[0] not in ('2012-01-22','2012-02-02','2012-02-18','2012-05-11'):
                    raise ValueError('training requires training-date frames')
                ids = candidate_indices(model, frame, ray_map, metadata, map_meta)
                label = labels[metadata['frame_key']]
                truth = torch.tensor(label['pose'], dtype=torch.float64)
                association = None
                if 'landmark_ids' in label:
                    association = torch.full((len(ids),), -100, dtype=torch.long)
                    for row, track in enumerate(label['landmark_ids']):
                        hit = torch.nonzero(ids[row] == track, as_tuple=False)
                        if track >= 0 and len(hit):
                            association[row] = hit[0,0]
                        elif track == -1:
                            association[row] = ids.shape[1]
                loss, gradient = train_step(model, optimizer, frame, ray_map, ids, truth, association)
                history.append(dict(step=step+1, loss=loss, gradient=gradient))
                print(json.dumps(history[-1]), flush=True)
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), steps=args.steps,
                            map_sha256=map_meta['sha256'], frame_sha256=[sha256(p) for p in args.frames],
                            labels_sha256=sha256(args.labels), noise_model_sha256=sha256(args.noise_model) if args.noise_model else None), out/'last.pt')
            (out/'history.json').write_text(json.dumps(history, indent=2))
        else:
            model.eval()
            records = []
            for index, path in enumerate(args.frames):
                budget.check()
                try:
                    frame, metadata = read_frame(path)
                    ids = candidate_indices(model, frame, ray_map, metadata, map_meta)
                    with torch.no_grad():
                        result = model(frame, ray_map, ids, hard=True)
                    arrays = {k:result[k] for k in ('pose','probabilities','candidates','active_ids','points','point_covariance','selected')}
                    arrays.update(origins=frame['origins'], directions=frame['directions'])
                    save_arrays(out/(str(index).zfill(5)+'.npz'), arrays, dict(frame_key=metadata['frame_key'], map_sha256=map_meta['sha256'],
                                frame_sha256=metadata['sha256'], model_sha256=sha256(args.model) if args.model else None,
                                noise_model_sha256=sha256(args.noise_model) if args.noise_model else None))
                    records.append(dict(frame_key=metadata['frame_key'], status='success'))
                except Exception as error:
                    records.append(dict(input=str(path), status='failed', error=str(error)))
            (out/'manifest.json').write_text(json.dumps(dict(total=len(args.frames), records=records), indent=2))
            print(json.dumps(dict(total=len(records), successful=sum(r['status']=='success' for r in records))))


if __name__ == '__main__':
    main()

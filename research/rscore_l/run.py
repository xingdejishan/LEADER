import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor'))
sys.path.insert(0, str(ROOT.parent))
WORKSPACE = ROOT.parents[2]
sys.path.insert(0, str(WORKSPACE / 'rscore-assets/hloc'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'features', 'geometry', 'multiframe', 'multiframe-report', 'topk', 'topk-report', 'diagnose-geometry', 'overlap', 'node2vec', 'train', 'retrieval', 'export', 'evaluate', 'reliability', 'benchmark', 'freeze', 'check', 'smoke', 'status', 'all'])
    parser.add_argument('--root', type=Path, default=Path('/home/zhang/rscore-l-local'))
    parser.add_argument('--bundle', type=Path, default=WORKSPACE / 'glace-local')
    parser.add_argument('--variant', default='depth')
    parser.add_argument('--graph', choices=['pose', 'lidar'], default='pose')
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--exclude-session', default='')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    args = parser.parse_args()
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    import torch
    torch.set_num_threads(4)
    from rscore_l.prepare import prepare_data, prepare_features, prepare_geometry, save_json
    data = args.root / 'data'
    output = args.root / 'outputs'
    args.root.mkdir(parents=True, exist_ok=True)
    if args.stage == 'topk-report':
        from rscore_l.topk_report import report_topk
        report_topk(args.root)
    elif args.stage == 'topk':
        from rscore_l.topk import run_topk
        run_topk(args.root, args.bundle, args.split)
    elif args.stage == 'multiframe-report':
        from rscore_l.multiframe import compare_multiframe
        compare_multiframe(args.root)
    elif args.stage == 'multiframe':
        from rscore_l.multiframe import prepare_multiframe, audit_multiframe
        prepare_multiframe(data)
        audit_multiframe(data)
    elif args.stage == 'freeze':
        from rscore_l.audit import freeze_inputs
        freeze_inputs(args.root)
    elif args.stage == 'diagnose-geometry':
        from rscore_l.diagnose_geometry import diagnose
        diagnose(data)
    elif args.stage == 'smoke':
        import shutil
        from rscore_l.train import make_config
        from scrstudio.scripts.train import main as official_train
        source = output / 'nclt-local/node2vec-pose/fixed-2089/scrstudio_models/head.pt'
        shutil.copy2(source, data / 'train/pose_n2c_smoke.pt')
        config = make_config(data, args.root / 'smoke', 'geometry', iterations=2)
        config.pipeline.datamanager.encoding = 'pose_n2c_smoke.pt'
        config.pipeline.datamanager.batch_size = 1024
        config.gradient_accumulation_steps = 1
        official_train(config)
    elif args.stage == 'status':
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        for path in output.glob('nclt-local/*/fixed-2089'):
            events = EventAccumulator(str(path), size_guidance={'scalars': 2}).Reload()
            values = {tag: [{'step': v.step, 'value': v.value} for v in events.Scalars(tag)[-1:]] for tag in events.Tags()['scalars'] if tag in ('Train Loss', 'Train Rays / Sec', 'GPU Memory (MB)', 'Train Metrics Dict/loss')}
            print(json.dumps(dict(method=path.parent.name, values=values)))
    elif args.stage == 'check':
        import unittest
        from rscore_l.test_contracts import Contracts
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
        if not result.wasSuccessful():
            raise RuntimeError('Contract tests failed')
    elif args.stage == 'prepare':
        prepare_data(args.bundle, data)
        save_json(args.root / 'protocol.json', dict(seed=2089, splits=dict(train=907, val=303, test=148),
            iterations=args.iterations, effective_batch=8192, microbatch=4096, accumulation=2,
            node2vec_iterations=5000, pca_dim=128, global_dim=256, head_width=768, samples_per_image=1024,
            node2vec_batches=dict(pose=[32, 8], lidar=[64, 4]),
            training_buffer=907*1024, official_scrstudio_revision='a2b40bd19f63803be2e527b829ddb71b02b0a473',
            official_glace_revision='e704a8c718f25dea026a3d5b70776363da7dd665',
            training='Official Trainer, AdamW/OneCycle, original two-stage head; user-authorized 10000-step local budget',
            deviations=['One GPU; effective batch 8192 instead of 327680', '928768 cached samples instead of 128000000 total',
                'Rotation augmentation disabled for noncentral intrinsics and mask alignment', 'PCA exact covariance eigendecomposition instead of cuML'],
            test_status='Previously used development replay; not a blind benchmark', torch=torch.__version__))
    elif args.stage == 'features':
        prepare_features(data)
    elif args.stage == 'geometry':
        prepare_geometry(data)
    elif args.stage == 'overlap':
        from scrstudio.scripts.overlap_score import ComputeOverlap
        if not (data / 'train/pose_overlap.npz').exists():
            ComputeOverlap(data=data / 'train', max_depth=50).main()
    elif args.stage == 'node2vec':
        from rscore_l.train import train_graph
        train_graph(data, output, args.graph)
    elif args.stage == 'train':
        from rscore_l.train import train
        train(data, output, args.variant, args.iterations, args.exclude_session)
    elif args.stage == 'retrieval':
        from scrstudio.scripts.retrieval_feat import ComputeNetVLAD
        complete = (data / args.split / 'netvlad_feats.npy').exists()
        complete &= args.split != 'train' or (data / 'train/netvlad_feats_pq.pkl').exists()
        if not complete:
            ComputeNetVLAD(data=data / args.split, pq=args.split == 'train', num_workers=0).main()
    elif args.stage == 'export':
        from rscore_l.evaluate import export
        export(args.root, args.variant, args.split)
    elif args.stage == 'evaluate':
        from rscore_l.evaluate import evaluate
        evaluate(args.root, args.bundle, args.variant, args.split)
    elif args.stage == 'reliability':
        from rscore_l.reliability import train_reliability
        train_reliability(args.root)
    elif args.stage == 'benchmark':
        from rscore_l.benchmark import benchmark
        benchmark(args.root, args.bundle, args.variant)
    else:
        from rscore_l.pipeline import run_all
        run_all(args)


if __name__ == '__main__':
    main()

import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .evaluate import visual_pose
from .prepare import save_json


def adapt_frame(job):
    source, destination = map(Path, job)
    camera = dict(np.load(source))
    best_count, best_hypothesis = -1, None
    for hypothesis, points in enumerate(camera['xyz']):
        _, count = visual_pose(dict(uv=camera['uv'], xyz=points, K=camera['K'], image_size_hw=camera['image_size_hw']))
        if count > best_count:
            best_count, best_hypothesis = count, hypothesis
    np.savez_compressed(destination, uv=camera['uv'], xyz=camera['xyz'][best_hypothesis], K=camera['K'])
    return dict(frame_id=source.stem, selected_hypothesis=best_hypothesis, pnp_inliers=best_count,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), adapted_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())


def original_fusion(root, bundle):
    destination = root / 'original_fusion'
    if (destination / 'evaluation').exists():
        raise FileExistsError('Original backend evaluation already exists: ' + str(destination / 'evaluation'))
    coordinates = destination / 'coordinates'
    coordinates.mkdir(parents=True, exist_ok=True)
    rows = json.loads((root / 'data/manifest.json').read_text())['test']
    solver = bundle / 'code/research/glace_fusion'
    source_hashes = {name: hashlib.sha256((solver / name).read_bytes()).hexdigest()
        for name in ('cached_joint_eval.py', 'joint_solver.py', 'pose_boundary.py', 'real_candidate_eval.py')}
    save_json(destination / 'adapter_protocol.json', dict(frames=len(rows), source_sha256=source_hashes,
        hypothesis_selection='Maximum official PnP inliers over 10 whole hypotheses, ties resolved by original order; all 5000 points retained',
        reliability='Not injected: the original cached_joint_eval uses uniform camera weights',
        backend='Unmodified original cached_joint_eval / JointSolverConfig(camera_scale_px=10.0) / joint_refine',
        baseline='Exactly the same per-frame cached v1_two_stage used for initialization and rejection fallback',
        ground_truth='Not used by adapter or solver; original runner reads pool GT for both error measurements',
        scope='Rerun original fusion backend on frozen frontend outputs, not raw-sensor end-to-end runtime'))
    jobs = [(root / 'exports/reliable/test' / (row['frame_id']+'.npz'), coordinates / (row['frame_id']+'.npz')) for row in rows]
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as executor:
        records = list(executor.map(adapt_frame, jobs))
    save_json(destination / 'adapter_records.json', records)
    evaluation = destination / 'evaluation'
    if evaluation.exists():
        raise FileExistsError('Original backend evaluation already exists: ' + str(evaluation))
    env = dict(os.environ, PYTHONPATH=str(bundle / 'code'), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='2')
    with (destination / 'evaluation.log').open('w') as log:
        subprocess.run([sys.executable, '-m', 'research.glace_fusion.cached_joint_eval', '--coordinates', str(coordinates),
            '--out', str(evaluation), '--bundle', str(bundle)], cwd=bundle / 'code', env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    summarize_original(destination)


def summarize_original(destination):
    evaluation = destination / 'evaluation'
    results = [json.loads(line) for line in (evaluation / 'records.jsonl').read_text().splitlines()]
    before = np.array([r['baseline_error'] for r in results])
    after = np.array([r['error'] for r in results])
    assert len(results) == 148 and len(set(r['image'] for r in results)) == 148
    delta = after-before
    delta[np.abs(delta) <= 1e-9] = 0
    samples = np.random.default_rng(2089).integers(0, len(results), size=(10000, len(results)))
    ci = np.percentile(delta[samples].mean(1), [2.5, 97.5], axis=0)
    paired = dict(frames=148, comparison_tolerance_m_and_deg=1e-9, baseline_MPE_MOE=before.mean(0).tolist(), multimodal_MPE_MOE=after.mean(0).tolist(),
        delta_multimodal_minus_baseline=delta.mean(0).tolist(), paired_bootstrap_95ci=ci.tolist(),
        baseline_median=np.median(before, axis=0).tolist(), multimodal_median=np.median(after, axis=0).tolist(),
        baseline_p95=np.percentile(before, 95, axis=0).tolist(), multimodal_p95=np.percentile(after, 95, axis=0).tolist(),
        improved=(delta < -1e-9).sum(0).tolist(), degraded=(delta > 1e-9).sum(0).tolist(), unchanged=(np.abs(delta) <= 1e-9).sum(0).tolist(),
        accepted=sum(r['accepted'] for r in results))
    save_json(destination / 'paired.json', paired)

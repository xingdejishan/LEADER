import argparse
import importlib.util
import json
import os
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
from scipy.spatial import cKDTree

import e2e_online as E
from visual_accelerated import AcceleratedBranch


def stats(values):
    a = np.asarray(values, dtype=np.float64) * 1000
    if not len(a):
        return dict(n=0)
    return dict(n=len(a), mean=float(a.mean()), p50=float(np.percentile(a, 50)),
                p95=float(np.percentile(a, 95)), maximum=float(a.max()),
                over_100ms=int(np.sum(a > 100)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for name in args.inputs for line in Path(name).read_text().splitlines()]
    if not 1 <= len(rows) <= 192 or len({r['key'] for r in rows}) != len(rows):
        parser.error('expected 1..192 unique finite benchmark frames')
    config = E.parser().parse_args([])
    E.configure(config)
    guard = E.ResourceGuard(config)
    try:
        sys.path.insert(0, str(E.PROJECT / 'stages/fast_search'))
        import fast_lib as F
        import pipeline as P

        path = '/home/zhangcy/projects/sp-lightglue-hard1502-20261013-v1/common.py'
        spec = importlib.util.spec_from_file_location('selection_reference_common', path)
        common = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(common)
        library = common.load(common.STORE / 'library.json')
        refs = json.loads((E.PROJECT / 'inputs/refs.json').read_text())
        cfg = P.Config(workers=1)
        from light_selection import SparseSelector
        from gpu_selection import GpuSelector
        selector = SparseSelector(refs, library['references'], points=1000, map_points_per_ref=15000, k=8.4)
        import torch
        torch.cuda.set_per_process_memory_fraction(0.15)
        gpu = GpuSelector(refs, library['references'], points=1000, map_points_per_ref=15000, k=8.4)
        torch = gpu.torch
        torch.cuda.reset_peak_memory_stats()
        memory_start = torch.cuda.memory_allocated()
        output = []
        for row in rows:
            guard.check()
            date, stamp = row['key'].split('/')
            raw = E.read_scan(E.DATA / 'NCLT' / date / 'velodyne_sync' / (stamp + '.bin'))[0]
            scan = E.prepare_scan(raw, cfg, F)
            results = {}
            for name, method in ([('cpu', selector), ('gpu', gpu)] if len(output)%2 == 0
                                 else [('gpu', gpu), ('cpu', selector)]):
                started = time.monotonic()
                pose, diag, visual = method.select(dict(pose=row['T_geometry']), row['proposal'], scan)
                elapsed = time.monotonic() - started
                results[name] = dict(T=pose.tolist(), arbitration=diag, seconds=elapsed)
            cpu, accelerated = results['cpu'], results['gpu']
            ca, ga = cpu['arbitration'], accelerated['arbitration']
            difference = max([abs(ca['costs'][key]-ga['costs'][key]) for key in ca['costs']] or [0])
            output.append(dict(key=row['key'], **results, max_score_difference=difference,
                               pose_equal=bool(np.allclose(cpu['T'], accelerated['T'],atol=1e-10,rtol=0)),
                               choice_equal=ca.get('chosen_hypothesis') == ga.get('chosen_hypothesis')))
        report = dict(scope='paired CPU/GPU identical shared evidence, no GT and no models',
                      frames=len(output), cpu_ms=stats([r['cpu']['seconds'] for r in output]),
                      gpu_ms=stats([r['gpu']['seconds'] for r in output]),
                      active_gpu_ms=stats([r['gpu']['seconds'] for r in output
                                           if r['gpu']['arbitration']['decided']]),
                      pose_mismatches=sum(not r['pose_equal'] for r in output),
                      choice_mismatches=sum(not r['choice_equal'] for r in output),
                      max_score_difference=max(r['max_score_difference'] for r in output),
                      peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_incremental_torch_bytes=torch.cuda.max_memory_allocated()-memory_start,
                      ground_truth_read=False, rows=output)
        with E.output_path(args.out).open('x') as stream:
            json.dump(report, stream, indent=2)
        print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
    finally:
        guard.close()


if __name__ == '__main__':
    main()

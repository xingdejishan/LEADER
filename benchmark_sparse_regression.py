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
        selector = SparseSelector(refs, library['references'], points=1000, map_points_per_ref=15000, k=8.4)
        output = []
        for row in rows:
            guard.check()
            date, stamp = row['key'].split('/')
            raw = E.read_scan(E.DATA / 'NCLT' / date / 'velodyne_sync' / (stamp + '.bin'))[0]
            scan = E.prepare_scan(raw, cfg, F)
            started = time.monotonic()
            pose, diag, visual = selector.select(dict(pose=row['T_geometry']), row['proposal'], scan)
            elapsed = time.monotonic() - started
            output.append(dict(key=row['key'], T=pose.tolist(), selection_seconds=elapsed,
                               arbitration=diag, T_geometry=row['T_geometry'], T0=row['T0']))
        report = dict(scope='96 previously inspected frames: regression only, not independent validation',
                      frames=len(output), selection_ms=stats([r['selection_seconds'] for r in output]),
                      active_ms=stats([r['selection_seconds'] for r in output if r['arbitration']['decided']]),
                      chosen=Counter(r['arbitration']['chosen'] for r in output),
                      notes=Counter(r['arbitration'].get('note','evaluated') for r in output),
                      ground_truth_read=False, rows=output)
        with E.output_path(args.out).open('x') as stream:
            json.dump(report, stream, indent=2)
        print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
    finally:
        guard.close()


if __name__ == '__main__':
    main()

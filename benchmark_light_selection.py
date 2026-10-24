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
        branch = AcceleratedBranch.__new__(AcceleratedBranch)
        branch.C, branch.references = common, library['references']
        branch.positions = np.asarray([r['body_position_world'] for r in branch.references])
        branch.verify_cfg = common.PLAN['verification']
        branch.verify_backend = 'kernel'
        branch._verification_maps, branch._verification_refs = OrderedDict(), OrderedDict()
        refs = json.loads((E.PROJECT / 'inputs/refs.json').read_text())
        cfg = P.Config(workers=1)
        maps = F.MapSource(refs, np.asarray([r['world_from_body'] for r in refs])[:, :3, 3],
                           max_refs=12, max_maps=4)
        output = []
        for row in rows:
            guard.check()
            date, stamp = row['key'].split('/')
            raw = E.read_scan(E.DATA / 'NCLT' / date / 'velodyne_sync' / (stamp + '.bin'))[0]
            scan = E.prepare_scan(raw, cfg, F)
            query = branch._query_cloud(raw, 0.4)
            T0 = np.asarray(row['T0'])
            ids = maps.nearby(T0[:3, 3])
            tree = cKDTree(maps.local(T0[:3, 3], ids)) if ids else None
            geometry = dict(pose=row['T_geometry'], tree=tree)
            started = time.monotonic()
            scored = []
            if len(query) >= 500:
                for hypothesis in row['proposal']['hypotheses'][:8]:
                    verification = branch.verify(hypothesis['pose'], query, hypothesis['reference_ids'])
                    scored.append((verification['geometry_score'], hypothesis))
            best = max(scored, key=lambda p: p[0]) if scored else None
            visual = None if best is None else dict(pose=best[1]['pose'])
            pose, arb = E.shared_arbitration(geometry, visual, scan, maps, cfg, config.k)
            elapsed = time.monotonic() - started
            output.append(dict(key=row['key'], full_T=pose.tolist(), light_T=row['T'],
                               full_chosen=arb['chosen'], light_chosen=row['arbitration']['chosen'],
                               full_selection_seconds=elapsed, light_selection_seconds=row['selection_seconds'],
                               light_decided=bool(row['arbitration']['decided']),
                               light_note=row.get('note'), generated=len(row['proposal']['hypotheses']),
                               full_visual_in_top2=bool(best is not None and any(np.allclose(
                                   best[1]['pose'], h['pose'], atol=1e-5, rtol=0)
                                   for h in row['proposal']['top2'])),
                               translation_change_m=float(np.linalg.norm(pose[:3, 3] - np.asarray(row['T'])[:3, 3]))))
        report = dict(scope='offline-replay: identical live-generated T0, geometry and PnP hypotheses',
                      frames=len(rows), full_selection_ms=stats([r['full_selection_seconds'] for r in output]),
                      light_selection_ms=stats([r['selection_seconds'] for r in rows]),
                      active_light_selection_ms=stats([r['selection_seconds'] for r in rows
                                                       if r['arbitration']['decided']]),
                      light_fallback_reasons=Counter(r['light_note'] or 'evaluated' for r in output),
                      full_visual_selected=sum(r['full_chosen'] == 'visual' for r in output),
                      light_visual_selected=sum(r['light_chosen'] == 'visual' for r in output),
                      changed_choice=sum(r['full_chosen'] != r['light_chosen'] for r in output),
                      old_visual_winner_preserved_in_top2=sum(r['full_visual_in_top2'] for r in output),
                      max_translation_change_m=max(r['translation_change_m'] for r in output),
                      mean_live_frame_work_seconds=float(np.mean([sum(r['seconds'].values()) for r in rows])),
                      ground_truth_read=False, model_loaded=False, rows=output)
        target = E.output_path(args.out)
        with target.open('x') as stream:
            json.dump(report, stream, indent=2)
        print(json.dumps({k: v for k, v in report.items() if k != 'rows'}))
    finally:
        guard.close()


if __name__ == '__main__':
    main()

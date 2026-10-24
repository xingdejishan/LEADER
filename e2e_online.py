import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
DATA = Path('/data2/zhangcy/leader-v2-full-trajectory-20261001')
ENVIRONMENT = '/home/zhangcy/.conda/envs/leader'
DEV_DATES = ('2012-02-12', '2012-02-19', '2012-03-31', '2012-05-26')


def valid_pose(value):
    import numpy as np

    pose = np.asarray(value, dtype=np.float64)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-3)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-3)):
        raise ValueError('expected a finite rigid world_from_body 4x4 transform')
    return pose.copy()


def sha_file(path):
    digest = hashlib.sha256()
    with open(str(path), 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def within(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    return path == root or root in path.parents


def output_path(path):
    if not within(PROJECT / 'results', PROJECT):
        raise ValueError('results directory escapes this project')
    path = Path(path)
    path = (path if path.is_absolute() else PROJECT / path).resolve()
    if not within(path, PROJECT / 'results') or path == PROJECT / 'results':
        raise ValueError('output must be inside this project results directory')
    return path


def read_scan(path, max_points=2000000):
    import numpy as np

    path = Path(path)
    if path.suffix == '.bin':
        size = path.stat().st_size
        if size == 0 or size % 8 or size // 8 > max_points:
            raise ValueError('invalid or oversized NCLT packed uint16 XYZI scan')
        dtype = np.dtype([('x', '<u2'), ('y', '<u2'), ('z', '<u2'), ('i', 'u1'), ('l', 'u1')])
        raw = np.fromfile(str(path), dtype=dtype)
        xyz = np.column_stack([raw[a] for a in ('x', 'y', 'z')])
        return (xyz.astype(np.float64) * 0.005 - 100.0,
                xyz.astype(np.float32) * 0.005 - 100.0, raw['i'].copy())
    if path.suffix == '.npy':
        if path.stat().st_size > max_points * 4 * 8 + 4096:
            raise ValueError('oversized XYZI array')
        values = np.load(str(path), mmap_mode='r', allow_pickle=False)
        if (values.ndim != 2 or values.shape[1] != 4 or not 0 < len(values) <= max_points
                or values.dtype.kind not in 'fiu' or not np.isfinite(values).all()):
            raise ValueError('npy input must be a finite Nx4 XYZI array in metres')
        return (np.asarray(values[:, :3], dtype=np.float64).copy(),
                np.asarray(values[:, :3], dtype=np.float32).copy(),
                np.asarray(values[:, 3], dtype=np.float32).copy())
    raise ValueError('supported point clouds: NCLT .bin or Nx4 XYZI .npy')


def prepare_scan(points, cfg, F):
    import numpy as np

    ranges = np.linalg.norm(points, axis=1)
    kept = points[(ranges > cfg.min_scan_range) & (ranges < cfg.max_scan_range)]
    if not len(kept):
        raise ValueError('no query points inside the geometry range')
    return F.voxel(kept, cfg.voxel)


def geometric_branch(T0, scan, record, maps, cfg, gate2, F, P,
                     min_refs=2, min_map=500):
    import numpy as np

    centre = T0[:3, 3]
    triple = maps.nearby(centre)[:cfg.reference_count]
    out = dict(pose=np.array(T0, dtype=np.float64), tree=None, u=None,
               gain=None, ran_fine=False, s=None, d=None, expanded=None, evals=None,
               rotated=False, triple=list(triple), map_points=0, scan_points=len(scan))
    if len(triple) < min_refs:
        out['skipped'] = 'no_reference'
        return out
    points = maps.local(centre, triple)
    out['map_points'] = len(points)
    if len(points) < min_map:
        out['skipped'] = 'thin_map'
        return out
    u = P.along_track_direction(T0)
    coarse = F.coarse_stage(points, scan, T0, u, workers=cfg.workers if hasattr(cfg, 'workers') else 1,
                            n_coarse=cfg.coarse_points, topk=cfg.topk)
    distances = coarse['tree'].query(coarse['base_c'] + centre, k=1, workers=1,
                                      distance_upper_bound=1.0)[0]
    gain = float(np.where(np.isfinite(distances), distances, 1.0).mean() - coarse['surface'].min())
    out.update(tree=coarse['tree'], u=u, gain=gain)
    if gain > gate2:
        pose, diag = P.localize(record, T0, maps, cfg, coarse=coarse, scan=scan)
        out.update(pose=np.asarray(pose, dtype=np.float64), ran_fine=True,
                   s=float(diag['s']), d=float(diag['d']),
                   expanded=diag['expanded'], evals=int(diag['evals']))
    return out


def shared_arbitration(geometry, visual, scan, maps, cfg, k):
    import numpy as np
    import arbitration as ARB

    geo_pose = valid_pose(geometry['pose'])
    diag = dict(chosen='route3', decided=False, visual_available=visual is not None,
                k=float(k), sigma_c=None, thresh=None, costs={}, switched=False)
    if visual is None:
        diag['note'] = 'visual_no_hypothesis'
        return geo_pose, diag
    visual_pose = valid_pose(visual['pose'])
    if not maps.nearby(visual_pose[:3, 3]):
        diag['note'] = 'no_reference_near_visual'
        return geo_pose, diag
    clouds = []
    geometry_tree = geometry.get('tree')
    if geometry_tree is not None:
        clouds.append(geometry_tree.data)
    for pose in ((visual_pose,) if geometry_tree is not None else (geo_pose, visual_pose)):
        ids = maps.nearby(pose[:3, 3])[:cfg.reference_count]
        if ids:
            cloud = maps.local(pose[:3, 3], ids)
            if len(cloud):
                clouds.append(cloud)
    if not clouds or sum(len(c) for c in clouds) < 500:
        diag['note'] = 'insufficient_shared_map'
        return geo_pose, diag
    evidence = ARB.Evidence(ARB.union_maps(*clouds), scan, n_points=4000)
    hypotheses = [ARB.Hypothesis(geo_pose, 'route3', {'incumbent': True}),
                  ARB.Hypothesis(visual_pose, 'visual')]
    chosen, arb = ARB.decide(hypotheses, evidence, k, scan=scan)
    if not all(math.isfinite(c) for c in arb['costs'].values()):
        raise ValueError('nonfinite arbitration cost')
    arb.update(visual_available=True, n_map=evidence.n_map, n_query=evidence.n_query)
    return valid_pose(chosen.pose), arb


def dataset_path(date, relative):
    from pathlib import PurePosixPath

    value = str(relative)
    rel = PurePosixPath(value)
    if (rel.is_absolute() or not rel.parts or any(p in ('', '.', '..') for p in rel.parts)
            or '\\' in value or ':' in value or '\x00' in value):
        raise ValueError('invalid manifest sensor path')
    path = DATA / 'NCLT' / date / rel
    if not any(within(path, root) for root in (DATA / 'NCLT', Path('/data2/dataset/NCLT'))):
        raise ValueError('sensor path escapes declared NCLT roots')
    return str(path.resolve())


def select_inputs(args):
    requested = None
    if args.key:
        requested = [args.key]
    elif args.keys_file:
        requested = json.loads(Path(args.keys_file).read_text())
    elif args.keys_from == 'frames' and not args.scan:
        requested = [r['key'] for r in json.loads((PROJECT / 'inputs/frames.json').read_text())]
    if requested is not None:
        if not isinstance(requested, list) or any(not isinstance(k, str) for k in requested):
            raise ValueError('keys file must be a JSON list of strings')
        if len(requested) != len(set(requested)):
            raise ValueError('duplicate frame keys')
    if args.scan:
        if not args.image or not args.key or not args.calibration:
            raise ValueError('--scan requires --image, --key and --calibration (T_BC JSON)')
        parts = args.key.split('/')
        if len(parts) != 2 or not parts[1].isdigit():
            raise ValueError('--key must be date/timestamp_us')
        payload = json.loads(Path(args.calibration).read_text())
        T_BC = payload['T_BC'] if isinstance(payload, dict) else payload
        return [dict(key=args.key, scan_path=str(Path(args.scan).resolve(strict=True)),
                     image_path=str(Path(args.image).resolve(strict=True)), T_BC=T_BC,
                     canonical_index=args.canonical_index, scan_sha256=None, image_sha256=None)]
    dates = [d for d in args.dates.split(',') if d] if args.dates else list(DEV_DATES)
    if any(d not in DEV_DATES for d in dates) or len(dates) != len(set(dates)):
        raise ValueError('--dates must select unique configured test dates')
    wanted = None if requested is None else set(requested)
    records, canonical = [], 0
    for date in DEV_DATES:
        payload = json.loads((DATA / 'NCLT' / date / 'manifest.json').read_text())
        if payload.get('split') != 'test' or payload.get('date') != date or not payload.get('complete'):
            raise ValueError('invalid public test manifest: ' + date)
        for row in payload['frames']:
            key = '%s/%s' % (date, row['timestamp_us'])
            if any(n in row for n in ('T_WC', 'T_WB', 'groundtruth', 'pose')):
                raise ValueError('test manifest must not contain query ground truth')
            if date in dates and (wanted is None or key in wanted):
                if row.get('camera_id') != 5 or row['scan']['timestamp_us'] != row['timestamp_us']:
                    raise ValueError('expected synchronized Cam5/LiDAR input')
                records.append(dict(key=key, canonical_index=canonical,
                                    scan_path=row['scan']['path'],
                                    image_path=row['image']['path'],
                                    scan_sha256=row['scan']['sha256'], image_sha256=row['image']['sha256'],
                                    T_BC=payload['roi']['T_BC']))
            canonical += 1
    if requested is not None:
        index = {r['key']: r for r in records}
        missing = [k for k in requested if k not in index]
        if missing:
            raise ValueError('unknown or excluded query keys: %r' % missing[:5])
        records = [index[k] for k in requested]
    if len({r['key'] for r in records}) != len(records):
        raise ValueError('duplicate test manifest keys')
    records = records[args.shard_index::args.shard_count][::args.stride]
    records = records[:args.limit] if args.limit else records
    for record in records:
        date = record['key'].split('/')[0]
        for name in ('scan_path', 'image_path'):
            record[name] = dataset_path(date, record[name])
    return records


def input_signature(spec):
    value = dict(spec)
    for name in ('scan_path', 'image_path'):
        stat = Path(spec[name]).stat()
        value[name + '_stat'] = (stat.st_size, stat.st_mtime_ns)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def configure(args):
    if os.path.realpath(sys.prefix) != os.path.realpath(ENVIRONMENT):
        raise RuntimeError('use only ' + ENVIRONMENT + '/bin/python')
    if PROJECT.stat().st_uid != os.getuid():
        raise RuntimeError('project is not owned by the current user')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(args.threads)
    sys.dont_write_bytecode = True
    runtime = PROJECT / '.runtime'
    if not within(runtime, PROJECT) or not within(PROJECT / 'results', PROJECT):
        raise RuntimeError('runtime/results directory escapes project')
    runtime.mkdir(mode=0o700, exist_ok=True)
    os.environ['TMPDIR'] = str(runtime)
    os.environ['XDG_CACHE_HOME'] = str(runtime / 'cache')
    os.environ['MPLCONFIGDIR'] = str(runtime / 'matplotlib')
    os.environ['CUDA_CACHE_PATH'] = str(runtime / 'cuda-cache')
    os.chdir(str(runtime))
    os.nice(max(0, 15 - os.nice(0)))

    def audit(event, values):
        target = None
        if event == 'open' and values and isinstance(values[0], (str, bytes, os.PathLike)):
            flags = values[2] if len(values) > 2 else 0
            mode = values[1] if len(values) > 1 else None
            if (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
                    or isinstance(mode, str) and any(c in mode for c in 'wax+')):
                target = values[0]
        elif event in ('os.remove', 'os.rmdir', 'os.mkdir', 'os.chmod', 'os.truncate'):
            target = values[0]
            if event == 'os.mkdir' and Path(os.fsdecode(target)).is_dir():
                return
        elif event in ('os.rename', 'os.link', 'os.symlink'):
            for value in values[:2]:
                if not any(within(os.fsdecode(value), root) for root in (PROJECT / 'results', runtime)):
                    raise PermissionError('write outside project prohibited')
        elif event == 'socket.connect':
            raise PermissionError('runtime network downloads prohibited; use existing weights')
        if target is not None:
            decoded = os.fsdecode(target)
            if decoded != '/dev/null' and not any(within(decoded, root) for root in (PROJECT / 'results', runtime)):
                raise PermissionError('write outside project results/.runtime prohibited: ' + decoded)

    sys.addaudithook(audit)
    return runtime


class ResourceGuard:
    def __init__(self, args):
        import threading

        self.args = args
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.watch, daemon=True)
        self.check()
        self.thread.start()

    def check(self):
        pages = int(Path('/proc/self/statm').read_text().split()[1])
        if pages * os.sysconf('SC_PAGE_SIZE') > self.args.max_rss_gib * 2 ** 30:
            raise MemoryError('process RSS safety limit reached')
        available = int(next(l for l in Path('/proc/meminfo').read_text().splitlines()
                             if l.startswith('MemAvailable:')).split()[1]) * 1024
        if available < 32 * 2 ** 30:
            raise MemoryError('server available RAM below 32 GiB safety floor')
        stat = os.statvfs(str(PROJECT))
        if stat.f_bavail * stat.f_frsize < self.args.min_free_gib * 2 ** 30:
            raise OSError('actual filesystem free space below safety floor')

    def watch(self):
        while not self.stop.wait(0.25):
            try:
                self.check()
            except Exception as exc:
                os.write(2, ('resource safety stop: %s\n' % exc).encode())
                os._exit(75)

    def close(self):
        self.stop.set()
        self.thread.join()


class ParallelLocalizer:
    def __init__(self, args, runtime_dir):
        import numpy as np
        from online_frontend import OnlineFrontend

        self.args = args
        for path in (PROJECT, PROJECT / 'stages', PROJECT / 'stages/fast_search'):
            sys.path.insert(0, str(path))
        import fast_lib as F
        import pipeline as P
        from visual_ordered import OrderedBranch
        from visual_accelerated import AcceleratedBranch
        from light_selection import ProposalBranch, LightSelector, SparseSelector

        self.F, self.P = F, P
        self.cfg = P.Config(workers=args.threads)
        refs = json.loads(Path(args.refs).read_text())
        if not refs:
            raise ValueError('empty reference library')
        if any(r.get('key', '').split('/')[0] in DEV_DATES for r in refs):
            raise ValueError('reference library must not use test-date query poses')
        positions = np.asarray([valid_pose(r['world_from_body'])[:3, 3] for r in refs])
        self.maps = F.MapSource(refs, positions, radius=self.cfg.crop_radius,
                                max_refs=12, max_maps=4)
        self.frontend = OnlineFrontend(args)
        light = args.selection_mode in ('light', 'sparse', 'gpu_sparse')
        branch_class = ProposalBranch if light else (AcceleratedBranch if args.verify_cache else OrderedBranch)
        branch_options = dict(verify_backend=args.verify_backend) if args.verify_cache and not light else {}
        self.visual = branch_class(device='cuda', arm=args.visual_arm, fast_pnp=bool(args.fast_pnp),
                                   torch_threads=args.threads,
                                   scratch=str(runtime_dir / ('visual-s%d' % args.shard_index)), **branch_options)
        if args.selection_mode == 'gpu_sparse':
            from gpu_selection import GpuSelector
            selector_class = GpuSelector
        else:
            selector_class = SparseSelector if args.selection_mode == 'sparse' else LightSelector
        self.light_selector = (selector_class(refs, self.visual.references, points=args.selection_points,
                                             map_points_per_ref=args.selection_map_points, k=args.k)
                               if light else None)

    def prepare_many(self, specs):
        prepared = []
        for spec in specs:
            started = time.monotonic()
            for name in ('scan', 'image'):
                expected = spec.get(name + '_sha256')
                if expected and sha_file(spec[name + '_path']) != expected:
                    raise ValueError(name + ' input SHA-256 mismatch')
            spec = dict(spec, T_BC=valid_pose(spec['T_BC']))
            points, points32, intensity = read_scan(spec['scan_path'])
            scan = prepare_scan(points, self.cfg, self.F)
            loaded = self.frontend.prepare(spec, points32, intensity)
            prepared.append((spec, points, scan, loaded, started))
        return prepared

    def predict_many(self, prepared):
        started = time.monotonic()
        predictions = self.frontend.predict_many([(p[0], p[3]) for p in prepared])
        rows = [self.predict(p[0], prepared=p, prediction=prediction)
                for p, prediction in zip(prepared, predictions)]
        elapsed = time.monotonic() - started
        finished = time.monotonic()
        for row in rows:
            row['batch_compute_seconds'] = elapsed
            row['batch_size'] = len(rows)
            row['batch_started_at'] = started
            row['batch_finished_at'] = finished
        return rows

    def predict(self, spec, prepared=None, prediction=None):
        import numpy as np

        if prepared is None:
            started = time.monotonic()
            for name in ('scan', 'image'):
                expected = spec.get(name + '_sha256')
                if expected and sha_file(spec[name + '_path']) != expected:
                    raise ValueError(name + ' input SHA-256 mismatch')
            spec = dict(spec, T_BC=valid_pose(spec['T_BC']))
            points, points32, intensity = read_scan(spec['scan_path'])
            scan = prepare_scan(points, self.cfg, self.F)
            T0, seconds, info = self.frontend.predict(spec, points32, intensity)
        else:
            spec, points, scan, loaded, started = prepared
            T0, seconds, info = prediction
        T0 = valid_pose(T0)
        tick = time.monotonic()
        geometry = geometric_branch(T0, scan, spec, self.maps, self.cfg,
                                    self.args.gate2, self.F, self.P)
        seconds['geometry'] = time.monotonic() - tick
        tick = time.monotonic()
        with self.frontend.torch.no_grad():
            if getattr(self.args, 'selection_mode', 'full') in ('light', 'sparse', 'gpu_sparse'):
                proposal = self.visual.propose(spec['key'], spec['image_path'])
                visual = None
            else:
                proposal = None
                visual = self.visual.hypothesize(spec['key'], spec['image_path'],
                                                budget=self.args.visual_budget,
                                                scan=points, scan_path=spec['scan_path'])
        seconds['visual'] = time.monotonic() - tick
        tick = time.monotonic()
        if proposal is not None:
            final, arb, visual = self.light_selector.select(geometry, proposal, scan)
            prefilter_seconds = proposal.get('prefilter_seconds', 0.0)
        else:
            final, arb = shared_arbitration(geometry, visual, scan, self.maps, self.cfg, self.args.k)
            prefilter_seconds = (((visual or {}).get('detail') or {}).get('seconds') or {}).get('verify', 0.0)
        seconds['arbitration'] = time.monotonic() - tick
        if self.frontend.runtime.denied:
            raise RuntimeError('test ground truth accessed')
        degraded = geometry.get('skipped') or arb.get('note')
        return dict(key=spec['key'], T=final.tolist(), convention='world_from_body',
                    status='degraded' if degraded else 'success', note=degraded,
                    T0=T0.tolist(), T_geometry=valid_pose(geometry['pose']).tolist(),
                    T_visual=None if visual is None else valid_pose(visual['pose']).tolist(),
                    experts='parallel', visual_source='live', upstream='live',
                    sam_source=self.args.sam_source, arbitration=arb, visual_attempted=True,
                    execution=dict(frontend_workers=getattr(self.args, 'frontend_workers', 1),
                                   fast_pnp=bool(getattr(self.args, 'fast_pnp', 0)),
                                   verify_cache=bool(getattr(self.args, 'verify_cache', 0)),
                                   verify_backend=getattr(self.args, 'verify_backend', 'tree'),
                                   selection_mode=getattr(self.args, 'selection_mode', 'full')),
                    geometry=dict(skipped=geometry.get('skipped'), ran_fine=geometry['ran_fine'],
                                  rotated=False, map_points=geometry['map_points'],
                                  scan_points=len(scan), gain=geometry['gain']),
                    visual=None if visual is None else dict(score=None if visual.get('score') is None else float(visual['score']),
                            verified=visual.get('verified'), candidates=int(visual['n_candidates']),
                            detail=visual.get('detail')),
                    proposal=proposal,
                    selection_seconds=float(prefilter_seconds + seconds['arbitration']),
                    seconds=seconds, total_seconds=time.monotonic() - started)


def read_done(stream):
    done, end = {}, 0
    stream.seek(0)
    while True:
        line = stream.readline()
        if not line:
            break
        if not line.endswith('\n'):
            stream.seek(end)
            stream.truncate()
            break
        row = json.loads(line)
        if row.get('status') in ('success', 'degraded'):
            valid_pose(row['T'])
            done[row['key']] = row['input_signature']
        end = stream.tell()
    stream.seek(0, os.SEEK_END)
    return done


def parser():
    from online_frontend import SAM_CHECKPOINT

    ap = argparse.ArgumentParser(description='live LEADER/V7 + parallel geometry/visual experts -> T_WB')
    group = ap.add_mutually_exclusive_group()
    group.add_argument('--key')
    group.add_argument('--keys-file')
    ap.add_argument('--scan')
    ap.add_argument('--image')
    ap.add_argument('--calibration', help='JSON 4x4 T_BC or {"T_BC": [...]} for explicit sensor paths')
    ap.add_argument('--canonical-index', type=int, default=0)
    ap.add_argument('--keys-from', choices=('records', 'frames'), default='records')
    ap.add_argument('--dates', default='')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--out', default='results/online_parallel.jsonl')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--check-inputs', action='store_true', help='validate selected sensor paths without models or CUDA')
    ap.add_argument('--refs', default=str(PROJECT / 'inputs/refs.json'))
    ap.add_argument('--gpu', default='3')
    ap.add_argument('--gpu-fraction', type=float, default=0.15)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--max-rss-gib', type=float, default=12)
    ap.add_argument('--min-free-gib', type=float, default=1)
    ap.add_argument('--sam-checkpoint', default=SAM_CHECKPOINT)
    ap.add_argument('--sam-source', choices=('cached', 'live'), default='cached')
    ap.add_argument('--sam-cache-root', default=str(DATA / 'sam'))
    ap.add_argument('--sam-device', choices=('cuda',), default='cuda')
    ap.add_argument('--visual-arm', default='frozen_noprune')
    ap.add_argument('--visual-budget', type=int, default=8)
    ap.add_argument('--frontend-workers', type=int, choices=(1, 4), default=4)
    ap.add_argument('--fast-pnp', type=int, choices=(0, 1), default=1)
    ap.add_argument('--verify-cache', type=int, choices=(0, 1), default=1)
    ap.add_argument('--verify-backend', choices=('tree', 'kernel'), default='kernel')
    ap.add_argument('--selection-mode', choices=('full', 'light', 'sparse', 'gpu_sparse'), default='full')
    ap.add_argument('--selection-points', type=int, choices=(500, 1000, 2000), default=1000)
    ap.add_argument('--selection-map-points', type=int, choices=(6000, 10000, 15000), default=15000)
    ap.add_argument('--gate2', type=float, default=-1)
    ap.add_argument('--k', type=float, default=8.4)
    return ap


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    try:
        args.shard_index, args.shard_count = map(int, args.shard.split('/'))
        if not 0 <= args.shard_index < args.shard_count:
            raise ValueError('invalid shard')
        if (args.limit < 0 or args.stride < 1 or not 1 <= args.threads <= 4
                or not 0 < args.gpu_fraction <= 0.15 or not 0 < args.max_rss_gib <= 12
                or not math.isfinite(args.min_free_gib) or args.min_free_gib < 1
                or args.visual_budget < 1 or args.canonical_index < 0
                or not math.isfinite(args.k) or args.k < 0 or not math.isfinite(args.gate2)
                or not args.gpu.isdigit()):
            raise ValueError('invalid arguments or resource limit exceeds safety policy')
        for name in ('scan', 'image', 'calibration', 'keys_file', 'refs', 'sam_checkpoint', 'sam_cache_root'):
            value = getattr(args, name)
            if name == 'sam_checkpoint' and args.sam_source != 'live':
                continue
            if name == 'sam_cache_root' and args.sam_source != 'cached':
                continue
            if value:
                setattr(args, name, str(Path(value).resolve(strict=True)))
        runtime_dir = configure(args)
        selected = select_inputs(args)
        if not selected:
            raise ValueError('no selected frames')
        if args.check_inputs:
            missing = []
            for spec in selected:
                valid_pose(spec['T_BC'])
                for name in ('scan_path', 'image_path'):
                    if not Path(spec[name]).is_file():
                        missing.append(dict(key=spec['key'], path=spec[name]))
            print(json.dumps(dict(frames=len(selected), missing_count=len(missing),
                                  missing=missing[:20], models_loaded=False, upstream='live',
                                  experts='parallel', dates=sorted({r['key'].split('/')[0] for r in selected}))))
            return 1 if missing else 0
        path = output_path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta_path = path.with_suffix(path.suffix + '.meta.json')
        config = {k: v for k, v in vars(args).items() if k not in ('resume', 'check_inputs')}
        config['code_sha256'] = {name: sha_file(PROJECT / name) for name in
                                ('e2e_online.py', 'online_frontend.py', 'arbitration.py',
                                 'visual_branch.py', 'visual_ordered.py', 'visual_accelerated.py', 'stages/fast_search/fast_lib.py',
                                 'bounded_kernel.py', 'kernels/libgrid.so',
                                 'light_selection.py', 'gpu_selection.py',
                                 'stages/fast_search/pipeline.py', 'stages/fast_search/adaptive.py')}
        config['refs_sha256'] = sha_file(args.refs)
        if args.sam_source == 'live':
            config['sam_sha256'] = sha_file(args.sam_checkpoint)
        else:
            config['sam_cache_manifests'] = {date: sha_file(Path(args.sam_cache_root) / date / 'cache_manifest.json')
                                           for date in sorted({r['key'].split('/')[0] for r in selected})}
        if args.resume:
            if json.loads(meta_path.read_text()) != config:
                raise ValueError('resume configuration differs; use a new output path')
        elif path.exists() or meta_path.exists():
            raise ValueError('output exists; use --resume or a new --out')
        guard = ResourceGuard(args)
        localizer = None
        try:
            import fcntl
            with path.open('r+' if args.resume else 'x+', encoding='utf-8') as sink:
                fcntl.flock(sink.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                if not args.resume:
                    with meta_path.open('x', encoding='utf-8') as meta:
                        json.dump(config, meta, sort_keys=True, allow_nan=False)
                done = read_done(sink)
                localizer = None
                failures = 0
                todo = []
                for spec in selected:
                    signature = input_signature(spec)
                    if spec['key'] in done:
                        if done[spec['key']] != signature:
                            raise ValueError('completed input changed: ' + spec['key'])
                    else:
                        todo.append((spec, signature))
                if todo:
                    from concurrent.futures import ThreadPoolExecutor

                    localizer = ParallelLocalizer(args, runtime_dir)
                    width = args.frontend_workers
                    batches = [todo[i:i + width] for i in range(0, len(todo), width)]
                    with ThreadPoolExecutor(max_workers=1) as prefetch:
                        pending = prefetch.submit(localizer.prepare_many, [r[0] for r in batches[0]])
                        for index, batch in enumerate(batches):
                            guard.check()
                            try:
                                prepared = pending.result()
                                if index + 1 < len(batches):
                                    pending = prefetch.submit(localizer.prepare_many,
                                                              [r[0] for r in batches[index + 1]])
                                rows = localizer.predict_many(prepared)
                            except Exception as exc:
                                for spec, signature in batch:
                                    row = dict(key=spec['key'], status='error', T=None,
                                               input_signature=signature,
                                               error='%s: %s' % (type(exc).__name__, exc))
                                    sink.write(json.dumps(row, allow_nan=False) + '\n')
                                sink.flush()
                                raise
                            for row, (spec, signature) in zip(rows, batch):
                                row['input_signature'] = signature
                                sink.write(json.dumps(row, allow_nan=False) + '\n')
                                sink.flush()
                                print(json.dumps(dict(key=row['key'], status=row['status'], T=row['T'],
                                                      chosen=row['arbitration']['chosen'])), flush=True)
                return 1 if failures else 0
        finally:
            try:
                if localizer is not None:
                    localizer.frontend.close()
            finally:
                guard.close()
    except (ValueError, RuntimeError) as exc:
        ap.error(str(exc))


if __name__ == '__main__':
    sys.exit(main())

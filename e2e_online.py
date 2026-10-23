#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""e2e_online.py -- single-file ONLINE end-to-end driver, three-stage software pipeline.

Per frame, one pass, everything in memory, no pose table and no frame cache:

    raw sensors (velodyne .bin | cam5 .jpg | SAM cache)
      T1  decode .......... one .bin read+decode (shared), cam5 jpg, SAM lookup, scan prep
      T2  GPU ............. LEADER forward -> refine -> T0
      T3  geometric ....... along-track search (V7 rotation preserved) -> final pose -> jsonl/npz
      (f) arbitration ..... only on gated frames (--ratio 1 plus an ambiguity ratio below
                            the threshold): a live visual hypothesis (visual_branch.py:
                            image -> retrieval -> SuperPoint/LightGlue -> pose -> laser
                            score) replaces the geometric pose only if its truncated
                            scan-to-map cost is lower by --delta; default off (--visual 0)

T1/T2/T3 run as three threads with bounded queues (depth 2-4).  Frame i+1 is
decoded while frame i is on the GPU, and frame i's geometric stage runs while
frame i+1 is on the GPU, so neither the GPU nor the CPUs sit idle waiting for
each other.  The GPU stage stays on the main thread; T1 and T3 are workers.
`--queue 0` runs the three stages inline in one thread (the old serial control).

`inputs/initial.npz` never feeds a stage in `--mode online`; it is opened after
the loop to score the result, and keys missing from it are skipped (the full
104,732-frame set mostly is not in that file).

Modes
  --mode online       the loop above (needs GPU)
  --mode integration  stages (d)+(e) only, T0 from inputs/initial.npz, compared
                      bit-exactly with the translation-only search pipeline (no GPU)

Two script-local shims, in memory for the life of this process only:
  1. data.leader_nclt_manifest._safe_child() is replaced by a resolver that keeps
     every lexical rule and the containment rule, but evaluates containment
     against the two declared real dataset roots after resolving symlinks.  Since
     2026-10-15 <date>/velodyne_sync is a symlink into /data2/dataset/NCLT while
     cam5 and the SAM store live in the leader tree, and no single data_root holds
     both, the literal frozen check rejects every scan path.  Containment is not
     dropped: a resolved path outside those two roots still raises.
  2. get_velo() and ManifestLEADERDataset._verify_file() read from a one-frame
     memo, so the .bin is read and decoded exactly once per frame; the digest
     compared against the frozen manifest is the sha256 of those exact bytes.
"""
import argparse
import hashlib
import json
import math
import os
import queue
import resource
import sys
import threading
import time
import traceback

PROJECT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
SCRATCH = '/home/zhangcy/.dsh_scratch'
EVAL = '/home/zhangcy/projects/leader-v5-eval-throughput-20261003'
V7FULL = '/home/zhangcy/projects/v7-full-20261006'
V7PLAN = '/data2/zhangcy/v7-full-20261006/plan.json'
CKPT = '/data2/zhangcy/v7-scratch-continue-20261009-v1/checkpoints/full-epoch7.pt'
CKPT_SHA = '4df0c1c98292edec43acde531bd57d0dfbea9db2c0c4530bba9c5aff78a939d8'
DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001'
REAL_DATASET_ROOTS = (os.path.join(DATA, 'NCLT'), '/data2/dataset/NCLT')

FRAMES_JSON = os.path.join(PROJECT, 'inputs', 'frames.json')
REFS_JSON = os.path.join(PROJECT, 'inputs', 'refs.json')
INITIAL_NPZ = os.path.join(PROJECT, 'inputs', 'initial.npz')
STAGES = ('decode', 'leader', 'refine', 'geometric')


# --------------------------------------------------------------------------- #
def set_env(gpu='2'):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    os.chdir(SCRATCH)


def sha_file(path):
    digest = hashlib.sha256()
    with open(str(path), 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def jnum(value):
    value = float(value)
    return value if math.isfinite(value) else None


def median(values):
    values = sorted(values)
    n = len(values)
    if not n:
        return float('nan')
    return values[n // 2] if n % 2 else 0.5 * (values[n // 2 - 1] + values[n // 2])


def rss_mib():
    with open('/proc/self/statm') as stream:
        return int(stream.read().split()[1]) * (os.sysconf('SC_PAGE_SIZE') / 2. ** 20)


def peak_rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.


# --------------------------------------------------------------------------- #
# script-local shims over the frozen loader                                     #
# --------------------------------------------------------------------------- #
def install_loader_shims():
    """Containment-preserving path resolver + one-read decode memo (in memory)."""
    from pathlib import Path, PurePosixPath
    import data.leader_nclt_manifest as lnm

    allowed = []
    for root in REAL_DATASET_ROOTS:
        try:
            allowed.append(Path(root).resolve(strict=True))
        except OSError:
            pass

    def safe_child(root, date, relative_path):
        value = str(relative_path)
        relative = PurePosixPath(value)
        if ('\\' in value or ':' in value or '\x00' in value or relative.is_absolute()
                or not relative.parts or any(p in ('', '.', '..') for p in relative.parts)):
            raise ValueError('manifest paths must be clean relative paths')
        base = Path(root).resolve(strict=True)
        date_root = (base / date).resolve(strict=True)
        if date_root != base and base not in date_root.parents:
            raise ValueError('date path escapes the dataset root')
        target = date_root.joinpath(*relative.parts).resolve(strict=True)
        if not any(target == r or r in target.parents for r in [base] + allowed):
            raise ValueError('manifest path escapes the dataset roots')
        return target

    memo = {'velo': {}, 'sha256': {}}
    frozen_get_velo = lnm.get_velo
    frozen_verify = lnm.ManifestLEADERDataset._verify_file

    def get_velo(path):
        entry = memo['velo'].get(os.path.realpath(str(path)))
        return frozen_get_velo(path) if entry is None else entry

    def verify_file(self, path, expected_sha):
        key = os.path.realpath(str(path))
        digest = memo['sha256'].get(key)
        if digest is None:
            return frozen_verify(self, path, expected_sha)
        if digest != expected_sha:
            raise ValueError('NCLT input SHA-256 mismatch: {}'.format(path))
        stat = Path(path).stat()
        self._validated_files[str(path)] = (stat.st_size, stat.st_mtime_ns)

    lnm._safe_child = safe_child
    lnm.get_velo = get_velo
    lnm.ManifestLEADERDataset._verify_file = verify_file
    return memo


# --------------------------------------------------------------------------- #
# stage (a): decode the raw inputs, once                                        #
# --------------------------------------------------------------------------- #
def decode_raw(scan_path, memo, F):
    """One read + one decode of the .bin, shared by T1 and T3.

    Returns the float64 cloud that fast_lib.decode() would have produced, and
    publishes the same bytes to the loader memo as the float32 cloud that
    get_velo() returns, together with their digest.
    """
    import numpy as np
    with open(scan_path, 'rb') as stream:
        blob = stream.read()
    size = F.SCAN_DTYPE.itemsize
    raw = np.frombuffer(blob, dtype=F.SCAN_DTYPE, count=len(blob) // size)

    # float64 metres -- bit-identical to fast_lib.decode(np.fromfile(path))
    points = np.empty((len(raw), 3), np.float64)
    for col, name in enumerate(('x', 'y', 'z')):
        np.multiply(raw[name], 0.005, out=points[:, col])
        points[:, col] -= 100.0

    # float32 XYI -- bit-identical to data2xyzi()/get_velo() over the same bytes
    xyz = np.hstack([raw[axis].reshape([-1, 1]) for axis in ('x', 'y', 'z')])
    key = os.path.realpath(scan_path)
    memo['velo'][key] = (xyz.astype(np.float32) * 0.005 - 100.0, raw['i'])
    memo['sha256'][key] = hashlib.sha256(blob).hexdigest()
    del blob, raw, xyz
    return points


def prepare_scan(points, cfg, F):
    """pipeline.prepare_scan()'s post-decode half (range annulus + voxel)."""
    import numpy as np
    radius = np.linalg.norm(points, axis=1)
    kept = points[(radius > cfg.min_scan_range) & (radius < cfg.max_scan_range)]
    return F.voxel(kept, cfg.voxel)


# --------------------------------------------------------------------------- #
# stages (d)+(e): geometric branch (V7 rotation preserved)                   #
# --------------------------------------------------------------------------- #
def cost(tree, points, np, trunc=1.0):
    distance = tree.query(points, k=1, workers=1, distance_upper_bound=trunc)[0]
    return float(np.where(np.isfinite(distance), distance, trunc).mean())


def geometric_branch(T0, scan, record, maps, cfg, gate2, F, P, rot_only=None,
                     min_refs=2, min_map=500, gate_on=False):
    """three_route_unified.one() without rotation-only refinement.

    Coverage guard: the reference library was selected around the 1502 hard frames,
    so a frame far from all of them can have an empty or very thin local map. Scoring
    against that yields a correction that is worse than no correction, and an empty
    triple would crash np.concatenate. In that case keep the front-end pose and flag
    the frame, so the trigger rate can be counted instead of guessed.
    """
    import numpy as np

    centre = T0[:3, 3]
    triple = maps.nearby(centre)[:cfg.reference_count]
    if len(triple) < min_refs:
        return dict(pose=np.array(T0, dtype=np.float64), tree=None, u=None,
                    gain=float('nan'), ran_fine=False, s=None, d=None, expanded=None,
                    evals=None, rotated=False, triple=list(triple), map_points=0,
                    scan_points=int(len(scan)), skipped='no_reference')
    map_points = maps.local(centre, triple)
    if len(map_points) < min_map:
        return dict(pose=np.array(T0, dtype=np.float64), tree=None, u=None,
                    gain=float('nan'), ran_fine=False, s=None, d=None, expanded=None,
                    evals=None, rotated=False, triple=list(triple),
                    map_points=int(len(map_points)), scan_points=int(len(scan)),
                    skipped='thin_map')
    u = P.along_track_direction(T0)

    coarse = F.coarse_stage(map_points, scan, T0, u, workers=1,
                            n_coarse=cfg.coarse_points, topk=cfg.topk)
    gain = cost(coarse['tree'], coarse['base_c'] + centre, np) - float(coarse['surface'].min())

    pose = np.array(T0, dtype=np.float64)
    ran_fine, s_value, d_value, expanded, evals = False, None, None, None, None
    if gain > gate2:
        pose, diag = P.localize(record, T0, maps, cfg, coarse=coarse, scan=scan)
        pose = np.asarray(pose, dtype=np.float64)
        ran_fine = True
        s_value = float(diag['s'])
        d_value = float(diag['d'])
        expanded = diag['expanded']
        evals = int(diag['evals'])

    # rotation-only refinement: translation frozen, step accepted on the held-out half
    # only when the improvement exceeds eps. Applied only to frames that actually ran
    # the search (that is the configuration measured best on the full set: the search
    # fixes the translation, which is what makes the rotation fit informative).
    rotated = False
    rotated_pose = pose
    if rot_only is not None and ran_fine:
        q = scan[np.linalg.norm(scan, axis=1) <= 50]
        cells = np.floor(q / 0.3).astype(np.int64)
        _, ids = np.unique(cells, axis=0, return_index=True)
        rotated_pose = rot_only(coarse['tree'], map_points, q[np.sort(ids)], pose)
        rotated = bool(np.abs(rotated_pose[:3, :3] - pose[:3, :3]).max() > 1e-12)

    # confidence gate: keep the rotation, decide the translation
    confidence = None
    if gate_on and ran_fine:
        try:
            import confidence as CONF
            lat = np.cross(u, [0.0, 0.0, 1.0])
            lat /= np.linalg.norm(lat)
            clouds = []
            for j in triple:
                c = maps._ref_cloud(j)
                clouds.append(c[np.linalg.norm(c - centre, axis=1) <= maps.radius])
            accept_c, score_c, _ = CONF.decide(
                coarse['tree'], map_points, scan, T0, rotated_pose, u, lat,
                surface=coarse['surface'], s=s_value, d=d_value, refs=clouds)
            confidence = dict(score=float(score_c), accept=bool(accept_c))
            if not accept_c:
                rotated_pose = np.array(rotated_pose, dtype=np.float64)
                rotated_pose[:3, 3] = T0[:3, 3]
        except Exception as exc:                       # never fail a frame on the gate
            confidence = dict(error=repr(exc))

    return dict(pose=rotated_pose, tree=coarse['tree'], u=u, gain=float(gain),
                ran_fine=bool(ran_fine), s=s_value, d=d_value, expanded=expanded,
                evals=evals, rotated=rotated, triple=list(triple),
                map_points=int(len(map_points)), scan_points=int(len(scan)),
                confidence=confidence)


def load_geometry():
    """The project's own verified geometric code (nothing reimplemented)."""
    if PROJECT not in sys.path:
        sys.path.insert(0, PROJECT)
    from three_route_unified import cost_at, cost, rot_only
    sys.path.insert(0, os.path.join(PROJECT, 'stages', 'fast_search'))
    sys.path.insert(0, os.path.join(PROJECT, 'stages'))
    import fast_lib as F
    import pipeline as P
    import ambiguity as A
    return F, P, A, cost_at, cost, rot_only


# --------------------------------------------------------------------------- #
# mode: integration check (no GPU)                                              #
# --------------------------------------------------------------------------- #
def mode_integration(args):
    import numpy as np
    F, P, A, cost_at, cost, rot_only = load_geometry()
    cfg = P.Config(solver='adaptive', coarse_points=1000, fine_points=1500,
                   workers=args.threads)
    refs = json.load(open(REFS_JSON))
    for row in refs:
        row['scan_path'] = os.path.realpath(row['scan_path'])
    maps = F.MapSource(refs, np.asarray([r['world_from_body'] for r in refs])[:, :3, 3],
                       radius=cfg.crop_radius)

    frames = {row['key']: row for row in json.load(open(FRAMES_JSON))}
    with np.load(INITIAL_NPZ, allow_pickle=False) as archive:
        keys = [str(k) for k in archive['keys']]
        t0s = np.asarray(archive['poses'], np.float64)

    worst = 0.0
    scan_worst = 0.0
    rows = []
    for i in range(args.check_keys):
        key = keys[i]
        scan_path = os.path.realpath(frames[key]['scan_path'])
        record = {'key': key, 'scan_path': scan_path}
        T0 = np.array(t0s[i], np.float64)

        reference_scan = P.prepare_scan(record, cfg)          # frozen decoder
        shared = prepare_scan(decode_raw(scan_path, {'velo': {}, 'sha256': {}}, F),
                              cfg, F)                          # driver's one-read decode
        if shared.shape != reference_scan.shape:
            scan_worst = float('inf')
        else:
            scan_worst = max(scan_worst, float(np.abs(shared - reference_scan).max()))

        out = geometric_branch(T0, reference_scan, record, maps, cfg, args.gate2,
                               F, P, gate_on=bool(getattr(args, 'confidence_gate', 0)))
        expected = (P.localize(record, T0, maps, cfg, scan=reference_scan)[0]
                    if out['ran_fine'] else T0)
        diff = float(np.abs(out['pose'] - expected).max())
        if not np.array_equal(out['pose'][:3, :3], T0[:3, :3]):
            raise AssertionError('geometric branch changed the V7 rotation')
        worst = max(worst, diff)
        rows.append(dict(key=key, max_abs_diff=diff, bit_exact=bool(diff == 0.0),
                         gain=round(out['gain'], 9), s=out['s'], d=out['d'],
                         expanded=out['expanded'], rotated=out['rotated'],
                         shift_m=round(float(np.linalg.norm(out['pose'][:3, 3] - T0[:3, 3])), 6),
                         shared_scan_max_abs_diff=scan_worst))
        print('[integration] ' + json.dumps(rows[-1], sort_keys=True), flush=True)

    ok = (worst == 0.0) and (scan_worst == 0.0) and len(rows) == args.check_keys
    print('[integration-summary] ' + json.dumps(
        dict(frames=len(rows), max_abs_diff=worst, shared_scan_max_abs_diff=scan_worst,
             bit_exact=bool(worst == 0.0), passed=bool(ok),
             against='translation-only pipeline.localize',
             stages='(d)+(e) only, T0 from inputs/initial.npz'), sort_keys=True))
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
def boot_online():
    """LEADER forward + appearance-field refine, bootstrapped as smoke_stage12.py."""
    import torch

    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.15)

    sys.path.insert(0, EVAL)
    from runtime import Runtime
    sys.path.insert(0, V7FULL)
    import v7_runtime as vr
    import cache_runtime_v2

    memo = install_loader_shims()
    c, parent, config, adaptive, Loader = vr.imports()
    denied = c.deny_test(config)

    class _NoCache(object):
        """No InputCache: no directory, no read, no write, ever."""

        def __init__(self, *a, **k):
            self.directory = None

        def load(self, *a, **k):
            raise AssertionError('InputCache disabled: the driver is online')

        def read(self, *a, **k):
            return None

        def write(self, *a, **k):
            raise AssertionError('InputCache disabled: the driver is online')

    cache_runtime_v2.InputCache = _NoCache
    testloader = vr.TestLoader(c, config)
    if len(testloader.records) != 104732:
        raise AssertionError('unexpected record count %d' % len(testloader.records))

    runtime = Runtime.__new__(Runtime)          # no Runtime.__init__ (no epoch models)
    runtime.config = config
    runtime.plan = testloader.plan
    runtime.datasets = testloader.datasets
    runtime.records = testloader.records
    runtime.denied = denied

    assert sha_file(config['base_checkpoint']) == runtime.plan['frozen_LEADER_sha256']
    from tools.magic_nclt_runtime import load_pretrained_model
    runtime.leader, saved = load_pretrained_model(config['base_checkpoint'], False,
                                                 torch.device('cuda:0'))
    runtime.leader.eval().requires_grad_(False)
    runtime.center = torch.as_tensor(saved['center_t'], device='cuda',
                                     dtype=torch.float32).reshape(3)
    del saved

    assert sha_file(CKPT) == CKPT_SHA, 'deployed checkpoint sha256 mismatch'
    state = torch.load(CKPT, map_location='cpu')
    model = vr.model(c, config, state['model'], False)
    with open(V7PLAN) as stream:
        inference_method = json.load(stream)['inference_method']
    return dict(c=c, config=config, runtime=runtime, model=model, memo=memo,
                denied=denied, inference_method=inference_method, torch=torch,
                state=state, vr=vr)


# --------------------------------------------------------------------------- #
# the three pipeline stages                                                     #
# --------------------------------------------------------------------------- #
def stage_decode(key, gid, ctx):
    """T1: one .bin read+decode (shared), cam5 jpg + SAM via the frozen loader."""
    import numpy as np
    started, t0 = time.time(), time.monotonic()
    rec = ctx['runtime'].records[gid]
    dsrec = ctx['runtime'].datasets[rec[0]].records[rec[1]]
    scan_path = os.path.realpath(str(dsrec.scan_path))
    image_path = os.path.realpath(str(dsrec.image_path))
    memo = ctx['memo']
    memo['velo'].clear()
    memo['sha256'].clear()

    points = decode_raw(scan_path, memo, ctx['F'])
    probe = ctx['probe']
    if probe is not None and probe.get('decode') is None:
        reference = ctx['F'].decode(scan_path)
        probe['decode'] = dict(max_abs_diff=float(np.abs(points - reference).max()),
                               same_shape=bool(points.shape == reference.shape),
                               image_resolved=os.path.isfile(image_path))
        del reference
    scan = prepare_scan(points, ctx['cfg'], ctx['F'])
    del points
    loaded = ctx['runtime'].load(rec)
    return dict(key=key, scan_path=scan_path, image_path=image_path, loaded=loaded, scan=scan,
                wall_start=started, decode=time.monotonic() - t0)


def stage_gpu(item, ctx):
    """T2: LEADER forward -> appearance-field refine -> T0.  GPU thread only."""
    torch = ctx['torch']
    runtime = ctx['runtime']
    t0 = time.monotonic()
    initial = runtime.initial(item['loaded'])
    geometry = runtime.geometry(item['loaded'], initial[1].detach().cpu())
    frame = runtime.frame(item['loaded'], *initial, geometry)
    item['loaded'] = None                       # drop the CPU sample as early as possible
    torch.cuda.synchronize()
    t_leader = time.monotonic() - t0

    t1 = time.monotonic()
    observed = ctx['model'].observation_batch([frame['sam']], [frame['rgb']],
                                              [frame['pixels']], [frame['hw']])[0]
    refined, _, _, info = ctx['c'].selfverify.refine(ctx['model'], frame, observed,
                                                     ctx['inference'])
    torch.cuda.synchronize()
    t_refine = time.monotonic() - t1

    T0 = refined.detach().cpu().double().numpy().copy()
    T0[:3, 3] += ctx['center']
    item.update(T0=T0, leader=t_leader, refine=t_refine, info=info)
    del initial, geometry, frame, observed, refined
    return item


def stage_gpu_batch(items, ctx):
    """T2 with N refine replicas: leader per frame, then N-way parallel refine."""
    torch = ctx['torch']
    runtime = ctx['runtime']
    frames = []
    for item in items:
        t0 = time.monotonic()
        initial = runtime.initial(item['loaded'])
        geometry = runtime.geometry(item['loaded'], initial[1].detach().cpu())
        frame = runtime.frame(item['loaded'], *initial, geometry)
        item['loaded'] = None
        torch.cuda.synchronize()
        item['leader'] = time.monotonic() - t0
        frames.append(frame)
        del initial, geometry
    t0 = time.monotonic()
    results = ctx['parallel'].run(frames)
    torch.cuda.synchronize()
    share = (time.monotonic() - t0) / len(items)
    for item, result in zip(items, results):
        refined, _, _, info = result
        T0 = refined.detach().cpu().double().numpy().copy()
        T0[:3, 3] += ctx['center']
        item.update(T0=T0, refine=share, info=info)
    del frames, results
    return items


def stage_geom(item, ctx):
    """T3: geometric branch, with the V7 rotation preserved."""
    t0 = time.monotonic()
    out = geometric_branch(item['T0'], item['scan'],
                           {'key': item['key'], 'scan_path': item['scan_path']},
                           ctx['maps'], ctx['cfg'], ctx['gate2'], ctx['F'], ctx['P'],
                           rot_only=(lambda t, g, q, T: ctx['rot_only'](t, g, q, T, eps=ctx['rot_eps']))
                           if ctx.get('use_rot') else None,
                           gate_on=bool(ctx.get('confidence_gate', False)))
    item['geometric'] = time.monotonic() - t0
    item['out'] = out
    return item


def make_visual_branch(ctx):
    """Load retrieval + matcher + verifier once, on the first gated frame."""
    import torch
    from visual_branch import VisualBranch
    before = torch.cuda.memory_allocated() / 2. ** 20
    branch = VisualBranch(device='cuda', arm=ctx['visual_arm'],
                          scratch=os.path.join(SCRATCH, 'vb'))
    after = torch.cuda.memory_allocated() / 2. ** 20
    ctx['branch'] = branch
    print('[visual] branch ready arm=%s gpu_allocated=%.0f MiB (+%.0f) cap=%.0f MiB'
          % (ctx['visual_arm'], after, after - before,
             0.15 * torch.cuda.get_device_properties(0).total_memory / 2. ** 20), flush=True)
    return branch


def visual_arbitration(item, ctx):
    """Ambiguity gate -> live visual hypothesis -> the same-cost arbitration.

    Same rule as the offline three_route_unified `--mode unified`: the frame's truncated
    scan-to-map cost is measured for both poses on each pose's own local map, and the visual
    pose replaces the geometric one only if it is cheaper by `delta`.
    """
    import numpy as np
    record = dict(ratio=None, gated=False, attempted=False, accepted=False, chosen='route3',
                  score=None, verified=None, n_candidates=None, channel=None, refs=None,
                  c_route3=None, c_visual=None, dcost=None, seconds=0.0, note='')
    ratio = None
    if ctx['ratio_gate']:                      # --ratio 1: the gate diagnostic on its own
        _, ambiguity = ctx['A'].decide(item['out']['tree'], item['scan'], item['T0'],
                                       item['out']['u'])
        ratio = jnum(ambiguity['ratio'])
        record['ratio'] = ratio
    if not ctx['visual']:
        return record
    if ratio is None or not np.isfinite(ratio) or ratio >= ctx['amb_threshold']:
        return record
    record['gated'] = True
    branch = ctx['branch'] or make_visual_branch(ctx)
    started = time.monotonic()
    try:
        hypothesis = branch.hypothesize(item['key'], item['image_path'],
                                        budget=ctx['visual_budget'],
                                        pose_prior=item['T0'])
    except BaseException as exc:                    # a visual failure never voids the frame
        record['seconds'] = time.monotonic() - started
        record['note'] = 'error:%s' % type(exc).__name__
        return record
    record['seconds'] = time.monotonic() - started
    if hypothesis is None:
        record['note'] = 'no_hypothesis'
        return record
    record['attempted'] = True
    record['score'] = float(hypothesis['score'])
    record['verified'] = bool(hypothesis.get('verified'))
    record['n_candidates'] = int(hypothesis['n_candidates'])
    record['channel'] = hypothesis.get('channel')
    record['refs'] = [int(x) for x in (hypothesis.get('reference_ids') or [])]
    query = item['scan'][:min(ctx['arb_points'], len(item['scan']))]
    c3 = float(ctx['cost_at'](item['out']['pose'], query, ctx['maps'], ctx['cfg']))
    cv = float(ctx['cost_at'](hypothesis['pose'], query, ctx['maps'], ctx['cfg']))
    record['c_route3'] = c3
    record['c_visual'] = cv
    record['dcost'] = cv - c3
    if cv < c3 - ctx['delta']:
        record['accepted'] = True
        record['chosen'] = 'visual'
        item['final_pose'] = np.asarray(hypothesis['pose'], dtype=np.float64)
    return record


def make_row(item, ctx):
    out = item['out']
    info = item['info']
    seconds = {'decode': item['decode'], 'leader': item['leader'],
               'refine': item['refine'], 'geometric': item['geometric']}
    visual = item.get('visual') or {}
    final = item.get('final_pose')
    final = out['pose'] if final is None else final
    return dict(key=item['key'], pose=[float(x) for x in final.ravel()],
                route3_pose=[float(x) for x in out['pose'].ravel()],
                t0=[float(x) for x in item['T0'].ravel()],
                seconds=dict((k, round(v, 6)) for k, v in seconds.items()),
                total=round(sum(seconds.values()), 6),
                t_start=round(item['wall_start'], 3), t_end=round(time.time(), 3),
                rss_MiB=round(rss_mib(), 1),
                gain=jnum(out['gain']), ratio=visual.get('ratio'),
                s=None if out['s'] is None else jnum(out['s']),
                d=None if out['d'] is None else jnum(out['d']),
                expanded=out['expanded'], rotated=bool(out['rotated']),
                ran_fine=out['ran_fine'], solver=ctx['cfg'].solver,
                confidence=(out.get('confidence') or {}).get('score'),
                confidence_accept=(out.get('confidence') or {}).get('accept'),
                chosen=visual.get('chosen', 'route3'), visual=bool(visual.get('accepted')),
                visual_gated=bool(visual.get('gated')),
                visual_attempted=bool(visual.get('attempted')),
                visual_score=visual.get('score'), visual_verified=visual.get('verified'),
                visual_candidates=visual.get('n_candidates'),
                visual_channel=visual.get('channel'), visual_refs=visual.get('refs'),
                visual_note=visual.get('note'),
                visual_seconds=round(float(visual.get('seconds') or 0.0), 6),
                c_route3=visual.get('c_route3'), c_visual=visual.get('c_visual'),
                dcost=visual.get('dcost'), delta=ctx['delta'],
                gate2=ctx['gate2'], rounds=int(info['rounds']),
                accepted=int(info['accepted']), status=str(info['status']),
                map_points=out['map_points'], scan_points=out['scan_points'],
                skipped=out.get('skipped'))


# --------------------------------------------------------------------------- #
# mode: online loop                                                             #
# --------------------------------------------------------------------------- #
def resolve_out(prefix):
    if not os.path.isabs(prefix):
        prefix = os.path.join(PROJECT, prefix)
    prefix = os.path.abspath(prefix)
    allowed = (os.path.join(PROJECT, 'results'), SCRATCH)
    if not any(prefix.startswith(root + os.sep) for root in allowed):
        raise SystemExit('refusing to write outside results/ or the scratch dir: ' + prefix)
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    return prefix


def select_pairs(args, runtime):
    """[(key, record index)] in canonical order, already sharded and limited."""
    dates = [d for d in (args.dates or '').split(',') if d]
    if args.keys_file:
        index = {rec[2]: i for i, rec in enumerate(runtime.records)}
        wanted = [str(k) for k in json.load(open(args.keys_file))]
        pairs = [(k, index[k]) for k in wanted if k in index]
        pairs = pairs[args.shard_index::args.shard_count]
        return pairs[:args.limit] if args.limit > 0 else pairs
    if args.keys_from == 'records':
        pairs = [(rec[2], i) for i, rec in enumerate(runtime.records)
                 if not dates or rec[0] in dates]
    else:
        index = {rec[2]: i for i, rec in enumerate(runtime.records)}
        pairs = [(row['key'], index[row['key']]) for row in json.load(open(FRAMES_JSON))
                 if (not dates or row['key'].split('/')[0] in dates)]
    pairs = pairs[args.shard_index::args.shard_count]
    return pairs[:args.limit] if args.limit > 0 else pairs


def read_done(jsonl_path):
    done = {}
    if not os.path.exists(jsonl_path):
        return done
    with open(jsonl_path) as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue                       # torn tail after a kill
            seconds = row.get('seconds') or {}
            if 'key' in row and 'pose' in row and not (set(STAGES) - set(seconds)):
                done[row['key']] = row
    return done


def mode_online(args):
    import numpy as np

    boot = boot_online()
    torch = boot['torch']
    runtime = boot['runtime']
    F, P, A, cost_at, cost, rot_only = load_geometry()

    cfg = P.Config(solver='adaptive', coarse_points=1000, fine_points=1500,
                   workers=args.threads)
    refs = json.load(open(REFS_JSON))
    for row in refs:
        row['scan_path'] = os.path.realpath(row['scan_path'])
    maps = F.MapSource(refs, np.asarray([r['world_from_body'] for r in refs])[:, :3, 3],
                       radius=cfg.crop_radius)

    pairs = select_pairs(args, runtime)
    prefix = resolve_out(args.out_prefix) + ('' if getattr(args, 'rot_only', 1) else '-no-rotation')
    npz_path = '%s-shard%d.npz' % (prefix, args.shard_index)
    jsonl_path = '%s-shard%d.jsonl' % (prefix, args.shard_index)
    done = read_done(jsonl_path)
    todo = [p for p in pairs if p[0] not in done]

    ctx = dict(runtime=runtime, model=boot['model'], c=boot['c'], torch=torch,
               memo=boot['memo'], F=F, P=P, A=A, maps=maps,
               cost_at=cost_at, arb_points=4000, amb_threshold=A.THRESHOLD,
               visual=bool(args.visual), ratio_gate=bool(args.ratio),
               visual_arm=args.visual_arm, visual_budget=args.visual_budget,
               branch=None, rot_only=rot_only, use_rot=bool(getattr(args, 'rot_only', 0)),
   rot_eps=float(getattr(args, 'rot_eps', 0.01)),
               cfg=cfg, gate2=args.gate2, delta=args.delta,
               confidence_gate=bool(getattr(args, 'confidence_gate', 0)),
               center=runtime.center.double().cpu().numpy(),
               inference=boot['inference_method'],
               probe=dict(decode=None) if args.check_decode else None,
               error=None, parallel=None)
    if args.refine_replicas > 1:
        import infer_batch_engine_v2 as engine
        del boot['model']
        ctx['model'] = None
        torch.cuda.empty_cache()
        ctx['parallel'] = engine.ParallelFrames(
            lambda: boot['vr'].model(boot['c'], boot['config'], boot['state']['model'], False),
            boot['c'].selfverify.refine, boot['inference_method'], args.refine_replicas)
        print('[boot] refine replicas=%d  gpu_allocated=%.0f MiB (cap %.0f)'
              % (args.refine_replicas, torch.cuda.memory_allocated() / 2. ** 20,
                 0.15 * torch.cuda.get_device_properties(0).total_memory / 2. ** 20),
              flush=True)
    print('[boot] refs=%d records=%d threads=%d gpu=%s mem_fraction=0.15 pipeline=%s depth=%d'
          % (len(refs), len(runtime.records), args.threads,
             os.environ['CUDA_VISIBLE_DEVICES'], 'on' if args.queue > 0 else 'off',
             args.queue), flush=True)
    print('[plan] keys-from=%s shard=%d/%d dates=%s selected=%d done=%d todo=%d -> %s'
          % (args.keys_from, args.shard_index, args.shard_count, args.dates or 'all',
             len(pairs), len(done), len(todo), jsonl_path), flush=True)

    def put_block(target, item, raise_on_error=True):
        while True:
            try:
                target.put(item, timeout=0.2)
                return True
            except queue.Full:
                if raise_on_error and ctx['error']:
                    return False

    def write_skip(key, reason, note=''):
        """Record a frame that could not be produced, so the shard moves on."""
        row = dict(key=str(key), pose=[0.0] * 16, t0=[0.0] * 16, seconds={},
                   total=0.0, skipped=reason, note=str(note)[:200])
        sink.write(json.dumps(row) + '\n')
        sink.flush()
        done[row['key']] = row
        finished.append(row)
        print('[skip] %s %s' % (key, reason), flush=True)

    def decode_worker():
        try:
            for key, gid in todo:
                try:
                    item = stage_decode(key, gid, ctx)
                except BaseException as exc:
                    write_skip(key, 'decode:%s' % type(exc).__name__, exc)
                    continue
                if not put_block(q1, item):
                    break
        except BaseException:
            ctx['error'] = traceback.format_exc()
        finally:
            put_block(q1, None, raise_on_error=False)

    def geom_worker():
        while True:
            item = q2.get()
            if item is None:
                break
            try:
                stage_geom(item, ctx)
                item['visual'] = visual_arbitration(item, ctx)
                row = make_row(item, ctx)
                sink.write(json.dumps(row) + '\n')
                sink.flush()
                done[row['key']] = row
                finished.append(row)
                if len(finished) % 20 == 0:
                    print('[frame] %d/%d %s serial=%.3fs window=%.3fs'
                          % (len(finished), len(todo), row['key'], row['total'],
                             row['t_end'] - finished[0]['t_end']), flush=True)
            except BaseException as exc:
                write_skip(item.get('key', '?'), 'geom:%s' % type(exc).__name__, exc)

    finished = []
    q1 = queue.Queue(maxsize=max(1, args.queue))
    q2 = queue.Queue(maxsize=max(1, args.queue))
    threads = []

    with open(jsonl_path, 'a') as sink, torch.no_grad():
        if args.queue <= 0:
            # serial control: the same three stages, inline, no overlap.
            # A frame that raises (empty point set, corrupt scan, CUDA hiccup) is
            # recorded as skipped and the shard keeps going: one bad frame must not
            # cost the remaining tens of thousands.
            for key, gid in todo:
                try:
                    item = stage_decode(key, gid, ctx)
                    if ctx['parallel'] is None:
                        item = stage_gpu(item, ctx)
                    else:
                        stage_gpu_batch([item], ctx)
                    stage_geom(item, ctx)
                    item['visual'] = visual_arbitration(item, ctx)
                    row = make_row(item, ctx)
                except BaseException as exc:
                    row = dict(key=key, pose=[0.0] * 16, t0=[0.0] * 16,
                               seconds={}, total=0.0,
                               skipped='error:%s' % type(exc).__name__,
                               note=str(exc)[:200])
                    print('[skip] %s %s' % (key, type(exc).__name__), flush=True)
                sink.write(json.dumps(row) + '\n')
                sink.flush()
                done[row['key']] = row
                finished.append(row)
                if len(finished) % 20 == 0:
                    print('[frame] %d/%d %s serial=%.3fs'
                          % (len(finished), len(todo), row['key'], row.get('total', 0.0)),
                          flush=True)
        else:
            threads = [threading.Thread(target=decode_worker, name='T1-decode'),
                       threading.Thread(target=geom_worker, name='T3-geometric')]
            for thread in threads:
                thread.daemon = True
                thread.start()
            width = max(1, args.refine_replicas)
            try:
                stop = False
                while not stop:
                    item = q1.get()
                    if item is None:
                        break
                    batch, sentinel = [item], False
                    while len(batch) < width and not ctx['error']:
                        try:
                            nxt = q1.get(timeout=0.05)
                        except queue.Empty:
                            break
                        if nxt is None:
                            sentinel = True
                            break
                        batch.append(nxt)
                    try:
                        if ctx['parallel'] is None:
                            batch = [stage_gpu(one, ctx) for one in batch]
                        else:
                            stage_gpu_batch(batch, ctx)
                    except BaseException as exc:
                        # one bad frame must not stall the whole pipeline: record it
                        # and keep the queues moving
                        for one in batch:
                            write_skip(one.get('key', '?'),
                                       'gpu:%s' % type(exc).__name__, exc)
                        continue
                    for one in batch:
                        if not put_block(q2, one):
                            stop = True
                            break
                    stop = stop or sentinel
            except BaseException:
                ctx['error'] = traceback.format_exc()
            finally:
                put_block(q2, None, raise_on_error=False)
                for thread in threads:
                    thread.join()
    if ctx['parallel'] is not None:
        ctx['parallel'].close()
    if ctx['error']:
        raise RuntimeError('pipeline failed:\n' + ctx['error'])

    # ---- (g) npz for the whole shard (resumed keys come from the jsonl)
    shard_keys = [k for k, _ in pairs if k in done]
    poses = (np.stack([np.asarray(done[k]['pose'], np.float64).reshape(4, 4)
                       for k in shard_keys]) if shard_keys
             else np.zeros((0, 4, 4), np.float64))
    np.savez(npz_path, keys=np.array(shard_keys), poses=poses)

    # ---- scoring against inputs/initial.npz (measurement only, never an input)
    with np.load(INITIAL_NPZ, allow_pickle=False) as archive:
        ref_poses = np.asarray(archive['poses'], np.float64)
        lag = {str(k): i for i, k in enumerate(archive['keys'])}
    t0_error, final_error = [], []
    for key in shard_keys:
        if key not in lag:
            continue                           # keys outside the scored subset
        i = lag[key]
        t0_error.append(float(np.linalg.norm(
            np.asarray(done[key]['t0'], np.float64).reshape(4, 4)[:3, 3] - ref_poses[i][:3, 3])))
        final_error.append(float(np.linalg.norm(
            np.asarray(done[key]['pose'], np.float64).reshape(4, 4)[:3, 3]
            - ref_poses[i][:3, 3])))

    rows = [done[k] for k in shard_keys]
    steady = rows[1:]                          # drop frame 0: pipeline fill + CUDA warm-up
    totals = {name: 0.0 for name in STAGES}
    per_frame, rss_trace = [], []
    for row in (steady or rows):
        if not row.get('seconds'):             # a skipped frame has no stage timings
            continue
        for name in STAGES:
            totals[name] += float(row['seconds'].get(name, 0.0))
        per_frame.append(float(row['total']))
        rss_trace.append(float(row['rss_MiB']))
    count = len([r for r in (steady or rows) if r.get('seconds')])
    gated = [r for r in rows if r.get('visual_gated')]
    attempted = [r for r in gated if r.get('visual_attempted')]
    accepted = [r for r in gated if r.get('visual')]
    visual_seconds = sorted(float(r.get('visual_seconds') or 0.0) for r in gated)
    visual_summary = dict(
        gated=len(gated), attempted=len(attempted), accepted=len(accepted),
        seconds_median=(round(median(visual_seconds), 4) if visual_seconds else None),
        seconds_p90=(round(visual_seconds[min(len(visual_seconds) - 1,
                                              int(0.9 * (len(visual_seconds) - 1)))], 4)
                     if visual_seconds else None),
        decisions=[dict(key=r['key'], ratio=r.get('ratio'), c_route3=r.get('c_route3'),
                        c_visual=r.get('c_visual'), dcost=r.get('dcost'),
                        chosen=r.get('chosen'), score=r.get('visual_score'),
                        seconds=r.get('visual_seconds'), note=r.get('visual_note'))
                   for r in gated])
    if count:
        mean = {name: totals[name] / count for name in STAGES}
        # steady state = wall clock between the first and the last completed frame
        ends = sorted(row['t_end'] for row in (steady or rows))
        span = ends[-1] - ends[0]
        span = span if span > 0 else sum(per_frame)
        period = span / (count - 1) if count > 1 else sum(mean.values())
        util = {name: round(mean[name] / period, 3) for name in STAGES}
        report = dict(
            mode='online', frames=count, resumed=len(pairs) - len(todo),
            keys_from=args.keys_from, shard='%d/%d' % (args.shard_index, args.shard_count),
            dates=args.dates, threads=args.threads, gate2=args.gate2, delta=args.delta,
            pipeline_depth=args.queue, ratio_diagnostic=bool(args.ratio),
            visual=bool(args.visual), visual_budget=args.visual_budget,
            visual_arm=args.visual_arm,
            stage_seconds_mean={k: round(v, 6) for k, v in mean.items()},
            serial_work_mean=round(sum(mean.values()), 6),
            stage_utilization=util,
            steady_span_seconds=round(span, 3),
            steady_ms_per_frame=round(1000.0 * period, 1),
            steady_frames_per_second=round(1.0 / period, 4),
            per_frame_total_median=round(median(per_frame), 6),
            per_frame_total_p90=round(sorted(per_frame)[min(count - 1, int(0.9 * (count - 1)))], 6),
            peak_gpu_allocated_MiB=round(torch.cuda.max_memory_allocated() / 2. ** 20, 1),
            peak_gpu_reserved_MiB=round(torch.cuda.max_memory_reserved() / 2. ** 20, 1),
            rss_MiB=dict(first=round(rss_trace[0], 1), mid=round(rss_trace[count // 2], 1),
                         last=round(rss_trace[-1], 1), peak=round(peak_rss_mib(), 1)),
            scored_against_initial=len(t0_error),
            skipped_not_in_initial=len(rows) - len(t0_error),
            t0_position_error_m=dict(mean=round(sum(t0_error) / len(t0_error), 4),
                                     median=round(median(t0_error), 4))
            if t0_error else None,
            final_position_error_m=dict(mean=round(sum(final_error) / len(final_error), 4),
                                        median=round(median(final_error), 4))
            if final_error else None,
            visual_summary=visual_summary,
            decode_identity=ctx['probe']['decode'] if ctx['probe'] else None,
            gt_online_attempts=len(runtime.denied), npz=npz_path, jsonl=jsonl_path)
    else:
        report = dict(mode='online', frames=0, note='all keys already done',
                      npz=npz_path, jsonl=jsonl_path)
    print('[report] ' + json.dumps(report, sort_keys=True))
    if runtime.denied:
        raise AssertionError('test GT was touched: %r' % (runtime.denied,))
    return 0


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description='online single-file e2e relocalisation driver')
    parser.add_argument('--mode', choices=('online', 'integration'), default='online')
    parser.add_argument('--shard', default='0/1', help='i/n')
    parser.add_argument('--limit', type=int, default=0, help='0 = all selected frames')
    parser.add_argument('--dates', default='2012-02-12,2012-02-19')
    parser.add_argument('--keys-from', choices=('frames', 'records'), default='frames',
                        help='frames: inputs/frames.json (1502); records: all 104732')
    parser.add_argument('--out-prefix', default='results/online')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--queue', type=int, default=3,
                        help='pipeline queue depth; 0 = serial (no overlap)')
    parser.add_argument('--refine-replicas', type=int, default=1,
                        help='>1 drives the refine stage with the project%ss ParallelFrames'
                             ' (N model replicas, N streams); changes T0 by 3-5 cm' % "'")
    parser.add_argument('--gate2', type=float, default=0.01,
                        help='descent depth needed before the search runs; 0.01 is the full-set working point')
    parser.add_argument('--confidence-gate', type=int, default=0,
                        help='1 = decide the translation with the cross-view consistency score '
                             '(rotation is kept either way); off by default until certified')
    parser.add_argument('--delta', type=float, default=0.05)
    parser.add_argument('--visual', type=int, default=0,
                        help='1 = live visual branch on gated frames (needs --ratio 1)')
    parser.add_argument('--visual-budget', type=int, default=8,
                        help='visual hypotheses laser-scored per gated frame')
    parser.add_argument('--visual-arm', default='frozen_noprune',
                        help='LightGlue arm: frozen_noprune (as the frozen table) or v8_noprune')
    parser.add_argument('--keys-file', default='',
                        help='JSON list of frame keys to run, in that order')
    parser.add_argument('--ratio', type=int, default=0,
                        help='1 = also run the ambiguity sweep (diagnostic only; off when'
                             ' the visual branch is disabled)')
    parser.add_argument('--check-decode', type=int, default=1,
                        help='1 = prove the shared decode equals fast_lib.decode() once')
    parser.add_argument('--rot-eps', type=float, default=0.01,
                        help='cost improvement needed to accept a rotation step')
    parser.add_argument('--rot-only', type=int, default=1,
                        help='1 = also run the rotation-only block (the completed full '
                             'run used 1; off by default because it costs orientation)')
    parser.add_argument('--check-keys', type=int, default=5)
    parser.add_argument('--gpu', default='2')
    args = parser.parse_args()
    args.shard_index, args.shard_count = (int(x) for x in args.shard.split('/'))
    if not 0 <= args.shard_index < args.shard_count:
        parser.error('--shard must be i/n with 0 <= i < n')
    if args.visual and not args.ratio:
        parser.error('--visual 1 needs --ratio 1: the visual branch is only queried on frames'
                     ' the ambiguity gate flags')

    set_env(args.gpu)
    return mode_integration(args) if args.mode == 'integration' else mode_online(args)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)








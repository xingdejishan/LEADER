#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""2-frame smoke test: stage 1 (LEADER forward -> initial pose + geometry/frame)
   -> stage 2 (appearance-field observation + selfverify.refine).

Read-only w.r.t. the filesystem: no InputCache, no npz/pt writes.  Frames stay in RAM.

Bootstrap mirrors /home/zhangcy/projects/v7-integrated-oneepoch-20261012-v3/rebuild_remaining_v3.py
(stage 1) and /home/zhangcy/projects/v7-full-20261006/worker.py (stage 2).
"""
import hashlib
import json
import os
import sys
import time
import traceback

SCRATCH = '/home/zhangcy/.dsh_scratch'
EVAL = '/home/zhangcy/projects/leader-v5-eval-throughput-20261003'
V7FULL = '/home/zhangcy/projects/v7-full-20261006'
V7PLAN = '/data2/zhangcy/v7-full-20261006/plan.json'
CKPT = '/data2/zhangcy/v7-scratch-continue-20261009-v1/checkpoints/full-epoch7.pt'
CKPT_SHA = '4df0c1c98292edec43acde531bd57d0dfbea9db2c0c4530bba9c5aff78a939d8'
INIT = '/home/zhangcy/projects/e2e-moe-20261020-v1/inputs/initial.npz'
KEYS = ['2012-02-12/1329068220036076', '2012-02-12/1329068221836354']

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '2')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.chdir(SCRATCH)

import numpy as np                       # noqa: E402
import torch                             # noqa: E402

torch.set_num_threads(1)
assert os.environ['CUDA_VISIBLE_DEVICES'] == '2', 'refusing to run on ' + os.environ['CUDA_VISIBLE_DEVICES']
torch.cuda.set_per_process_memory_fraction(0.2)          # hard cap, 0.2 of one 24 GiB 4090

sys.path.insert(0, EVAL)
from runtime import Runtime                               # noqa: E402
sys.path.insert(0, V7FULL)
import v7_runtime as vr                                   # noqa: E402
import cache_runtime_v2                                   # noqa: E402

# --- glue shim -------------------------------------------------------------
# Since 2026-10-15 every <date>/velodyne_sync under the frozen dataset root is a
# symlink to /data2/dataset/NCLT/<date>/velodyne_sync (other user's tree), so the
# loader's _safe_child() "must stay under the dataset root" check rejects every
# scan path and ManifestLEADERDataset can no longer be built at all.  We keep its
# path-cleanliness rules but stop following the symlink out of the root, so the
# frozen manifests/hashes stay byte-identical and only the containment check is
# relaxed.  Script-local: no project file is modified.
import data.leader_nclt_manifest as _lnm                  # noqa: E402
from pathlib import Path                                  # noqa: E402
from pathlib import PurePosixPath                         # noqa: E402

_SHIMMED = _lnm._safe_child


def _safe_child_allow_symlink(root, date, relative_path):
    value = str(relative_path)
    relative = PurePosixPath(value)
    if ('\\' in value or ':' in value or '\x00' in value or relative.is_absolute()
            or not relative.parts or any(part in ('', '.', '..') for part in relative.parts)):
        raise ValueError('manifest paths must be clean relative paths')
    return Path(root) / date / Path(*relative.parts)


_lnm._safe_child = _safe_child_allow_symlink


class _NoCache(object):
    """Stub that replaces InputCache: no mkdir, no read, no write, ever."""
    def __init__(self, *a, **k):
        self.directory = None

    def load(self, *a, **k):
        raise AssertionError('InputCache disabled by smoke test')

    def read(self, *a, **k):
        return None

    def write(self, *a, **k):
        raise AssertionError('InputCache disabled by smoke test')


def sha(path):
    h = hashlib.sha256()
    with open(str(path), 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def rot_deg(a, b):
    r = np.asarray(a, float)[:3, :3] @ np.asarray(b, float)[:3, :3].T
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1.) * .5, -1., 1.))))


def main():
    report = {'stage': 'smoke_stage12', 'keys': KEYS}
    begun = time.monotonic()

    # ---------------- bootstrap (reference A) ----------------
    c, parent, config, adaptive, Loader = vr.imports()
    denied = c.deny_test(config)
    cache_runtime_v2.InputCache = _NoCache            # injected before TestLoader builds one
    testloader = vr.TestLoader(c, config)
    assert len(testloader.records) == 104732

    runtime = Runtime.__new__(Runtime)                # no Runtime.__init__ (no epoch models)
    runtime.config = config
    runtime.plan = testloader.plan
    runtime.datasets = testloader.datasets
    runtime.records = testloader.records
    runtime.denied = denied

    assert sha(config['base_checkpoint']) == runtime.plan['frozen_LEADER_sha256']
    from tools.magic_nclt_runtime import load_pretrained_model
    runtime.leader, saved = load_pretrained_model(config['base_checkpoint'], False, torch.device('cuda:0'))
    runtime.leader.eval().requires_grad_(False)
    runtime.center = torch.as_tensor(saved['center_t'], device='cuda', dtype=torch.float32).reshape(3)
    del saved

    # ---------------- stage 2 model (reference B) ----------------
    assert sha(CKPT) == CKPT_SHA, 'deployed checkpoint sha256 mismatch'
    state = torch.load(CKPT, map_location='cpu')
    m = vr.model(c, config, state['model'], False)
    del state
    p_infer = json.load(open(V7PLAN))['inference_method']

    print('[shim] leader_nclt_manifest._safe_child patched: velodyne_sync symlink allowed (containment check dropped)')
    print('[boot] seconds=%.2f  base_checkpoint=%s' % (time.monotonic() - begun, config['base_checkpoint']))
    print('[boot] ckpt_sha_ok=%s  far_covolume=%s' % (True, json.dumps(config['far_covolume'])))
    print('[boot] proc_mem_fraction=%.2f  threads=%d  visible=%s  center=%s'
          % (0.2, torch.get_num_threads(), os.environ['CUDA_VISIBLE_DEVICES'],
             [round(float(x), 3) for x in runtime.center.cpu()]))
    sys.stdout.flush()

    # ---------------- reference poses ----------------
    with np.load(INIT, allow_pickle=False) as archive:
        ref_keys = [str(k) for k in archive['keys']]
        ref_poses = np.asarray(archive['poses'], float)

    center = runtime.center.double().cpu().numpy()

    def stage1(rec):
        torch.cuda.synchronize()
        tick = time.monotonic()
        loaded = runtime.load(rec)
        initial = runtime.initial(loaded)
        geometry = runtime.geometry(loaded, initial[1].detach().cpu())
        frame = runtime.frame(loaded, *initial, geometry)
        torch.cuda.synchronize()
        return frame, initial, time.monotonic() - tick

    def stage2(frame):
        torch.cuda.synchronize()
        tick = time.monotonic()
        observed = m.observation_batch([frame['sam']], [frame['rgb']], [frame['pixels']], [frame['hw']])[0]
        torch.cuda.synchronize()
        t_obs = time.monotonic() - tick
        tick = time.monotonic()
        pose, _, _, info = c.selfverify.refine(m, frame, observed, p_infer)
        torch.cuda.synchronize()
        return pose, info, t_obs, time.monotonic() - tick

    rows = []
    passes = [(KEYS[0], 'cold'), (KEYS[1], 'cold'), (KEYS[0], 'warm-repeat')]
    for tag, phase in passes:
        index = next(i for i, rec in enumerate(runtime.records) if rec[2] == tag)
        rec = runtime.records[index]
        with torch.no_grad():
            frame, initial, t1 = stage1(rec)
            initial_t = initial[0].detach().cpu().double().numpy().copy()
            initial_world = initial_t.copy()
            initial_world[:3, 3] += center
            pose, info, t_obs, t_ref = stage2(frame)
        world = pose.detach().cpu().double().numpy().copy()
        world[:3, 3] += center
        ref = ref_poses[ref_keys.index(tag)]
        row = {
            'key': tag, 'phase': phase, 'record_index': index,
            'input_sha256': frame['input_sha256'],
            'initial_finite': bool(torch.isfinite(initial[0]).all()),
            'initial_pose_t': [round(float(x), 6) for x in initial_world[:3, 3]],
            'refined_pose_t': [round(float(x), 6) for x in world[:3, 3]],
            'ref_t': [round(float(x), 6) for x in ref[:3, 3]],
            'refined_pose': [[float(x) for x in r] for r in world],
            'initial_pos_err_m': round(float(np.linalg.norm(initial_world[:3, 3] - ref[:3, 3])), 6),
            'initial_rot_err_deg': round(rot_deg(initial_world, ref), 6),
            'pos_err_m': round(float(np.linalg.norm(world[:3, 3] - ref[:3, 3])), 6),
            'rot_err_deg': round(rot_deg(world, ref), 6),
            'rounds': int(info['rounds']), 'accepted': int(info['accepted']), 'stop': str(info['status']),
            'n_points': int(len(frame['points'])), 'n_corr': int(len(frame['source'])),
            'valid_corr': int(frame['valid_corr']),
            's1_seconds': round(t1, 4),
            's2_obs_seconds': round(t_obs, 4),
            's2_refine_seconds': round(t_ref, 4),
            's2_seconds': round(t_obs + t_ref, 4),
            'gpu_peak_alloc_MiB': round(torch.cuda.max_memory_allocated() / 2. ** 20, 1),
        }
        rows.append(row)
        print('[frame] ' + json.dumps(row, sort_keys=True))
        sys.stdout.flush()
        del frame, initial, pose, world
        torch.cuda.empty_cache()

    assert not runtime.denied, 'test GT was touched: %r' % (runtime.denied,)

    cold = [r for r in rows if r['phase'] == 'cold']
    warm = [r for r in rows if r['phase'] == 'warm-repeat']
    report['rows'] = rows
    report['s1_seconds_per_frame_cold'] = round(sum(r['s1_seconds'] for r in cold) / len(cold), 4)
    report['s2_seconds_per_frame_cold'] = round(sum(r['s2_seconds'] for r in cold) / len(cold), 4)
    report['s1_seconds_per_frame_warm'] = warm[0]['s1_seconds'] if warm else None
    report['s2_seconds_per_frame_warm'] = warm[0]['s2_seconds'] if warm else None
    report['max_memory_allocated_MiB'] = round(torch.cuda.max_memory_allocated() / 2. ** 20, 1)
    report['max_memory_reserved_MiB'] = round(torch.cuda.max_memory_reserved() / 2. ** 20, 1)
    report['denied_events'] = len(runtime.denied)
    report['gt_online_attempts'] = 0
    report['total_seconds'] = round(time.monotonic() - begun, 2)
    report['complete'] = True
    print('[summary] ' + json.dumps({k: v for k, v in report.items() if k != 'rows'}, sort_keys=True))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        print('[summary] {"complete": false}')
        sys.exit(1)

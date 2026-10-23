"""Frozen constants for the single-frame along-track ambiguity gate.

Frozen 2026-10-20. Changing anything here invalidates the acceptance numbers
in README.md and requires re-running replay_check.py on both frozen sets.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'

# ---- frozen decision constant
THRESHOLD = 0.21          # ratio < THRESHOLD  ->  severe, escalate to the visual route

# ---- frozen geometry
S_GRID = __import__('numpy').round(__import__('numpy').arange(-10.0, 10.0001, 0.5), 3)
FAR_MIN = 5.0             # "competing offset" = |s| > FAR_MIN
TRUNC = 1.0               # per-point cost cap, metres
VOXEL = 0.3
N_REF = 3                 # nearest reference scans used for the local map
KEEP_RADIUS = 60.0        # map points kept within this radius of the pose
QCAP = 4000               # query point cap
QSEED = 20261019

# ---- frozen inputs (read-only; nothing here is written to)
SEL = '/data2/zhangcy/v7-raw-geometry-hard1502-20261013-v1/selection.json'
HARD_SCORED = '/data2/zhangcy/v7-raw-geometry-hard1502-20261013-v1/scored.npz'
ORD_KEYS = '/data2/zhangcy/v8-selector-20261016-v1/provider-ordinary.npz'
ORD_BASE = '/data2/zhangcy/v8-selector-20261016-v1/baseline-e7.npz'
NCLT = '/data2/zhangcy/leader-v2-full-trajectory-20261001/NCLT/%s/velodyne_sync/%s.bin'
E2E_PROJECT = '/home/zhangcy/projects/e2e-three-route-20261020-v1'

# ---- acceptance numbers produced by `replay_check.py`
EXPECT = {'hard': dict(flagged=84), 'ordinary': dict(flagged=0)}

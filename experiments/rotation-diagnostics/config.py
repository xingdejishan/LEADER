"""Paths for the rotation diagnostics. Edit this file for your machine.

Two things are read from outside this directory:

  * the frozen scorer, which owns the truth policy (including the 255 substituted
    frames) and is the only place MPE/MOE is defined;
  * the current model directory, for the frame list, the reference library, the
    initial poses and the search stages the pipeline actually runs.

Nothing here writes outside its own directory.
"""
from pathlib import Path

MODEL = Path('/home/zhangcy/projects/e2e-moe-20261020-v1')
SCORER = Path('/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')

FRAMES = MODEL / 'inputs/frames.json'
REFS = MODEL / 'inputs/refs.json'
INITIAL = MODEL / 'inputs/initial.npz'

# optional: a per-frame ambiguity-ratio table, if you want the correlation panel
DIAG = MODEL / 'results/three_route_no_rotation_diag.jsonl'

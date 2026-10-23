"""Thin adapter: run the frozen gate on the three-route end-to-end pipeline.

Nothing in the e2e project is modified or imported for writing; this module only
reads its frozen artifacts and re-derives the scan path for a frame key.

Integration point in the e2e pipeline (one line, once the caller supplies the
route-1 pose): replace the stage-1 rule with

    import adapters
    escalate, record = adapters.gate_frame(key, leader_pose)

and feed `escalate` where the current stage-1 `severe` flag is used. The rest of
the pipeline (visual route acceptance, field refinement) is untouched.
"""
import json
import sys

import numpy as np

import config as C
import gate

_LIB = None


def library():
    global _LIB
    if _LIB is None:
        _LIB = gate.ReferenceLibrary()
    return _LIB


def scan_path_of(key):
    """Resolve the frame's raw scan from the frozen selection table if present,
    otherwise from the NCLT naming convention."""
    sel = json.load(open(C.SEL))
    for r in sel['queries']:
        if str(r['key']) == str(key):
            return r['scan_path']
    return C.NCLT % tuple(str(key).split('/'))


def gate_frame(key, pose, threshold=None):
    """escalate: bool (True -> hand the frame to the independent visual route)."""
    return gate.decide(library(), scan_path_of(key), pose, threshold)


if __name__ == '__main__':
    z = np.load(C.HARD_SCORED, allow_pickle=True)
    keys = [str(k) for k in z['keys']]
    poses = np.asarray(z['refined_V7_poses'], float)
    for i in (0, 700):
        escalate, rec = gate_frame(keys[i], poses[i])
        rec = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in rec.items()}
        print(keys[i], json.dumps(rec, indent=1, default=float))

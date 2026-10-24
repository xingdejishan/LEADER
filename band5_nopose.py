"""The visual expert with the pose prior removed (variants A and B of the >5 m study).

The deployed expert (`visual_branch.VisualBranch`) is handed the frame image and the
frame's own initial pose T0.  T0 is used in exactly one place: retrieval unions the CLS
top-5 with the 5 library references nearest to T0's camera centre.  The solver never
touches T0 (EPnP + RANSAC + LM from scratch), so deleting T0 means deleting that second
channel and nothing else.

`NoPoseVisualBranch.retrieve` takes no pose argument at all; anything that looks like one
is refused rather than ignored.  `retrieve_cls_only` is global DINOv2 CLS cosine over the
whole library, top-k.

Both subclasses ride on `visual_ordered.OrderedBranch`, the class the live path builds,
so the caches behave as deployed (`visual_branch.py` evicts with `dict.popitem(last=False)`,
which a plain dict cannot do past 64 entries).

Variants used in the study:
    A5   NoPoseVisualBranch(k=5)          pose-free, CLS top-5
    B20  NoPoseVisualBranch(k=20)         pose-free, CLS top-20
    C5   OrderedBranch                    deployed, CLS top-5 union position 5-NN, T0 in

Restored from bytecode, 2026-10-24.  由字节码还原，仅用于复现历史实验。

The source of this module was never committed and had already left the work tree; the only
surviving copy was `__pycache__/band5_nopose.cpython-38.pyc` (2026-10-23 12:01).  What
follows is rebuilt from that bytecode: same docstring, same function order, same defaults
(`build` still 256 / 0.4 / False), same error strings, same variant table, same
`CaptureMixin` capture points.  Two things could not stay byte-identical, both because of
commit b3e65d6 (SALAD global cosine top-5 replaces the two-channel retrieval):

1. Pose plumbing.  The bytecode's `hypothesize` override still carried a pose keyword,
   asserted it was empty, and forwarded it verbatim into `super().hypothesize(...)`.  The
   parent no longer accepts that keyword, so the forward raised `TypeError` before any
   work happened.  The override is therefore gone: the inherited `hypothesize` carries the
   pose-free signature by construction, and `retrieve` refuses any extra argument exactly
   as `visual_branch.VisualBranch.retrieve` does.  `reject_pose_like` below is kept, and
   is still the name `probe_branch.py` imports.  The study never passed a pose, so nothing
   it measured changes.

2. A5 and B20 no longer execute.  `retrieve_cls_only` reads `self.reference_cls` and
   `self.embed`; b3e65d6 deleted both together with the CLS channel.  The rule is kept
   verbatim so it can be read, and re-run against a pre-b3e65d6 `visual_branch.py`, but on
   the current parent those two variants raise `AttributeError`.  C5 (the deployed branch)
   and the capture harness work; `mw_cls.py` holds a CLS embedder whose top-20 was
   verified identical to the removed one.
"""
import numpy as np

from visual_ordered import OrderedBranch


def reject_pose_like(value, name='argument'):
    """Refuse anything that looks like a pose or a position."""
    if value is None:
        return
    if isinstance(value, (bool, int, np.integer)):
        return
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (4, 4):
        raise TypeError('%s: refusing a 4x4 pose matrix, this expert is pose-free' % name)
    if array.shape in ((3,), (3, 1), (1, 3)):
        raise TypeError('%s: refusing a 3-vector position, this expert is pose-free' % name)
    raise TypeError('%s: refusing non-scalar argument of shape %r' % (name, array.shape))


def retrieve_cls_only(branch, image_path, k):
    """Global DINOv2 CLS cosine over the whole library, top-k.  Image in, ids out."""
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        reject_pose_like(k, 'k')
        raise TypeError('k: must be a positive integer, got %r' % (k,))
    k = int(k)
    if k <= 0:
        raise ValueError('k must be positive')
    similarity = branch.reference_cls @ branch.embed(image_path).astype(np.float32)
    ids = [int(i) for i in np.argsort(-similarity)[:k]]
    detail = dict(visual=ids, visual_cosine=[float(similarity[i]) for i in ids],
                  position=[], mode='cls-only', k=k, pose_free=True)
    return ids, detail


class NoPoseVisualBranch(OrderedBranch):
    """The deployed branch with the position channel deleted and asserted away."""

    def __init__(self, k=5, **kwargs):
        super(NoPoseVisualBranch, self).__init__(**kwargs)
        self.k = int(k)

    def retrieve(self, image_path, *extra):
        for value in extra:
            reject_pose_like(value, 'retrieve')
        if extra:
            raise TypeError('retrieve takes one image path, got %d extra arguments'
                            % len(extra))
        return retrieve_cls_only(self, image_path, self.k)


class CaptureMixin(object):
    """Records every hypothesis and every fair score the deployed code computes."""

    def reset_capture(self):
        self.cap = dict(hypotheses=[], accepted=[], pnp_calls=None, capped=None,
                        verify=[], retrieval=None)

    def retrieve(self, image_path, *extra):
        ids, detail = super(CaptureMixin, self).retrieve(image_path, *extra)
        self.cap['retrieval'] = detail
        return ids, detail

    def _hypotheses(self, q_model, candidates, by_reference):
        out = super(CaptureMixin, self)._hypotheses(q_model, candidates, by_reference)
        self.cap['hypotheses'], self.cap['accepted'] = out[0], out[1]
        self.cap['pnp_calls'], self.cap['capped'] = out[2], out[3]
        return out

    def verify(self, pose, query_cloud, excluded):
        row = super(CaptureMixin, self).verify(pose, query_cloud, excluded)
        self.cap['verify'].append((np.asarray(pose, dtype=np.float64).copy(), row))
        return row


CapturedNoPose = type('CapturedNoPose', (CaptureMixin, NoPoseVisualBranch), {})
CapturedDeployed = type('CapturedDeployed', (CaptureMixin, OrderedBranch), {})


def build(variant, scratch, device='cuda', keypoint_budget=256, score_voxel=0.4,
          verbose=False):
    """variant: 'A5' | 'B20' | 'C5'; returns a branch whose .cap holds the last frame."""
    common = dict(device=device, arm='frozen_noprune', torch_threads=1, scratch=scratch,
                  keypoint_budget=keypoint_budget, fast_pnp=False, score_voxel=score_voxel,
                  verbose=verbose)
    if variant == 'A5':
        branch = CapturedNoPose(k=5, **common)
    elif variant == 'B20':
        branch = CapturedNoPose(k=20, **common)
    elif variant == 'C5':
        branch = CapturedDeployed(**common)
    else:
        raise ValueError(variant)
    branch.reset_capture()
    return branch

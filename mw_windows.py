"""Where the geometric search starts: candidate windows from two independent channels.

The deployed geometry expert runs one along-track/lateral search window centred on the
frame's own initial pose T0 and reaches +-8 m along the track.  On the frames this study
targets (t0 off by 5-20 m along the road) that window does not contain the truth, so no
amount of searching inside it can recover.

This module changes only the list of centres; the search, the map assembly, the cost and
the arbitration are untouched.

  channel 1  position prior   the 5 library references nearest to T0's camera centre
  channel 2  CLS              global DINOv2 CLS cosine over the whole library, top 20,
                              no pose anywhere in it

The two lists are merged (a reference that appears in both is kept once), then greedily
non-maximum suppressed at 15 m: a centre within 15 m of an already kept centre would
search the same 16 m stretch of road, so it is dropped.  Ordering is by channel
reliability -- position first (nearest to T0 first), then CLS (highest cosine first) --
and at most MAX_CENTERS windows survive.

No truth is read here, no constant is tuned, and no pose other than T0 and the library
positions is used.
"""
import numpy as np

NMS_RADIUS_M = 15.0      # two windows closer than this search the same stretch of road
MAX_CENTERS = 20         # the study's ceiling on windows per frame
POSITION_CHANNEL = 5     # the deployed retrieval's position channel width
CLS_CHANNEL = 20         # the pose-free channel width


def candidate_windows(t0_position, reference_positions, position_ids, cls_ids,
                      nms=NMS_RADIUS_M, cap=MAX_CENTERS):
    """Candidate window centres for one frame.

    t0_position         (3,) camera centre of the frame's initial pose
    reference_positions (N,3) library reference positions
    position_ids        reference ids sorted by distance to t0 (the 5 nearest are used)
    cls_ids             reference ids sorted by descending CLS cosine (top 20 are used)

    Returns a list of dicts, ordered position-first then CLS:
        ref, channel ('position'|'cls'), also_cls (the other channel proposed it too),
        rank (position: 0 = nearest; cls: 0 = highest cosine), centre (3,),
        distance_to_t0
    """
    t0 = np.asarray(t0_position, dtype=np.float64).reshape(3)
    refs = np.asarray(reference_positions, dtype=np.float64)
    if refs.ndim != 2 or refs.shape[1] != 3:
        raise ValueError('reference_positions must be (N,3), got %r' % (refs.shape,))

    position_list = [int(r) for r in list(position_ids)[:POSITION_CHANNEL]]
    cls_list = [int(r) for r in list(cls_ids)[:CLS_CHANNEL]]
    in_position = set(position_list)
    in_cls = set(cls_list)

    rows, order = {}, []
    for rank, rid in enumerate(position_list):
        if rid in rows:
            continue
        rows[rid] = dict(ref=rid, channel='position', rank=rank,
                         also_cls=rid in in_cls)
        order.append(rid)
    for rank, rid in enumerate(cls_list):
        if rid in rows:           # already proposed by the position channel
            continue
        rows[rid] = dict(ref=rid, channel='cls', rank=rank, also_cls=False)
        order.append(rid)

    kept = []
    for rid in order:
        centre = refs[rid]
        if any(np.linalg.norm(centre - refs[k['ref']]) < nms for k in kept):
            continue
        kept.append(rows[rid])
        if len(kept) >= int(cap):
            break

    out = []
    for row in kept:
        centre = refs[row['ref']].copy()
        out.append(dict(ref=row['ref'], channel=row['channel'],
                        also_cls=bool(row['also_cls']), rank=int(row['rank']),
                        centre=centre,
                        distance_to_t0=float(np.linalg.norm(centre - t0))))
    return out

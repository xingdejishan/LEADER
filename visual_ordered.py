"""The frozen visual branch with both of its caches usable past 64 entries.

`visual_branch.VisualBranch` builds `self._features` and `self._projections` as plain
dicts and evicts with `dict.popitem(last=False)`, which only exists on OrderedDict: as
soon as either cache passes 64 entries every call dies with

    TypeError: popitem() takes no keyword arguments

This subclass swaps only those two containers. No rule, constant, solver or cached value
changes; `visual_branch.py` itself stays byte-for-byte unchanged. It is the class the
live visual path builds (`three_route_unified.py --visual-source live` and
`visual_live.py`).
"""
from collections import OrderedDict

import visual_branch as VB


class OrderedBranch(VB.VisualBranch):
    """VisualBranch with its two caches as OrderedDict (same values, same policy)."""

    def __init__(self, *args, **kwargs):
        super(OrderedBranch, self).__init__(*args, **kwargs)
        self._features = OrderedDict()
        self._projections = OrderedDict()

"""One algorithm per module, named after its class.

    CFS   coarse-to-fine along-track x lateral search (no learned parameters)
    MoE   the per-frame mixture-of-experts localiser built on top of CFS

Both are importable without side effects: no file writes, no argument parsing,
no environment mutation, nothing executed at import time beyond reading globals.
"""
from .cfs import CFS
from .moe import MoE

__all__ = ['CFS', 'MoE']

"""Minimal helper extracted from pose_replay.py (selection rule only, no Matcher subclass)."""

import torch


def selected_indices(prediction: torch.Tensor) -> torch.Tensor:
    """Top-k reliability selection, identical to pose_replay.selected_indices."""
    n = len(prediction)
    return prediction[:, 3].topk(max(min(50, n), n // 2)).indices

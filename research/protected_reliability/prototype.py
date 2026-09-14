"""Standalone design prototype; NOT a trained or repository-integrated model.

Input: ONE frame's existing cached LiDAR features, already sampled visual
features, ORIGINAL LEADER predictions and the existing projection-valid mask.
Do not re-encode LiDAR, recompute projection masks, or substitute a fine-tuned
regression head to produce these inputs.

The module changes only the last (raw reliability) channel. Keep the repository's
original TRR implementation, Cartesian/coarse coordinate conventions and Matcher.

Proposed training objective:
    original_trr(gt, out.pred[:, :3], out.pred[:, 3], batch_idx).mean()
    + 0.1 * boundary_ranking_loss(out, base_pred, gt)
Adapt the TRR call/return unpacking to the pinned repository implementation.
The original coordinates/features and image backbone are intentionally detached.
For batches, call per frame and average losses per frame, not per point.

Run only structural self-checks with:
    python protected_visual_reliability.py
These checks do NOT establish localization performance.
"""
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class FusionResult:
    pred: Tensor                 # [N, 4]; XYZ identical to base_pred
    protected: Tensor            # [N]; high-reliability core, never rescored
    editable: Tensor             # [N]; image-valid, non-core points
    delta: Tensor                # [N]; actual reliability residual
    gap: Tensor                  # scalar; detached baseline score gap
    k: int                       # original candidate count; no selection here


class ProtectedVisualReliability(nn.Module):
    """A 641 -> 32 -> 1 score-only head with 20,577 trainable parameters.

    The raw LEADER reliability is a ranking score, not a [0, 1] probability.
    alpha=0.25 is a prespecified conservative design value, not a tuned result.
    The guarantee is preservation of the protected candidate subset, NOT a
    guarantee that the estimated pose or the remaining selected points improve.
    """

    def __init__(self, hidden_dim: int = 32, alpha: float = 0.25) -> None:
        super().__init__()
        if hidden_dim <= 0 or not 0.0 < alpha < 1.0:
            raise ValueError('hidden_dim must be positive and 0 < alpha < 1.')
        self.alpha = float(alpha)
        self.head = nn.Sequential(
            nn.Linear(512 + 128 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        lidar_features: Tensor,
        visual_features: Tensor,
        base_pred: Tensor,
        image_valid: Tensor,
    ) -> FusionResult:
        if base_pred.ndim != 2 or base_pred.shape[1] != 4:
            raise ValueError('base_pred must be [N, 4] from original LEADER.')
        n = base_pred.shape[0]
        if n < 1:
            raise ValueError('An empty frame is not supported.')
        if lidar_features.shape != (n, 512) or visual_features.shape != (n, 128):
            raise ValueError('Expected aligned [N,512] and [N,128] features.')
        if image_valid.shape != (n,) or image_valid.dtype != torch.bool:
            raise ValueError('image_valid must be a boolean [N] mask.')
        tensors = (lidar_features, visual_features, image_valid)
        if any(t.device != base_pred.device for t in tensors):
            raise ValueError('All inputs must be on the same device.')
        if base_pred.dtype != torch.float32:
            raise ValueError('Use float32 original predictions, without re-encoding.')
        if self.head[0].weight.device != base_pred.device:
            raise ValueError('Move the head and inputs to the same device.')
        if self.head[0].weight.dtype != torch.float32:
            raise ValueError('Use a float32 head for this diagnostic prototype.')
        if not bool(torch.isfinite(base_pred).all()):
            raise ValueError('Original predictions must be finite.')

        base = base_pred.detach()
        u0 = base[:, 3]
        k = max(min(50, n), int(0.5 * n))  # original LEADER inference convention
        empty = torch.zeros(n, dtype=torch.bool, device=base.device)
        zero_delta = torch.zeros_like(u0)
        zero_gap = u0.new_zeros(())
        if k == n:
            # All points are already selected; do not introduce ordering-only effects.
            return FusionResult(base, ~empty, empty, zero_delta, zero_gap, k)

        sorted_u = torch.sort(u0, descending=True).values
        core_count = max(1, k // 2)
        core_threshold = sorted_u[core_count - 1]
        cutoff = sorted_u[k - 1]
        gap = (core_threshold - cutoff).detach()
        # Include ties at the core threshold rather than assigning arbitrary identities.
        protected = u0 >= core_threshold
        if not bool(gap > 1e-8):
            return FusionResult(base, protected, empty, zero_delta, gap, k)

        editable = image_valid & ~protected
        indices = torch.nonzero(editable, as_tuple=False).flatten()
        if indices.numel() == 0:
            return FusionResult(base, protected, editable, zero_delta, gap, k)

        lidar = lidar_features.detach()[indices].float()
        visual = visual_features.detach()[indices].float()
        if not bool(torch.isfinite(lidar).all() and torch.isfinite(visual).all()):
            raise ValueError('Editable points need finite sampled features.')
        # Parameter-free normalization: it does not fit PCA or data statistics.
        lidar = F.layer_norm(lidar, (512,))
        visual = F.layer_norm(visual, (128,))
        score_context = ((u0[indices] - cutoff) / gap).clamp(-4.0, 4.0)
        x = torch.cat((lidar, visual, score_context[:, None]), dim=1)
        z = self.head(x).squeeze(-1)
        learned_delta = self.alpha * gap * torch.tanh(z)
        delta = zero_delta.index_copy(0, indices, learned_delta)
        u_new = u0 + delta
        pred = torch.cat((base[:, :3], u_new[:, None]), dim=1)
        return FusionResult(pred, protected, editable, delta, gap, k)


def boundary_ranking_loss(
    out: FusionResult,
    base_pred: Tensor,
    gt_world: Tensor,
    max_pairs: int = 256,
    min_error_gap_m: float = 0.01,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Small auxiliary loss; use ALONGSIDE the unchanged original TRR loss.

    Sample pairs straddling the original selection boundary, from approximately
    the 25%-50% and 50%-75% reliability ranks. At least one point must be editable.
    The label is which FROZEN baseline coordinate has smaller continuous error,
    NOT whether a point passes a coarse 2 m inlier threshold. GT is training-only.
    Random sampling is on the same device as predictions; pass a matching-device
    torch.Generator for a fully reproducible training pipeline.
    """
    if gt_world.shape != base_pred[:, :3].shape:
        raise ValueError('GT must match the original coarse-voxel [N,3] targets.')
    if gt_world.device != out.pred.device or base_pred.device != out.pred.device:
        raise ValueError('Predictions and GT must share a device.')
    if max_pairs <= 0 or min_error_gap_m < 0:
        raise ValueError('Invalid pair count or error-gap threshold.')
    zero = out.pred[:, 3].sum() * 0.0
    n, k = base_pred.shape[0], out.k
    if k == n or not bool(out.gap > 1e-8) or not bool(out.editable.any()):
        return zero

    order = torch.argsort(base_pred.detach()[:, 3], descending=True)
    core_count = max(1, k // 2)
    inside = order[core_count:k]
    inside = inside[~out.protected[inside]]
    outside = order[k:min(n, k + (k - core_count))]
    if inside.numel() == 0 or outside.numel() == 0:
        return zero
    draw_count = max_pairs * 4
    left = inside[torch.randint(inside.numel(), (draw_count,),
                               device=order.device, generator=generator)]
    right = outside[torch.randint(outside.numel(), (draw_count,),
                                 device=order.device, generator=generator)]
    error = (base_pred.detach()[:, :3] - gt_world.detach()).norm(dim=-1)
    error_difference = error[right] - error[left]
    keep = (out.editable[left] | out.editable[right]) & (
        error_difference.abs() > min_error_gap_m
    )
    left, right = left[keep][:max_pairs], right[keep][:max_pairs]
    direction = error_difference[keep][:max_pairs].sign()
    if left.numel() == 0:
        return zero
    score_difference = (out.pred[left, 3] - out.pred[right, 3]) / out.gap
    return F.softplus(-direction * score_difference).mean()


def _self_check() -> None:
    """Synthetic invariance checks, not a localization experiment."""
    torch.manual_seed(17)
    torch.set_num_threads(1)
    n = 800
    lidar = torch.randn(n, 512, requires_grad=True)
    visual = torch.randn(n, 128, requires_grad=True)
    base = torch.randn(n, 4)
    base[:, 3] = torch.linspace(5.0, -5.0, n)
    base.requires_grad_()
    valid = torch.rand(n) < 0.22
    module = ProtectedVisualReliability()
    assert sum(p.numel() for p in module.parameters()) == 20577

    # Zero initialization is an exact numerical identity.
    initial = module(lidar, visual, base, valid)
    assert torch.equal(initial.pred, base.detach())

    # Force arbitrary nonzero residuals to check protection beyond initialization.
    with torch.no_grad():
        nn.init.normal_(module.head[-1].weight, std=0.5)
        nn.init.constant_(module.head[-1].bias, 0.3)
    out = module(lidar, visual, base, valid)
    assert torch.equal(out.pred[:, :3], base.detach()[:, :3])
    unchanged = ~valid | out.protected
    assert torch.equal(out.pred[unchanged, 3], base.detach()[unchanged, 3])
    assert bool((out.delta.abs() <= module.alpha * out.gap + 1e-7).all())
    selected = torch.zeros(n, dtype=torch.bool)
    selected[out.pred[:, 3].topk(out.k).indices] = True
    assert bool(selected[out.protected].all())

    # Missing images, tied scores, and frames with <=50 points fall back to baseline.
    no_image = module(lidar, visual, base, torch.zeros_like(valid))
    assert torch.equal(no_image.pred, base.detach())
    tied_base = base.detach().clone()
    tied_base[:, 3] = 2.0
    assert torch.equal(module(lidar, visual, tied_base, valid).pred, tied_base)
    assert torch.equal(module(lidar[:30], visual[:30], base[:30], valid[:30]).pred,
                       base.detach()[:30])
    invalid_visual = visual.detach().clone()
    invalid_visual[~valid] = float('nan')
    assert bool(torch.isfinite(module(lidar, invalid_visual, base, valid).pred).all())

    # Only the new module receives gradients, not any baseline/visual input.
    target = base.detach()[:, :3] + 0.1 * torch.randn(n, 3)
    rank_loss = boundary_ranking_loss(out, base, target)
    loss = rank_loss + out.delta.square().mean()
    loss.backward()
    assert lidar.grad is None and visual.grad is None and base.grad is None
    assert module.head[-1].weight.grad is not None
    assert bool(torch.isfinite(module.head[-1].weight.grad).all())

    # Check the algebraic core guarantee under many worst-direction perturbations.
    for _ in range(100):
        score = torch.randn(n)
        k, core_count = n // 2, n // 4
        ordered = score.sort(descending=True).values
        core_threshold, cutoff = ordered[core_count - 1], ordered[k - 1]
        gap = core_threshold - cutoff
        core = score >= core_threshold
        delta = 0.25 * gap * (2 * torch.rand(n) - 1)
        delta[core] = 0
        new_selected = torch.zeros(n, dtype=torch.bool)
        new_selected[(score + delta).topk(k).indices] = True
        assert bool(new_selected[core].all())

    print('PASS: 20,577 parameters; identity initialization; frozen coordinates;')
    print('PASS: missing-image/core-score identity; bounded residual; core retention;')
    print('PASS: tied/small-frame fallback; masked NaNs; gradients only in new head.')
    print('NOT RUN: real-data training, original TRR integration, Matcher, pose evaluation.')


if __name__ == '__main__':
    _self_check()

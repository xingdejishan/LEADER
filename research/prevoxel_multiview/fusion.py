"""Model-side fusion: ViewAdapter + Transformer weighting + differentiable
voxel aggregation. ALL trainable modules live here and run inside the
training loop's model.forward so that LEADER's scene-coordinate loss
backpropagates through v_i into the Adapter/Transformer/NULL (the gradient
chain the whole design exists for).

Dataset provides fixed observations (dataset_hook.PerFrameObservation); this
module consumes tensors.
"""
import torch
import torch.nn as nn


class ViewAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        va = config["view_adapter"]
        dq = 9
        img_dim = config["image_feature"]["dim"]
        self.net = nn.Sequential(
            nn.Linear(img_dim + dq, va["hidden"]),
            nn.GELU(),
            nn.LayerNorm(va["hidden"]),
            nn.Linear(va["hidden"], va["out_dim"]),
        )

    def forward(self, img_feat, quality):
        """img_feat: (N,6,C) | quality: (N,6,Dq) -> (N,6,d)"""
        return self.net(torch.cat([img_feat, quality], dim=-1))


class ViewWeighting(nn.Module):
    def __init__(self, config):
        super().__init__()
        tf = config["transformer"]
        d = tf["d_model"]
        self.d = d
        self.null_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=tf["nhead"], dim_feedforward=tf["ffn_dim"],
            dropout=tf["dropout"], batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=tf["num_layers"])
        self.score_head = nn.Linear(d, 1)

    def forward(self, z, valid, force_null=None):
        """z: (N,6,d) | valid: (N,6) bool | force_null: (N,) bool or None.

        Invalid views are excluded from self-attention via
        src_key_padding_mask (they cannot be attended to inside the
        transformer), AND their output scores get -inf so softmax yields
        exactly alpha=0 (Check 4). Degenerate rows (no valid view / forced
        NULL) bypass the transformer entirely (Check 5).

        Returns (alpha_null (N,), alpha_views (N,6), v (N,d)).
        """
        n = z.shape[0]
        device = z.device
        any_valid = valid.any(dim=1)
        if force_null is None:
            force_null = torch.zeros(n, dtype=torch.bool, device=device)
        bypass = force_null | ~any_valid

        tokens = torch.cat([self.null_token.expand(n, 1, self.d), z], dim=1)
        # padding mask: True = key excluded from attention
        key_mask = torch.zeros(n, 7, dtype=torch.bool, device=device)
        key_mask[:, 1:] = ~valid
        safe = tokens.masked_fill(key_mask[..., None], 0.0)
        h = self.encoder(safe, src_key_padding_mask=key_mask)
        h = torch.where(key_mask[..., None], torch.zeros_like(h), h)

        scores = self.score_head(h).squeeze(-1)  # (N,7)
        neg = torch.tensor(float("-inf"), device=device)
        scores = scores.masked_fill(key_mask, neg)
        alpha = torch.softmax(scores, dim=-1)
        alpha_views = alpha[:, 1:]
        alpha_null = alpha[:, 0]

        # structural overrides (exact, not learned)
        alpha_views = torch.where(bypass[:, None], torch.zeros_like(alpha_views), alpha_views)
        alpha_null = torch.where(bypass, torch.ones_like(alpha_null), alpha_null)
        v = (alpha_views[..., None] * z).sum(dim=1)
        v = torch.where(bypass[:, None], torch.zeros_like(v), v)
        return alpha_null, alpha_views, v


class PreVoxelMultiViewFusion(nn.Module):
    """Point-level: observation tensors -> v_i (N,d), r_i (N,). Differentiable."""

    def __init__(self, config):
        super().__init__()
        self.adapter = ViewAdapter(config)
        self.weighting = ViewWeighting(config)

    def forward(self, img_feat, quality, valid, force_null=None):
        z = self.adapter(img_feat, quality)
        z = z * valid[..., None]  # zero content for invalid views
        return self.weighting(z, valid, force_null)


def aggregate_to_voxel(v_point, inverse, counts):
    """Differentiable mean aggregation raw points -> quantized rows.

    v_point: (N,K) float tensor (may require grad)
    inverse: (N,) long tensor — ME sparse_quantize's own mapping (same as LEADER)
    counts:  (M,) float tensor
    Returns (M,K): mean of v_point per cell. Grad flows via index_add.
    """
    m = int(counts.shape[0])
    sums = torch.zeros(m, v_point.shape[1], dtype=v_point.dtype,
                       device=v_point.device)
    sums.index_add_(0, inverse, v_point)
    return sums / counts[:, None]


class LEADERFirstLayerExtension(nn.Module):
    """Wrap the frozen RPGE stem's first Linear to accept +65 visual channels.

    Spec section 12: W_new = [W_L, W_V] with W_V = 0 at init so step-0 output
    is numerically identical to original LEADER. W_L frozen; W_V trainable in
    stage 1 (caller freezes everything else).
    """

    def __init__(self, original_linear, extra_channels=65):
        super().__init__()
        W_L = original_linear.weight.data  # (out, in)
        out_f, in_f = W_L.shape
        self.extra_channels = extra_channels
        self.weight_orig = nn.Parameter(W_L.clone(), requires_grad=False)
        self.weight_vis = nn.Parameter(torch.zeros(out_f, extra_channels))
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

    def forward(self, x):
        if x.shape[-1] == self.weight_orig.shape[1]:
            return nn.functional.linear(x, self.weight_orig, self.bias)
        w = torch.cat([self.weight_orig, self.weight_vis], dim=1)
        return nn.functional.linear(x, w, self.bias)


class MultiViewLeaderForward(nn.Module):
    """Full differentiable forward used by the training loop.

    Inputs (per frame, from dataset_hook):
      obs: dict with img_feat (N,6,C) / quality (N,6,9) / valid (N,6) as
           torch tensors (raw-point aligned, fixed observations);
           index (M,) long — ME sparse_quantize's own representative mapping
           (voxel_mapping with return_index; verified: ME's quantized feats
           are feats[index] exactly, a representative selection, NOT mean —
           probes diag_mapping2 + probe_quantize_mapping2/7, real frame).
      lidar_feats: (N,3) [high, range, label] per raw point (LEADER's own
           pre-quantize feature columns).

    Returns (feats_ext (M, 68), diagnostics):
      columns 0..2  = lidar_feats[index] — bitwise equal to ME's own
                      quantized feats (sanity_check Check 7 asserts this)
      columns 3..66 = v_voxel = v[index] (representative point's visual
                      feature; differentiable gather)
      column  67    = r_voxel = (1-alpha_NULL)[index]
    """

    def __init__(self, config):
        super().__init__()
        self.fusion = PreVoxelMultiViewFusion(config)

    def forward(self, obs, lidar_feats, all_camera_dropout_prob=0.0, rng=None):
        dev = lidar_feats.device
        n = lidar_feats.shape[0]
        force_null = None
        if all_camera_dropout_prob > 0 and rng is not None:
            force_null = torch.from_numpy(
                rng.random(n) < all_camera_dropout_prob).to(dev)
        alpha_null, alpha_views, v = self.fusion(
            obs["img_feat"].to(dev), obs["quality"].to(dev),
            obs["valid"].to(dev), force_null)
        index = obs["index"].to(dev)          # (M,) representative raw point per voxel row
        feats_l_voxel = lidar_feats[index]    # == ME's own quantized feats
        v_voxel = v[index]                    # differentiable gather, same rows
        r_point = (1.0 - alpha_null)[:, None]
        r_voxel = r_point[index]
        return torch.cat([feats_l_voxel, v_voxel, r_voxel], dim=1), \
            dict(alpha_null=alpha_null, alpha_views=alpha_views)



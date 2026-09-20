"""IRLS-Huber 鲁棒刚体拟合 —— 替代 SC2-PCR 的推理期求解器。

接口与 models/sc2pcr.py 的 Matcher.estimator 完全一致：
    输入: src_keypts_corr [bs, n, 3], tgt_keypts_corr [bs, n, 3]  (torch, 通常 float32)
    输出: pred_trans       [bs, 4, 4]

约定：返回 T 使得 tgt ~= R @ src + t（与官方 SC2_PCR 同向）。

本模块无任何可训练参数，不涉及训练。
超参：delta=0.5 (m), iters=10。来源：Huber 共识报告 §7 推荐配置。
"""
import torch


def huber_rigid_torch(A, B, delta=0.5, iters=10):
    """加权 Umeyama 刚体拟合 + IRLS-Huber 重加权。

    A, B: [bs, n, 3] torch 张量（同一 device）。
    返回: [bs, 4, 4]
    """
    bs, n, _ = A.shape
    dev, dt = A.device, A.dtype

    # ---- 初始化：等权拟合（等价于一次朴素 SVD）----
    w = torch.ones(bs, n, device=dev, dtype=dt)
    R = None
    t = None

    for _ in range(max(1, int(iters))):
        ws = w.sum(dim=1, keepdim=True)                      # [bs,1]
        ws = ws.clamp_min(1e-12)
        cA = (A * w[:, :, None]).sum(dim=1, keepdim=True) / ws[:, :, None]
        cB = (B * w[:, :, None]).sum(dim=1, keepdim=True) / ws[:, :, None]
        Am = A - cA
        Bm = B - cB
        H = (Am * w[:, :, None]).transpose(1, 2) @ Bm        # [bs,3,3]
        U, S, Vt = torch.linalg.svd(H)
        # 反射修正：保证 det(R) = +1
        det = torch.det(Vt.transpose(1, 2) @ U.transpose(1, 2))   # [bs]
        D = torch.eye(3, device=dev, dtype=dt).unsqueeze(0).repeat(bs, 1, 1)
        D[:, 2, 2] = torch.sign(det)
        R = Vt.transpose(1, 2) @ D @ U.transpose(1, 2)       # [bs,3,3]
        cA_v = cA.squeeze(1)                                 # [bs,3]
        cB_v = cB.squeeze(1)                                 # [bs,3]
        t = cB_v - (R @ cA_v.unsqueeze(2)).squeeze(2)        # [bs,3]

        # ---- 残差与 Huber 权重 ----
        pred = (R @ A.transpose(1, 2)).transpose(1, 2) + t[:, None, :]
        r = torch.linalg.norm(pred - B, dim=2)                # [bs,n]
        w = torch.where(r <= delta, torch.ones_like(r), delta / r.clamp_min(1e-12))

    T = torch.eye(4, device=dev, dtype=dt).unsqueeze(0).repeat(bs, 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    return T


class HuberMatcher:
    """与 models/sc2pcr.py 的 Matcher 同接口，仅替换 estimator。"""

    def __init__(self, delta=0.5, iters=10):
        self.delta = delta
        self.iters = iters

    def estimator(self, src_keypts_corr, tgt_keypts_corr):
        return huber_rigid_torch(src_keypts_corr, tgt_keypts_corr,
                                 delta=self.delta, iters=self.iters)

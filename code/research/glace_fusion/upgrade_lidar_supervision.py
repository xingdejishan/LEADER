"""Upgrade an already-LiDAR-patched vendor trainer to the stage-4 recipe.

Chains onto a vendor whose trainer carries the stage-2 auxiliary (camera-frame
Smooth L1 with a fixed weight literal, buffer lidar keys, progress metrics):

    lidar_error = smooth_l1(pred_cam_coords_b31.squeeze(-1), lidar_camera).sum(1)
    lidar_loss = (lidar_error * lidar_valid.reshape(-1)).sum()
    loss = loss + W * lidar_loss

and replaces that single metric residual with the decomposed objective

    depth:   Smooth L1(pred_z/target_z - 1, beta=0.25, clamped)            * depth_weight
    bearing: Smooth L1(pred_px, target_px, beta=bearing_beta_px) * bearing_weight

plus the optional per-cell ReliabilityHead (depth-agreement BCE on supported
samples, detached inputs, own Adam) saved as <head>.rel.pt next to the head
checkpoint. Anchors are matched exactly once; any drift raises.
"""
import shutil
from pathlib import Path

from .retrain_rgb_baseline import replace_once


def upgrade_lidar_supervision(vendor, overall_weight=5.0, depth_weight=5.0,
                              bearing_weight=1.0, bearing_beta_px=1.0,
                              bearing_clamp_px=50.0,
                              reliability_head=True, reliability_lr=1e-3,
                              depth_ratio_tol=1.25):
    vendor = Path(vendor)
    path = vendor / 'ace_trainer.py'
    text = path.read_text()

    weight_literal = repr(float(overall_weight))
    old_residual = ("        lidar_error = torch.nn.functional.smooth_l1_loss(\n"
                    "            pred_cam_coords_b31.squeeze(-1), lidar_camera, reduction='none').sum(1)\n"
                    "        lidar_loss = (lidar_error * lidar_valid.reshape(-1)).sum()\n"
                    f"        loss = loss + {weight_literal} * lidar_loss\n")
    new_residual = (
        "        target_depth = lidar_camera[:, 2].clamp_min(1e-3)\n"
        "        rel_err = torch.clamp(pred_cam_coords_b31[:, 2, 0] / target_depth - 1.0,\n"
        "                              -1.0, 9.0)\n"
        "        depth_error = torch.nn.functional.smooth_l1_loss(\n"
        "            rel_err, torch.zeros_like(rel_err), beta=0.25, reduction='none')\n"
        "        bearing_delta = torch.clamp(pred_px_b21.squeeze(2) - target_px_b2,\n"
        f"                                    -{float(bearing_clamp_px)!r}, {float(bearing_clamp_px)!r})\n"
        "        bearing_error = torch.nn.functional.smooth_l1_loss(\n"
        "            bearing_delta, torch.zeros_like(bearing_delta),\n"
        f"            beta={float(bearing_beta_px)!r}, reduction='none').sum(1)\n"
        "        lidar_error = "
        f"{float(depth_weight)!r} * depth_error + {float(bearing_weight)!r} * bearing_error\n"
        "        lidar_loss = (lidar_error * lidar_valid.reshape(-1)).sum()\n"
        f"        loss = loss + {weight_literal} * lidar_loss\n")

    reliability_block = ''
    rel_save = ''
    if reliability_head:
        text = replace_once(text, 'from lidar_supervision import load_camera_targets\n',
                            'from lidar_supervision import load_camera_targets\n'
                            'from reliability_head import ReliabilityHead\n')
        text = replace_once(text,
            '    def training_step(self, features_bC, target_px_b2, gt_inv_poses_b34, Ks_b33, invKs_b33, lidar_camera, lidar_valid):',
            '    def training_step(self, features_bC, target_px_b2, gt_inv_poses_b34, Ks_b33, invKs_b33, lidar_camera, lidar_valid):'
            '\n        if getattr(self, "rel_head", None) is None:'
            '\n            self.rel_head = ReliabilityHead(int(features_bC.shape[1])).to(self.device)'
            '\n            self.rel_optimizer = torch.optim.Adam(self.rel_head.parameters(),'
            f'\n                                                  lr={float(reliability_lr)!r})\n')
        reliability_block = (
            '        rel_supported = lidar_valid.reshape(-1) > 0\n'
            '        if rel_supported.any():\n'
            '            with torch.no_grad():\n'
            '                rel_label = ReliabilityHead.consistency_labels(\n'
            '                    pred_cam_coords_b31[:, :, 0].detach(), lidar_camera.detach(),\n'
            '                    rel_supported.float(), depth_ratio_tol='
            f'{float(depth_ratio_tol)!r})\n'
            '            rel_logit = self.rel_head(features_bC.detach(),\n'
            '                                      pred_cam_coords_b31[:, :, 0].detach())\n'
            '            rel_loss = torch.nn.functional.binary_cross_entropy_with_logits(\n'
            '                rel_logit[rel_supported], rel_label[rel_supported])\n'
            '            self.rel_optimizer.zero_grad()\n'
            '            rel_loss.backward()\n'
            '            self.rel_optimizer.step()\n'
            '        else:\n'
            '            rel_loss = torch.zeros((), device=features_bC.device)\n'
            '            rel_label = torch.zeros((), device=features_bC.device)\n')
        rel_save = (
            '        if getattr(self, "rel_head", None) is not None:\n'
            '            rel_path = str(self.options.output_map_file) + \'.rel.pt\'\n'
            '            torch.save(self.rel_head.save_payload(), rel_path + \'.tmp\')\n'
            '            os.replace(rel_path + \'.tmp\', rel_path)\n')

    text = replace_once(text, old_residual, reliability_block + new_residual)
    text = replace_once(text,
        '                                lidar_fraction=float(lidar_valid.mean()),\n'
        '                                lidar_loss_per_sample=float(lidar_loss / batch_size))',
        '                                lidar_fraction=float(lidar_valid.mean()),\n'
        '                                lidar_loss_per_sample=float(lidar_loss / batch_size)'
        + (',\n                                rel_loss=float(rel_loss),\n'
           '                                rel_positive_fraction=float('
           'rel_label[rel_supported].mean()) if rel_supported.any() else 0.0'
           if reliability_head else '') + ')')
    if reliability_head:
        text = replace_once(text,
            '        head_state_dict = self.regressor.heads.module.state_dict()',
            rel_save + '        head_state_dict = self.regressor.heads.module.state_dict()')
    path.write_text(text)
    shutil.copyfile(Path(__file__).with_name('reliability_head.py'),
                    vendor / 'reliability_head.py')
    return dict(upgraded=True, aux_mode='decomposed', overall_weight=float(overall_weight),
                depth_weight=float(depth_weight), bearing_weight=float(bearing_weight),
                bearing_beta_px=float(bearing_beta_px),
                bearing_clamp_px=float(bearing_clamp_px),
                reliability_head=bool(reliability_head),
                depth_ratio_tol=float(depth_ratio_tol))

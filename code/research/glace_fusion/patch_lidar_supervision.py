from pathlib import Path
import shutil

from .retrain_rgb_baseline import replace_once


def patch_lidar_supervision(vendor, weight=1., relative_storage=False, log_depth=False,
                            aux_mode=None, bearing_weight=1., depth_weight=1.,
                            bearing_beta_px=1.0, reliability_head=False,
                            reliability_lr=1e-3, depth_ratio_tol=1.25):
    """Patch a vendor copy (already carrying the retrain/valid-region patches)
    to add sparse training-only LiDAR supervision.

    relative_storage: loader for the full-scene compact layout (nominal camera
    frame float16 cloud + nominal T_WC sidecar, see fullscene_lidar_targets).
    log_depth: legacy single-term log-depth residual (superseded by
    aux_mode='decomposed', kept for reproducibility).
    aux_mode:
      'l1'         - legacy camera-frame Smooth L1 (beta=1m, weight);
      'log_depth'  - single log-depth Smooth L1 term;
      'decomposed' - pixel-space bearing Smooth L1 (beta=bearing_beta_px,
                     weight=bearing_weight) + log-depth Smooth L1
                     (weight=depth_weight). Bearing is depth-scale free, so
                     angular supervision is uniform across the FOV, unlike
                     the metric L1 whose bearing gradient shrinks for near
                     points. The pixel-space term matches the 10px scoring
                     scale used at evaluation.
    reliability_head: train a small per-cell reliability MLP (depth-agreement
    labels within depth_ratio_tol, BCE on supported samples, detached inputs,
    own Adam optimizer) saved next to the head checkpoint as <head>.rel.pt.
    """
    vendor = Path(vendor)
    loader = 'load_camera_targets_rel' if relative_storage else 'load_camera_targets'
    if aux_mode is None:
        aux_mode = 'log_depth' if log_depth else 'l1'
    if aux_mode not in ('l1', 'log_depth', 'decomposed'):
        raise ValueError(f'Unknown aux_mode {aux_mode}')
    path = vendor / 'ace_trainer.py'
    text = path.read_text()

    imports = 'import os\nimport math\nfrom lidar_supervision import ' + loader
    if reliability_head:
        imports += '\nfrom reliability_head import ReliabilityHead'
    text = replace_once(text, 'import os\n', imports + '\n')

    text = replace_once(text, '        self.training_buffer = {',
        "        self.lidar_folder = self.options.scene / 'train/lidar_world'\n"
        "        if not self.lidar_folder.is_dir():\n"
        "            raise ValueError('Missing training-only LiDAR supervision directory')\n"
        '        self.training_buffer = {\n'
        "            'lidar_camera': torch.empty((self.options.training_buffer_size, 3), dtype=torch.float32, device='cpu'),\n"
        "            'lidar_valid': torch.empty((self.options.training_buffer_size, 1), dtype=torch.float32, device='cpu'),")

    text = replace_once(text, 'intrinsics_inv_B33, _, _, _,idx in training_dataloader:',
        'intrinsics_inv_B33, _, image_paths, _,idx in training_dataloader:')

    text = replace_once(text, '                    batch_data = {',
        '                    if B != 1:\n'
        "                        raise ValueError('LiDAR supervision currently requires buffer image batch size one')\n"
        '                    lidar_camera, lidar_valid = ' + loader + '(\n'
        '                        self.lidar_folder, image_paths[0],\n'
        '                        normalize_shape(pixel_positions_B2HW).cpu().numpy(),\n'
        '                        intrinsics_B33[0].cpu().numpy(), gt_pose_inv_B44[0].cpu().numpy(),\n'
        '                        image_B1HW.shape[-2], image_B1HW.shape[-1])\n'
        '                    batch_data = {\n'
        "                        'lidar_camera': torch.from_numpy(lidar_camera).to(self.device),\n"
        "                        'lidar_valid': torch.from_numpy(lidar_valid[:, None]).to(self.device),")

    text = replace_once(text,
        "                self.training_buffer['intrinsics_inv'][random_batch_indices].to(self.device).contiguous()",
        "                self.training_buffer['intrinsics_inv'][random_batch_indices].to(self.device).contiguous(),\n"
        "                self.training_buffer['lidar_camera'][random_batch_indices].to(self.device),\n"
        "                self.training_buffer['lidar_valid'][random_batch_indices].to(self.device)")

    signature_extra = ''
    if reliability_head:
        signature_extra = (
            '\n        if getattr(self, "rel_head", None) is None:\n'
            '            self.rel_head = ReliabilityHead(int(features_bC.shape[1])).to(self.device)\n'
            '            self.rel_optimizer = torch.optim.Adam(self.rel_head.parameters(),\n'
            f'                                                  lr={float(reliability_lr)!r})\n')
    text = replace_once(text,
        '    def training_step(self, features_bC, target_px_b2, gt_inv_poses_b34, Ks_b33, invKs_b33):',
        '    def training_step(self, features_bC, target_px_b2, gt_inv_poses_b34, Ks_b33, invKs_b33, lidar_camera, lidar_valid):'
        + signature_extra)

    if aux_mode == 'l1':
        residual = ('        lidar_error = torch.nn.functional.smooth_l1_loss(\n'
                    "            pred_cam_coords_b31.squeeze(-1), lidar_camera, reduction='none').sum(1)\n")
        label = 'Smooth L1 camera-frame 3D residual, beta=1m; sum / whole batch size'
    elif aux_mode == 'log_depth':
        residual = ('        pred_depth = pred_cam_coords_b31[:, 2, 0].clamp_min(1e-3)\n'
                    '        target_depth = lidar_camera[:, 2].clamp_min(1e-3)\n'
                    '        lidar_error = torch.nn.functional.smooth_l1_loss(\n'
                    "            torch.log(pred_depth), torch.log(target_depth), reduction='none')\n")
        label = 'log-depth Smooth L1 on camera-frame ray targets'
    else:
        residual = ('        pred_depth = pred_cam_coords_b31[:, 2, 0].clamp_min(1e-3)\n'
                    '        target_depth = lidar_camera[:, 2].clamp_min(1e-3)\n'
                    '        depth_error = torch.nn.functional.smooth_l1_loss(\n'
                    "            torch.log(pred_depth), torch.log(target_depth), reduction='none')\n"
                    '        bearing_error = torch.nn.functional.smooth_l1_loss(\n'
                    '            pred_px_b21.squeeze(2), target_px_b2,\n'
                    f"            beta={float(bearing_beta_px)!r}, reduction='none').sum(1)\n"
                    '        lidar_error = '
                    f'{float(depth_weight)!r} * depth_error + {float(bearing_weight)!r} * bearing_error\n')
        label = (f'decomposed: pixel bearing Smooth L1 (beta={bearing_beta_px}px, w='
                 f'{bearing_weight}) + log-depth Smooth L1 (w={depth_weight}); sum / batch size')

    progress_call = (
        '                                valid_fraction=float(valid_mask_b1.float().mean()),\n'
        '                                lidar_fraction=float(lidar_valid.mean()),\n'
        '                                lidar_loss_per_sample=float(lidar_loss / batch_size)')
    if reliability_head:
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
        progress_call += (',\n'
                          '                                rel_loss=float(rel_loss),\n'
                          '                                rel_positive_fraction=float('
                          'rel_label[rel_supported].mean()) if rel_supported.any() else 0.0')
    else:
        reliability_block = ''

    text = replace_once(text, '        loss /= batch_size',
        reliability_block
        + residual
        + '        lidar_loss = (lidar_error * lidar_valid.reshape(-1)).sum()\n'
        + f'        loss = loss + {float(weight)!r} * lidar_loss\n'
        + '        loss /= batch_size')

    text = replace_once(text,
        '                                valid_fraction=float(valid_mask_b1.float().mean()))',
        progress_call + ')')

    text = replace_once(text,
        '        self.write_progress("buffer_complete", int(counts.sum()), self.options.training_buffer_size)',
        '        self.write_progress("buffer_complete", int(counts.sum()), self.options.training_buffer_size,\n'
        '                            lidar_fraction=float(self.training_buffer["lidar_valid"].mean()))\n'
        '        if self.training_buffer["lidar_valid"].sum() == 0:\n'
        '            raise ValueError("No sparse LiDAR supervision reached the buffer")')

    if reliability_head:
        text = replace_once(text,
            '        head_state_dict = self.regressor.heads.module.state_dict()',
            '        if getattr(self, "rel_head", None) is not None:\n'
            '            rel_path = str(self.options.output_map_file) + \'.rel.pt\'\n'
            '            torch.save(self.rel_head.save_payload(), rel_path + \'.tmp\')\n'
            '            os.replace(rel_path + \'.tmp\', rel_path)\n'
            '        head_state_dict = self.regressor.heads.module.state_dict()')

    path.write_text(text)
    shutil.copyfile(Path(__file__).with_name('lidar_supervision.py'), vendor / 'lidar_supervision.py')
    if reliability_head:
        shutil.copyfile(Path(__file__).with_name('reliability_head.py'),
                        vendor / 'reliability_head.py')

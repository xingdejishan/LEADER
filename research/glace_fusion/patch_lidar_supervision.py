from pathlib import Path
import shutil

from .retrain_rgb_baseline import replace_once


def patch_lidar_supervision(vendor, weight=1.):
    vendor = Path(vendor)
    path = vendor / 'ace_trainer.py'
    text = path.read_text()
    text = replace_once(text, 'import os\n', 'import os\nfrom lidar_supervision import load_camera_targets\n')
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
        '                    lidar_camera, lidar_valid = load_camera_targets(\n'
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
    text = replace_once(text,
        '    def training_step(self, features_bC, target_px_b2, gt_inv_poses_b34, Ks_b33, invKs_b33):',
        '    def training_step(self, features_bC, target_px_b2, gt_inv_poses_b34, Ks_b33, invKs_b33, lidar_camera, lidar_valid):')
    text = replace_once(text, '        loss /= batch_size',
        '        lidar_error = torch.nn.functional.smooth_l1_loss(\n'
        "            pred_cam_coords_b31.squeeze(-1), lidar_camera, reduction='none').sum(1)\n"
        '        lidar_loss = (lidar_error * lidar_valid.reshape(-1)).sum()\n'
        f'        loss = loss + {float(weight)!r} * lidar_loss\n'
        '        loss /= batch_size')
    text = replace_once(text,
        '                                valid_fraction=float(valid_mask_b1.float().mean()))',
        '                                valid_fraction=float(valid_mask_b1.float().mean()),\n'
        '                                lidar_fraction=float(lidar_valid.mean()),\n'
        '                                lidar_loss_per_sample=float(lidar_loss / batch_size))')
    text = replace_once(text,
        '        self.write_progress("buffer_complete", int(counts.sum()), self.options.training_buffer_size)',
        '        self.write_progress("buffer_complete", int(counts.sum()), self.options.training_buffer_size,\n'
        '                            lidar_fraction=float(self.training_buffer["lidar_valid"].mean()))\n'
        '        if self.training_buffer["lidar_valid"].sum() == 0:\n'
        '            raise ValueError("No sparse LiDAR supervision reached the buffer")')
    path.write_text(text)
    shutil.copyfile(Path(__file__).with_name('lidar_supervision.py'), vendor / 'lidar_supervision.py')

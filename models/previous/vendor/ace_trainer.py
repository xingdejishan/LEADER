# Copyright © Niantic, Inc. 2022.

import logging
import random
import time
import os
from lidar_supervision import load_camera_targets
from valid_region import grid_valid_region
import json

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms.functional as TF
from sklearn.cluster import KMeans
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader,sampler,WeightedRandomSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from ace_util import get_pixel_grid, to_homogeneous
from ace_loss import ReproLoss
from ace_network import Regressor
from dataset import CamLocDataset
from room_dataset import RoomDataset


_logger = logging.getLogger(__name__)


def set_seed(seed):
    """
    Seed all sources of randomness.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class TrainerACE:
    def __init__(self, options):
        self.options = options
        dist.init_process_group(backend='nccl', init_method="env://")
        device_id = int(os.environ["LOCAL_RANK"])
        self.device = torch.device('cuda',device_id)
        torch.cuda.set_device(device_id)

        # The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
        # torch.backends.cuda.matmul.allow_tf32 = False

        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        # torch.backends.cudnn.allow_tf32 = False

        # Setup randomness for reproducibility.
        self.base_seed = int(os.environ["GLACE_SEED"]) + device_id
        set_seed(self.base_seed)

        # Used to generate batch indices.
        self.batch_generator = torch.Generator()
        self.batch_generator.manual_seed(self.base_seed + 1023)

        # Dataloader generator, used to seed individual workers by the dataloader.
        self.loader_generator = torch.Generator()
        self.loader_generator.manual_seed(self.base_seed + 511)

        # Generator used to sample random features (runs on the GPU).
        self.sampling_generator = torch.Generator(device=self.device)
        self.sampling_generator.manual_seed(self.base_seed + 4095)

        # Generator used to permute the feature indices during each training epoch.
        self.training_generator = torch.Generator()
        self.training_generator.manual_seed(self.base_seed + 8191)

        # Generator for global feature noise
        self.gn_generator = torch.Generator(device=self.device)
        self.gn_generator.manual_seed(self.base_seed + 24601)

        self.iteration = 0
        self.training_start = None
        self.num_data_loader_workers = 3


        # Create dataset.
        ds_args=dict(
            mode=0,  # Default for ACE, we don't need scene coordinates/RGB-D.
            use_half=self.options.use_half,
            image_height=self.options.image_resolution,
            augment=self.options.use_aug,
            aug_rotation=self.options.aug_rotation,
            aug_scale_max=self.options.aug_scale,
            aug_scale_min=1 / self.options.aug_scale,
            num_clusters=self.options.num_clusters,  # Optional clustering for Cambridge experiments.
            cluster_idx=self.options.cluster_idx,    # Optional clustering for Cambridge experiments.
            feat_name=self.options.feat_name,
        )

        if self.options.scene.suffix=='.txt':
            self.dataset=RoomDataset(self.options.scene,
                                       training=True, **ds_args)
        else:
            self.dataset = CamLocDataset(
                root_dir=self.options.scene / "train",
                **ds_args
            )
        self.global_feat_dim=self.dataset.global_feat_dim
        self.global_feats=torch.tensor(self.dataset.global_feats,
                dtype=(torch.float32, torch.float16)[self.options.use_half],device=self.device)
        
        if self.options.num_decoder_clusters>1:
            if dist.get_rank() == 0:
                _logger.info(f"Clustering camera centers into {self.options.num_decoder_clusters} clusters for position decoder.")
            kmeans=KMeans(n_clusters=self.options.num_decoder_clusters,random_state=0).fit(self.dataset.pose_values[:,:3,3].astype(np.float32))
            mean=torch.from_numpy(kmeans.cluster_centers_).float()
        else:
            mean=self.dataset.mean_cam_center

        # Create network using the state dict of the pretrained encoder.
        encoder_state_dict = torch.load(self.options.encoder_path, map_location="cpu")
        self.regressor = Regressor.create_from_encoder(
            encoder_state_dict,
            mean=mean,
            num_head_blocks=self.options.num_head_blocks,
            use_homogeneous=self.options.use_homogeneous,
            global_feat_dim=self.global_feat_dim if self.options.global_feat else 0,
            head_channels=self.options.head_channels,
            mlp_ratio=self.options.mlp_ratio,
        )
        if dist.get_rank() == 0:
            _logger.info(f"Loaded pretrained encoder from: {self.options.encoder_path}")

        self.regressor = self.regressor.to(self.device)
        torch.backends.cudnn.benchmark = True
        self.regressor.train()
        self.regressor.heads = DistributedDataParallel(self.regressor.heads, device_ids=[device_id],output_device=device_id)

        # Setup optimization parameters.
        self.optimizer = optim.AdamW(self.regressor.heads.parameters(), lr=self.options.learning_rate_min)

        # Setup learning rate scheduler.
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                       max_lr=self.options.learning_rate_max,
                                                       total_steps=self.options.max_iterations,
                                                       cycle_momentum=False)

        # Gradient scaler in case we train with half precision.
        self.scaler = GradScaler(enabled=self.options.use_half)

        # Generate grid of target reprojection pixel positions.
        pixel_grid_2HW = get_pixel_grid(self.regressor.OUTPUT_SUBSAMPLE)
        self.pixel_grid_2HW = pixel_grid_2HW.to(self.device)

        # Compute total number of iterations.
        self.iterations = self.options.max_iterations
        self.iterations_output = 100 # print loss every n iterations, and (optionally) write a visualisation frame

        # Setup reprojection loss function.
        self.repro_loss = ReproLoss(
            total_iterations=self.options.max_iterations,
            soft_clamp=self.options.repro_loss_soft_clamp,
            soft_clamp_min=self.options.repro_loss_soft_clamp_min,
            type=self.options.repro_loss_type,
            circle_schedule=(self.options.repro_loss_schedule == 'circle')
        )

        # Will be filled at the beginning of the training process.
        self.training_buffer = None

        # Generate video of training process
        if self.options.render_visualization and dist.get_rank() == 0:
            import ace_vis_util as vutil
            from ace_visualizer import ACEVisualizer
            # infer rendering folder from map file name
            target_path = vutil.get_rendering_target_path(
                self.options.render_target_path,
                self.options.output_map_file)
            self.ace_visualizer = ACEVisualizer(
                target_path,
                self.options.render_flipped_portrait,
                self.options.render_map_depth_filter,
                mapping_vis_error_threshold=self.options.render_map_error_threshold)
        else:
            self.ace_visualizer = None

    def train(self):
        """
        Main training method.

        Fills a feature buffer using the pretrained encoder and subsequently trains a scene coordinate regression head.
        """

        if self.ace_visualizer is not None:

            # Setup the ACE render pipeline.
            self.ace_visualizer.setup_mapping_visualisation(
                self.dataset.pose_values,
                self.dataset.rgb_files,
                self.iterations // self.iterations_output + 1,
                self.options.render_camera_z_offset
            )

        creating_buffer_time = 0.
        training_time = 0.

        self.training_start = time.time()

        # Create training buffer.
        buffer_start_time = time.time()
        self.create_training_buffer()
        buffer_end_time = time.time()
        creating_buffer_time += buffer_end_time - buffer_start_time
        _logger.info(f"Filled training buffer in {buffer_end_time - buffer_start_time:.1f}s.")

        # Train the regression head.
        self.epoch=0
        while self.iteration<self.options.max_iterations:
            epoch_start_time = time.time()
            self.run_epoch()
            training_time += time.time() - epoch_start_time
            self.epoch+=1

        # Save trained model if main process.
        if dist.get_rank() == 0:
            self.save_model()
            end_time = time.time()
            _logger.info(f'Done without errors. '
                        f'Creating buffer time: {creating_buffer_time:.1f} seconds. '
                        f'Training time: {training_time:.1f} seconds. '
                        f'Total time: {end_time - self.training_start:.1f} seconds.')

        if self.ace_visualizer is not None:

            # Finalize the rendering by animating the fully trained map.
            vis_dataset = CamLocDataset(
                root_dir=self.options.scene / "train",
                mode=0,
                use_half=self.options.use_half,
                image_height=self.options.image_resolution,
                augment=False,
                feat_name=self.options.feat_name,
                ) # No data augmentation when visualizing the map

            vis_dataset_loader = torch.utils.data.DataLoader(
                vis_dataset,
                shuffle=False, # Process data in order for a growing effect later when rendering
                num_workers=self.num_data_loader_workers)

            self.ace_visualizer.finalize_mapping(self.regressor, vis_dataset_loader)

    def write_progress(self, stage, completed, total, **extra):
        path = str(self.options.output_map_file) + '.progress.json'
        data = dict(stage=stage, completed=completed, total=total,
                    time=time.time(), started=self.training_start,
                    gpu_peak_bytes=torch.cuda.max_memory_allocated(), **extra)
        with open(path + '.tmp', 'w') as handle:
            json.dump(data, handle, indent=2)
        os.replace(path + '.tmp', path)

    def create_training_buffer(self):
        # Disable benchmarking, since we have variable tensor sizes.
        torch.backends.cudnn.benchmark = False

        # Sampler.
        if self.options.scene.suffix=='.txt':
            # use weighted sampler make each subset have same number of samples
            ds_lens=[len(ds) for ds in self.dataset.datasets]
            weights=[]
            for ds_len in ds_lens:
                weights.extend([1./ds_len]*ds_len)
            batch_sampler = WeightedRandomSampler(weights, max(ds_lens),
                                                replacement=True,generator=self.batch_generator)
        else:
            batch_sampler = sampler.RandomSampler(self.dataset, generator=self.batch_generator)

        # Used to seed workers in a reproducible manner.
        def seed_worker(worker_id):
            # Different seed per epoch. Initial seed is generated by the main process consuming one random number from
            # the dataloader generator.
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        # Batching is handled at the dataset level (the dataset __getitem__ receives a list of indices, because we
        # need to rescale all images in the batch to the same size).
        training_dataloader = DataLoader(dataset=self.dataset,
                                         sampler=batch_sampler,
                                         batch_size=1,
                                         drop_last=False,
                                         worker_init_fn=seed_worker,
                                         generator=self.loader_generator,
                                         pin_memory=True,
                                         num_workers=self.num_data_loader_workers,
                                         persistent_workers=self.num_data_loader_workers > 0,
                                         timeout=60 if self.num_data_loader_workers > 0 else 0,
                                         )
        
        if dist.get_rank() == 0:
            _logger.info("Starting creation of the training buffer.")
        feature_dim=self.regressor.feature_dim
        # Create a training buffer that lives on the GPU.
        self.lidar_folder = self.options.scene / 'train/lidar_world'
        if not self.lidar_folder.is_dir():
            raise ValueError('Missing training-only LiDAR supervision directory')
        self.training_buffer = {
            'lidar_camera': torch.empty((self.options.training_buffer_size, 3), dtype=torch.float32, device='cpu'),
            'lidar_valid': torch.empty((self.options.training_buffer_size, 1), dtype=torch.float32, device='cpu'),
            'features': torch.empty((self.options.training_buffer_size, feature_dim),
                                    dtype=(torch.float32, torch.float16)[self.options.use_half], device='cpu'),
            'target_px': torch.empty((self.options.training_buffer_size, 2), dtype=torch.float32, device='cpu'),
            'gt_poses_inv': torch.empty((self.options.training_buffer_size, 3, 4), dtype=torch.float32,
                                        device='cpu'),
            'intrinsics': torch.empty((self.options.training_buffer_size, 3, 3), dtype=torch.float32,
                                      device='cpu'),
            'intrinsics_inv': torch.empty((self.options.training_buffer_size, 3, 3), dtype=torch.float32,
                                          device='cpu'),
            'img_idx': torch.empty((self.options.training_buffer_size,), dtype=torch.int64, device='cpu'),
        }

        # Features are computed in evaluation mode.
        self.regressor.eval()

        # The encoder is pretrained, so we don't compute any gradient.
        with torch.no_grad():
            # Iterate until the training buffer is full.
            buffer_idx = 0
            dataset_passes = 0

            while buffer_idx < self.options.training_buffer_size:
                dataset_passes += 1
                for image_B1HW, image_mask_B1HW, gt_pose_B44, gt_pose_inv_B44, intrinsics_B33, intrinsics_inv_B33, _, image_paths, _,idx in training_dataloader:

                    # Copy to device.
                    image_B1HW = image_B1HW.to(self.device, non_blocking=True)
                    image_mask_B1HW = image_mask_B1HW.to(self.device, non_blocking=True)
                    gt_pose_inv_B44 = gt_pose_inv_B44.to(self.device, non_blocking=True)
                    intrinsics_B33 = intrinsics_B33.to(self.device, non_blocking=True)
                    intrinsics_inv_B33 = intrinsics_inv_B33.to(self.device, non_blocking=True)

                    # Compute image features.
                    with autocast(enabled=self.options.use_half):
                        features_BCHW = self.regressor.get_features(image_B1HW)

                    # Dimensions after the network's downsampling.
                    B, C, H, W = features_BCHW.shape

                    # The image_mask needs to be downsampled to the actual output resolution and cast to bool.
                    image_mask_B1HW = grid_valid_region(image_mask_B1HW, H, W)

                    # If the current mask has no valid pixels, continue.
                    if image_mask_B1HW.sum() == 0:
                        raise ValueError('Image has no valid GLACE output pixel centers')

                    # Create a tensor with the pixel coordinates of every feature vector.
                    pixel_positions_B2HW = self.pixel_grid_2HW[:, :H, :W].clone()  # It's 2xHxW (actual H and W) now.
                    pixel_positions_B2HW = pixel_positions_B2HW[None]  # 1x2xHxW
                    pixel_positions_B2HW = pixel_positions_B2HW.expand(B, 2, H, W)  # Bx2xHxW

                    # Bx3x4 -> Nx3x4 (for each image, repeat pose per feature)
                    gt_pose_inv = gt_pose_inv_B44[:, :3]
                    gt_pose_inv = gt_pose_inv.unsqueeze(1).expand(B, H * W, 3, 4).reshape(-1, 3, 4)

                    # Bx3x3 -> Nx3x3 (for each image, repeat intrinsics per feature)
                    intrinsics = intrinsics_B33.unsqueeze(1).expand(B, H * W, 3, 3).reshape(-1, 3, 3)
                    intrinsics_inv = intrinsics_inv_B33.unsqueeze(1).expand(B, H * W, 3, 3).reshape(-1, 3, 3)

                    def normalize_shape(tensor_in):
                        """Bring tensor from shape BxCxHxW to NxC"""
                        return tensor_in.transpose(0, 1).flatten(1).transpose(0, 1)

                    if B != 1:
                        raise ValueError('LiDAR supervision currently requires buffer image batch size one')
                    lidar_camera, lidar_valid = load_camera_targets(
                        self.lidar_folder, image_paths[0],
                        normalize_shape(pixel_positions_B2HW).cpu().numpy(),
                        intrinsics_B33[0].cpu().numpy(), gt_pose_inv_B44[0].cpu().numpy(),
                        image_B1HW.shape[-2], image_B1HW.shape[-1])
                    batch_data = {
                        'lidar_camera': torch.from_numpy(lidar_camera).to(self.device),
                        'lidar_valid': torch.from_numpy(lidar_valid[:, None]).to(self.device),
                        'features': normalize_shape(features_BCHW),
                        'target_px': normalize_shape(pixel_positions_B2HW),
                        'gt_poses_inv': gt_pose_inv,
                        'intrinsics': intrinsics,
                        'intrinsics_inv': intrinsics_inv
                    }

                    # Turn image mask into sampling weights (all equal).
                    image_mask_B1HW = image_mask_B1HW.float()
                    image_mask_N1 = normalize_shape(image_mask_B1HW)

                    # Over-sample according to image mask.
                    features_to_select = self.options.samples_per_image * B
                    features_to_select = min(features_to_select, self.options.training_buffer_size - buffer_idx)

                    # Sample indices uniformly, with replacement.
                    sample_idxs = torch.multinomial(image_mask_N1.view(-1),
                                                    features_to_select,
                                                    replacement=True,
                                                    generator=self.sampling_generator)

                    # Select the data to put in the buffer.
                    for k in batch_data:
                        batch_data[k] = batch_data[k][sample_idxs]

                    # Write to training buffer. Start at buffer_idx and end at buffer_offset - 1.
                    buffer_offset = buffer_idx + features_to_select
                    for k in batch_data:
                        self.training_buffer[k][buffer_idx:buffer_offset] = batch_data[k].cpu()
                    self.training_buffer['img_idx'][buffer_idx:buffer_offset]=idx.item()

                    buffer_idx = buffer_offset
                    if buffer_idx % (1024 * 100) == 0 or buffer_idx == self.options.training_buffer_size:
                        self.write_progress("buffer", buffer_idx, self.options.training_buffer_size)
                    if buffer_idx >= self.options.training_buffer_size:
                        break

        buffer_memory = sum([v.element_size() * v.nelement() for k, v in self.training_buffer.items()])
        buffer_memory /= 1024 * 1024 * 1024
        if dist.get_rank() == 0:
            _logger.info(f"Created buffer of {buffer_memory:.2f}GB with {dataset_passes} passes over the training data.")
        counts = torch.bincount(self.training_buffer["img_idx"], minlength=len(self.dataset))
        if torch.any(counts != self.options.samples_per_image):
            raise RuntimeError("Buffer does not contain the requested samples for every image")
        self.write_progress("buffer_complete", int(counts.sum()), self.options.training_buffer_size,
                            lidar_fraction=float(self.training_buffer["lidar_valid"].mean()))
        if self.training_buffer["lidar_valid"].sum() == 0:
            raise ValueError("No sparse LiDAR supervision reached the buffer")
        self.regressor.train()

    def run_epoch(self):
        """
        Run one epoch of training, shuffling the feature buffer and iterating over it.
        """
        # Enable benchmarking since all operations work on the same tensor size.
        torch.backends.cudnn.benchmark = True

        # Shuffle indices.
        random_indices = torch.randperm(self.options.training_buffer_size, generator=self.training_generator)

        # Iterate with mini batches.
        for batch_start in range(0, self.options.training_buffer_size, self.options.batch_size):
            batch_end = batch_start + self.options.batch_size

            # Drop last batch if not full.
            if batch_end > self.options.training_buffer_size:
                continue

            # Sample indices.
            random_batch_indices = random_indices[batch_start:batch_end]

            # create features batch by concatenating global features and local features
            if self.options.global_feat:
                features_batch=torch.empty((self.options.batch_size,self.regressor.feature_dim + self.dataset.global_feat_dim ),
                                           dtype=self.training_buffer['features'].dtype,device=self.device)
                features_batch[:,:self.dataset.global_feat_dim]=self.global_feats[self.training_buffer['img_idx'][random_batch_indices]]
                features_batch[:,self.dataset.global_feat_dim:]=self.training_buffer['features'][random_batch_indices]
            else:
                features_batch=self.training_buffer['features'][random_batch_indices]

            # Call the training step with the sampled features and relevant metadata.
            self.training_step(
                features_batch,
                self.training_buffer['target_px'][random_batch_indices].to(self.device).contiguous(),
                self.training_buffer['gt_poses_inv'][random_batch_indices].to(self.device).contiguous(),
                self.training_buffer['intrinsics'][random_batch_indices].to(self.device).contiguous(),
                self.training_buffer['intrinsics_inv'][random_batch_indices].to(self.device).contiguous(),
                self.training_buffer['lidar_camera'][random_batch_indices].to(self.device),
                self.training_buffer['lidar_valid'][random_batch_indices].to(self.device)
            )
            self.iteration += 1
            if self.iteration >= self.options.max_iterations:
                break

    def training_step(self, features_bC, target_px_b2, gt_inv_poses_b34, Ks_b33, invKs_b33, lidar_camera, lidar_valid):
        """
        Run one iteration of training, computing the reprojection error and minimising it.
        """
        if self.options.feat_noise_std > 0.0:
            # first self.global_feat_dim is global feature, add gaussian noise and normalize to unit length
            features_bC[:,:self.global_feat_dim]+=torch.empty_like(features_bC[:,:self.global_feat_dim]).normal_(
                                                            mean=0,std=self.options.feat_noise_std,generator=self.gn_generator)
            features_bC[:,:self.global_feat_dim]=torch.nn.functional.normalize(features_bC[:,:self.global_feat_dim],dim=1)
            

        batch_size = features_bC.shape[0]
        channels = features_bC.shape[1]

        # Reshape to a "fake" BCHW shape, since it's faster to run through the network compared to the original shape.
        features_bCHW = features_bC[None, None, ...].view(-1, 16, 32, channels).permute(0, 3, 1, 2)
        with autocast(enabled=self.options.use_half):
            pred_scene_coords_b3HW = self.regressor.get_scene_coordinates(features_bCHW)

        # Back to the original shape. Convert to float32 as well.
        pred_scene_coords_b31 = pred_scene_coords_b3HW.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(-1).float()

        # Make 3D points homogeneous so that we can easily matrix-multiply them.
        pred_scene_coords_b41 = to_homogeneous(pred_scene_coords_b31)

        # Scene coordinates to camera coordinates.
        pred_cam_coords_b31 = torch.bmm(gt_inv_poses_b34, pred_scene_coords_b41)

        # Project scene coordinates.
        pred_px_b31 = torch.bmm(Ks_b33, pred_cam_coords_b31)

        # Avoid division by zero.
        # Note: negative values are also clamped at +self.options.depth_min. The predicted pixel would be wrong,
        # but that's fine since we mask them out later.
        pred_px_b31[:, 2].clamp_(min=self.options.depth_min)

        # Dehomogenise.
        pred_px_b21 = pred_px_b31[:, :2] / pred_px_b31[:, 2, None]

        # Measure reprojection error.
        reprojection_error_b2 = pred_px_b21.squeeze() - target_px_b2
        reprojection_error_b1 = torch.norm(reprojection_error_b2, dim=1, keepdim=True, p=1)

        #
        # Compute masks used to ignore invalid pixels.
        #
        # Predicted coordinates behind or close to camera plane.
        depth = pred_cam_coords_b31[:, 2] 
        invalid_min_depth_b1 = depth < self.options.depth_min
        # Very large reprojection errors.
        invalid_repro_b1 = reprojection_error_b1 > self.options.repro_loss_hard_clamp
        # Predicted coordinates beyond max distance.
        invalid_max_depth_b1 = depth > self.options.depth_max

        # Invalid mask is the union of all these. Valid mask is the opposite.
        invalid_mask_b1 = (invalid_min_depth_b1 | invalid_repro_b1 | invalid_max_depth_b1)
        valid_mask_b1 = ~invalid_mask_b1

        # Reprojection error for all valid scene coordinates.
        valid_reprojection_error_b1 = reprojection_error_b1[valid_mask_b1]
        # Compute the loss for valid predictions.
        loss_valid = self.repro_loss.compute(valid_reprojection_error_b1, self.iteration)

        # Handle the invalid predictions: generate proxy coordinate targets with constant depth assumption.
        pixel_grid_crop_b31 = to_homogeneous(target_px_b2.unsqueeze(2))
        target_camera_coords_b31 = self.options.depth_target * torch.bmm(invKs_b33, pixel_grid_crop_b31)

        # Compute the distance to target camera coordinates.
        invalid_mask_b11 = invalid_mask_b1.unsqueeze(2)
        loss_invalid = torch.abs(target_camera_coords_b31 - pred_cam_coords_b31).masked_select(invalid_mask_b11).sum()

        # Final loss is the sum of all 2.
        loss = loss_valid + loss_invalid
        lidar_error = torch.nn.functional.smooth_l1_loss(
            pred_cam_coords_b31.squeeze(-1), lidar_camera, reduction='none').sum(1)
        lidar_loss = (lidar_error * lidar_valid.reshape(-1)).sum()
        loss = loss + 1.0 * lidar_loss
        loss /= batch_size
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")

        # We need to check if the step actually happened, since the scaler might skip optimisation steps.
        old_optimizer_step = self.optimizer._step_count

        # Optimization steps.
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.iteration > 0 and self.iteration % 5000 == 0 and dist.get_rank() == 0:
            self.save_model()
        if self.iteration % self.iterations_output == 0 or self.iteration + 1 == self.options.max_iterations:
            self.write_progress("training", self.iteration + 1, self.options.max_iterations,
                                loss=float(loss), optimizer_steps=int(self.optimizer._step_count),
                                valid_fraction=float(valid_mask_b1.float().mean()),
                                lidar_fraction=float(lidar_valid.mean()),
                                lidar_loss_per_sample=float(lidar_loss / batch_size))
            # Print status.
            time_since_start = time.time() - self.training_start
            fraction_valid = float(valid_mask_b1.sum() / batch_size)
            # median_depth = float(pred_cam_coords_b31[:, 2].median())
            if dist.get_rank() == 0:
                _logger.info(f'Iter {self.iteration:6d}|{self.options.max_iterations:06d}, '
                         f'Loss: {loss:.1f}, Valid: {fraction_valid * 100:.1f}%, Time: {time_since_start:.2f}s')

            if self.ace_visualizer is not None:
                vis_scene_coords = pred_scene_coords_b31.detach().cpu().squeeze().numpy()
                vis_errors = reprojection_error_b1.detach().cpu().squeeze().numpy()
                self.ace_visualizer.render_mapping_frame(vis_scene_coords, vis_errors)

        # Only step if the optimizer stepped and if we're not over-stepping the total_steps supported by the scheduler.
        if old_optimizer_step < self.optimizer._step_count < self.scheduler.total_steps:
            self.scheduler.step()

    def save_model(self):
        # NOTE: This would save the whole regressor (encoder weights included) in full precision floats (~30MB).
        # torch.save(self.regressor.state_dict(), self.options.output_map_file)

        # This saves just the head weights as half-precision floating point numbers for a total of ~4MB, as mentioned
        # in the paper. The scene-agnostic encoder weights can then be loaded from the pretrained encoder file.
        head_state_dict = self.regressor.heads.module.state_dict()
        head_state_dict = {k: v.detach().float().cpu() for k, v in head_state_dict.items()}
        temporary = str(self.options.output_map_file) + ".tmp"
        torch.save(head_state_dict, temporary)
        os.replace(temporary, self.options.output_map_file)
        _logger.info(f"Saved trained head weights to: {self.options.output_map_file}")

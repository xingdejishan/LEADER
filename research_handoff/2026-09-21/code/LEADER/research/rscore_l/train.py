from copy import deepcopy
from dataclasses import fields
import json

import torch
import time

from scrstudio.configs.method_configs import method_configs
from scrstudio.scripts.train import main
from scrstudio.engine.trainer import Trainer

from .datamanager import GeometryBufferConfig
from .losses import PersistentGeometryLossConfig, FiniteRobustLossConfig
from .prepare import save_json


class AuditedTrainer(Trainer):
    def train_iteration(self, step):
        if not hasattr(self, 'audit_started'):
            self.audit_started = time.perf_counter()
            self.optimizer_updates = 0
            self.audit_start_step = step
        before = self.grad_scaler.get_scale()
        result = super().train_iteration(step)
        self.optimizer_updates += self.grad_scaler.get_scale() >= before
        if step % 100 == 0 or step == self.config.max_num_iterations-1:
            loss = float(result[0].detach())
            if not torch.isfinite(result[0]):
                raise FloatingPointError(f'Nonfinite training loss at step {step}')
            completed = step - self.audit_start_step + 1
            elapsed = time.perf_counter() - self.audit_started
            save_json(self.config.get_base_dir() / 'training_progress.json', dict(step=step, total=self.config.max_num_iterations,
                loss=loss, successful_optimizer_updates_this_run=self.optimizer_updates, resumed_start_step=self.audit_start_step,
                elapsed_seconds=elapsed, seconds_per_step=elapsed/completed,
                estimated_remaining_seconds=elapsed/completed*(self.config.max_num_iterations-step-1),
                peak_memory_mb=torch.cuda.max_memory_allocated()/2**20))
        return result


def finite_reprojection(config):
    config = deepcopy(config)
    original = config.robust_loss
    config.robust_loss = FiniteRobustLossConfig(**{f.name: getattr(original, f.name) for f in fields(original) if f.name != '_target'})
    return config


def make_config(data, output, variant, iterations=10000, exclude_session=''):
    config = deepcopy(method_configs['depth-scrfacto' if variant == 'depth' else 'scrfacto'])
    config._target = AuditedTrainer
    config.method_name = variant
    config.experiment_name = 'nclt-local'
    config.timestamp = 'fixed-2089'
    config.output_dir = output
    config.data = data
    config.machine.num_devices = 1
    config.machine.seed = 2089
    config.max_num_iterations = iterations
    config.gradient_accumulation_steps = 2
    config.steps_per_eval_all_images = 0
    config.steps_per_save = 1000
    config.logging.steps_per_log = 100
    config.pipeline.datamanager = GeometryBufferConfig(data=data, batch_size=4096,
        graph='lidar_overlap.npz' if variant.startswith('lidar') else 'pose_overlap.npz',
        encoding='lidar_n2c.pt' if variant.startswith('lidar') else 'pose_n2c.pt', exclude_session=exclude_session)
    if variant == 'lidar-multiframe':
        config.pipeline.datamanager.geometry_folder = 'multiframe_training_features'
    config.pipeline.model.max_num_iterations = iterations
    config.pipeline.model.losses = [finite_reprojection(loss) for loss in config.pipeline.model.losses]
    config.optimizers['head']['scheduler'].max_steps = iterations
    if variant in ('geometry', 'lidar', 'lidar-multiframe') or variant.startswith('lidar-fold'):
        config.pipeline.model.losses = [PersistentGeometryLossConfig(
            final_reprojection=finite_reprojection(method_configs['scrfacto'].pipeline.model.losses[0]),
            coarse_reprojection=finite_reprojection(method_configs['scrfacto'].pipeline.model.losses[1]))]
    return config


def train(data, output, variant, iterations=10000, exclude_session=''):
    config = make_config(data, output, variant, iterations, exclude_session)
    config.save_only_head = False
    existing = config.get_base_dir() / 'scrstudio_models'
    progress_path = config.get_base_dir() / 'training_progress.json'
    if progress_path.exists() and (existing / 'head.pt').exists():
        progress = json.loads(progress_path.read_text())
        if progress['step'] == iterations-1 and progress['total'] == iterations:
            return
    if existing.exists() and list(existing.glob('step-*.ckpt')):
        config.load_dir = existing
    main(config)
    progress = json.loads((config.get_base_dir() / 'training_progress.json').read_text())
    if progress['step'] != iterations - 1:
        raise RuntimeError('Training returned before the requested final iteration')
    head = torch.load(existing / 'head.pt', weights_only=True)
    if not all(torch.isfinite(value).all() for value in head.values() if torch.is_tensor(value)):
        raise FloatingPointError('Saved scene head contains nonfinite values')


def train_graph(data, output, graph):
    config = deepcopy(method_configs['node2vec'])
    config.data = data
    config.output_dir = output
    config.experiment_name = 'nclt-local'
    config.method_name = 'node2vec-' + graph
    config.timestamp = 'fixed-2089'
    config.machine.seed = 2089
    config.machine.num_devices = 1
    config.steps_per_eval_all_images = 0
    config.pipeline.model.graph = graph + '_overlap.npz'
    config.pipeline.datamanager.batch_size = 64 if graph == 'lidar' else 32
    config.gradient_accumulation_steps = 4 if graph == 'lidar' else 8
    config.logging.steps_per_log = 500
    config.save_only_head = False
    config.steps_per_save = 1000
    target = data / 'train' / (graph + '_n2c.pt')
    if target.exists():
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        events = EventAccumulator(str(config.get_base_dir()), size_guidance={'scalars': 1}).Reload()
        timing = events.Scalars('Train Rays / Sec')
        if timing and timing[-1].step == config.max_num_iterations - 1:
            return
    checkpoint_dir = config.get_base_dir() / 'scrstudio_models'
    if checkpoint_dir.exists() and list(checkpoint_dir.glob('step-*.ckpt')):
        config.load_dir = checkpoint_dir
    main(config)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events = EventAccumulator(str(config.get_base_dir()), size_guidance={'scalars': 1}).Reload()
    timing = events.Scalars('Train Rays / Sec')
    if not timing or timing[-1].step != config.max_num_iterations - 1:
        raise RuntimeError('Node2Vec returned before the requested final iteration')
    import shutil
    shutil.copy2(config.get_base_dir() / 'scrstudio_models/head.pt', target)

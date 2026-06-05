from __future__ import annotations

import os
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.seq2seq_model import build_model as build_trajectory_model, HybridLoss
from models.trajectory_dataset import create_dataloaders
from utils.config import deep_merge, load_config_bundle, validate_time_config_contract

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("Warning: TensorBoard is not available; logs will be printed to console only.")


def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_teacher_forcing_ratio(epoch: int, config: dict) -> float:

    if not config['teacher_forcing']['enabled']:
        return 0.0

    strategy = config['teacher_forcing']['strategy']
    initial_ratio = config['teacher_forcing']['initial_ratio']
    final_ratio = config['teacher_forcing']['final_ratio']

    if strategy == 'inverse_sigmoid':
        k = config['teacher_forcing']['k']
        total_epochs = config['training']['epochs']
        x = k * (epoch / total_epochs - 0.5)
        sigmoid = 1 / (1 + np.exp(-x))
        ratio = initial_ratio - (initial_ratio - final_ratio) * sigmoid

    elif strategy == 'linear' or strategy == 'linear_decay':
        decay_steps = config['teacher_forcing']['decay_steps']
        ratio = initial_ratio - (initial_ratio - final_ratio) * min(epoch / decay_steps, 1.0)

    elif strategy == 'exponential':
        decay_steps = max(float(config['teacher_forcing'].get('decay_steps', 1)), 1.0)
        ratio = final_ratio + (initial_ratio - final_ratio) * np.exp(-epoch / decay_steps)

    else:
        ratio = initial_ratio

    return max(final_ratio, min(initial_ratio, ratio))


def compute_regression_metrics(pred_pos: torch.Tensor,
                               target_seq: torch.Tensor,
                               pred_len: int,
                               fps: float) -> Dict[str, float]:

    target_pos = target_seq[:, :, 0:2]
    pos_error = torch.linalg.norm(pred_pos - target_pos, dim=-1)

    ade = pos_error.mean()
    fde = pos_error[:, -1].mean()
    rmse_final = torch.sqrt(torch.mean(pos_error[:, -1] ** 2))

    if fps <= 0:
        raise ValueError(f"fps must be positive for RMSE horizon computation, got {fps}.")
    horizon_5s_frames = int(round(5.0 * fps))
    if horizon_5s_frames <= 0:
        raise ValueError(
            f"RMSE@5s horizon must resolve to a positive frame count, got {horizon_5s_frames} from fps={fps}."
        )
    if pred_len < horizon_5s_frames:
        raise ValueError(
            f"Configured pred_len={pred_len} is shorter than the 5s RMSE horizon "
            f"({horizon_5s_frames} frames at fps={fps})."
        )
    if pos_error.shape[1] < horizon_5s_frames:
        raise ValueError(
            f"Target sequence length={pos_error.shape[1]} is shorter than the 5s RMSE horizon "
            f"({horizon_5s_frames} frames at fps={fps})."
        )

    horizon_5s_idx = horizon_5s_frames - 1
    rmse_5s = torch.sqrt(torch.mean(pos_error[:, horizon_5s_idx] ** 2))

    return {
        'ade': float(ade.item()),
        'fde': float(fde.item()),
        'rmse_final': float(rmse_final.item()),
        'rmse_5s': float(rmse_5s.item()),
    }


class Trainer:


    def __init__(self, config: dict, data_config: dict, distributed: bool = False, rank: int = 0, world_size: int = 1, local_rank: int = 0):
        self.config = config
        self.data_config = data_config
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.dynamic_geo_enabled = bool(self.config.get('model', {}).get('dynamic_geo', {}).get('enabled', False))
        self.scene_name_to_id = self._build_scene_name_to_id() if self.dynamic_geo_enabled else {}

        set_seed(config['seed'])
        self.setup_device()
        self.model = self.build_model()
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.criterion = self.build_loss()
        scaler_device = 'cuda' if self.device.type == 'cuda' else 'cpu'
        self.scaler = GradScaler(
            scaler_device,
            enabled=bool(config['distributed'].get('mixed_precision', True)) and self.device.type == 'cuda'
        )
        self.setup_logging()

        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.checkpoint_metric_name = str(self.config['checkpoint'].get('metric', 'val_loss')).lower()
        self.scheduler_monitor_name = str(self.config.get('scheduler', {}).get('monitor', self.checkpoint_metric_name)).lower()
        self.early_stopping_monitor_name = str(
            self.config.get('training', {}).get('early_stopping', {}).get('monitor', self.checkpoint_metric_name)
        ).lower()
        self.best_monitor_value = float('inf')
        early_stopping_cfg = self.config.get('training', {}).get('early_stopping', {})
        tf_cfg = self.config.get('teacher_forcing', {})
        explicit_start_epoch = early_stopping_cfg.get('start_epoch', None)
        if explicit_start_epoch is not None:
            self.early_stopping_start_epoch = max(0, int(explicit_start_epoch))
        else:
            tf_strategy = str(tf_cfg.get('strategy', '')).lower()
            if bool(tf_cfg.get('enabled', False)) and tf_strategy in {'linear', 'linear_decay'}:
                self.early_stopping_start_epoch = max(0, int(tf_cfg.get('decay_steps', 0)))
            else:
                self.early_stopping_start_epoch = 0
        self.obs_len = int(self.data_config.get('sliding_window', {}).get('obs_len', 0))
        self.pred_len = int(self.data_config.get('sliding_window', {}).get('pred_len', 0))
        try:
            self.fps = float(self.data_config.get('rts_smoother', {}).get('fps', 25))
        except (TypeError, ValueError):
            self.fps = 25.0
        self.rmse_5s_frames = int(round(5.0 * self.fps)) if self.fps > 0 else 0
        if self.fps <= 0:
            raise ValueError(f"Configured fps must be positive, got {self.fps}.")
        if self.pred_len < self.rmse_5s_frames:
            raise ValueError(
                f"Configured pred_len={self.pred_len} is shorter than the 5s RMSE horizon "
                f"({self.rmse_5s_frames} frames at fps={self.fps})."
            )
        self.rmse_5s_seconds = self.rmse_5s_frames / self.fps

    def _build_scene_name_to_id(self) -> dict[str, int]:
        ordered_scenes: list[str] = []

        def _append(scene_name) -> None:
            scene_name = str(scene_name).strip()
            if scene_name and scene_name not in ordered_scenes:
                ordered_scenes.append(scene_name)

        for scene_name in self.config.get('scenes', []) or []:
            _append(scene_name)
        split_cfg = self.data_config.get('split_by_scene', {}) if isinstance(self.data_config, dict) else {}
        for key in ('train_scenes', 'val_scenes', 'test_scenes'):
            for scene_name in split_cfg.get(key, []) or []:
                _append(scene_name)

        if not ordered_scenes:
            raise ValueError('model.dynamic_geo.enabled=true but no scene names are configured.')

        return {scene_name: idx for idx, scene_name in enumerate(ordered_scenes)}

    def _encode_scene_batch(self, scene_batch) -> torch.Tensor:
        if isinstance(scene_batch, str):
            scene_names = [scene_batch]
        else:
            scene_names = [str(scene_name) for scene_name in scene_batch]
        try:
            scene_ids = [self.scene_name_to_id[scene_name] for scene_name in scene_names]
        except KeyError as exc:
            known_scenes = sorted(self.scene_name_to_id)
            raise KeyError(
                f'Encountered unknown scene "{exc.args[0]}" while dynamic_geo is enabled. '
                f'Known scenes: {known_scenes}'
            ) from exc
        return torch.tensor(scene_ids, dtype=torch.long, device=self.device)

    def _forward_batch(self, batch: dict, teacher_forcing_ratio: float):
        obs_seq = batch['obs_seq'].to(self.device)
        pred_seq = batch['pred_seq'].to(self.device)
        if not self.dynamic_geo_enabled:
            pred_pos, pred_vel = self.model(obs_seq, pred_seq, teacher_forcing_ratio)
            return obs_seq, pred_seq, pred_pos, pred_vel

        pred_pos, pred_vel = self.model(
            obs_seq,
            pred_seq,
            teacher_forcing_ratio,
            scene_ids=self._encode_scene_batch(batch['scene']),
            anchor_abs=batch['anchor_abs'].to(self.device),
            is_flipped=batch['is_flipped'].to(self.device),
        )
        return obs_seq, pred_seq, pred_pos, pred_vel

    def setup_device(self):
        training_cfg = self.config.get('training', {})
        requested_device = str(training_cfg.get('device', 'auto')).strip().lower()
        prefer_multi_gpu = bool(training_cfg.get('prefer_multi_gpu', True))
        min_gpus_for_multi = int(training_cfg.get('min_gpus_for_multi', 2))
        visible_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

        self.use_data_parallel = False
        self.use_ddp = False

        if requested_device == 'cpu' or visible_gpu_count == 0:
            self.device = torch.device('cpu')
            self.gpu_ids = []
            if self.rank == 0:
                if visible_gpu_count == 0:
                    print("No available GPU detected; falling back to CPU.")
                else:
                    print("CPU requested by configuration; GPU will be skipped.")
            return

        if self.distributed and dist.is_initialized():
            try:
                local_rank = int(os.environ.get('LOCAL_RANK', self.local_rank))
            except Exception:
                local_rank = self.local_rank
            torch.cuda.set_device(local_rank)
            self.device = torch.device(f'cuda:{local_rank}')
            self.gpu_ids = [local_rank]
            self.use_ddp = True
            if self.rank == 0:
                print(f"DDP: training on GPU local_rank={local_rank}")
            return

        if requested_device.startswith('cuda:'):
            requested_index = int(requested_device.split(':', 1)[1])
            if requested_index < visible_gpu_count:
                self.device = torch.device(f'cuda:{requested_index}')
                self.gpu_ids = [requested_index]
                if self.rank == 0:
                    print(f"Using configured single GPU {requested_index}")
            else:
                self.device = torch.device('cuda:0')
                self.gpu_ids = [0]
                if self.rank == 0:
                    print(f"Configured GPU {requested_index} is unavailable; falling back to GPU 0.")
            return

        if prefer_multi_gpu and visible_gpu_count >= min_gpus_for_multi:
            self.device = torch.device('cuda:0')
            self.gpu_ids = list(range(visible_gpu_count))
            self.use_data_parallel = True
            if self.rank == 0:
                gpu_names = [torch.cuda.get_device_name(i) for i in self.gpu_ids]
                print(f"Detected {visible_gpu_count} GPUs; enabling DataParallel: {self.gpu_ids}")
                print(f"GPU models: {gpu_names}")
            return

        self.device = torch.device('cuda:0')
        self.gpu_ids = [0]
        if self.rank == 0:
            if prefer_multi_gpu:
                print(f"Not enough GPUs for multi-GPU training (detected {visible_gpu_count}); falling back to GPU 0.")
            else:
                print("Using single GPU 0.")

    def build_model(self) -> nn.Module:
        model = build_trajectory_model(self.config)
        if self.rank == 0:
            print("Building trajectory prediction model")

        model = model.to(self.device)

        if self.use_ddp and self.config['distributed']['enabled']:
            model = DDP(
                model,
                device_ids=[self.gpu_ids[0]] if self.gpu_ids else None,
                output_device=self.gpu_ids[0] if self.gpu_ids else None
            )
        elif self.use_data_parallel and len(self.gpu_ids) > 1:
            model = nn.DataParallel(model, device_ids=self.gpu_ids)
            if self.rank == 0:
                print(f"DataParallel enabled on GPUs: {self.gpu_ids}")

        return model

    def build_optimizer(self) -> optim.Optimizer:
        return optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )

    def build_scheduler(self):
        scheduler_config = self.config['scheduler']
        scheduler_type = scheduler_config['type']

        if scheduler_type == 'ReduceLROnPlateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 'min',
                factor=float(scheduler_config['factor']),
                patience=int(scheduler_config['patience']),
                min_lr=float(scheduler_config['min_lr']),
                verbose=True
            )
        elif scheduler_type == 'StepLR':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config['factor']
            )
        elif scheduler_type == 'CosineAnnealingLR':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get('T_max', self.config['training']['epochs']),
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )
        return None

    def build_loss(self):
        loss_config = self.config['loss']
        try:
            position_weight = loss_config['position_weight']
            velocity_weight = loss_config['velocity_weight']
            endpoint_weight = loss_config['endpoint_weight']
            velocity_scale = loss_config['velocity_scale']
            dt = self.config['model']['dt']
        except KeyError as e:
            raise KeyError(f"Missing required loss/model parameter: {e}")

        boundary_weight = loss_config.get('boundary_weight', 0.0)
        criterion = HybridLoss(
            position_weight,
            velocity_weight,
            endpoint_weight,
            velocity_scale,
            dt,
            boundary_weight=boundary_weight,
        )
        if self.rank == 0:
            print("Loss: HybridLoss")
        return criterion


    def setup_logging(self):
        if self.rank == 0:
            log_dir = Path(self.config['logging']['log_dir'])
            log_dir.mkdir(parents=True, exist_ok=True)

            if self.config['logging']['tensorboard'] and HAS_TENSORBOARD:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                self.writer = SummaryWriter(log_dir / f'run_{timestamp}')
            else:
                self.writer = None

            checkpoint_dir = Path(self.config['checkpoint']['save_dir'])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir = checkpoint_dir

    def get_monitor_value(self, val_metrics: Dict[str, float], metric_name: Optional[str] = None) -> float:
        metric_name = (metric_name or self.checkpoint_metric_name).lower()
        metric_alias = {
            'val_loss': 'loss',
            'loss': 'loss',
            'ade': 'ade',
            'fde': 'fde',
            'rmse': 'rmse_final',
            'rmse_final': 'rmse_final',
            'rmse_5s': 'rmse_5s',
            'fde75': 'fde',
            'fde@75': 'fde',
        }
        metric_key = metric_alias.get(metric_name)
        if metric_key is None:
            raise ValueError(
                f"Unsupported monitor metric: {metric_name}. "
                "Allowed values: val_loss/loss/ade/fde/rmse_final/rmse_5s/"
                "fde75(deprecated alias of fde)/fde@75(deprecated alias of fde)"
            )
        if metric_key not in val_metrics:
            raise KeyError(f"Validation metrics are missing monitor key: {metric_key}")
        return float(val_metrics[metric_key])

    def maybe_resume(self):
        checkpoint_cfg = self.config.get('checkpoint', {})
        resume_enabled = bool(checkpoint_cfg.get('resume', False))
        resume_path_cfg = checkpoint_cfg.get('resume_path')

        if not resume_enabled and not resume_path_cfg:
            return

        if resume_path_cfg:
            resume_path = Path(resume_path_cfg)
        else:
            resume_path = Path(checkpoint_cfg['save_dir']) / 'latest.pth'

        if not resume_path.exists():
            if self.rank == 0:
                print(f"Resume checkpoint not found: {resume_path}. Training will start from scratch.")
            return

        checkpoint = torch.load(resume_path, map_location=self.device)
        model_state = checkpoint['model_state']
        if isinstance(self.model, (DDP, nn.DataParallel)):
            self.model.module.load_state_dict(model_state)
        else:
            self.model.load_state_dict(model_state)

        optimizer_state = checkpoint.get('optimizer_state')
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)

        scheduler_state = checkpoint.get('scheduler_state')
        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)

        scaler_state = checkpoint.get('scaler_state')
        if scaler_state is not None:
            self.scaler.load_state_dict(scaler_state)

        self.start_epoch = int(checkpoint.get('epoch', -1)) + 1
        self.best_val_loss = float(checkpoint.get('best_val_loss', self.best_val_loss))

        saved_metric = str(checkpoint.get('checkpoint_metric', self.checkpoint_metric_name)).lower()
        if saved_metric == self.checkpoint_metric_name:
            self.best_monitor_value = float(checkpoint.get('best_monitor_value', self.best_monitor_value))
        else:
            self.best_monitor_value = float('inf')
            if self.rank == 0:
                print(
                    f"Monitor metric changed during resume: checkpoint={saved_metric}, current={self.checkpoint_metric_name}."
                    "Best monitor value will be recomputed."
                )

        if self.rank == 0:
            print(
                f"Resumed training from checkpoint: {resume_path} | "
                f"start_epoch={self.start_epoch} | "
                f"best {self.checkpoint_metric_name}={self.best_monitor_value:.6f} | "
                f"best val loss={self.best_val_loss:.6f}"
            )

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        tf_ratio = get_teacher_forcing_ratio(epoch, self.config)
        total_loss = 0.0
        loss_components = defaultdict(float)
        num_batches = 0
        use_amp = bool(self.scaler is not None and self.scaler.is_enabled())

        if self.distributed and hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(train_loader):
            self.optimizer.zero_grad()
            if use_amp:
                with autocast(device_type='cuda'):
                    obs_seq, pred_seq, pred_pos, pred_vel = self._forward_batch(batch, tf_ratio)
                    loss, loss_dict = self.criterion(pred_pos, pred_vel, pred_seq)
                self.scaler.scale(loss).backward()
                if self.config['training']['grad_clip'] > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                obs_seq, pred_seq, pred_pos, pred_vel = self._forward_batch(batch, tf_ratio)
                loss, loss_dict = self.criterion(pred_pos, pred_vel, pred_seq)
                loss.backward()
                if self.config['training']['grad_clip'] > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip'])
                self.optimizer.step()

            total_loss += float(loss.item())
            for key, value in loss_dict.items():
                if key != 'total':
                    loss_components[key] += value
            num_batches += 1

            if batch_idx % self.config['logging']['print_every'] == 0 and self.rank == 0:
                print(
                    f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                    f"Total: {loss.item():.4f} | "
                    f"Pos: {loss_dict.get('pos', 0.0):.4f} | "
                    f"Vel: {loss_dict.get('vel', 0.0):.4f} | "
                    f"End: {loss_dict.get('end', 0.0):.4f} | "
                    f"Bdry: {loss_dict.get('bdry', 0.0):.4f} | "
                    f"TF: {tf_ratio:.2f}"
                )

        metrics = {'loss': total_loss / num_batches if num_batches > 0 else 0.0}
        for key, value in loss_components.items():
            metrics[key] = value / num_batches if num_batches > 0 else 0.0
        return metrics

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        loss_components = defaultdict(float)
        metric_sums = defaultdict(float)
        num_batches = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                obs_seq, pred_seq, pred_pos, pred_vel = self._forward_batch(batch, teacher_forcing_ratio=0.0)
                loss, loss_dict = self.criterion(pred_pos, pred_vel, pred_seq)
                metric_dict = compute_regression_metrics(pred_pos, pred_seq, pred_len=self.pred_len, fps=self.fps)

                batch_size = obs_seq.shape[0]
                total_loss += float(loss.item()) * batch_size
                for key, value in loss_dict.items():
                    if key != 'total':
                        loss_components[key] += value * batch_size
                for key, value in metric_dict.items():
                    metric_sums[key] += value * batch_size
                num_batches += 1
                total_samples += batch_size

        metrics = {'loss': total_loss / total_samples if total_samples > 0 else 0.0}
        for key, value in loss_components.items():
            metrics[key] = value / total_samples if total_samples > 0 else 0.0
        for key, value in metric_sums.items():
            metrics[key] = value / total_samples if total_samples > 0 else 0.0
        return metrics

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        if self.rank == 0:
            print(f"Starting training. Total epochs: {self.config['training']['epochs']}")

        for epoch in range(self.start_epoch, self.config['training']['epochs']):
            start_time = time.time()


            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)


            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler_monitor_value = self.get_monitor_value(val_metrics, self.scheduler_monitor_name)
                    self.scheduler.step(scheduler_monitor_value)
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

            if self.rank == 0:
                final_fde_label = 'FDE'
                if self.pred_len > 0 and self.fps > 0:
                    final_fde_label = f"FDE@{self.pred_len / self.fps:.1f}s"
                rmse_5s_label = 'RMSE@5.0s'
                if self.fps > 0:
                    rmse_5s_label = f"RMSE@{self.rmse_5s_seconds:.1f}s"
                print(f"\nEpoch {epoch} | Time: {time.time() - start_time:.1f}s | LR: {current_lr:.6f}")
                print(
                    "Train Loss: "
                    f"{train_metrics['loss']:.6f} | "
                    "Val Loss: "
                    f"{val_metrics['loss']:.6f} | "
                    f"ADE: {val_metrics.get('ade', 0.0):.6f} | "
                    f"{final_fde_label}: {val_metrics.get('fde', 0.0):.6f} | "
                    f"{rmse_5s_label}: {val_metrics.get('rmse_5s', 0.0):.6f}"
                )

                if self.writer:
                    self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
                    self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
                    self.writer.add_scalar('LR', current_lr, epoch)
                    if 'ade' in val_metrics:
                        self.writer.add_scalar('Metrics/ADE', val_metrics['ade'], epoch)
                    if 'fde' in val_metrics:
                        final_fde_tag = 'Metrics/FDE'
                        if self.pred_len > 0 and self.fps > 0:
                            final_fde_tag = f"Metrics/FDE@{self.pred_len / self.fps:.1f}s"
                        self.writer.add_scalar(final_fde_tag, val_metrics['fde'], epoch)
                    if 'rmse_5s' in val_metrics:
                        rmse_5s_tag = 'Metrics/RMSE@5.0s'
                        if self.fps > 0:
                            rmse_5s_tag = f"Metrics/RMSE@{self.rmse_5s_seconds:.1f}s"
                        self.writer.add_scalar(rmse_5s_tag, val_metrics['rmse_5s'], epoch)


                _min_delta = self.config['training']['early_stopping'].get('min_delta', 0.0)
                monitor_value = self.get_monitor_value(val_metrics)
                is_best = monitor_value < (self.best_monitor_value - _min_delta)
                if is_best:
                    self.best_monitor_value = monitor_value
                    self.best_val_loss = val_metrics['loss']
                    self.patience_counter = 0
                    if bool(self.config['checkpoint'].get('save_best', True)):
                        self.save_checkpoint(epoch, val_metrics['loss'], is_best=True, monitor_value=monitor_value)
                    print(
                        " >>> New Best Model! "
                        f"{self.checkpoint_metric_name}: {self.best_monitor_value:.6f} | "
                        f"Val Loss: {self.best_val_loss:.6f}"
                    )
                else:
                    if epoch >= self.early_stopping_start_epoch:
                        self.patience_counter += 1


                save_every = int(self.config['checkpoint'].get('save_every', 0) or 0)
                if save_every > 0 and epoch % save_every == 0:
                    self.save_checkpoint(epoch, val_metrics['loss'], monitor_value=monitor_value)

                self.save_checkpoint(epoch, val_metrics['loss'], filename='latest.pth', monitor_value=monitor_value)

                if self.config['training']['early_stopping']['enabled']:
                    if (
                        epoch >= self.early_stopping_start_epoch
                        and self.patience_counter >= self.config['training']['early_stopping']['patience']
                    ):
                        print(
                            f"Early stopping triggered on {self.early_stopping_monitor_name}. "
                            f"Best value: {self.best_monitor_value:.6f}"
                        )
                        break

    def save_checkpoint(self, epoch, val_loss, is_best=False, filename=None, monitor_value: Optional[float] = None):
        if filename is None:
            filename = f'epoch_{epoch}.pth'

        path = self.checkpoint_dir / filename
        if is_best:
            path = self.checkpoint_dir / 'best_model.pth'

        model_state = self.model.module.state_dict() if isinstance(self.model, (DDP, nn.DataParallel)) else self.model.state_dict()

        torch.save({
            'epoch': epoch,
            'model_state': model_state,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict() if self.scheduler is not None else None,
            'scaler_state': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_monitor_value': self.best_monitor_value,
            'monitor_value': float(monitor_value) if monitor_value is not None else None,
            'checkpoint_metric': self.checkpoint_metric_name,
            'config': self.config
        }, path)


def main(config_dir=None, runtime_overrides=None):
    cfg_root = Path(config_dir) if config_dir is not None else (PROJECT_ROOT / 'configs')
    config = load_config_bundle(cfg_root, ['data', 'model', 'train'])
    if runtime_overrides:
        config = deep_merge(config, runtime_overrides)
    validate_time_config_contract(config)
    data_config = config

    train_path = config['paths']['train_data']
    val_path = config['paths']['val_data']

    print(f"Data paths: \n  Train: {train_path}\n  Val: {val_path}")

    training_cfg = config.get('training', {})
    requested_workers = int(training_cfg.get('num_workers', 0 if os.name == 'nt' else 4))
    num_workers = max(0, requested_workers)
    if os.name == 'nt' and requested_workers > 0:
        print(f"Windows detected; num_workers={requested_workers} may cause memory issues, so it is set to 0.")
        num_workers = 0
    print(f"DataLoader configuration: num_workers={num_workers}")

    train_loader, val_loader = create_dataloaders(
        train_path=train_path,
        val_path=val_path,
        batch_size=config['training']['batch_size'],
        num_workers=num_workers,
        obs_len=data_config['sliding_window']['obs_len'],
        pred_len=data_config['sliding_window']['pred_len'],
        feature_dim=config['model']['encoder']['input_dim'],
        use_relative_coords=bool(config['training'].get('use_relative_coords', True))
    )

    trainer = Trainer(config, data_config)
    trainer.maybe_resume()
    trainer.train(train_loader, val_loader)

if __name__ == '__main__':
    main()

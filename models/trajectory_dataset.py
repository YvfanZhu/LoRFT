import os
import pickle
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class TrajectoryDataset(Dataset):
    def __init__(self, data_path: str, obs_len: Optional[int] = None, pred_len: Optional[int] = None, feature_dim: int = 10, use_relative_coords: bool = True):
        if obs_len is None or pred_len is None:
            raise ValueError('TrajectoryDataset requires explicit obs_len and pred_len.')
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.feature_dim = feature_dim
        self.use_relative_coords = use_relative_coords
        if not os.path.exists(data_path):
            raise FileNotFoundError(f'Dataset file not found: {data_path}')
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        self.samples = data['samples']
        self.data_feature_dim = self.samples[0]['obs_seq'].shape[1] if self.samples else 10
        print(f'Loaded dataset: {data_path}')
        print(f'  samples: {len(self.samples)}')
        print(f'  feature_dim: {self.data_feature_dim}')
        if self.data_feature_dim != feature_dim:
            print(f'Warning: dataset feature_dim={self.data_feature_dim}, model feature_dim={feature_dim}.')

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        raw_obs = sample['obs_seq'].astype(np.float32)
        raw_pred = sample['pred_seq'].astype(np.float32)
        obs_seq = raw_obs[:, :self.feature_dim]
        pred_seq = raw_pred[:, :self.feature_dim]
        origin_x = obs_seq[-1, 0]
        origin_y = obs_seq[-1, 1]
        start_pos_norm = torch.tensor([origin_x, origin_y], dtype=torch.float32)
        if self.use_relative_coords:
            obs_seq[:, 0] -= origin_x
            obs_seq[:, 1] -= origin_y
            pred_seq[:, 0] -= origin_x
            pred_seq[:, 1] -= origin_y
        anchor_abs = sample.get('anchor_abs')
        if anchor_abs is None:
            anchor_abs = sample.get('start_pos', [0.0, 0.0, 0.0, 0.0])
        return {
            'obs_seq': torch.from_numpy(obs_seq),
            'pred_seq': torch.from_numpy(pred_seq),
            'start_pos': start_pos_norm,
            'anchor_abs': torch.tensor(anchor_abs, dtype=torch.float32),
            'scene': sample.get('scene', ''),
            'track_id': sample.get('track_id', 0),
            'is_flipped': sample.get('is_flipped', False),
            'direction': sample.get('direction', 1),
        }


def create_dataloaders(train_path: str, val_path: str, batch_size: int = 64, num_workers: int = 4, obs_len: Optional[int] = None, pred_len: Optional[int] = None, feature_dim: int = 10, use_relative_coords: bool = True, distributed: bool = False, rank: int = 0, world_size: int = 1) -> Tuple[DataLoader, DataLoader]:
    if obs_len is None or pred_len is None:
        raise ValueError('create_dataloaders requires explicit obs_len and pred_len.')
    train_dataset = TrajectoryDataset(train_path, obs_len, pred_len, feature_dim, use_relative_coords)
    val_dataset = TrajectoryDataset(val_path, obs_len, pred_len, feature_dim, use_relative_coords)
    train_sampler = None
    val_sampler = None
    if distributed and world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader

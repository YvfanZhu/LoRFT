import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryEncoder(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=256, num_layers=2, dropout=0.3, bidirectional=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        self.projection = nn.Linear(hidden_dim * 2, hidden_dim) if bidirectional else None

    def forward(self, x):
        x_emb = self.embedding(x)
        _, (hidden, cell) = self.lstm(x_emb)
        if self.projection:
            hidden = self.projection(torch.cat([hidden[-2], hidden[-1]], dim=1)).unsqueeze(0).repeat(self.num_layers, 1, 1)
            cell = self.projection(torch.cat([cell[-2], cell[-1]], dim=1)).unsqueeze(0).repeat(self.num_layers, 1, 1)
        return hidden, cell


class TrajectoryDecoder(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=256, output_dim=2, num_layers=2, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x, hidden, cell):
        x_emb = self.embedding(x)
        output, (hidden, cell) = self.lstm(x_emb, (hidden, cell))
        prediction = self.fc_out(output)
        return prediction, hidden, cell


class TrajectorySeq2Seq(nn.Module):
    def __init__(self, config):
        super().__init__()
        enc_cfg = config['model']['encoder']
        dec_cfg = config['model']['decoder']
        try:
            loss_cfg = config['loss']
            model_cfg = config['model']
            self.velocity_scale = loss_cfg['velocity_scale']
            self.dt = model_cfg['dt']
            self.max_delta = float(model_cfg.get('decode', {}).get('max_delta', 0.1))
        except KeyError as exc:
            raise KeyError(f'Missing required model parameter: {exc}') from exc
        self.encoder = TrajectoryEncoder(
            input_dim=enc_cfg['input_dim'],
            hidden_dim=enc_cfg['hidden_dim'],
            num_layers=enc_cfg['num_layers'],
            dropout=enc_cfg['dropout'],
            bidirectional=enc_cfg['bidirectional'],
        )
        self.decoder = TrajectoryDecoder(
            input_dim=dec_cfg['input_dim'],
            hidden_dim=dec_cfg['hidden_dim'],
            output_dim=dec_cfg['output_dim'],
            num_layers=dec_cfg['num_layers'],
            dropout=dec_cfg['dropout'],
        )

    def forward(self, input_seq, target_seq, teacher_forcing_ratio=0.5, pred_len: Optional[int] = None):
        hidden, cell = self.encoder(input_seq)
        decoder_input = input_seq[:, -1, :].unsqueeze(1)
        current_pos = input_seq[:, -1, 0:2].unsqueeze(1)
        fixed_wh = input_seq[:, -1, 2:4].unsqueeze(1)
        context_geo = input_seq[:, -1, 6:].unsqueeze(1)
        pred_pos_list = []
        pred_vel_list = []
        if target_seq is None and pred_len is None:
            raise ValueError('TrajectorySeq2Seq.forward requires pred_len when target_seq is None.')
        target_len = target_seq.shape[1] if target_seq is not None else pred_len
        for t in range(target_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            output = torch.clamp(output, -self.max_delta, self.max_delta)
            pred_vel_list.append(output)
            next_pos = current_pos + output
            pred_pos_list.append(next_pos)
            use_teacher_forcing = target_seq is not None and random.random() < teacher_forcing_ratio
            if use_teacher_forcing:
                gt_pos = target_seq[:, t:t + 1, 0:2]
                gt_vel = target_seq[:, t:t + 1, 4:6]
                decoder_input = torch.cat([gt_pos, fixed_wh, gt_vel, context_geo], dim=2)
                current_pos = gt_pos
            else:
                velocity_per_sec = output / self.dt
                next_vel_feat = velocity_per_sec * self.velocity_scale
                decoder_input = torch.cat([next_pos, fixed_wh, next_vel_feat, context_geo], dim=2)
                current_pos = next_pos
        pred_pos = torch.cat(pred_pos_list, dim=1)
        pred_vel = torch.cat(pred_vel_list, dim=1)
        return pred_pos, pred_vel


class HybridLoss(nn.Module):
    def __init__(self, position_weight, velocity_weight, endpoint_weight, velocity_scale, dt, boundary_weight=0.0):
        super().__init__()
        if any(v is None for v in (position_weight, velocity_weight, endpoint_weight, velocity_scale, dt)):
            raise ValueError('HybridLoss requires position_weight, velocity_weight, endpoint_weight, velocity_scale, and dt.')
        self.pos_w = position_weight
        self.vel_w = velocity_weight
        self.end_w = endpoint_weight
        self.vel_scale = velocity_scale
        self.dt = dt
        self.bdry_w = boundary_weight
        self.mse = nn.MSELoss()

    def forward(self, pred_pos, pred_vel, target_seq):
        target_pos = target_seq[:, :, 0:2]
        target_v_feat = target_seq[:, :, 4:6]
        loss_pos = self.mse(pred_pos, target_pos)
        pred_v_scaled = (pred_vel / self.dt) * self.vel_scale
        loss_vel = F.smooth_l1_loss(pred_v_scaled, target_v_feat, beta=0.05)
        loss_end = self.mse(pred_pos[:, -1, :], target_pos[:, -1, :])
        u_pred = pred_pos[:, :, 0]
        loss_bdry = (torch.clamp(-u_pred, min=0.0) ** 2 + torch.clamp(u_pred - 1.0, min=0.0) ** 2).mean()
        total = self.pos_w * loss_pos + self.vel_w * loss_vel + self.end_w * loss_end + self.bdry_w * loss_bdry
        return total, {'pos': loss_pos.item(), 'vel': loss_vel.item(), 'end': loss_end.item(), 'bdry': loss_bdry.item()}


def build_model(config):
    dynamic_geo_cfg = config.get('model', {}).get('dynamic_geo', {})
    if bool(dynamic_geo_cfg.get('enabled', False)):
        from .seq2seq_dynamic_geo import TrajectorySeq2SeqDynamicGeo

        return TrajectorySeq2SeqDynamicGeo(config)
    return TrajectorySeq2Seq(config)


TrajectoryPredictionLoss = HybridLoss


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

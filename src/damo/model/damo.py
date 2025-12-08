import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf, DictConfig

import damo.utils as utils
from damo.model.layers.transforms import Permute
from damo.model.layers.blocks import ResConvBlock
from damo.model.layers.attention import PointAttention


class Damo(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        cfg: DictConfig = OmegaConf.create(kwargs)

        self.seq_len = seq_len = cfg.seq_len
        self.n_joints = n_joints = cfg.n_joints
        self.n_rep_joints = n_rep_joints = cfg.n_rep_joints

        default_act = cfg.default_act
        default_norm = cfg.default_norm

        assert seq_len % 2 == 1, ValueError(f"seq_len ({seq_len}) must be odd number.")

        cfg_me = cfg.marker_embedding
        cfg_ma = cfg.marker_attention
        cfg_pa = cfg.post_attention
        cfg_rjp = cfg.rep_joint_predictor
        cfg_rwp = cfg.rep_weight_predictor
        cfg_rop = cfg.rep_offset_predictor

        d_model = cfg_ma.d_model

        act = lambda c: c.get("act", default_act)
        norm = lambda c: c.get("norm", default_norm)

        # [SHAPE] input: markers_seq [B, S, M, 3]
        self.marker_embedding = nn.Sequential(
            Permute(0, 3, 2, 1),  # [B, 3, M, S]
            ResConvBlock(3, d_model, cfg_me.d_hidden, conv_dim=2, act=act(cfg_me), norm=norm(cfg_me))
            # [B, C, M, S] (C = d_model)
        )

        self.marker_attention = PointAttention(cfg_ma)  # [B, C, M, S] -> [B, C, M]

        self.post_attention = nn.Sequential(
            ResConvBlock(d_model, d_model, cfg_pa.d_hidden, conv_dim=1, act=act(cfg_pa), norm=norm(cfg_pa)),
            # [B, C, M] -> [B, C, M]
            ResConvBlock(d_model, n_joints * 4, cfg_pa.d_hidden, conv_dim=1, act=act(cfg_pa), norm=norm(cfg_pa))
            # [B, C, M] -> [B, 4J, M]
        )

        self.rep_joint_predictor = nn.Sequential(
            ResConvBlock(n_joints * 4, n_joints + 1, cfg_rjp.d_hidden, conv_dim=1, act=act(cfg_rjp), norm=norm(cfg_rjp)),
            # [B, 4J, M] -> [B, J+1, M]
        )

        self.rep_weight_predictor = nn.Sequential(
            ResConvBlock(n_joints * 2 + 1, n_rep_joints, cfg_rwp.d_hidden, conv_dim=1, act=act(cfg_rwp), norm=norm(cfg_rwp)),
            # [B, 2J+1, M] -> [B, Jr, M]
            nn.Softmax(dim=1),
            Permute(0, 2, 1)
            # [B, Jr, M] -> [B, M, Jr]
        )

        self.rep_offset_predictor = nn.Sequential(
            ResConvBlock(n_joints * 4 + 1, n_rep_joints * 3, cfg_rop.d_hidden, conv_dim=1, act=act(cfg_rwp), norm=norm(cfg_rwp)),
            # [B, 4J+1, M] -> [B, 3Jr, M]
            Permute(0, 2, 1)
            # [B, 3Jr, M] -> [B, M, 3Jr]
        )

    def forward(self, *, markers_seq, **kwargs):
        """
        markers_seq: (B, S, M, 3)
        markers_seq_mask: (B, S, M)  (1: valid, 0: padded)
        returns:
          weights: (B, M, J+1)
          rep_weights: (B, M, Jr)
          rep_offsets: (B, M, Jr, 3)
        """
        B, S, M, _ = markers_seq.shape
        J = self.n_joints
        Jr = self.n_rep_joints
        smi = S // 2

        markers_seq_mask = kwargs.get("markers_seq_mask", None)

        if markers_seq_mask is None:
            markers_seq_mask = Damo.make_markers_mask(markers_seq)


        markers_center_seq = Damo.compute_center(markers_seq, markers_seq_mask)  # [B, S, 1, 3]
        markers_centered_seq = markers_seq - markers_center_seq  # [B, S, M, 3]

        markers_feats_seq = self.marker_embedding(markers_centered_seq)  # [B, C, M, S]
        markers_attention = self.marker_attention(markers_feats_seq, markers_seq_mask)  # [B, C, M]

        seq_mid_mask = markers_seq_mask[:, smi, :].unsqueeze(1).float()
        markers_attention = markers_attention * seq_mid_mask  # [B, C, M]
        marker_config_feats = self.post_attention(markers_attention) * seq_mid_mask  # [B, 4J, M]

        weights = self.rep_joint_predictor(marker_config_feats)  # [B, J+1, M]

        weights_feats = torch.cat([weights, marker_config_feats[:, :J, :]], dim=1)
        rep_weights = self.rep_weight_predictor(weights_feats)  # [B, M, Jr]

        offsets_feats = torch.cat([weights, marker_config_feats[:, J:, :]], dim=1)
        rep_offsets = self.rep_offset_predictor(offsets_feats)  # [B, M, 3Jr]

        weights = weights.permute(0, 2, 1).contiguous()
        rep_offsets = rep_offsets.reshape(B, M, Jr, 3)

        outputs = {
            "weights": weights,
            "rep_weights": rep_weights,
            "rep_offsets": rep_offsets,
            "markers_mask": markers_seq_mask[:, smi]
        }

        return outputs


    @staticmethod
    def compute_center(markers_seq, markers_mask_seq, zero_eps=1e-6):
        """
        inputs:  markers_seq (B, S, M, 3)
        returns: center_seq (B, S, 1, 3)
        """
        B, S, M, C = markers_seq.shape
        device = markers_seq.device
        dtype = markers_seq.dtype

        is_zero = (markers_seq.abs() < zero_eps).all(dim=-1)
        valid = markers_mask_seq.bool() & (~is_zero)

        mid = S // 2
        ps_mid = markers_seq[:, mid]
        v_mid = valid[:, mid]

        ps_mid = ps_mid.masked_fill(~v_mid.unsqueeze(-1), float('nan'))
        center = torch.nanmedian(ps_mid, dim=1, keepdim=True).values
        center = torch.nan_to_num(center, nan=0.0)

        center_seq = center.view(B, 1, 1, C).expand(B, S, 1, C)
        return center_seq.to(device=device, dtype=dtype)

    @staticmethod
    def make_markers_mask(markers: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mask = (markers != 0).any(dim=-1)

        return mask
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import damo.utils as utils
from damo.model.layers.pe import make_time_pe


class DistBias(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        raw = math.log(math.exp(cfg["init_tau"]) - 1.0)
        self.raw_tau = nn.Parameter(torch.full((cfg["n_heads"],), raw))
        self.eps = 1e-6

    def forward(self, d2_bhms):
        tau = F.softplus(self.raw_tau).view(1, -1, 1, 1) + self.eps  # (1,H,1,1)
        return -d2_bhms / tau  # (B,H,Mq,Mk)


class PointAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.d_model = d_model = cfg["d_model"]
        self.n_heads = n_heads = cfg["n_heads"]
        self.seq_len = seq_len = cfg["seq_len"]
        self.n_layers = n_layers = cfg["n_layers"]

        self.use_time_pe = cfg["use_time_pe"]
        self.use_dist_bias = cfg["use_dist_bias"]

        self.layers = nn.ModuleList([
            PointAttentionBlock(cfg, first_block=(i == 0))
            for i in range(n_layers)
        ])

        if cfg["use_time_pe"]:
            pe = make_time_pe(seq_len, d_model, dtype=torch.float32)
            self.register_buffer("time_pe", pe, persistent=False)
        else:
            self.time_pe = None

        if cfg["use_dist_bias"]:
            self.dist_bias = DistBias(cfg["dist_bias"])
        else:
            self.dist_bias = None

    @staticmethod
    def _build_masks(points_mask, mid):
        """
        returns:
          kv_allowed: (B,1,1,S*M) bool
          q_allowed:  (B,1,M,1)   bool
        """
        B, S, M = points_mask.shape
        B, S, M = points_mask.shape
        kv_allowed = points_mask.bool().reshape(B, 1, 1, S * M)
        q_allowed = points_mask[:, mid].bool().reshape(B, 1, M, 1)
        return kv_allowed, q_allowed

    def _distance_bias_bh(self, points_seq, kv_allowed, q_allowed):
        """
        points_xyz: (B,S,M,3)
        kv_allowed: (B,1,1,S*M) bool
        q_allowed:  (B,1,M,1)   bool
        return attn_bias_bh: (B*H, M, S*M) float
        """
        B, S, M, _ = points_seq.shape
        H = self.n_heads
        S_mid = S // 2

        Pq = points_seq[:, S_mid]  # (B,M,3)
        Pkv = points_seq.reshape(B, S * M, 3)  # (B,S*M,3)

        d2 = (Pq.unsqueeze(2) - Pkv.unsqueeze(1)).pow(2).sum(-1)  # (B,M,S*M)

        # (B,1,M,1) & (B,1,1,S*M) -> (B,1,M,S*M)
        valid = (q_allowed & kv_allowed).bool().reshape(B, 1, M, S * M)
        valid_f = valid.float()

        denom = valid_f.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)  # (B,1,1,1)
        mean_d2 = (d2.view(B, 1, M, S * M) * valid_f).sum(dim=(-2, -1), keepdim=True) / denom
        d2_norm = d2.view(B, 1, M, S * M) / mean_d2.detach().clamp_min(1e-6)  # (B,1,M,S*M)

        d2_norm = d2_norm.masked_fill(~valid, 0.0)  # (B,1,M,S*M)

        d2_bh = d2_norm.expand(B, H, M, S * M)  # (B,H,M,S*M)
        bias_bh = self.dist_bias(d2_bh)  # (B,H,M,S*M)
        return bias_bh.reshape(B * H, M, S * M)  # (B*H,M,S*M)

    def forward(self, feats, points_mask, points_seq=None, is_training=True):
        B, C, M, S = feats.shape
        H = self.n_heads
        S_mid = S // 2
        assert C == self.d_model and S == self.seq_len, "channel/seq_len mismatch"

        # (B,S,M,C)
        x = feats.permute(0, 3, 2, 1).contiguous()

        # K/V: (B,S,M,C) + time PE → (B,C,S*M)
        x_kv = x + self.time_pe[:, :S] if (self.use_time_pe and self.time_pe is not None) else x
        kv = x_kv.reshape(B, S * M, C).transpose(1, 2).contiguous()  # (B,C,S*M)

        # Q(mid): (B,C,M)
        q_mid = x[:, S_mid].transpose(1, 2).contiguous()  # (B,C,M)

        # masks
        kv_allowed, q_allowed = self._build_masks(points_mask, S_mid)
        attn_mask = (~kv_allowed).expand(B, H, M, S * M).reshape(B * H, M, S * M)
        q_valid = q_allowed.squeeze(-1)  # (B,1,M)

        # distance bias (learnable tau)
        attn_bias = None
        if self.use_dist_bias and (points_seq is not None):
            attn_bias = self._distance_bias_bh(points_seq, kv_allowed, q_allowed)  # (B*H,M,S*M)

        z = q_mid
        for layer in self.layers:
            z = layer(
                q=z,
                kv=kv,
                attn_mask_bh=attn_mask,
                attn_bias_bh=attn_bias,
                q_valid=q_valid,
                is_training=is_training
            )

        z = z * q_valid
        return z  # (B,C,M)


class PointAttentionBlock(nn.Module):
    def __init__(self, cfg, first_block=False):
        super().__init__()

        d_model = cfg["d_model"]

        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(cfg, init_zero=first_block)
        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = FeedForward(cfg["ffn"])

    def forward(self, q, kv, attn_mask_bh=None, attn_bias_bh=None, q_valid=None, is_training=True):
        """
            q: (B,C,M)
            kv: (B,C,S*M)
            attn_mask_bh: (B*H,M,S*M) bool (True=mask)
            attn_bias_bh: (B*H,M,S*M) or None
            q_valid: (B,1,M) bool
        """
        q_ln = self.ln_q(q.transpose(1, 2)).transpose(1, 2)
        kv_ln = self.ln_kv(kv.transpose(1, 2)).transpose(1, 2)

        y = self.attn(q_ln, kv_ln, kv_ln, attn_mask=attn_mask_bh, attn_bias=attn_bias_bh, is_training=is_training)
        y = y + q
        if q_valid is not None:
            y = y * q_valid

        z = self.ln_ffn(y.transpose(1, 2)).transpose(1, 2)
        z = self.ffn(z)
        z = z + y
        if q_valid is not None:
            z = z * q_valid
        return z

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg, init_zero=False):
        super().__init__()

        self.d_model: int = cfg["d_model"]
        self.n_heads: int = cfg["n_heads"]

        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")

        self.d_head: int = self.d_model // self.n_heads
        self.attn_dropout: float = cfg["attn_dropout"]

        self.proj_q = nn.Conv1d(self.d_model, self.d_model, kernel_size=1, bias=True)
        self.proj_k = nn.Conv1d(self.d_model, self.d_model, kernel_size=1, bias=True)
        self.proj_v = nn.Conv1d(self.d_model, self.d_model, kernel_size=1, bias=True)
        self.merge = nn.Conv1d(self.d_model, self.d_model, kernel_size=1, bias=True)

        for w in [self.proj_q, self.proj_k, self.proj_v]:
            nn.init.xavier_uniform_(w.weight)
            nn.init.zeros_(w.bias)

        if init_zero:
            nn.init.zeros_(self.merge.weight)
            nn.init.zeros_(self.merge.bias)
        else:
            nn.init.xavier_uniform_(self.merge.weight)
            nn.init.zeros_(self.merge.bias)

    def forward(self, q_in, k_in, v_in, attn_mask=None, attn_bias=None, is_training=True):
        """
            q_in: (B, C, Mq)
            k_in: (B, C, Mk)
            v_in: (B, C, Mk)
            attn_mask: (B*H, Mq, Mk) bool, True=mask
            attn_bias: (B*H, Mq, Mk) float or None  (e.g., -d^2/tau)
            returns: (B, C, Mq)
        """
        B, C, Mq = q_in.shape
        Mk = k_in.shape[-1]

        Q = self.proj_q(q_in)  # (B,C,Mq)
        K = self.proj_k(k_in)  # (B,C,Mk)
        V = self.proj_v(v_in)  # (B,C,Mk)

        def split_heads(x):  # (B,C,M)->(B,H,M,d_h)
            B0, C0, M0 = x.shape
            x = x.view(B0, self.n_heads, self.d_head, M0).transpose(2, 3)
            return x

        Qh, Kh, Vh = map(split_heads, (Q, K, V))  # (B,H,Mq,dh), (B,H,Mk,dh)...
        Qr = Qh.reshape(B * self.n_heads, Mq, self.d_head)
        Kr = Kh.reshape(B * self.n_heads, Mk, self.d_head)
        Vr = Vh.reshape(B * self.n_heads, Mk, self.d_head)

        # scaled dot-product attention (handles 1/sqrt(d) + softmax)
        if attn_bias is not None:
            scores = torch.matmul(Qr, Kr.transpose(-2, -1)) / math.sqrt(self.d_head)
            scores = scores + attn_bias
            if attn_mask is not None:
                scores = scores.masked_fill(attn_mask, float('-inf'))
            attn = torch.softmax(scores, dim=-1)
            if is_training and self.attn_dropout and self.attn_dropout > 0.0:
                attn = F.dropout(attn, p=self.attn_dropout, training=True)
            out = torch.matmul(attn, Vr)  # (B*H,Mq,dh)
        else:
            out = F.scaled_dot_product_attention(
                Qr, Kr, Vr,
                attn_mask=attn_mask,  # True=mask
                dropout_p=self.attn_dropout if is_training else 0.0,
                is_causal=False
            )  # (B*H,Mq,dh)

        out = out.view(B, self.n_heads, Mq, self.d_head).transpose(2, 3).reshape(B, C, Mq)
        out = self.merge(out)  # (B,C,Mq)
        return out


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        d_model = cfg["d_model"]
        d_hidden = cfg["d_hidden"]
        dropout = cfg["dropout"]
        act = cfg["act"]

        self.fc1 = nn.Conv1d(d_model, d_hidden, 1)
        self.fc2 = nn.Conv1d(d_hidden, d_model, 1)
        self.act = utils.make_activation(act)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
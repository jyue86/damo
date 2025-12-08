import torch
import numpy as np
from typing import Dict, Any, List, Callable, Optional
import random

import damo.utils as utils


class MocapNoiseAugmentor:
    def __init__(self, cfg):
        self.cfg = cfg

        self.jitter = cfg["jitter"]["enabled"]
        self.ghost = cfg["ghost"]["enabled"]
        self.occlusion = cfg["occlusion"]["enabled"]
        self.shuffle = cfg["shuffle"]["enabled"]

    def __call__(self, markers_seq, marker_normals, transform_mat_seq):
        """
        markers_seq:   [S, M, 3]
        marker_normals: [M, 3]
        transform_mat_seq: [S, M, 4, 4]
        """
        S, M, _ = markers_seq.shape

        m_seq = markers_seq.copy()
        m_mask_seq = np.ones((S, M), dtype=bool)

        if self.ghost:
            m_seq, m_mask_seq = self._add_ghost(m_seq, m_mask_seq, marker_normals, transform_mat_seq)

        if self.jitter:
            m_seq = self._add_jitter(m_seq, m_mask_seq)

        if self.occlusion:
            occlusion_mask = self._make_occlusion_mask(markers_seq)
            m_seq[:, :M][occlusion_mask] = 0.0

        packed_m_seq, packed_mid_idx, num_valid = pack_markers(m_seq)

        if self.shuffle:
            packed_m_seq, packed_mid_idx = self._shuffle_markers(packed_m_seq, packed_mid_idx, num_valid)

        return packed_m_seq, packed_mid_idx, num_valid

    def _add_jitter(self, m_seq, m_mask_seq):
        cfg = self.cfg["jitter"]
        S, M, _ = m_seq.shape

        max_dist = cfg["max_distance"]
        beta_dist = cfg["distance_beta"]

        u = np.random.beta(a=1.0, b=beta_dist, size=(S, M))
        dist = max_dist * u
        dist[~m_mask_seq] = 0.0

        dirs = np.random.normal(size=(S, M, 3))
        norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
        dirs /= (norms + 1e-8)

        noise = dirs * dist[..., None]  # [S, M, 3]

        return m_seq + noise

    def _add_ghost(self, m_seq, m_mask_seq, mn, t_mat_seq):
        cfg = self.cfg["ghost"]
        S, M, _ = m_seq.shape

        tracks_mask = sample_tracks_mask(S, cfg)  # [S, Nt]
        Nt = tracks_mask.shape[1]

        if Nt == 0:
            return m_seq, m_mask_seq

        tracks = np.zeros((S, Nt, 3), dtype=m_seq.dtype)

        min_dist, max_dist = cfg["distance"]
        sigma_dist = cfg.get("distance_sigma", None)
        max_theta = cfg["max_deg_from_normal"]

        if sigma_dist is not None:
            dists = utils.random_utils.sample_continuous_gaussian(
                min_dist, max_dist, sigma_dist, size=Nt
            )
        else:
            dists = np.random.uniform(min_dist, max_dist, size=Nt)

        for ti in range(Nt):
            base_f = np.random.randint(0, S)
            base_m = np.random.randint(0, M)
            base_pos = m_seq[base_f, base_m]  # [3]

            R = t_mat_seq[base_f, base_m, :3, :3]
            n_rest = mn[base_m]
            n_posed = R @ n_rest
            n_posed /= (np.linalg.norm(n_posed) + 1e-8)

            dist = float(dists[ti])

            dir_vec = utils.random_utils.sample_direction_in_cone(
                n_posed, theta_max_deg=max_theta
            ).astype(m_seq.dtype)
            offset = dir_vec * dist  # [3]

            active = tracks_mask[:, ti]  # [S]
            tracks[active, ti] = base_pos + offset

        m_seq_aug = np.concatenate([m_seq, tracks], axis=1)
        m_mask_seq_aug = np.concatenate([m_mask_seq, tracks_mask], axis=1)

        return m_seq_aug, m_mask_seq_aug

    def _make_occlusion_mask(self, m_seq):
        cfg = self.cfg["occlusion"]
        S, M, _ = m_seq.shape

        tracks_mask = sample_tracks_mask(S, cfg)  # [S, Nt]
        Nt = tracks_mask.shape[1]

        mask = np.zeros((S, M), dtype=bool)

        if Nt == 0:
            return mask

        marker_idx = np.random.choice(M, size=Nt, replace=(Nt > M))
        for ti in range(Nt):
            mask[:, marker_idx[ti]] |= tracks_mask[:, ti]

        return mask

    def _shuffle_markers(self, m_seq, mid_idx, num_valid):
        cfg = self.cfg["shuffle"]
        S, M, _ = m_seq.shape
        smi = S // 2

        shuffled_m_seq = np.zeros_like(m_seq)
        shuffled_mid_idx = None

        for s in range(S):
            K = int(num_valid[s])
            if K <= 1:
                continue

            perm = np.random.permutation(K)
            shuffled_m_seq[s, :K] = m_seq[s, :K][perm]

            if s == smi:
                shuffled_mid_idx = mid_idx[perm]

        return shuffled_m_seq, shuffled_mid_idx

def sample_tracks_mask(seq_len: int, cfg: dict) -> np.ndarray:
    S = seq_len
    min_nt, max_nt = cfg["num_tracks"]

    sigma_nt = cfg.get("num_tracks_sigma", None)

    Nt = utils.random_utils.sample_discrete_gaussian(min_nt, max_nt, sigma_nt)
    if Nt <= 0:
        return np.zeros((S, 0), dtype=bool)

    min_L, max_L = cfg["track_len"]
    sigma_L = cfg.get("track_len_sigma", None)

    lengths = utils.random_utils.sample_discrete_gaussian(
        min_L, max_L, sigma_L, size=Nt
    ).astype(int)

    centers = np.random.randint(0, S, size=Nt)

    starts = centers - lengths // 2
    starts = np.clip(starts, 0, S - 1)
    ends = np.clip(starts + lengths, 0, S)

    mask = np.zeros((S, Nt), dtype=bool)
    for i in range(Nt):
        s = starts[i]
        e = ends[i]
        mask[s:e, i] = True

    return mask  # [S, Nt]


def pack_markers(m_seq, thresh=1e-6):
    S, M, _ = m_seq.shape
    smi = S // 2

    packed_seq = np.zeros_like(m_seq)
    packed_mid_idx = None
    num_valid = np.zeros(S, dtype=int)

    for s in range(S):
        pos = m_seq[s]  # [M, 3]
        norms = np.linalg.norm(pos, axis=-1)  # [M]

        valid = norms > thresh
        idx_valid = np.nonzero(valid)[0]
        K = idx_valid.size
        num_valid[s] = K

        if K > 0:
            packed_seq[s, :K] = pos[idx_valid]

            if s == smi:
                packed_mid_idx = idx_valid.copy()

    return packed_seq, packed_mid_idx, num_valid
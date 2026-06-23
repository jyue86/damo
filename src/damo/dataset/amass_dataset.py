import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, get_worker_info
from typing import Dict, Any, List, Callable
import numpy as np
import pickle
import math
import random
from pathlib import Path
from omegaconf import OmegaConf
from functools import lru_cache

import damo.utils as utils
from damo.dataset.body_cache import BodyCache
from damo.dataset.marker_sampler import MarkerSampler
from damo.dataset.mocap_noise_augmentor import MocapNoiseAugmentor


class AmassDataset(Dataset):
    GENDER2ID = {"male": 0, "female": 1, "neutral": 2}
    ID2GENDER = {0: "male", 1: "female", 2: "neutral"}

    CACHE_KEYS = {
        "smplh": ["gender", "betas", "trans", "poses"],
        "smplx": ["gender", "betas", "trans", "poses",
                  "markers_latent", "latent_labels", "markers_latent_vids",
                  "markers_obs", "markers_sim", "labels_obs"],
    }

    def __init__(self, cfg, train=True, amass_root=None):
        super().__init__()
        base_data_root = Path(amass_root).expanduser() if amass_root else utils.Paths.amass_data
        base_data_path = base_data_root / cfg["data_dir_name"].lower()
        self.base_data_path = base_data_path

        split = "train" if train else "val"
        self.file_paths = [
            p
            for ds in cfg["dataset_names"][split]
            for p in (base_data_path / ds).rglob(f"*{cfg['filename_pattern']}*.npz")
            if p.is_file()
        ]

        # --- Base configs ---
        self.seed = cfg["seed"]
        self.dtype = utils.get_dtype(cfg["dtype"])
        self.seq_len = cfg["seq_len"]
        self.genders = cfg["genders"]

        # --- Data types ---
        dt_probs_cfg = cfg["data_type_probs"]
        if split == "val" and cfg.get("val_data_type_probs", None) is not None:
            dt_probs_cfg = cfg["val_data_type_probs"]
        self.data_types = list(dt_probs_cfg.keys())
        self.data_type_probs = np.array(
            [dt_probs_cfg[k] for k in self.data_types],
            dtype=np.float64
        )
        self.data_type_probs = self.data_type_probs / self.data_type_probs.sum()
        self.dispatch: Dict[str, Callable[[int], Dict[str, torch.Tensor]]] = {
            "syn_arb": self._getitem_syn_arb,
            "syn_sup": self._getitem_syn_sup,
            "real": self._getitem_real,
        }

        # --- Synthesis configs ---
        marker_sampler_cfg = OmegaConf.to_container(cfg["marker_sampler"], resolve=True)
        self.marker_sampler = MarkerSampler(marker_sampler_cfg)

        # --- Noise ---
        noise_cfg = OmegaConf.to_container(cfg["noise"], resolve=True)
        self.noise_augmentor = MocapNoiseAugmentor(noise_cfg)

        # --- Cache ---
        self.cache_capacity = cfg["cache_capacity"]

        file_cache_keys = []
        for body_type in ["smplh", "smplx"]:
            for k in AmassDataset.CACHE_KEYS[body_type]:
                file_cache_keys.append(f"{body_type}_{k}")

        self._file_cache = utils.FileCache(keys=file_cache_keys, capacity=self.cache_capacity)
        self._global_index = utils.build_global_index(
            files=self.file_paths,
            sequential_data_key="smplx_trans",
            ratio=(0.1, 0.9),
            requires={"smplx_gender": self.genders},
            seq_len=self.seq_len,
        )

        self._body_cache = BodyCache(
            genders=self.genders,
            capacity=self.cache_capacity,
            synthetic_body_source=cfg.get("synthetic_body_source", "caesar_cache"),
            synthetic_body_pool_dir=cfg.get("synthetic_body_pool_dir", None),
        )

    def __len__(self):
        return len(self._global_index)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        data_type = np.random.choice(self.data_types, p=self.data_type_probs)
        getitem_fn = self.dispatch[data_type]
        return getitem_fn(idx)

    def _get_data_from_cache(self, idx):
        file_idx, seq_idx = self._global_index[idx]
        file_path = self.file_paths[file_idx]
        data_obj = self._file_cache.get(file_path)

        ssi = seq_idx - self.seq_len // 2  # sequence start index
        sei = seq_idx + self.seq_len // 2 + 1  # sequence end index

        return file_idx, file_path, data_obj, ssi, seq_idx, sei

    def _getitem_syn_arb(self, idx) -> Dict[str, torch.Tensor]:
        return self._getitem_syn(idx, syn_type="arbitrary")

    def _getitem_syn_sup(self, idx) -> Dict[str, torch.Tensor]:
        return self._getitem_syn(idx, syn_type="superset")

    def _getitem_syn(self, idx, syn_type):
        fi, fp, data, ssi, smi, sei = self._get_data_from_cache(idx)

        body_type = "smplh"
        pose_dim = 66
        get_data = lambda k: data[f"{body_type}_{k}"]

        trans = get_data("trans")[ssi:sei]  # [S, 3] (S = sei-ssi)
        poses = get_data("poses")[ssi:sei, :pose_dim]  # [S, 66]

        synthetic_body, gender = self._body_cache.get_random_synthetic_body(self.genders, to_torch=False)
        v_shaped = synthetic_body["vertices"]  # [V, 3]
        j_shaped = synthetic_body["joints"][:22]  # [J(22), 3]

        full_weights = self._body_cache.get_weights(body_type, gender, j22=True, to_torch=False)  # [V, J(22)]
        full_rep_jids = self._body_cache.get_topk_weight_jids(body_type, gender, j22=True, k=3, to_torch=False)  # [V, J(3)]

        if syn_type == "arbitrary":
            full_max_weight_jids = full_rep_jids[:, 0]  # [V]
            marker_sample = self.marker_sampler.get_arbitrary_sample(v_shaped, full_weights, full_max_weight_jids)
        elif syn_type == "superset":
            marker_sample = self.marker_sampler.get_superset_sample(v_shaped)
        else:
            raise ValueError(f"Unknown syn_type: {syn_type}")

        m_shaped = marker_sample.m_shaped  # [M, 3]
        markers_vids = marker_sample.markers_vids  # [M]

        weights = full_weights[markers_vids]  # [M, J(22)]
        offsets = m_shaped[:, None, :] - j_shaped[None, :, :]  # [M, J(22), 3]

        rep_jids = full_rep_jids[markers_vids]  # [M, J(3)]
        rep_weights = utils.gather_topk_joints(rep_jids[None, ...], weights[None, ...])[0]  # [M, J(3)]
        rep_offsets = utils.gather_topk_joints(rep_jids[None, ...], offsets[None, ...])[0]  # [M, J(3), 3]

        smpl_model = self._body_cache.get_smpl_model(body_type, gender)

        marker_normals = self.marker_sampler.get_vertex_normals()[markers_vids]  # [M, 3]
        lbs_output = smpl_model.fk_with_reference(
            vertices=self._to_torch(m_shaped),
            joints=self._to_torch(j_shaped),
            weights=self._to_torch(weights),
            trans=self._to_torch(trans),
            poses=self._to_torch(poses),
            vids=self._to_torch_index(markers_vids),
        )
        m_posed = lbs_output.vertices.numpy()  # [S, M, 3]
        j_posed = lbs_output.joints.numpy()  # [S, J(22), 3]
        t_mat_seq = lbs_output.transform_matrix.numpy()  # [S, M, 4, 4]

        M_clean = m_posed.shape[1]
        J = j_posed.shape[1]
        J_rep = rep_jids.shape[1]

        noisy_m_posed, noisy_marker_idx, num_markers = self.noise_augmentor(m_posed, marker_normals, t_mat_seq)

        M_noisy = noisy_marker_idx.size
        J_extend = J + 1

        noisy_weights = np.zeros((M_noisy, J_extend), dtype=weights.dtype)
        noisy_rep_weights = np.zeros((M_noisy, J_rep), dtype=rep_weights.dtype)
        noisy_rep_offsets = np.zeros((M_noisy, J_rep, 3), dtype=rep_offsets.dtype)

        valid_mask = noisy_marker_idx < M_clean
        ghost_mask = ~valid_mask

        valid_idx_in_noisy = np.nonzero(valid_mask)[0]
        valid_idx_in_clean = noisy_marker_idx[valid_mask]
        ghost_idx_in_noisy = np.nonzero(ghost_mask)[0]

        noisy_weights[valid_idx_in_noisy, :J] = weights[valid_idx_in_clean]
        noisy_rep_weights[valid_idx_in_noisy] = rep_weights[valid_idx_in_clean]
        noisy_rep_offsets[valid_idx_in_noisy] = rep_offsets[valid_idx_in_clean]

        noisy_weights[ghost_idx_in_noisy, J] = 1.0

        # --- Additional ---
        m_shaped_obs = np.zeros((M_noisy, 3), dtype=m_shaped.dtype)
        m_shaped_obs[valid_idx_in_noisy] = m_shaped[valid_idx_in_clean]

        markers_vids_obs = -np.ones(M_noisy, dtype=np.int64)
        markers_vids_obs[valid_idx_in_noisy] = markers_vids[valid_idx_in_clean]

        return {
            "markers_seq": self._to_torch(noisy_m_posed),  # [S, M_max, 3]
            "num_markers": self._to_torch_num(num_markers),  # [S] (n <= M_max)
            "weights": self._to_torch(noisy_weights),  # [M_noisy, J_extend(23)]
            "rep_weights": self._to_torch(noisy_rep_weights),  # [M_noisy, J_rep(3), 3]
            "rep_offsets": self._to_torch(noisy_rep_offsets),  # [M_noisy, J_rep(3), 3]

            # --- Additional ---
            "latent_markers": self._to_torch(m_shaped_obs),  # [M_noisy, 3]
            "latent_markers_vids": self._to_torch_index(markers_vids_obs),  # [M_noisy]
        }


    def _getitem_real(self, idx) -> Dict[str, torch.Tensor]:
        fi, fp, data, ssi, smi, sei = self._get_data_from_cache(idx)

        body_type = "smplx"
        get_data = lambda k: data[f"{body_type}_{k}"]

        gender = get_data("gender")
        markers_latent_vids = get_data("markers_latent_vids")
        latent_labels = list(markers_latent_vids.keys())

        M_max = len(latent_labels)
        S = sei - ssi

        markers_seq_list = get_data("markers_obs")[ssi:sei]
        num_markers = np.zeros(S, dtype=int)
        markers_seq = np.zeros((S, M_max, 3), dtype=float)
        for s in range(S):
            Ms = markers_seq_list[s].shape[0]
            num_markers[s] = Ms
            markers_seq[s, :Ms, :] = markers_seq_list[s]

        latent_vids = np.stack([markers_latent_vids[k] for k in latent_labels], axis=0)  # [M_max]
        labels_to_idx = {k: i for i, k in enumerate(latent_labels)}
        labels = get_data("labels_obs")[smi]  # [M_mid]
        idx_labels = np.array([labels_to_idx[l] for l in labels], dtype=np.int64)
        markers_vids = latent_vids[idx_labels]  # [M_mid]

        full_weights = self._body_cache.get_weights(body_type, gender, j22=True, to_torch=False)  # [V, J(22)]
        weights = full_weights[markers_vids]  # [M_mid, J(22)]

        latent_markers = get_data("markers_latent")[idx_labels]  # [M_mid, 3]
        betas = get_data("betas")
        j_shaped = self._body_cache.get_smpl_body(body_type, gender, betas, to_torch=False)["joints"][:22]  # [J(22), 3]
        offsets = latent_markers[:, None, :] - j_shaped[None, :, :]  # [M_mid, J(22), 3]

        full_rep_idx = self._body_cache.get_topk_weight_jids(body_type, gender, j22=True, k=3, to_torch=False)  # [V, J(3)]
        rep_idx = full_rep_idx[markers_vids]  # [M_mid, J(3)]

        rep_weights = utils.mocap_utils.gather_topk_joints(rep_idx, weights)  # [M_mid, J(3)]
        rep_offsets = utils.mocap_utils.gather_topk_joints(rep_idx, offsets)  # [M_mid, J(3), 3]

        M_mid, J = weights.shape
        weights_extend = np.zeros((M_mid, J+1), dtype=weights.dtype)
        weights_extend[:, :J] = weights

        return {
            "markers_seq": self._to_torch(markers_seq),
            "num_markers": self._to_torch_num(num_markers),
            "weights": self._to_torch(weights_extend),
            "rep_weights": self._to_torch(rep_weights),
            "rep_offsets": self._to_torch(rep_offsets),

            # --- Additional ---
            "latent_markers": self._to_torch(latent_markers),
            "latent_markers_vids": self._to_torch_index(markers_vids),
        }

    def _to_torch(self, x):
        return torch.from_numpy(x).to(dtype=self.dtype)

    def _to_torch_mask(self, x):
        return torch.from_numpy(x).bool()

    def _to_torch_index(self, x):
        return torch.from_numpy(x).long()

    def _to_torch_num(self, x):
        return torch.from_numpy(x).int()


def collate_variable_markers(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    B = len(batch)
    S = batch[0]["markers_seq"].shape[0]

    M_seq_list = [item["markers_seq"].shape[1] for item in batch]
    M_seq_max = max(M_seq_list)

    markers_dtype = batch[0]["markers_seq"].dtype

    markers_seq = torch.zeros((B, S, M_seq_max, 3), dtype=markers_dtype)
    num_markers = torch.stack([item["num_markers"] for item in batch], dim=0)  # [B, S]

    for b, item in enumerate(batch):
        m_seq = item["markers_seq"]  # [S, M_i, 3]
        _, M_i, _ = m_seq.shape
        markers_seq[b, :, :M_i] = m_seq

    idx = torch.arange(M_seq_max, device=num_markers.device)[None, None, :]
    markers_seq_mask = idx < num_markers.unsqueeze(-1)

    M_mid_list = [item["weights"].shape[0] for item in batch]
    M_mid_max = max(M_mid_list)

    J_ext = batch[0]["weights"].shape[1]
    J_rep = batch[0]["rep_weights"].shape[1]

    weights = torch.zeros(
        (B, M_mid_max, J_ext),
        dtype=batch[0]["weights"].dtype,
    )
    rep_weights = torch.zeros(
        (B, M_mid_max, J_rep),
        dtype=batch[0]["rep_weights"].dtype,
    )
    rep_offsets = torch.zeros(
        (B, M_mid_max, J_rep, 3),
        dtype=batch[0]["rep_offsets"].dtype,
    )

    latent_markers = torch.zeros(
        (B, M_mid_max, 3),
        dtype=batch[0]["latent_markers"].dtype,
    )
    latent_markers_vids = torch.full(
        (B, M_mid_max),
        fill_value=-1,
        dtype=batch[0]["latent_markers_vids"].dtype,
    )
    markers_mask = torch.zeros(
        (B, M_mid_max),
        dtype=torch.bool,
    )

    for b, item in enumerate(batch):
        M_i = item["weights"].shape[0]

        weights[b, :M_i] = item["weights"]
        rep_weights[b, :M_i] = item["rep_weights"]
        rep_offsets[b, :M_i] = item["rep_offsets"]
        markers_mask[b, :M_i] = True

        latent_markers[b, :M_i] = item["latent_markers"]
        latent_markers_vids[b, :M_i] = item["latent_markers_vids"]


    return {
        "markers_seq": markers_seq,  # [B, S, M_seq_max, 3]
        "markers_seq_mask": markers_seq_mask,  # [B, S, M_seq_max]  bool

        "num_markers": num_markers,  # [B, S]
        "weights": weights,  # [B, M_mid_max, J_ext]
        "rep_weights": rep_weights,  # [B, M_mid_max, J_rep, 3]
        "rep_offsets": rep_offsets,  # [B, M_mid_max, J_rep, 3]
        "markers_mask": markers_mask,  # [B, M_mid_max]  bool

        "latent_markers": latent_markers,  # [B, M_mid_max, 3]
        "latent_markers_vids": latent_markers_vids,  # [B, M_mid_max]
    }

def seed_everything_worker(worker_id: int):
    worker_info = torch.utils.data.get_worker_info()
    base = worker_info.seed
    random.seed(base)
    np.random.seed(base % (2**32 - 1))
    torch.manual_seed(base)

def make_dataloader(cfg: dict, train: bool = False):
    ds = AmassDataset(
        cfg["dataset"],
        train=train,
        amass_root=cfg.get("amass_root"),
    )

    if len(ds) == 0:
        split = "train" if train else "val"
        dataset_names = cfg["dataset"]["dataset_names"][split]
        raise ValueError(
            "AMASS dataset produced zero samples after indexing. "
            f"split={split}, root={ds.base_data_path}, datasets={dataset_names}, "
            f"pattern=*{cfg['dataset']['filename_pattern']}*.npz, "
            f"genders={cfg['dataset']['genders']}"
        )

    loader_cfg = cfg["dataloader"]
    loader = DataLoader(
        ds,
        batch_size=loader_cfg["batch_size"],
        shuffle=train,
        num_workers=loader_cfg["num_workers"],
        pin_memory=loader_cfg["pin_memory"],
        drop_last=loader_cfg["drop_last"],
        collate_fn=collate_variable_markers,
        worker_init_fn=seed_everything_worker,
        persistent_workers=(loader_cfg["num_workers"] > 0),
    )

    return ds, loader

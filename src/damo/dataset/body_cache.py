import torch
import numpy as np
import random
from pathlib import Path
from typing import Dict, Any, List, Callable

import damo.utils as utils
from damo.smpl.body_model import create_smpl_model

class BodyCache:
    def __init__(
        self,
        genders,
        capacity,
        synthetic_body_source="caesar_cache",
        synthetic_body_pool_dir=None,
    ):
        self.genders = genders
        self.capacity = capacity
        self.synthetic_body_source = synthetic_body_source
        self.synthetic_body_pool_dir = self._resolve_pool_dir(synthetic_body_pool_dir)
        self.dtype = torch.float32
        self.device = utils.get_device(use_cuda=False)

        self.models = {}
        self.caesar = {}
        self.amass_betas = {}
        self.lbs_weights = {}
        self.topk_weight_jids = {}

        self._cache: Dict[int, Any] = {}
        self._order: List[int] = []

    def _resolve_pool_dir(self, synthetic_body_pool_dir):
        if synthetic_body_pool_dir is None:
            return utils.Paths.data / "synthetic_bodies"

        path = Path(synthetic_body_pool_dir)
        if path.is_absolute():
            return path
        return utils.Paths.root / path

    def get_smpl_model(self, body_type, gender):
        model_key = f"{body_type}_{gender}"

        if model_key not in self.models:
            self.models[model_key] = create_smpl_model(
                model_dir=utils.Paths.smpl_models,
                model_type=body_type,
                gender=gender,
                num_betas=16,
                enable_hand=True
            ).to(dtype=self.dtype, device=self.device).eval()

        return self.models[model_key]

    def get_topk_weight_jids(self, body_type, gender, j22=True, k=3, to_torch=False):
        topk_weight_jids_key = f"{body_type}_{gender}" + ('_j22' if j22 else '') + f"_k{k}"

        if topk_weight_jids_key not in self.topk_weight_jids:
            weights = self.get_weights(body_type, gender, j22, to_torch)
            topk_weight_jids, _ = utils.topk_weight_joints(weights=weights, k=k, normalize=True)
            self.topk_weight_jids[topk_weight_jids_key] = topk_weight_jids

        return self.topk_weight_jids[topk_weight_jids_key]  # [V, 3]

    def get_weights(self, body_type, gender, j22=True, to_torch=False):
        model_key = f"{body_type}_{gender}"
        weights_key = model_key + ('_j22' if j22 else '')

        if weights_key not in self.lbs_weights:
            if j22 and model_key in self.lbs_weights:
                self.lbs_weights[weights_key] = utils.compress_weights_to_j22(
                    weights=self.lbs_weights[model_key],
                    body_type=body_type
                )
            else:
                weights = self.get_smpl_model(body_type, gender).get_weights(to_numpy=not to_torch)
                if j22:
                    weights = utils.compress_weights_to_j22(weights=weights, body_type=body_type)
                self.lbs_weights[weights_key] = weights

        if to_torch:
            return utils.ensure_torch(self.lbs_weights[weights_key], dtype=self.dtype, device=self.device)
        else:
            return utils.ensure_numpy(self.lbs_weights[weights_key])

    def get_caesar(self, gender):
        if gender not in self.caesar:
            path = utils.Paths.smpl_models / f"caesar_{gender}.npz"
            self.caesar[gender] = utils.io_utils.load_npz_to_dict(path)

        return self.caesar[gender]

    def get_random_caesar_body(self, genders: List[str], to_torch=False):
        gender = random.choice(genders)
        caesar = self.get_caesar(gender)
        num_bodies = caesar["vertices"].shape[0]

        body_idx = np.random.randint(0, num_bodies)

        body = self.get_caesar_body(gender, body_idx, to_torch=to_torch)

        return body, gender

    def get_amass_beta_pool(self, gender):
        if gender not in self.amass_betas:
            path = self.synthetic_body_pool_dir / f"amass_smplh_betas_{gender}.npz"
            self.amass_betas[gender] = utils.io_utils.load_npz_to_dict(path)

        return self.amass_betas[gender]

    def get_random_amass_beta_body(self, genders: List[str], to_torch=False):
        gender = random.choice(genders)
        pool = self.get_amass_beta_pool(gender)
        betas = pool["betas"]
        num_bodies = betas.shape[0]
        body_idx = np.random.randint(0, num_bodies)

        body = self.get_smpl_body("smplh", gender, betas[body_idx], to_torch=to_torch)
        return body, gender

    def get_random_synthetic_body(self, genders: List[str], to_torch=False):
        if self.synthetic_body_source == "amass_betas":
            return self.get_random_amass_beta_body(genders, to_torch=to_torch)
        if self.synthetic_body_source == "caesar_cache":
            return self.get_random_caesar_body(genders, to_torch=to_torch)
        raise ValueError(f"Unknown synthetic_body_source: {self.synthetic_body_source}")


    def get_caesar_body(self, gender, idx, to_torch=False):
        vertices = self.get_caesar(gender)["vertices"][idx]
        joints = self.get_caesar(gender)["joints"][idx, :22]

        if to_torch:
            vertices = utils.ensure_torch(vertices, dtype=self.dtype, device=self.device)
            joints = utils.ensure_torch(joints, dtype=self.dtype, device=self.device)

        return {
            "vertices": vertices,
            "joints": joints
        }

    def get_smpl_body(self, body_type, gender, betas, to_torch=False):
        key = utils.betas_to_key(betas)

        if key in self._cache:
            self._order.remove(key)
            self._order.append(key)
            return self._cache[key]

        obj = self._forward(body_type, gender, betas, to_torch)
        self._cache[key] = obj
        self._order.append(key)
        if len(self._order) > self.capacity:
            evict = self._order.pop(0)
            self._cache.pop(evict, None)

        if to_torch:
            vertices = utils.ensure_torch(obj["vertices"], dtype=self.dtype, device=self.device)
            joints = utils.ensure_torch(obj["joints"], dtype=self.dtype, device=self.device)
        else:
            vertices = utils.ensure_numpy(obj["vertices"])
            joints = utils.ensure_numpy(obj["joints"])

        return {
            "vertices": vertices,
            "joints": joints
        }

    def _forward(self, body_type, gender, betas, to_torch=False):
        betas = utils.ensure_torch(betas, device=self.device, dtype=self.dtype)
        model = self.get_smpl_model(body_type, gender)
        v_shaped, joints = model.get_v_shaped(betas)

        if not to_torch:
            v_shaped = utils.ensure_numpy(v_shaped)
            joints = utils.ensure_numpy(joints)

        return {
            "vertices": v_shaped,
            "joints": joints
        }

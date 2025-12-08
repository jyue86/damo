import numpy as np
import torch
from typing import List, Dict, Sequence, Tuple, Optional, Any

from .ensure_types import ensure_device, ensure_torch, ensure_numpy

SMPLX_FINGER_TO_WRIST = {

}

def build_joint_groups(joint2num: Dict[str, int]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}

    head_keywords = {"head", "eye", "jaw"}
    hand_keywords = {"thumb", "index", "middle", "ring", "pinky"}
    left_prefix = "l_"
    right_prefix = "r_"

    head_idxs = [
        idx for name, idx in joint2num.items()
        if any(kw in name.lower() for kw in head_keywords)
    ]
    if head_idxs:
        groups["Head"] = sorted(set(head_idxs))

    left_hand_idxs = [
        idx for name, idx in joint2num.items()
        if any(kw in name.lower() for kw in hand_keywords)
           and name.lower().startswith(left_prefix)
    ]
    if left_hand_idxs:
        groups["L_Hand"] = sorted(set(left_hand_idxs))

    right_hand_idxs = [
        idx for name, idx in joint2num.items()
        if any(kw in name.lower() for kw in hand_keywords)
           and name.lower().startswith(right_prefix)
    ]
    if right_hand_idxs:
        groups["R_Hand"] = sorted(set(right_hand_idxs))

    already = set().union(*groups.values() if groups else set())
    for name, idx in joint2num.items():
        if idx in already:
            continue
        groups[name] = [idx]

    return groups


def betas_to_key(betas, decimals=4):
    b = ensure_numpy(betas)
    b = np.round(b * (10 ** decimals)) / (10 ** decimals)
    return tuple(b.tolist())


def compress_weights_to_j22(weights, body_type):
    if body_type.lower() == "smplx":
        return compress_weights_j55_to_j22(weights)
    elif body_type.lower() == "smplh":
        return compress_weights_j52_to_j22(weights)
    else:
        raise ValueError(f"Invalid body_type: {body_type}")

def compress_weights_j55_to_j22(weights):
    assert weights.ndim == 2 and weights.shape[1] == 55

    targets = {
        20: (25, 40),  # Left fingers (25~39) -> L_Wrist (20)
        21: (40, 55),  # Right fingers (40~54) -> R_Wrist (21)
        15: (22, 25),  # Face (22~24) -> Head (15)
    }

    if torch.is_tensor(weights):
        return _compress_weights_to_j22_torch(weights, targets)
    elif isinstance(weights, np.ndarray):
        return _compress_weights_to_j22_np(weights, targets)
    else:
        raise TypeError(f"Invalid type: {type(weights)}")

def compress_weights_j52_to_j22(weights):
    assert weights.ndim == 2 and weights.shape[1] == 52

    targets = {
        20: (22, 37),  # Left fingers (22~36) -> L_Wrist (20)
        21: (37, 52),  # Right fingers (37~51) -> R_Wrist (21)
    }

    if torch.is_tensor(weights):
        return _compress_weights_to_j22_torch(weights, targets)
    elif isinstance(weights, np.ndarray):
        return _compress_weights_to_j22_np(weights, targets)
    else:
        raise TypeError(f"Invalid type: {type(weights)}")

def _compress_weights_to_j22_torch(weights: torch.Tensor, targets: Dict[int, Tuple[int, int]]) -> torch.Tensor:
    w = weights.clone()

    for k, v in targets.items():
        w[:, k] += w[:, v[0]:v[1]].sum(dim=1)
        w[:, v[0]:v[1]] = 0.0

    w_reduced = w[:, :22].contiguous()

    s = w_reduced.sum(dim=1, keepdim=True).clamp_min(1e-8)
    w_reduced = w_reduced / s

    return w_reduced

def _compress_weights_to_j22_np(weights: np.ndarray, targets: Dict[int, Tuple[int, int]]) -> np.ndarray:
    w = weights.copy()

    for k, v in targets.items():
        w[:, k] += w[:, v[0]:v[1]].sum(axis=1)
        w[:, v[0]:v[1]] = 0.0

    w_reduced = w[:, :22].copy()

    s = np.sum(w_reduced, axis=1, keepdims=True)
    s = np.clip(s, 1e-8, None)
    w_reduced = w_reduced / s

    return w_reduced
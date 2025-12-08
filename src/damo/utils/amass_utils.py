import numpy as np
import torch
from typing import List, Dict, Sequence, Tuple, Optional, Any

from .ensure_types import ensure_device, ensure_torch


def _get_params_for_smplh(
        data,
        betas=True,
        transl=True,
        global_orient=True,
        body_pose=True,
        hand_pose=True,
        dtype=torch.float32,
        device="cpu",
        **kwargs
):
    device = ensure_device(device)
    to_torch = lambda x: ensure_torch(x, dtype=dtype, device=device)
    params = {}

    if betas:
        params["betas"] = to_torch(data["betas"])
    if transl:
        params["transl"] = to_torch(data["trans"])

    if global_orient or body_pose or hand_pose:
        poses = to_torch(data["poses"])

        if global_orient:
            params["global_orient"] = poses[:, :3]
        if body_pose:
            params["body_pose"] = poses[:, 3:66]
        if hand_pose:
            params["left_hand_pose"] = poses[:, 66:111]
            params["right_hand_pose"] = poses[:, 111:]

    return params

def _get_params_for_smplx(
        data,
        betas=True,
        transl=True,
        global_orient=True,
        body_pose=True,
        hand_pose=True,
        dtype=torch.float32,
        device="cpu",
        **kwargs
):
    device = ensure_device(device)
    to_torch = lambda x: ensure_torch(x, dtype=dtype, device=device)
    params = {}

    if betas:
        params["betas"] = to_torch(data["betas"])
    if transl:
        params["transl"] = to_torch(data["trans"])

    if global_orient:
        params["global_orient"] = to_torch(data["root_orient"])
    if body_pose:
        params["body_pose"] = to_torch(data["pose_body"])
    if hand_pose:
        hand_pose = to_torch(data["pose_hand"])
        params["left_hand_pose"] = hand_pose[:, :45]
        params["right_hand_pose"] = hand_pose[:, 45:]

    return params

def get_params_from_amass(
        data,
        body_type=None,
        **kwargs
):
    if body_type is None:
        if "pose_eye" in data:
            body_type = "smplx"
        elif "dmpls" in data:
            body_type = "smplh"
        else:
            raise ValueError("Body type not recognized. Warning: Unknown data.")

    if body_type.lower() == "smplx":
        return _get_params_for_smplx(
            data=data,
            **kwargs
        )
    elif body_type.lower() == "smplh":
        return _get_params_for_smplh(
            data=data,
            **kwargs
        )
    else:
        raise ValueError(f"Invalid body type: {body_type}")
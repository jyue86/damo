from typing import Union, Optional, Tuple, Callable
from dataclasses import dataclass, asdict, fields
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_float(x: torch.Tensor) -> torch.Tensor:
    # For AMP/half
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


def get_device(use_cuda: bool) -> str:
    return "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"


def to_tensor(
        array: Union[np.ndarray, torch.Tensor], dtype=torch.float32
) -> torch.Tensor:
    if torch.is_tensor(array):
        return array
    else:
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


DTYPE_REGISTRY = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "uint16": torch.uint16,
    "uint32": torch.uint32,
    "uint64": torch.uint64,
    "bool": torch.bool,
}
def get_dtype(type_name: str) -> torch.dtype:
    assert type_name in DTYPE_REGISTRY
    return DTYPE_REGISTRY[type_name]

def l2_loss(pred, target, target_mask=None, dim_reduce=None, eps=1e-6):
    if target_mask is None:
        return F.mse_loss(pred, target)
    else:
        return _masked_reduction(
            pred, target, target_mask,
            diff_fn=lambda x: x.pow(2),
            dim_reduce=dim_reduce,
            eps=eps
        )

def l1_loss(pred, target, target_mask=None, dim_reduce=None, eps=1e-6):
    if target_mask is None:
        return (pred - target).abs().mean()
    else:
        return _masked_reduction(
            pred, target, target_mask,
            diff_fn=lambda x: x.abs(),
            dim_reduce=dim_reduce,
            eps=eps
        )

def _masked_reduction(
        pred,
        target,
        mask,
        diff_fn,
        dim_reduce=None,
        eps=1e-6
) -> torch.Tensor:
    m = mask
    M = mask.shape[1]

    while m.dim() < pred.dim():
        m = m.unsqueeze(-1)

    diff = diff_fn(pred[:, :M, ...] - target) * m

    if dim_reduce is None:
        denom = m.sum().clamp_min(eps)
        return diff.sum() / denom
    else:
        dim_reduce = tuple(sorted(dim_reduce))
        num = m.sum(dim=dim_reduce).clamp_min(eps)
        s = diff.sum(dim=dim_reduce)
        return (s / num).mean()

def kl_loss(pred, target, target_mask=None):
    if target_mask is None:
        log_probs = F.log_softmax(pred, dim=-1)
        loss = F.kl_div(log_probs, target, reduction="batchmean")
        return loss

    B, M = target_mask.shape

    logits = pred[:, :M, ...]
    gt = target
    gt = gt / (gt.sum(dim=-1, keepdim=True) + 1e-8)
    valid = target_mask.bool()

    logits_valid = logits[valid]
    gt_valid = gt[valid]

    log_probs_valid = F.log_softmax(logits_valid, dim=-1)

    loss = F.kl_div(log_probs_valid, gt_valid, reduction="batchmean")
    return loss

LOSS_FN_REGISTRY = {
    "l1": l1_loss,
    "l2": l2_loss,
    "kl": kl_loss,
}
def get_loss_fn(fn_name: str) -> Callable:
    return LOSS_FN_REGISTRY[fn_name]


def make_activation(name: str, **kwargs):
    key = name.strip().lower().replace("-", "").replace("_", "")

    if key in ("none", "identity"):
        return nn.Identity()

    if key == "relu":
        return nn.ReLU(inplace=kwargs.get("inplace", False))

    if key in ("silu", "swish"):
        return nn.SiLU(inplace=kwargs.get("inplace", False))

    if key in ("gelu", "gelutanh"):
        approx = kwargs.get("approximate", "tanh" if key != "gelu" else "none")
        if approx not in ("none", "tanh"):
            raise ValueError("GELU approximate must be 'none' or 'tanh'")
        return nn.GELU(approximate=approx)

    if key in ("leakyrelu", "lrelu"):
        return nn.LeakyReLU(
            negative_slope=kwargs.get("negative_slope", 0.01),
            inplace=kwargs.get("inplace", False),
        )

    if key == "elu":
        return nn.ELU(alpha=kwargs.get("alpha", 1.0), inplace=kwargs.get("inplace", False))

    if key == "selu":
        return nn.SELU(inplace=kwargs.get("inplace", False))

    if key == "tanh":
        return nn.Tanh()

    if key == "sigmoid":
        return nn.Sigmoid()

    if key == "mish":
        return nn.Mish()

    raise ValueError(f"Unknown activation: {name}")
import numpy as np
import torch
from typing import Optional

from .ensure_types import ensure_numpy, ensure_torch

def topk_weight_joints(weights, markers_mask=None, k=3, normalize=True):
    """
    weights:      (S, M_max, J)
    markers_mask: (S, M_max)  True = Valid
    return:
        top_idx:     (S, M_max, k)  Invalid -> -1
        top_weights: (S, M_max, k)  Invalid -> 0.0
    """
    assert weights.ndim in [2, 3]
    is_sequential = True

    if weights.ndim == 2:
        is_sequential = False
        w = weights[None, ...]
    else:
        w = weights

    if markers_mask is not None:
        assert markers_mask.shape[-1] == weights.shape[-2]
        if not is_sequential:
            mask = markers_mask[None, ...]
        else:
            mask = markers_mask

        if torch.is_tensor(w):
            idx, vals = _topk_weight_joints_masked_torch(w, mask, k, normalize)
        elif isinstance(w, np.ndarray):
            idx, vals = _topk_weight_joints_masked_np(w, mask, k, normalize)
        else:
            raise TypeError('weights must be a tensor or numpy array')
    else:
        if torch.is_tensor(w):
            idx, vals = _topk_weight_joints_torch(w, k, normalize)
        elif isinstance(w, np.ndarray):
            idx, vals = _topk_weight_joints_np(w, k, normalize)
        else:
            raise TypeError('weights must be a tensor or numpy array')

    if not is_sequential:
        return idx[0], vals[0]

    return idx, vals

def gather_topk_joints(top_idx, values):
    """
    top_idx: (S, M, K)
    values:  (S, M, J, ...)
    return:  (S, M, K, ...)
    """
    assert top_idx.ndim in [2, 3]
    assert values.shape[0] == top_idx.shape[0]

    is_sequential = True
    if top_idx.ndim == 2:
        is_sequential = False
        idx = top_idx[None, ...]
        v = values[None, ...]
    else:
        idx = top_idx
        v = values

    if torch.is_tensor(v):
        result =  _gather_topk_joints_torch(idx, v)
    elif isinstance(v, np.ndarray):
        result =  _gather_topk_joints_np(idx, v)
    else:
        raise TypeError('weights must be a tensor or numpy array')

    if not is_sequential:
        result = result[0]

    return result

def _topk_weight_joints_masked_np(weights: np.ndarray, markers_mask: np.ndarray, k=3, normalize=True):
    w = np.where(markers_mask[..., None], weights, -np.inf)  # (S, M, J)

    idx, vals = _topk_weight_joints_np(weights=w, k=k, normalize=normalize)

    invalid = ~markers_mask  # (S, M) bool
    idx[invalid] = -1
    vals[invalid] = 0.0

    return idx, vals

def _topk_weight_joints_np(weights: np.ndarray, k=3, normalize=True):
    idx = np.argpartition(-weights, k - 1, axis=-1)[..., :k]  # (S, M, k)
    vals = np.take_along_axis(weights, idx, axis=-1)  # (S, M, k)

    order = np.argsort(-vals, axis=-1)  # (S, M, k)
    idx = np.take_along_axis(idx, order, axis=-1)
    vals = np.take_along_axis(vals, order, axis=-1)

    if normalize:
        denom = vals.sum(axis=-1, keepdims=True)  # (S, M, 1)
        denom = np.clip(denom, 1e-8, None)
        vals = vals / denom

    return idx, vals

def _topk_weight_joints_masked_torch(weights: torch.Tensor, markers_mask: torch.Tensor, k=3, normalize=True):
    w = torch.where(markers_mask.unsqueeze(-1), weights,
                    torch.full_like(weights, float('-inf')))

    vals, idx = _topk_weight_joints_torch(weights=w, k=k, normalize=normalize)

    mask_k = markers_mask.unsqueeze(-1).expand_as(vals)
    vals = torch.where(mask_k, vals, torch.zeros_like(vals))
    idx = torch.where(
        mask_k,
        idx,
        torch.full_like(idx, -1),
    )

    return idx, vals

def _topk_weight_joints_torch(weights: torch.Tensor, k=3, normalize=True):

    vals, idx = torch.topk(weights, k, dim=-1)  # (S, M, k)

    if normalize:
        denom = vals.sum(dim=-1, keepdim=True).clamp_min(1e-8)  # (S, M, 1)
        vals = vals / denom

    return idx, vals

def _gather_topk_joints_np(top_idx: np.ndarray, values: np.ndarray):
    S, M, K = top_idx.shape

    safe_idx = np.where(top_idx >= 0, top_idx, 0)  # (S, M, K)
    s_idx = np.arange(S)[:, None, None]  # (S, 1, 1)
    m_idx = np.arange(M)[None, :, None]  # (1, M, 1)

    gathered = values[s_idx, m_idx, safe_idx]

    return gathered

def _gather_topk_joints_torch(top_idx: torch.Tensor, values: torch.Tensor):
    S, M, K = top_idx.shape

    safe_idx = top_idx.clamp_min(0)  # (S, M, K)
    extra_dims = values.shape[3:]
    safe_idx_exp = safe_idx.view(S, M, K, *([1] * len(extra_dims))).expand(
        S, M, K, *extra_dims
    )

    gathered = torch.gather(values, dim=2, index=safe_idx_exp)  # (S, M, K, ...)

    return gathered


def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray, eps: float = 1e-8):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    v0 = vertices[faces[:, 0], :]
    v1 = vertices[faces[:, 1], :]
    v2 = vertices[faces[:, 2], :]

    face_normals = np.cross(v1 - v0, v2 - v0)
    normals = np.zeros_like(vertices)

    np.add.at(normals, faces[:, 0], face_normals)
    np.add.at(normals, faces[:, 1], face_normals)
    np.add.at(normals, faces[:, 2], face_normals)

    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    norm = np.maximum(norm, eps)
    normals /= norm

    return normals.astype(vertices.dtype)

def compute_vertex_area(vertices, faces):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)  # [F]

    area_per_vertex = np.zeros(vertices.shape[0], dtype=np.float64)
    np.add.at(area_per_vertex, faces[:, 0], face_areas / 3.0)
    np.add.at(area_per_vertex, faces[:, 1], face_areas / 3.0)
    np.add.at(area_per_vertex, faces[:, 2], face_areas / 3.0)

    return area_per_vertex  # [V]

def compute_density_weights(vertex_areas, valid_mask, power=1.0):
    weights = np.zeros_like(vertex_areas, dtype=np.float64)

    a = vertex_areas[valid_mask]
    if a.size == 0:
        return weights

    w = a ** power
    w_sum = w.sum()
    if w_sum <= 0:
        return weights

    w /= w_sum
    weights[valid_mask] = w
    return weights

def build_marker_vertex_candidates(
        vertices: np.ndarray,  # [V, 3]
        faces: np.ndarray,
        sole_height: float = 0.02,
        offset_dist: float = 0.01,
        sdf_margin: float = 0.008
):
    normals = compute_vertex_normals(vertices, faces)
    vertex_areas = compute_vertex_area(vertices, faces)

    mask_excl_sole = _exclude_sole_vertices(vertices, sole_height)
    mask_excl_blocked = _exclude_blocked_vertices(vertices, faces, normals, offset_dist, sdf_margin)

    exclude_mask = mask_excl_sole | mask_excl_blocked
    valid_mask = ~exclude_mask

    sample_weights = compute_density_weights(vertex_areas, valid_mask, power=1.0)

    return valid_mask, sample_weights

def _exclude_sole_vertices(vertices, sole_height):
    y = vertices[:, 1]
    min_y = y.min()
    mask_exclude = y < (min_y + sole_height)

    return mask_exclude

def _exclude_blocked_vertices(vertices, faces, normals, offset_dist, sdf_margin):
    import trimesh

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    p_out = vertices + offset_dist * normals  # [V, 3]
    sd = trimesh.proximity.signed_distance(mesh, p_out)  # [V]
    sd = -sd

    mask_exclude = sd < sdf_margin
    return mask_exclude
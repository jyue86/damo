import os, random, hashlib
import numpy as np
import torch
import math

_DISCRETE_GAUSS_CACHE = {}
_CONT_GAUSS_CACHE = {}

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def stable_hash(s: str, seed: int) -> int:
    h = hashlib.blake2b(digest_size=8, person=str(seed).encode())
    h.update(s.encode())
    return int.from_bytes(h.digest(), byteorder="little")

def find_sigma_for_target_prob0(num_values: int, target_p0: float, lo: float=0.1, hi: float=10.0, iters: int=60, print_pdf=False) -> float:
    values = np.arange(num_values)

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        p0 = gaussian_pdf(values, mid)[0]
        if p0 > target_p0:
            lo = mid
        else:
            hi = mid

    sigma = 0.5 * (lo + hi)

    if print_pdf:
        pdf = gaussian_pdf(values, sigma)
        for i, v in enumerate(values):
            print(f"[{v}]: {pdf[i]}")

        print(f"sigma: {sigma}")

    return sigma

def gaussian_pdf(values: np.ndarray, sigma: float) -> np.ndarray:
    pdfs = np.exp(-0.5 * (values / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
    # discrete normalize
    pdfs /= pdfs.sum()
    return pdfs

def _get_discrete_gauss_table(num_values: int, sigma: float):
    key = (int(num_values), float(sigma))
    if key not in _DISCRETE_GAUSS_CACHE:
        ks = np.arange(num_values, dtype=np.int64)
        w = np.exp(-0.5 * (ks / sigma) ** 2)
        w /= w.sum()
        _DISCRETE_GAUSS_CACHE[key] = (ks, w)
    return _DISCRETE_GAUSS_CACHE[key]

def _get_cont_gauss_table(num_bins: int, sigma: float):
    key = (int(num_bins), float(sigma))
    if key not in _CONT_GAUSS_CACHE:
        ks = np.arange(num_bins, dtype=np.int64)
        w = np.exp(-0.5 * (ks / sigma) ** 2)
        w /= w.sum()
        _CONT_GAUSS_CACHE[key] = (ks, w)
    return _CONT_GAUSS_CACHE[key]

def sample_discrete_gaussian(
    min_v: int,
    max_v: int,
    sigma: float,
    size: int | tuple | None = None,
):
    assert min_v <= max_v
    num_values = max_v - min_v + 1

    ks, w = _get_discrete_gauss_table(num_values, sigma)  # ks: [0..num_values-1]

    if size is None:
        k = np.random.choice(ks, p=w)
        return int(min_v + int(k))
    else:
        k = np.random.choice(ks, size=size, p=w)
        return (min_v + k).astype(int)

def sample_continuous_gaussian(
    min_v: float,
    max_v: float,
    sigma: float,
    num_bins: int = 32,
    size: int | tuple | None = None,
):
    if max_v <= min_v:
        if size is None:
            return float(min_v)
        else:
            return np.full(size, float(min_v), dtype=float)

    ks, w = _get_cont_gauss_table(num_bins, sigma)

    if size is None:
        k = np.random.choice(ks, p=w)
        u = np.random.rand()
    else:
        k = np.random.choice(ks, size=size, p=w)
        u = np.random.rand(*((size,) if isinstance(size, int) else size))

    frac = (k + u) / (num_bins - 1)
    return min_v + frac * (max_v - min_v)

def sample_distance_beta(max_dist: float, theta: float) -> float:
    if theta <= 0:
        raise ValueError("theta must be > 0")
    u = np.random.beta(a=1.0, b=theta)
    return float(max_dist * u)


def sample_direction_in_cone(normal: np.ndarray, theta_max_deg: float) -> np.ndarray:
    n = np.asarray(normal, dtype=np.float64)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-8:
        dir_vec = np.random.normal(size=3)
        d_norm = np.linalg.norm(dir_vec)
        if d_norm < 1e-8:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return (dir_vec / d_norm).astype(np.float64)

    n /= n_norm

    if abs(n[0]) < 0.9:
        tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    u = np.cross(n, tmp)
    u_norm = np.linalg.norm(u)
    u /= (u_norm + 1e-8)
    v = np.cross(n, u)

    theta_max = np.deg2rad(theta_max_deg)
    cos_max = math.cos(theta_max)

    u0 = np.random.rand()
    cos_theta = 1.0 - u0 * (1.0 - cos_max)  # [cos_max, 1]
    sin_theta_sq = max(0.0, 1.0 - cos_theta * cos_theta)
    sin_theta = math.sqrt(sin_theta_sq)

    phi = 2.0 * math.pi * np.random.rand()
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)

    dir_vec = (
        cos_theta * n +
        sin_theta * (cos_phi * u + sin_phi * v)
    )
    return dir_vec.astype(np.float64)


def test_code():
    find_sigma_for_target_prob0(7, 0.4, print_pdf=True)

if __name__ == "__main__":
    test_code()
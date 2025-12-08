from pathlib import Path
import torch
import numpy as np

def check_types(value, allowed_types: tuple[type, ...], var_name: str = "variable"):
    if not isinstance(value, allowed_types):
        allowed_names = ", ".join(t.__name__ for t in allowed_types)
        raise TypeError(
            f"Invalid type for {var_name}: {type(value).__name__} "
            f"(expected {allowed_names})"
        )
    return value

def ensure_path(path: str | Path) -> Path:
    if isinstance(path, str):
        path = Path(path)
    elif not isinstance(path, Path):
        raise TypeError(f"Invalid path type: {type(path)} (expected str or Path)")
    return path

def ensure_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    elif torch.is_tensor(x):
        return x.detach().cpu().numpy()
    else:
        raise TypeError(f"Invalid type: {type(x)} (expected torch.Tensor or np.ndarray)")

def ensure_torch(x, device=None, dtype=None):
    if torch.is_tensor(x):
        if device is not None or dtype is not None:
            return x.to(device=device, dtype=dtype)
        return x
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
        if device is not None or dtype is not None:
            return t.to(device=device, dtype=dtype)
        return t
    else:
        raise TypeError(f"Invalid type: {type(x)} (expected np.ndarray or torch.Tensor)")

def ensure_str(s: str | Path | np.ndarray) -> str:
    if isinstance(s, str):
        return s
    elif isinstance(s, Path):
        return str(s)
    elif isinstance(s, np.ndarray):
        return s.item()
    else:
        raise TypeError(f"Invalid type: {type(s)} (expected str or Path or np.ndarray)")

def ensure_device(d: str | torch.device) -> torch.device:
    if d is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(d, torch.device):
        return d
    if isinstance(d, str):
        return torch.device(d)
    raise TypeError(f"Invalid type: {type(d)} (expected str, torch.device or None)")
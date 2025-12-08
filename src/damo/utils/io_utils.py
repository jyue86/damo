from typing import Dict, Any, List
from pathlib import Path
import os
import json
import numpy as np
from omegaconf import OmegaConf

from .ensure_types import ensure_path, check_types

def get_files(
        path: str | Path,
        ext: str | None = None,
        verbose: int | bool = 0
) -> list[Path]:

    path = ensure_path(path)
    check_types(verbose, (int, bool), "verbose")

    if not path.exists() or not path.is_dir():
        print(f"[Error] Invalid path: {path.resolve()}")

    if ext is None:
        pattern = "*"
    else:
        if not ext.startswith("."):
            ext = "." + ext
        pattern = f"*{ext}"

    files = sorted(path.glob(pattern))
    files = [f for f in files if f.is_file()]

    if type(verbose) is int:
        if verbose > 0:
            for i, f in enumerate(files[:verbose]):
                print(f"[{i+1:02d}] {f.name}")

            if len(files) > verbose:
                print("...")

    if verbose > 0:
        print(f"Directory: {path.resolve()}")
        print(f"Number of {'' if ext is None else ext+' '}files: {len(files)}")

    return files

def save_config(cfg, path="config_full.yaml"):
    with open(path, "w") as f:
        OmegaConf.save(config=cfg, f=f.name)

def save_ckpt(model, step: int, path="ckpt.pt"):
    ckpt = {
        "step": step,
        "model": model.state_dict(),
    }
    import torch
    torch.save(ckpt, path)

def load_npz_to_dict(path, allow_pickle=True):
    with np.load(path, allow_pickle=allow_pickle) as f:
        d = {k: f[k] for k in f.files}

    return d

def load_nested_npz(path, delimiter="/"):
    data = np.load(path)
    root = {}

    for flat_key in data.files:
        parts = flat_key.split(delimiter)
        cur = root

        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]

        cur[parts[-1]] = data[flat_key]

    return root

def _flatten_nested(d: Dict[str, Any], prefix: str = "", delimiter: str = "/"):
    flat = {}
    for k, v in d.items():
        key = f"{prefix}{delimiter}{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten_nested(v, key, delimiter))
        else:
            flat[key] = v
    return flat


def save_nested_npz(path: str, nested: Dict[str, Any], delimiter: str = "/") -> None:
    flat = _flatten_nested(nested, delimiter=delimiter)
    np.savez(path, **flat)

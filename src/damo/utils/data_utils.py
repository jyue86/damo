from typing import List, Tuple, Dict, Optional, Any, get_type_hints
from pathlib import Path
import numpy as np
import warnings

class Struct(object):
    def __init__(self, **kwargs):
        hints: dict[str, Any] = get_type_hints(self.__class__)

        for key, val in kwargs.items():
            setattr(self, key, val)

        for name in hints:
            if not hasattr(self, name):
                warnings.warn(
                    f"{self.__class__.__name__}: missing field '{name}' in init kwargs",
                    stacklevel=2,
                )


class FileCache:
    def __init__(self, keys: List[str], capacity: int = 32):
        self.keys = keys
        self.capacity = capacity
        self._cache: Dict[Path, Any] = {}
        self._order: List[Path] = []

    def get(self, path: Path):
        path = path.resolve()

        if path in self._cache:
            self._order.remove(path)
            self._order.append(path)
            return self._cache[path]

        obj = self._load(path)
        self._cache[path] = obj
        self._order.append(path)
        if len(self._order) > self.capacity:
            evict = self._order.pop(0)
            self._cache.pop(evict, None)
        return obj

    def _load(self, path: Path):
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=True) as npz:
                out = {}
                for k in npz.files:
                    if k not in self.keys:
                        continue
                    arr = npz[k]
                    out[k] = arr.item() if arr.ndim == 0 else arr
                return out
        else:
            raise ValueError(f"Unsupported file: {path}")


def build_global_index(
        files: List[Path],
        sequential_data_key: str,
        ratio: Tuple[float, float],
        requires: Optional[Dict[str, List[Any]]] = None,
) -> List[Tuple[int, int]]:

    index: List[Tuple[int, int]] = []

    for fi, fp in enumerate(files):
        with np.load(fp, allow_pickle=True) as npz:

            if requires is not None:
                for k, v in requires.items():
                    if k not in npz:
                        continue

                    arr = npz[k]
                    d = arr.item() if arr.ndim == 0 else arr
                    if d not in v:
                        continue

            t = npz[sequential_data_key].shape[0]
            si = int(t * max(ratio[0], 0))
            ei = int(t * min(ratio[1], 1))

        for i in range(si, ei):
            index.append((fi, i))

    return index


def normalize_ndarray(x):
    x_min = x.min()
    x_max = x.max()
    x_norm = (x - x_min) / (x_max - x_min + 1e-8)

    return x_norm
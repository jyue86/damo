from typing import Union, Optional
from dataclasses import dataclass, asdict, fields
import numpy as np
import torch

from copy import deepcopy

@dataclass
class ModelOutput:
    vertices: Optional[torch.Tensor] = None
    joints: Optional[torch.Tensor] = None
    full_pose: Optional[torch.Tensor] = None
    global_orient: Optional[torch.Tensor] = None
    transl: Optional[torch.Tensor] = None
    v_shaped: Optional[torch.Tensor] = None
    betas: Optional[torch.Tensor] = None
    body_pose: Optional[torch.Tensor] = None
    hand_pose: Optional[torch.Tensor] = None

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __iter__(self):
        return self.keys()

    def keys(self):
        keys = [t.name for t in fields(self)]
        return iter(keys)

    def values(self):
        values = [getattr(self, t.name) for t in fields(self)]
        return iter(values)

    def items(self):
        data = [(t.name, getattr(self, t.name)) for t in fields(self)]
        return iter(data)


@dataclass
class LbsOutput:
    vertices: Optional[torch.Tensor] = None
    joints: Optional[torch.Tensor] = None
    transform_matrix: Optional[torch.Tensor] = None
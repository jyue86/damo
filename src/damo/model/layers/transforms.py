import torch
import torch.nn as nn
import torch.nn.functional as F

class Transpose(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims
        self._name = 'transpose'

    def forward(self, x):
        return x.transpose(*self.dims)


class SDivide(nn.Module):
    def __init__(self, scale):
        super(SDivide, self).__init__()
        self.scale = scale
        self._name = 'scalar_divide'

    def forward(self, x):
        return x / self.scale


class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims
        self._name = 'permute'
    def forward(self, x):
        return x.permute(*self.dims).contiguous()
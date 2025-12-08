import torch
import torch.nn as nn
import torch.nn.functional as F

import damo.utils as utils


class ResConvBlock(nn.Module):
    def __init__(
            self, c_in, c_out, c_hidden, dropout=0.1,
            conv_dim=1, norm="ln", act="gelu_tanh"):
        super().__init__()

        assert conv_dim == 1 or 2, f"conv_dim {conv_dim} must be 1 or 2"
        assert norm in ("bn", "ln"), f"norm {norm} must be 'bn' or 'ln'"

        def make_norm(c):
            if norm == "bn":
                return nn.BatchNorm1d(c) if conv_dim == 1 else nn.BatchNorm2d(c)
            else:
                return nn.GroupNorm(1, c)

        def make_conv(c1, c2):
            return nn.Conv1d(c1, c2, 1, 1) if conv_dim == 1 \
                else nn.Conv2d(c1, c2, 1, 1)

        self.pre = make_norm(c_in)
        self.conv1 = make_conv(c_in, c_hidden)
        self.act = utils.make_activation(act)
        self.conv2 = make_conv(c_hidden, c_out)
        self.drop = nn.Dropout(dropout)
        self.short = make_conv(c_in, c_out) if c_in != c_out else nn.Identity()

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)

        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

        if isinstance(self.short, nn.Conv1d):
            nn.init.xavier_uniform_(self.short.weight)
            nn.init.zeros_(self.short.bias)

    def forward(self, x):
        y = self.pre(x)
        y = self.conv1(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.conv2(y)
        return y + self.short(x)
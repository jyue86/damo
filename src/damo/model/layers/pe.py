import torch
import math

def make_time_pe(S: int, C: int, device=None, dtype=None):
    half = C // 2
    n_freq = half + (C % 2)

    pos = torch.arange(S, device=device, dtype=dtype).unsqueeze(1)  # (S,1)

    div = torch.exp(
        torch.arange(0, n_freq, device=device, dtype=dtype)
        * (-math.log(10000.0) / n_freq)
    )

    pe = torch.zeros(S, C, device=device, dtype=dtype)
    pe[:, 0:2*half:2] = torch.sin(pos * div[:half])
    pe[:, 1:2*half:2] = torch.cos(pos * div[:half])

    if C % 2 == 1:
        pe[:, -1] = torch.sin(pos.squeeze(1) * div[-1])

    return pe.view(1, S, 1, C)  # (1,S,1,C)
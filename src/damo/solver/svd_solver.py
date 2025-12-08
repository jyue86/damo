import torch
import torch.nn as nn
import torch.nn.functional as F


class SVD_Solver(nn.Module):
    def __init__(self, n_joints):
        super().__init__()
        self.n_joints = n_joints

    def forward(self, points, weight, offset):
        """
        points: (B, M, 3)
        weight: (B, M, J)
        offset: (B, M, J, 3)
        returns: R (B, J, 3, 3), t (B, J, 3, 1)
        """
        X = points.permute(0, 2, 1).unsqueeze(1)   # (B,1,3,M)
        Z = offset.permute(0, 2, 3, 1)             # (B,J,3,M)
        w = weight.permute(0, 2, 1).unsqueeze(2)   # (B,J,1,M)
        R, t = SVD_Solver.svd_rot(Z, X, w)
        return R, t

    @staticmethod
    def svd_rot(P, Q, w):
        """
        P,Q: (..., d, n), w: (..., 1, n)
        returns R (..., d, d), t (..., d, 1)
        """
        d, n = P.shape[-2:]
        Pw = torch.sum(P * w, dim=-1) / torch.sum(w, dim=-1)         # (..., d)
        Qw = torch.sum(Q * w, dim=-1) / torch.sum(w, dim=-1)         # (..., d)
        X = P - Pw[..., None]                                        # (..., d, n)
        Y = Q - Qw[..., None]                                        # (..., d, n)
        Yt = Y.transpose(-1, -2)                                     # (..., n, d) -> (..., d, n)^T

        S = X @ Yt                                                   # (..., d, d)
        U, Svals, Vh = torch.linalg.svd(S)                           # U @ diag(S) @ Vh
        V = Vh.transpose(-1, -2)

        det = torch.det(V @ U.transpose(-1, -2))                     # (...,)
        D = torch.eye(d, device=S.device, dtype=S.dtype).expand(S.shape[:-2] + (d, d)).clone()
        D[..., -1, -1] = det
        R = V @ D @ U.transpose(-1, -2)                              # (..., d, d)

        t = Qw[..., None] - R @ Pw[..., None]                        # (..., d, 1)
        return R, t
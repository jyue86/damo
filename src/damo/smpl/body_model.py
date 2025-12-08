import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import pickle
from typing import Union, Optional, Any, Dict

from ..utils.ensure_types import ensure_path, ensure_str
from ..utils.torch_utils import to_tensor, to_np
from ..utils.data_utils import Struct
from .model_output import ModelOutput, LbsOutput


def create_smpl_model(
        model_dir: Union[str, Path],
        model_type: str,
        gender: str = "male",
        ext: Optional[str] = "npz",
        **kwargs
        # gender: str,
        # num_betas: int,
        # batch_size: Optional[int] = 1,
        # v_template: Optional[Union[np.ndarray, torch.Tensor]] = None,
        # dtype: Optional[torch.dtype] = torch.float32,
        # flat_hand: Optional[bool] = True,
        # enable_hand: Optional[bool] = True,
):
    gender = ensure_str(gender)
    model_path = ensure_path(model_dir) / model_type.lower() / f'{model_type.upper()}_{gender.upper()}.{ext}'

    if not model_path.exists():
        ext = "pkl" if ext == "npz" else "npz"
        model_path = ensure_path(model_dir) / model_type.lower() / f'{model_type.upper()}_{gender.upper()}.{ext}'
        if not model_path.exists():
            raise FileNotFoundError()

    if model_type.lower() == 'smplh':
        return Smplh(model_path, gender, **kwargs)
    elif model_type.lower() == 'smplx':
        return Smplx(model_path, gender, **kwargs)


class Smplh(nn.Module):
    NUM_BODY_JOINTS = 21
    NUM_HAND_JOINTS = 15
    NUM_JOINTS = NUM_BODY_JOINTS + 2 * NUM_HAND_JOINTS

    GENDER2ID = {"male": 0, "female": 1}
    ID2GENDER = {0: "male", 1: "female"}

    def __init__(
            self,
            model_path: Union[str, Path],
            gender: str,
            num_betas: int,
            batch_size: Optional[int] = 1,
            v_template: Optional[Union[np.ndarray, torch.Tensor]] = None,
            dtype: Optional[torch.dtype] = torch.float32,
            flat_hand: Optional[bool] = True,
            enable_hand: Optional[bool] = True,
            model_data: Optional[Struct] = None,
            **kwargs
    ):
        super().__init__()

        if not flat_hand:
            assert enable_hand is True

        self.gender = gender
        self.num_betas = num_betas
        self.dtype = dtype
        self.flat_hand = flat_hand
        self.enable_hand = enable_hand

        self.faces: np.ndarray

        self.shapedirs: torch.Tensor
        self.faces_tensor: torch.Tensor
        self.v_template: torch.Tensor
        self.j_regressor: torch.Tensor
        self.posedirs: torch.Tensor
        self.parents: torch.Tensor
        self.lbs_weights: torch.Tensor
        self.pose_mean: torch.Tensor

        self.left_hand_mean: torch.Tensor
        self.right_hand_mean: torch.Tensor

        self.betas: torch.Tensor
        self.global_orient: torch.Tensor
        self.body_pose: torch.Tensor
        self.left_hand_pose: torch.Tensor
        self.right_hand_pose: torch.Tensor
        self.transl: torch.Tensor

        if model_data is None:
            model_data = load_model(model_path)

        shapedirs = model_data.shapedirs[:, :, :self.num_betas]  # [V, 3, 16]
        self.register_buffer(
            'shapedirs',
            to_tensor(to_np(shapedirs), dtype=dtype)
        )
        self.faces = model_data.f
        self.register_buffer(
            'faces_tensor',
            to_tensor(to_np(self.faces, dtype=np.int64), dtype=torch.long)
        )

        if v_template is None:
            v_template = model_data.v_template
        if not torch.is_tensor(v_template):
            v_template = to_tensor(to_np(v_template), dtype=dtype)
        self.register_buffer('v_template', v_template)

        j_regressor = to_tensor(to_np(model_data.J_regressor), dtype=dtype)
        self.register_buffer('j_regressor', j_regressor)

        num_pose_basis = self.NUM_BODY_JOINTS * 9
        if self.enable_hand:
            num_pose_basis = self.NUM_JOINTS * 9

        posedirs = np.reshape(model_data.posedirs[:, :, :num_pose_basis], [-1, num_pose_basis]).T
        self.register_buffer('posedirs', to_tensor(to_np(posedirs), dtype=dtype))

        parents = to_tensor(to_np(model_data.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)

        lbs_weights = to_tensor(to_np(model_data.weights), dtype=dtype)
        self.register_buffer('lbs_weights', lbs_weights)

        # --- Parameters ---
        default_betas = torch.zeros([batch_size, self.num_betas], dtype=dtype)
        self.register_buffer(
            'betas',
            default_betas
        )
        default_global_orient = torch.zeros([batch_size, 3], dtype=dtype)
        self.register_buffer(
            'global_orient',
            default_global_orient
        )
        default_body_pose = torch.zeros([batch_size, self.NUM_BODY_JOINTS * 3], dtype=dtype)
        self.register_buffer(
            'body_pose',
            default_body_pose
        )
        default_left_hand_pose = torch.zeros([batch_size, self.NUM_HAND_JOINTS * 3], dtype=dtype)
        self.register_buffer(
            'left_hand_pose',
            default_left_hand_pose
        )
        default_right_hand_pose = torch.zeros([batch_size, self.NUM_HAND_JOINTS * 3], dtype=dtype)
        self.register_buffer(
            'right_hand_pose',
            default_right_hand_pose
        )
        default_transl = torch.zeros([batch_size, 3], dtype=dtype)
        self.register_buffer(
            'transl',
            default_transl
        )

        # --- Mean pose ---
        if self.enable_hand:
            if not self.flat_hand:
                left_hand_mean = model_data.hands_meanl
                right_hand_mean = model_data.hands_meanr
            else:
                left_hand_mean = np.zeros_like(model_data.hands_meanl)
                right_hand_mean = np.zeros_like(model_data.hands_meanr)

            self.register_buffer('left_hand_mean', to_tensor(left_hand_mean, dtype=self.dtype))
            self.register_buffer('right_hand_mean', to_tensor(right_hand_mean, dtype=self.dtype))

        pose_mean_tensor = self.create_mean_pose()
        if not torch.is_tensor(pose_mean_tensor):
            pose_mean_tensor = to_tensor(to_np(pose_mean_tensor), dtype=dtype)
        self.register_buffer('pose_mean', pose_mean_tensor)

    def get_weights(self, to_numpy=False):
        if to_numpy:
            return self.lbs_weights.detach().clone().cpu().numpy()
        else:
            return self.lbs_weights.detach().clone()

    def get_regressor(self, to_numpy=False):
        if to_numpy:
            return self.j_regressor.detach().clone().cpu().numpy()
        else:
            return self.j_regressor.detach().clone()

    def create_mean_pose(self):
        global_orient_mean = torch.zeros([3], dtype=self.dtype)
        body_pose_mean = torch.zeros([self.NUM_BODY_JOINTS * 3], dtype=self.dtype)

        if self.enable_hand:
            pose_mean = np.concatenate([
                global_orient_mean,
                body_pose_mean,
                self.left_hand_mean,
                self.right_hand_mean
            ], axis=0)
        else:
            pose_mean = np.concatenate([global_orient_mean, body_pose_mean], axis=0)

        return pose_mean

    def forward(
            self,
            betas: Optional[torch.Tensor] = None,
            transl: Optional[torch.Tensor] = None,
            global_orient: Optional[torch.Tensor] = None,
            body_pose: Optional[torch.Tensor] = None,
            left_hand_pose: Optional[torch.Tensor] = None,
            right_hand_pose: Optional[torch.Tensor] = None,
            apply_trans: Optional[bool] = True,
    ):
        if self.enable_hand:
            if not self.flat_hand:
                assert left_hand_pose is None and right_hand_pose is None

        if transl is None:
            transl = self.transl
        if betas is None:
            betas = self.betas
        if global_orient is None:
            global_orient = self.global_orient
        if body_pose is None:
            body_pose = self.body_pose

        if betas.ndim == 1:
            betas = betas.unsqueeze(0)

        batch_size = max(betas.shape[0], global_orient.shape[0], body_pose.shape[0])

        if left_hand_pose is None:
            left_hand_pose = self.left_hand_pose.expand(batch_size, -1)
        if right_hand_pose is None:
            right_hand_pose = self.right_hand_pose.expand(batch_size, -1)

        pose_dim = 3 + self.NUM_BODY_JOINTS * 3  # 66
        if self.enable_hand:
            pose_dim = 3 + self.NUM_JOINTS * 3  # 168

        full_pose = torch.cat([
            global_orient.reshape(-1, 1, 3),
            body_pose.reshape(-1, self.NUM_BODY_JOINTS, 3)
        ], dim=1)

        if self.enable_hand:
            if self.flat_hand:
                full_pose = torch.cat([
                    full_pose,
                    left_hand_pose.reshape(-1, self.NUM_HAND_JOINTS, 3),
                    right_hand_pose.reshape(-1, self.NUM_HAND_JOINTS, 3)
                ], dim=1)
            else:
                full_pose = torch.cat([
                    full_pose,
                    torch.zeros((batch_size, self.NUM_FACE_JOINTS, 3), dtype=full_pose.dtype, device=full_pose.device),
                    torch.zeros((batch_size, self.NUM_HAND_JOINTS * 2, 3), dtype=full_pose.dtype, device=full_pose.device)
                ], dim=1)

        full_pose = full_pose.reshape(-1, pose_dim)  # [B, Jx3]

        # full_pose += self.pose_mean

        if betas.shape[0] == 1:
            betas = betas.expand(batch_size, -1)

        # [V, 3] + [B, V, 3] -> [B, V, 3]
        v_shaped = self.v_template + blend_shapes(betas, self.shapedirs)  # [B, V, 3]

        vertices, joints = self.lbs(betas, full_pose, v_shaped)

        if apply_trans:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)

        output = ModelOutput(
            v_shaped=v_shaped,
            vertices=vertices,
            joints=joints,
            betas=betas,
            full_pose=full_pose
        )
        return output

    def get_v_shaped(self, betas):
        if len(betas.shape) == 1:
            betas = betas.unsqueeze(0)

        v_shaped = self.v_template + blend_shapes(betas, self.shapedirs)
        joints = vertices2joints(self.j_regressor, v_shaped)

        return v_shaped[0], joints[0]

    def lbs(self, betas, pose, v_shaped):
        batch_size = max(betas.shape[0], pose.shape[0])
        device, dtype = betas.device, betas.dtype

        # [V, 3] + [B, V, 3] -> [B, V, 3]
        # v_shaped = self.v_template + blend_shapes(betas, self.shapedirs)  # [B, V, 3]
        j = vertices2joints(self.j_regressor, v_shaped)  # [B, J(55), 3]

        ident = torch.eye(3, dtype=dtype, device=device)
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view(
            [batch_size, -1, 3, 3])  # [B, J(22 or 55), 3, 3]

        # P: 189 or 486
        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])  # [B, P]
        pose_feature_dim = pose_feature.shape[-1]  # P

        # [B, P] @ [P, Vx3] -> [B, Vx3]
        pose_offsets = torch.matmul(
            pose_feature, self.posedirs[:pose_feature_dim, :]
        ).view(batch_size, -1, 3)  # [B, V, 3]

        v_posed = pose_offsets + v_shaped  # [B, V, 3]

        if rot_mats.shape[1] < j.shape[1]:
            ident = torch.eye(3, dtype=dtype, device=device)
            j_diff = j.shape[1] - rot_mats.shape[1]
            rot_mats = torch.cat([
                rot_mats,
                ident.view(1, 1, 3, 3).expand(batch_size, j_diff, 3, 3)
            ], dim=1)  # [B, J(55), 3, 3]

        # j_transformed: [B, J(55), 3]
        # t_j_rel: [B, J(55), 4, 4]
        j_transformed, t_j_rel = batch_rigid_transform(rot_mats, j, self.parents, dtype=dtype)

        w = self.lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])  # [B, V, J(55)]
        num_joints = self.j_regressor.shape[0]
        t = torch.matmul(w, t_j_rel.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)  # [B, V, 4, 4]

        homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1], dtype=dtype, device=device)  # [B, V, 1]
        v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)  # [B, V, 4]
        v_homo = torch.matmul(t, torch.unsqueeze(v_posed_homo, dim=-1))  # [B, V, 4, 1]
        v = v_homo[:, :, :3, 0]

        return v, j_transformed

    def fk_with_reference(
            self,
            vertices: torch.Tensor,
            joints: torch.Tensor,
            weights: torch.Tensor,
            trans: torch.Tensor,
            poses: torch.Tensor,
            vids: torch.Tensor,
    ):
        """
        vertices: (V, 3)
        joints: (J, 3)
        weights: (V, J)
        trans: (B, 3)
        poses: (B, 3 x J)
        vids: (V)

        return
            v: (B, V, 3)
            j: (B, J, 3)
        """
        batch_size = poses.shape[0]
        V, J = weights.shape
        pose_dim = J * 3
        assert poses.shape[1] == pose_dim

        device, dtype = poses.device, poses.dtype

        v_shaped = vertices.clone()
        j_shaped = joints.clone()

        # --- LBS ---
        if v_shaped.ndim == 2:
            v_shaped = v_shaped.unsqueeze(0).expand(batch_size, -1, -1)
        if j_shaped.ndim == 2:
            j_shaped = j_shaped.unsqueeze(0).expand(batch_size, -1, -1)

        ident = torch.eye(3, dtype=dtype, device=device)
        rot_mats = batch_rodrigues(poses.view(-1, 3)).view(
            [batch_size, -1, 3, 3])

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        pose_feature_dim = pose_feature.shape[-1]

        vids_offsets = torch.arange(3, device=device)
        flat_vids_extended = (vids[:, None] * 3 + vids_offsets[None, :]).reshape(-1)
        picked_posedirs = self.posedirs[:pose_feature_dim, :].index_select(dim=1, index=flat_vids_extended)

        pose_offsets = torch.matmul(pose_feature, picked_posedirs).view(batch_size, -1, 3)

        v_posed = pose_offsets + v_shaped

        if rot_mats.shape[1] < j_shaped.shape[1]:
            ident = torch.eye(3, dtype=dtype, device=device)
            j_diff = j_shaped.shape[1] - rot_mats.shape[1]
            rot_mats = torch.cat([
                rot_mats,
                ident.view(1, 1, 3, 3).expand(batch_size, j_diff, 3, 3)
            ], dim=1)  # [B, J, 3, 3]

        # j_transformed: [B, J, 3]
        # t_j_rel: [B, J, 4, 4]
        j_transformed, t_j_rel = batch_rigid_transform(rot_mats, j_shaped, self.parents[:J], dtype=dtype)

        w = weights.unsqueeze(dim=0).expand([batch_size, -1, -1])  # [B, V, J]
        t = torch.matmul(w, t_j_rel.view(batch_size, J, 16)).view(batch_size, -1, 4, 4)  # [B, V, 4, 4]

        homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1], dtype=dtype, device=device)  # [B, V, 1]
        v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)  # [B, V, 4]
        v_homo = torch.matmul(t, torch.unsqueeze(v_posed_homo, dim=-1))  # [B, V, 4, 1]
        v = v_homo[:, :, :3, 0]

        output = LbsOutput(
            vertices=v,
            joints=j_transformed,
            transform_matrix=t,
        )
        return output


class Smplx(Smplh):
    NUM_FACE_JOINTS = 3
    NUM_JOINTS = Smplh.NUM_BODY_JOINTS + 2 * Smplh.NUM_HAND_JOINTS + NUM_FACE_JOINTS

    GENDER2ID = {"male": 0, "female": 1, "neutral": 2}
    ID2GENDER = {0: "male", 1: "female", 2: "neutral"}

    def __init__(
            self,
            model_path: Union[str, Path],
            gender: str,
            num_betas: int,
            batch_size: Optional[int] = 1,
            v_template: Optional[Union[np.ndarray, torch.Tensor]] = None,
            dtype: Optional[torch.dtype] = torch.float32,
            flat_hand: Optional[bool] = True,
            enable_hand: Optional[bool] = True,
            enable_expression: Optional[bool] = False,
            **kwargs
    ):
        model_data = load_model(model_path)

        super(Smplx, self).__init__(
            model_path=model_path,
            gender=gender,
            num_betas=num_betas,
            batch_size=batch_size,
            v_template=v_template,
            dtype=dtype,
            flat_hand=flat_hand,
            enable_hand=enable_hand,
            model_data=model_data
        )

    def create_mean_pose(self):
        global_orient_mean = torch.zeros([3], dtype=self.dtype)
        body_pose_mean = torch.zeros([self.NUM_BODY_JOINTS * 3], dtype=self.dtype)
        face_pose_mean = torch.zeros([self.NUM_FACE_JOINTS * 3], dtype=self.dtype)

        if self.enable_hand:
            pose_mean = np.concatenate([
                global_orient_mean,
                body_pose_mean,
                face_pose_mean,
                self.left_hand_mean,
                self.right_hand_mean
            ], axis=0)
        else:
            pose_mean = np.concatenate([global_orient_mean, body_pose_mean], axis=0)

        return pose_mean

    def forward(
            self,
            betas: Optional[torch.Tensor] = None,
            transl: Optional[torch.Tensor] = None,
            global_orient: Optional[torch.Tensor] = None,
            body_pose: Optional[torch.Tensor] = None,
            left_hand_pose: Optional[torch.Tensor] = None,
            right_hand_pose: Optional[torch.Tensor] = None,
            apply_trans: Optional[bool] = True,
    ):
        if self.enable_hand:
            if not self.flat_hand:
                assert left_hand_pose is None and right_hand_pose is None

        if transl is None:
            transl = self.transl
        if betas is None:
            betas = self.betas
        if global_orient is None:
            global_orient = self.global_orient
        if body_pose is None:
            body_pose = self.body_pose

        if betas.ndim == 1:
            betas = betas.unsqueeze(0)

        batch_size = max(betas.shape[0], global_orient.shape[0], body_pose.shape[0])

        if left_hand_pose is None:
            left_hand_pose = self.left_hand_pose.expand(batch_size, -1)
        if right_hand_pose is None:
            right_hand_pose = self.right_hand_pose.expand(batch_size, -1)

        pose_dim = 3 + self.NUM_BODY_JOINTS * 3  # 66
        if self.enable_hand:
            pose_dim = 3 + self.NUM_JOINTS * 3  # 168

        full_pose = torch.cat([
            global_orient.reshape(-1, 1, 3),
            body_pose.reshape(-1, self.NUM_BODY_JOINTS, 3)
        ], dim=1)

        if self.enable_hand:
            if self.flat_hand:
                full_pose = torch.cat([
                    full_pose,
                    torch.zeros((batch_size, self.NUM_FACE_JOINTS, 3), dtype=full_pose.dtype, device=full_pose.device),
                    left_hand_pose.reshape(-1, self.NUM_HAND_JOINTS, 3),
                    right_hand_pose.reshape(-1, self.NUM_HAND_JOINTS, 3)
                ], dim=1)
            else:
                full_pose = torch.cat([
                    full_pose,
                    torch.zeros((batch_size, self.NUM_FACE_JOINTS, 3), dtype=full_pose.dtype, device=full_pose.device),
                    torch.zeros((batch_size, self.NUM_HAND_JOINTS * 2, 3), dtype=full_pose.dtype, device=full_pose.device)
                ], dim=1)

        full_pose = full_pose.reshape(-1, pose_dim)  # [B, Jx3]
        # full_pose += self.pose_mean

        if betas.shape[0] == 1:
            betas = betas.expand(batch_size, -1)

        # [V, 3] + [B, V, 3] -> [B, V, 3]
        v_shaped = self.v_template + blend_shapes(betas, self.shapedirs)  # [B, V, 3]

        vertices, joints = self.lbs(betas, full_pose, v_shaped)

        if apply_trans:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)

        output = ModelOutput(
            v_shaped=v_shaped,
            vertices=vertices,
            joints=joints,
            betas=betas,
            full_pose=full_pose
        )
        return output


def blend_shapes(betas, shapedirs):
    blend_shape = torch.einsum('bl,mkl->bmk', [betas, shapedirs])
    return blend_shape

def vertices2joints(j_regressor, vertices):
    joints = torch.einsum('bik,ji->bjk', [vertices, j_regressor])
    return joints

def batch_rodrigues(rot_vecs, epsilon=1e-8):
    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
    """
    Applies a batch of rigid transformations to the joints

        Parameters
        ----------
        rot_mats : torch.tensor BxNx3x3
            Tensor of rotation matrices
        joints : torch.tensor BxNx3
            Locations of joints
        parents : torch.tensor N
            The kinematic tree of each object
        dtype : torch.dtype, optional:
            The data type of the created tensors, the default is torch.float32

        Returns
        -------
        posed_joints : torch.tensor BxNx3
            The locations of the joints after applying the pose rotations
        rel_transforms : torch.tensor BxNx4x4
            The relative (with respect to the root joint) rigid transformations
            for all the joints
    """
    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms

def transform_mat(R, t):
    """
    Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    """
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def load_model(path):
    path = ensure_path(path)

    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as z:
            data = Struct(**z)

        return data
    else:
        with open(path, "rb") as f:
            data = Struct(**pickle.load(f))

        return data

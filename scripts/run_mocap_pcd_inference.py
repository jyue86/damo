#!/usr/bin/env python
"""Run DAMO inference and videos for mocap point-cloud CSV exports."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import damo.utils as utils
from damo.smpl.body_model import create_smpl_model
from damo.solver.svd_solver import SVD_Solver


@dataclass
class MocapPcdTake:
    path: Path
    points: np.ndarray
    mask: np.ndarray
    frames: np.ndarray
    times: np.ndarray
    fps: float


def load_mocap_pcd_csv(path: str | Path, max_frames: int | None = None) -> MocapPcdTake:
    path = Path(path)
    fps = _read_export_fps(path)
    df = pd.read_csv(path, header=5)
    if max_frames is not None:
        df = df.iloc[:max_frames]

    coords = df.iloc[:, 2:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    if coords.shape[1] % 3 != 0:
        raise ValueError(f"{path} has {coords.shape[1]} coordinate columns, not divisible by 3")

    points_nan = coords.reshape(coords.shape[0], coords.shape[1] // 3, 3)
    mask = np.isfinite(points_nan).all(axis=-1)
    points = np.nan_to_num(points_nan, nan=0.0).astype(np.float32)

    frames = pd.to_numeric(df["Frame"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    times = pd.to_numeric(df["Time (Seconds)"], errors="coerce").to_numpy(dtype=np.float32)
    if not np.isfinite(fps) and len(times) > 1:
        dt = np.diff(times[np.isfinite(times)])
        fps = float(1.0 / np.median(dt)) if dt.size else 30.0
    if not np.isfinite(fps):
        fps = 30.0

    return MocapPcdTake(path=path, points=points, mask=mask, frames=frames, times=times, fps=fps)


def build_marker_windows(points: np.ndarray, mask: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    if seq_len % 2 != 1:
        raise ValueError(f"seq_len must be odd, got {seq_len}")
    if points.shape[:2] != mask.shape:
        raise ValueError(f"points/mask shape mismatch: {points.shape[:2]} vs {mask.shape}")

    half = seq_len // 2
    frame_idx = np.arange(points.shape[0])
    window_idx = np.clip(frame_idx[:, None] + np.arange(-half, half + 1)[None, :], 0, points.shape[0] - 1)
    return points[window_idx], mask[window_idx]


@torch.no_grad()
def run_model(model, windows: np.ndarray, window_mask: np.ndarray, device: torch.device, batch_size: int):
    outputs = []
    model.eval()
    for start in tqdm(range(0, len(windows), batch_size), desc="infer", leave=False):
        end = start + batch_size
        markers_seq = torch.as_tensor(windows[start:end], dtype=torch.float32, device=device)
        markers_seq_mask = torch.as_tensor(window_mask[start:end], dtype=torch.bool, device=device)
        out = model(markers_seq=markers_seq, markers_seq_mask=markers_seq_mask)
        outputs.append({k: v.detach().cpu() for k, v in out.items()})

    return {
        k: torch.cat([out[k] for out in outputs], dim=0)
        for k in outputs[0]
    }


@torch.no_grad()
def estimate_joint_pose(points: np.ndarray, model_out: dict[str, torch.Tensor], n_joints: int):
    mid_points = torch.as_tensor(points, dtype=torch.float32)
    markers_mask = model_out["markers_mask"].bool()
    logits = model_out["weights"]
    rep_weights = model_out["rep_weights"]
    rep_offsets = model_out["rep_offsets"]

    joint_probs = F.softmax(logits, dim=-1)[..., :n_joints]
    k = rep_offsets.shape[2]
    top_idx = torch.topk(joint_probs, k=k, dim=-1).indices

    full_weights = torch.zeros((*rep_weights.shape[:2], n_joints), dtype=rep_weights.dtype)
    full_offsets = torch.zeros((*rep_weights.shape[:2], n_joints, 3), dtype=rep_offsets.dtype)

    valid = markers_mask.unsqueeze(-1)
    full_weights.scatter_add_(2, top_idx, rep_weights * valid)
    full_offsets.scatter_(2, top_idx.unsqueeze(-1).expand_as(rep_offsets), rep_offsets * valid.unsqueeze(-1))

    support = (full_weights > 1e-4).sum(dim=1)
    valid_joints = support >= 3

    rotations = torch.full((points.shape[0], n_joints, 3, 3), torch.nan, dtype=torch.float32)
    translations = torch.full((points.shape[0], n_joints, 3, 1), torch.nan, dtype=torch.float32)
    if valid_joints.any():
        X = mid_points.permute(0, 2, 1).unsqueeze(1).expand(-1, n_joints, -1, -1)
        Z = full_offsets.permute(0, 2, 3, 1)
        w = full_weights.permute(0, 2, 1).unsqueeze(2)
        rot_valid, trans_valid = SVD_Solver.svd_rot(Z[valid_joints], X[valid_joints], w[valid_joints])
        rotations[valid_joints] = rot_valid
        translations[valid_joints] = trans_valid

    joint_positions = translations.squeeze(-1)

    return {
        "joint_rotations": rotations.numpy(),
        "joint_positions": joint_positions.numpy(),
        "joint_valid": valid_joints.numpy(),
        "joint_probs": joint_probs.numpy(),
        "top_joint_idx": top_idx.numpy(),
    }


def mocap_to_view_coords(points: np.ndarray) -> np.ndarray:
    """Map mocap coordinates (X left, Y up, Z forward) to plot coordinates."""
    p = np.asarray(points)
    return np.stack((-p[..., 0], p[..., 2], p[..., 1]), axis=-1)


def rotations_to_smpl_params(
    joint_rotations: np.ndarray,
    joint_positions: np.ndarray,
    joint_valid: np.ndarray,
    parents: np.ndarray,
    rest_root: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T, J = joint_rotations.shape[:2]
    local_rot = np.tile(np.eye(3, dtype=np.float32), (T, J, 1, 1))

    finite_rot = np.isfinite(joint_rotations).all(axis=(-2, -1))
    valid = joint_valid & finite_rot
    for t in range(T):
        for j in range(J):
            if not valid[t, j]:
                continue
            parent = parents[j]
            if parent < 0 or not valid[t, parent]:
                local_rot[t, j] = joint_rotations[t, j]
            else:
                local_rot[t, j] = joint_rotations[t, parent].T @ joint_rotations[t, j]

    rotvec = Rotation.from_matrix(local_rot.reshape(-1, 3, 3)).as_rotvec().astype(np.float32).reshape(T, J, 3)
    root_pos = joint_positions[:, 0].copy()
    root_ok = valid[:, 0] & np.isfinite(root_pos).all(axis=-1)
    fallback_root = np.nanmedian(np.where(np.isfinite(joint_positions), joint_positions, np.nan), axis=1)
    root_pos[~root_ok] = np.nan_to_num(fallback_root[~root_ok], nan=0.0)
    transl = root_pos - rest_root.reshape(1, 3)

    return rotvec[:, 0], rotvec[:, 1:22].reshape(T, -1), transl.astype(np.float32)


@torch.no_grad()
def build_smpl_body_sequence(
    joint_rotations: np.ndarray,
    joint_positions: np.ndarray,
    joint_valid: np.ndarray,
    *,
    body_model_type: str,
    gender: str,
    betas: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    body_model = create_smpl_model(
        model_dir=utils.Paths.smpl_models,
        model_type=body_model_type,
        gender=gender,
        num_betas=int(betas.shape[0]),
        enable_hand=False,
    ).to(device).eval()
    parents = body_model.parents[:22].detach().cpu().numpy()

    zero = torch.zeros(1, int(betas.shape[0]), dtype=torch.float32, device=device)
    rest = body_model(betas=zero, apply_trans=False)
    rest_root = rest.joints[0, 0].detach().cpu().numpy()

    global_orient, body_pose, transl = rotations_to_smpl_params(
        joint_rotations,
        joint_positions,
        joint_valid,
        parents,
        rest_root,
    )

    vertices = []
    joints = []
    betas_t = torch.as_tensor(betas, dtype=torch.float32, device=device).view(1, -1)
    for start in tqdm(range(0, len(global_orient), batch_size), desc="smpl", leave=False):
        end = start + batch_size
        out = body_model(
            betas=betas_t,
            transl=torch.as_tensor(transl[start:end], dtype=torch.float32, device=device),
            global_orient=torch.as_tensor(global_orient[start:end], dtype=torch.float32, device=device),
            body_pose=torch.as_tensor(body_pose[start:end], dtype=torch.float32, device=device),
        )
        vertices.append(out.vertices.detach().cpu().numpy().astype(np.float32))
        joints.append(out.joints[:, :22].detach().cpu().numpy().astype(np.float32))

    return {
        "vertices": np.concatenate(vertices, axis=0),
        "joints": np.concatenate(joints, axis=0),
        "faces": np.asarray(body_model.faces, dtype=np.int32),
        "parents": parents,
        "global_orient": global_orient,
        "body_pose": body_pose,
        "transl": transl,
        "betas": betas.astype(np.float32),
    }


def write_point_video(
    path: str | Path,
    points: np.ndarray,
    mask: np.ndarray,
    *,
    joints: np.ndarray | None = None,
    joint_valid: np.ndarray | None = None,
    parents: np.ndarray | None = None,
    fps: float = 30.0,
    title: str = "",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    view_points = mocap_to_view_coords(points)
    view_joints = mocap_to_view_coords(joints) if joints is not None else None
    limits = _axis_limits(view_points, mask, view_joints, joint_valid)
    writer = FFMpegWriter(fps=fps, bitrate=2400)

    with writer.saving(fig, str(path), dpi=120):
        for t in tqdm(range(points.shape[0]), desc=path.name, leave=False):
            ax.clear()
            _style_axis(ax, limits, title)
            pts = view_points[t, mask[t]]
            if pts.size:
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=10, c="#2f6fed", alpha=0.8)

            if joints is not None and joint_valid is not None:
                j = view_joints[t]
                v = joint_valid[t] & np.isfinite(j).all(axis=-1)
                if v.any():
                    ax.scatter(j[v, 0], j[v, 1], j[v, 2], s=24, c="#d13f31", alpha=0.95)
                if parents is not None:
                    for child, parent in enumerate(parents):
                        if parent < 0 or child >= len(v) or parent >= len(v) or not (v[child] and v[parent]):
                            continue
                        seg = j[[parent, child]]
                        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], c="#202020", linewidth=1.8)

            writer.grab_frame()

    plt.close(fig)


def write_body_video(
    path: str | Path,
    points: np.ndarray,
    mask: np.ndarray,
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray,
    joint_valid: np.ndarray,
    parents: np.ndarray,
    fps: float = 30.0,
    title: str = "",
    mesh_face_stride: int = 8,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    view_points = mocap_to_view_coords(points)
    view_vertices = mocap_to_view_coords(vertices)
    view_joints = mocap_to_view_coords(joints)
    limits = _axis_limits(view_points, mask, view_joints, joint_valid)
    plot_faces = faces[::max(1, mesh_face_stride)]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    writer = FFMpegWriter(fps=fps, bitrate=3000)

    with writer.saving(fig, str(path), dpi=120):
        for t in tqdm(range(points.shape[0]), desc=path.name, leave=False):
            ax.clear()
            _style_axis(ax, limits, title)
            pts = view_points[t, mask[t]]
            if pts.size:
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=7, c="#2f6fed", alpha=0.35)

            mesh = Poly3DCollection(
                view_vertices[t][plot_faces],
                facecolor=(0.75, 0.72, 0.62, 0.48),
                edgecolor=(0.24, 0.24, 0.24, 0.22),
                linewidth=0.18,
            )
            ax.add_collection3d(mesh)

            j = view_joints[t]
            v = joint_valid[t] & np.isfinite(j).all(axis=-1)
            if v.any():
                ax.scatter(j[v, 0], j[v, 1], j[v, 2], s=18, c="#d13f31", alpha=0.95)
            for child, parent in enumerate(parents):
                if parent < 0 or child >= len(v) or parent >= len(v) or not (v[child] and v[parent]):
                    continue
                seg = j[[parent, child]]
                ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], c="#111111", linewidth=1.5)

            writer.grab_frame()

    plt.close(fig)


def load_model_and_cfg(config_dir: Path, checkpoint: Path, device: torch.device):
    with initialize_config_dir(version_base=None, config_dir=str(config_dir.resolve())):
        cfg = compose(config_name="config")
    model = instantiate(cfg.model).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    return model, cfg


def load_body_parents(body_model_type: str, gender: str) -> np.ndarray:
    model = create_smpl_model(
        model_dir=utils.Paths.smpl_models,
        model_type=body_model_type,
        gender=gender,
        num_betas=16,
        enable_hand=False,
    )
    return model.parents[:22].detach().cpu().numpy()


def load_betas(path: Path | None, num_betas: int = 16) -> np.ndarray:
    if path is None:
        return np.zeros(num_betas, dtype=np.float32)
    data = np.load(path)
    if isinstance(data, np.lib.npyio.NpzFile):
        if "betas" not in data:
            raise KeyError(f"{path} does not contain a 'betas' array")
        betas = data["betas"]
    else:
        betas = data
    betas = np.asarray(betas, dtype=np.float32).reshape(-1)
    if betas.shape[0] < num_betas:
        betas = np.pad(betas, (0, num_betas - betas.shape[0]))
    return betas[:num_betas]


def process_take(args, model, cfg, device: torch.device, parents: np.ndarray, csv_path: Path) -> list[Path]:
    take = load_mocap_pcd_csv(csv_path, max_frames=args.max_frames)
    windows, window_mask = build_marker_windows(take.points, take.mask, cfg.model.seq_len)
    model_out = run_model(model, windows, window_mask, device=device, batch_size=args.batch_size)
    pose = estimate_joint_pose(take.points, model_out, n_joints=cfg.model.n_joints)
    body = build_smpl_body_sequence(
        pose["joint_rotations"],
        pose["joint_positions"],
        pose["joint_valid"],
        body_model_type=args.body_model,
        gender=args.gender,
        betas=args.betas,
        device=device,
        batch_size=args.batch_size,
    )

    pred_dir = args.out_dir / "predictions"
    video_dir = args.out_dir / "videos"
    pred_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    pred_path = pred_dir / f"{csv_path.stem}_predictions.npz"
    np.savez_compressed(
        pred_path,
        frames=take.frames,
        times=take.times,
        points=take.points,
        marker_mask=take.mask,
        weights=model_out["weights"].numpy(),
        rep_weights=model_out["rep_weights"].numpy(),
        rep_offsets=model_out["rep_offsets"].numpy(),
        smpl_vertices=body["vertices"],
        smpl_joints=body["joints"],
        smpl_faces=body["faces"],
        smpl_global_orient=body["global_orient"],
        smpl_body_pose=body["body_pose"],
        smpl_transl=body["transl"],
        smpl_betas=body["betas"],
        **pose,
    )

    video_fps = args.video_fps or min(float(take.fps), 60.0)
    raw_video = video_dir / f"{csv_path.stem}_raw_pcd.mp4"
    pred_video = video_dir / f"{csv_path.stem}_pred_joints.mp4"
    if not args.no_video:
        write_point_video(raw_video, take.points, take.mask, fps=video_fps, title=f"{csv_path.stem} raw")
        write_body_video(
            pred_video,
            take.points,
            take.mask,
            vertices=body["vertices"],
            faces=body["faces"],
            joints=body["joints"],
            joint_valid=np.isfinite(body["joints"]).all(axis=-1),
            parents=body["parents"],
            fps=video_fps,
            title=f"{csv_path.stem} predicted {args.body_model.upper()} body",
            mesh_face_stride=args.mesh_face_stride,
        )

    return [pred_path, raw_video, pred_video]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/mocap-pcd/26-6-20"))
    parser.add_argument("--checkpoint", type=Path, default=Path("ckpts/ckpt_best.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/mocap-pcd-26-6-20"))
    parser.add_argument("--config-dir", type=Path, default=Path("conf"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--body-model", choices=["smplh", "smplx"], default="smplh")
    parser.add_argument("--gender", choices=["male", "female", "neutral"], default="male")
    parser.add_argument("--betas", type=Path, default=None, help="Optional .npy/.npz file containing shape betas.")
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--mesh-face-stride", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=None, help="Optional smoke-test frame cap per CSV.")
    parser.add_argument("--no-video", action="store_true", help="Write predictions only.")
    args = parser.parse_args()
    args.betas = load_betas(args.betas)
    if args.body_model == "smplh" and args.gender == "neutral":
        raise ValueError("SMPL-H neutral model is not available in this repo; use male/female or --body-model smplx.")
    return args


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device(utils.get_device(use_cuda=True))
    else:
        device = torch.device(args.device)

    csv_paths = sorted(args.input_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {args.input_dir}")

    model, cfg = load_model_and_cfg(args.config_dir, args.checkpoint, device)
    parents = load_body_parents(args.body_model, args.gender)

    print(f"Processing {len(csv_paths)} CSVs on {device}; outputs -> {args.out_dir}")
    all_outputs: list[Path] = []
    for csv_path in csv_paths:
        print(f"[take] {csv_path}")
        all_outputs.extend(process_take(args, model, cfg, device, parents, csv_path))

    print("Wrote:")
    for path in all_outputs:
        if args.no_video and path.suffix == ".mp4":
            continue
        print(f"  {path}")


def _read_export_fps(path: Path) -> float:
    with path.open() as f:
        first = f.readline().strip().split(",")
    for key in ("Export Frame Rate", "Capture Frame Rate"):
        if key in first:
            idx = first.index(key)
            if idx + 1 < len(first):
                try:
                    return float(first[idx + 1])
                except ValueError:
                    pass
    return float("nan")


def _axis_limits(points: np.ndarray, mask: np.ndarray, joints: np.ndarray | None, joint_valid: np.ndarray | None):
    clouds = [points[mask]]
    if joints is not None and joint_valid is not None:
        jmask = joint_valid & np.isfinite(joints).all(axis=-1)
        if jmask.any():
            clouds.append(joints[jmask])
    data = np.concatenate([c for c in clouds if c.size], axis=0)
    center = data.mean(axis=0)
    radius = max(float(np.ptp(data, axis=0).max()) * 0.55, 0.5)
    return np.stack([center - radius, center + radius], axis=1)


def _style_axis(ax, limits: np.ndarray, title: str) -> None:
    ax.set_title(title)
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.set_zlim(limits[2])
    ax.set_xlabel("right (-mocap X)")
    ax.set_ylabel("forward (mocap Z)")
    ax.set_zlabel("up (mocap Y)")
    ax.view_init(elev=16, azim=-68)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


if __name__ == "__main__":
    main()

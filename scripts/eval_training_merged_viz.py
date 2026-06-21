#!/usr/bin/env python
"""Run DAMO inference videos on merged AMASS training files, excluding CMU by default."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import damo.utils as utils
from damo.smpl.body_model import create_smpl_model

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_mocap_pcd_inference import (
    build_marker_windows,
    build_smpl_body_sequence,
    load_betas,
    load_model_and_cfg,
    run_model,
    write_body_video,
    write_point_video,
    estimate_joint_pose,
)


@dataclass
class MergedTake:
    path: Path
    points: np.ndarray
    mask: np.ndarray
    frames: np.ndarray
    times: np.ndarray
    betas: np.ndarray
    smplh_poses: np.ndarray
    smplh_trans: np.ndarray
    gender: str


def list_merged_files(root: Path, include: list[str] | None, exclude: list[str]) -> list[Path]:
    merged_root = root / "merged" if (root / "merged").is_dir() else root
    include_set = set(include) if include else {p.name for p in merged_root.iterdir() if p.is_dir()}
    exclude_set = set(exclude)
    datasets = sorted(include_set - exclude_set)
    return [
        p
        for ds in datasets
        for p in sorted((merged_root / ds).rglob("*_merged.npz"))
        if p.is_file()
    ]


def dataset_name_for_path(path: Path, amass_root: Path) -> str:
    merged_root = amass_root / "merged" if (amass_root / "merged").is_dir() else amass_root
    if merged_root in path.parents:
        return path.relative_to(merged_root).parts[0]
    return path.parts[-3]


def _scalar_string(value) -> str:
    return str(np.asarray(value).item()).lower()


def filter_files_by_gender(files: list[Path], gender: str | None) -> list[Path]:
    if gender is None or gender == "any":
        return files

    kept = []
    for path in files:
        with np.load(path, allow_pickle=True) as data:
            if _scalar_string(data["smplx_gender"]) == gender:
                kept.append(path)
    return kept


def select_per_subset(files: list[Path], samples_per_subset: int, seed: int, amass_root: Path) -> list[Path]:
    by_dataset: dict[str, list[Path]] = {}
    for path in files:
        by_dataset.setdefault(dataset_name_for_path(path, amass_root), []).append(path)

    rng = np.random.default_rng(seed)
    selected = []
    for dataset in sorted(by_dataset):
        subset_files = sorted(by_dataset[dataset])
        if samples_per_subset >= len(subset_files):
            selected.extend(subset_files)
            continue
        idx = np.sort(rng.choice(len(subset_files), size=samples_per_subset, replace=False))
        selected.extend(subset_files[i] for i in idx)
    return selected


def select_evenly_spaced(files: list[Path], count: int, seed: int) -> list[Path]:
    if count >= len(files):
        return files
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(files), size=count, replace=False))
    return [files[i] for i in idx]


def load_merged_take(path: Path, max_frames: int | None = None, stride: int = 1) -> MergedTake:
    data = np.load(path, allow_pickle=True)
    markers_obs = data["smplx_markers_obs"]
    frame_slice = slice(None, max_frames * stride if max_frames is not None else None, stride)
    if max_frames is not None:
        markers_obs = markers_obs[frame_slice]
    elif stride > 1:
        markers_obs = markers_obs[frame_slice]

    lengths = [np.asarray(m).shape[0] for m in markers_obs]
    m_max = max(lengths)
    points = np.zeros((len(markers_obs), m_max, 3), dtype=np.float32)
    mask = np.zeros((len(markers_obs), m_max), dtype=bool)
    for t, markers in enumerate(markers_obs):
        arr = np.asarray(markers, dtype=np.float32)
        n = arr.shape[0]
        points[t, :n] = arr
        mask[t, :n] = True

    return MergedTake(
        path=path,
        points=points,
        mask=mask,
        frames=np.arange(len(markers_obs), dtype=np.int64) * int(stride),
        times=np.arange(len(markers_obs), dtype=np.float32) * float(stride) / 30.0,
        betas=np.asarray(data["smplh_betas"], dtype=np.float32).reshape(-1),
        smplh_poses=np.asarray(data["smplh_poses"][frame_slice], dtype=np.float32),
        smplh_trans=np.asarray(data["smplh_trans"][frame_slice], dtype=np.float32),
        gender=_scalar_string(data["smplx_gender"]),
    )


@torch.no_grad()
def build_gt_smplh_body_sequence(
    take: MergedTake,
    *,
    gender: str,
    num_betas: int,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    body_model = create_smpl_model(
        model_dir=utils.Paths.smpl_models,
        model_type="smplh",
        gender=gender,
        num_betas=num_betas,
        enable_hand=False,
    ).to(device).eval()

    betas_t = torch.as_tensor(take.betas[:num_betas], dtype=torch.float32, device=device).view(1, -1)
    poses = take.smplh_poses
    trans = take.smplh_trans
    vertices = []
    joints = []
    for start in tqdm(range(0, len(poses), batch_size), desc="gt-smpl", leave=False):
        end = start + batch_size
        out = body_model(
            betas=betas_t,
            transl=torch.as_tensor(trans[start:end], dtype=torch.float32, device=device),
            global_orient=torch.as_tensor(poses[start:end, :3], dtype=torch.float32, device=device),
            body_pose=torch.as_tensor(poses[start:end, 3:66], dtype=torch.float32, device=device),
        )
        vertices.append(out.vertices.detach().cpu().numpy().astype(np.float32))
        joints.append(out.joints[:, :22].detach().cpu().numpy().astype(np.float32))

    return {
        "vertices": np.concatenate(vertices, axis=0),
        "joints": np.concatenate(joints, axis=0),
        "faces": np.asarray(body_model.faces, dtype=np.int32),
        "parents": body_model.parents[:22].detach().cpu().numpy(),
    }


def compute_joint_error_summary(pred_joints: np.ndarray, gt_joints: np.ndarray) -> dict[str, float | int]:
    if pred_joints.shape != gt_joints.shape:
        raise ValueError(f"pred/gt joint shape mismatch: {pred_joints.shape} vs {gt_joints.shape}")

    valid = np.isfinite(pred_joints).all(axis=-1) & np.isfinite(gt_joints).all(axis=-1)
    err = np.linalg.norm(pred_joints - gt_joints, axis=-1)
    root_aligned_pred = pred_joints - pred_joints[:, :1]
    root_aligned_gt = gt_joints - gt_joints[:, :1]
    ra_err = np.linalg.norm(root_aligned_pred - root_aligned_gt, axis=-1)

    valid_err = err[valid]
    valid_ra_err = ra_err[valid]
    return {
        "frames": int(pred_joints.shape[0]),
        "joints": int(pred_joints.shape[1]),
        "valid_joint_observations": int(valid.sum()),
        "mpjpe_mm": float(np.nanmean(valid_err) * 1000.0),
        "median_mpjpe_mm": float(np.nanmedian(valid_err) * 1000.0),
        "root_aligned_mpjpe_mm": float(np.nanmean(valid_ra_err) * 1000.0),
        "root_aligned_median_mpjpe_mm": float(np.nanmedian(valid_ra_err) * 1000.0),
    }


def process_take(args, model, cfg, device: torch.device, path: Path) -> list[Path]:
    take = load_merged_take(path, max_frames=args.max_frames, stride=args.frame_stride)
    windows, window_mask = build_marker_windows(take.points, take.mask, cfg.model.seq_len)
    model_out = run_model(model, windows, window_mask, device=device, batch_size=args.batch_size)
    pose = estimate_joint_pose(take.points, model_out, n_joints=cfg.model.n_joints)
    body = build_smpl_body_sequence(
        pose["joint_rotations"],
        pose["joint_positions"],
        pose["joint_valid"],
        body_model_type=args.body_model,
        gender=args.gender,
        betas=take.betas[: args.num_betas],
        device=device,
        batch_size=args.batch_size,
    )
    gt_body = build_gt_smplh_body_sequence(
        take,
        gender=args.gender,
        num_betas=args.num_betas,
        device=device,
        batch_size=args.batch_size,
    )
    error_summary = compute_joint_error_summary(body["joints"], gt_body["joints"])

    dataset_name = dataset_name_for_path(path, args.amass_root)
    stem = f"{dataset_name}_{path.stem}"
    pred_dir = args.out_dir / "predictions"
    video_dir = args.out_dir / "videos"
    pred_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    pred_path = pred_dir / f"{stem}_predictions.npz"
    np.savez_compressed(
        pred_path,
        source_path=str(path),
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
        gt_smpl_vertices=gt_body["vertices"],
        gt_smpl_joints=gt_body["joints"],
        gt_smpl_faces=gt_body["faces"],
        gt_smplh_poses=take.smplh_poses,
        gt_smplh_trans=take.smplh_trans,
        error_metric_names=np.array(list(error_summary.keys())),
        error_metric_values=np.array(list(error_summary.values()), dtype=np.float64),
        **pose,
    )
    error_path = pred_dir / f"{stem}_errors.csv"
    with error_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "take", "source_path", *error_summary.keys()])
        writer.writeheader()
        writer.writerow({"dataset": dataset_name, "take": path.stem, "source_path": str(path), **error_summary})

    raw_video = video_dir / f"{stem}_raw_markers.mp4"
    pred_video = video_dir / f"{stem}_pred_body.mp4"
    if not args.no_video:
        write_point_video(raw_video, take.points, take.mask, fps=args.video_fps, title=f"{stem} raw markers")
        write_body_video(
            pred_video,
            take.points,
            take.mask,
            vertices=body["vertices"],
            faces=body["faces"],
            joints=body["joints"],
            joint_valid=np.isfinite(body["joints"]).all(axis=-1),
            parents=body["parents"],
            fps=args.video_fps,
            title=f"{stem} predicted {args.body_model.upper()} body",
            mesh_face_stride=args.mesh_face_stride,
        )

    print(
        f"[error] {stem}: MPJPE={error_summary['mpjpe_mm']:.1f} mm, "
        f"root-aligned={error_summary['root_aligned_mpjpe_mm']:.1f} mm"
    )
    return [pred_path, error_path, raw_video, pred_video]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amass-root", type=Path, default=Path("/mnt/bcc-data/proj/ego-exo-collect/amass"))
    parser.add_argument("--checkpoint", type=Path, default=Path("ckpts/ckpt_best.pt"))
    parser.add_argument("--config-dir", type=Path, default=Path("conf"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/train-merged-exclude-cmu"))
    parser.add_argument("--include-datasets", nargs="*", default=None)
    parser.add_argument("--exclude-datasets", nargs="*", default=["CMU"])
    parser.add_argument("--num-takes", type=int, default=4)
    parser.add_argument("--samples-per-subset", type=int, default=None)
    parser.add_argument("--gender-filter", choices=["male", "female", "neutral", "any"], default="male")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--body-model", choices=["smplh", "smplx"], default="smplh")
    parser.add_argument("--gender", choices=["male", "female", "neutral"], default="male")
    parser.add_argument("--num-betas", type=int, default=16)
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--mesh-face-stride", type=int, default=8)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    args.betas = load_betas(None, num_betas=args.num_betas)
    return args


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    files = list_merged_files(args.amass_root, args.include_datasets, args.exclude_datasets)
    files = filter_files_by_gender(files, None if args.gender_filter == "any" else args.gender_filter)
    if not files:
        raise FileNotFoundError(f"No merged AMASS files found under {args.amass_root} with exclude={args.exclude_datasets}")
    if args.samples_per_subset is not None:
        selected = select_per_subset(files, args.samples_per_subset, args.seed, args.amass_root)
    else:
        selected = select_evenly_spaced(files, args.num_takes, args.seed)

    model, cfg = load_model_and_cfg(args.config_dir, args.checkpoint, device)
    print(f"Processing {len(selected)} merged takes on {device}; outputs -> {args.out_dir}")
    print("Datasets:", sorted({dataset_name_for_path(p, args.amass_root) for p in selected}))
    outputs: list[Path] = []
    errors = []
    for path in tqdm(selected, desc="takes"):
        print(f"[take] {path}")
        outputs.extend(process_take(args, model, cfg, device, path))
        err_csv = outputs[-3]
        with err_csv.open() as f:
            rows = list(csv.DictReader(f))
            errors.extend(rows)

    if errors:
        summary_path = args.out_dir / "error_summary.csv"
        fieldnames = list(errors[0].keys())
        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(errors)
        outputs.append(summary_path)

        print("Error summary by dataset:")
        for dataset in sorted({row["dataset"] for row in errors}):
            rows = [row for row in errors if row["dataset"] == dataset]
            mpjpe = np.mean([float(row["mpjpe_mm"]) for row in rows])
            ra_mpjpe = np.mean([float(row["root_aligned_mpjpe_mm"]) for row in rows])
            print(f"  {dataset}: n={len(rows)}, MPJPE={mpjpe:.1f} mm, root-aligned={ra_mpjpe:.1f} mm")

    print("Wrote:")
    for path in outputs:
        if args.no_video and path.suffix == ".mp4":
            continue
        print(f"  {path}")


if __name__ == "__main__":
    main()

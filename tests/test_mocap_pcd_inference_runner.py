import numpy as np
import importlib.util
import sys
import torch
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_mocap_pcd_inference.py"
_SPEC = importlib.util.spec_from_file_location("run_mocap_pcd_inference", _SCRIPT)
runner = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = runner
_SPEC.loader.exec_module(runner)

build_marker_windows = runner.build_marker_windows
load_mocap_pcd_csv = runner.load_mocap_pcd_csv
estimate_joint_pose = runner.estimate_joint_pose
mocap_to_view_coords = runner.mocap_to_view_coords
rotations_to_smpl_params = runner.rotations_to_smpl_params


def test_load_mocap_pcd_csv_reads_repeated_xyz_marker_columns(tmp_path):
    src = tmp_path / "take.csv"
    src.write_text(
        "\n".join(
            [
                "Format Version,1.23,Export Frame Rate,100.000000",
                ",Type,Marker,Marker,Marker,Marker,Marker,Marker",
                ",Name,A,A,A,B,B,B",
                ",ID,1,1,1,2,2,2",
                ",,Position,Position,Position,Position,Position,Position",
                "Frame,Time (Seconds),X,Y,Z,X,Y,Z",
                "0,0.00,1,2,3,4,5,6",
                "1,0.01,7,8,9,,,",
            ]
        )
    )

    take = load_mocap_pcd_csv(src)

    assert take.points.shape == (2, 2, 3)
    np.testing.assert_allclose(take.points[0, 0], [1, 2, 3])
    np.testing.assert_allclose(take.points[1, 1], [0, 0, 0])
    np.testing.assert_array_equal(take.mask, [[True, True], [True, False]])
    assert take.fps == 100.0


def test_build_marker_windows_edge_pads_sequence():
    points = np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3)
    mask = np.ones((4, 2), dtype=bool)

    windows, window_mask = build_marker_windows(points, mask, seq_len=3)

    assert windows.shape == (4, 3, 2, 3)
    np.testing.assert_allclose(windows[0], points[[0, 0, 1]])
    np.testing.assert_allclose(windows[-1], points[[2, 3, 3]])
    np.testing.assert_array_equal(window_mask[0], mask[[0, 0, 1]])


def test_estimate_joint_pose_marks_unsupported_joints_invalid():
    points = np.zeros((1, 4, 3), dtype=np.float32)
    points[0, :, 0] = np.arange(4, dtype=np.float32)
    logits = torch.full((1, 4, 4), -10.0)
    logits[..., 0] = 10.0
    model_out = {
        "markers_mask": torch.ones((1, 4), dtype=torch.bool),
        "weights": logits,
        "rep_weights": torch.ones((1, 4, 1), dtype=torch.float32),
        "rep_offsets": torch.zeros((1, 4, 1, 3), dtype=torch.float32),
    }

    pose = estimate_joint_pose(points, model_out, n_joints=3)

    np.testing.assert_array_equal(pose["joint_valid"], [[True, False, False]])
    assert np.isfinite(pose["joint_positions"][0, 0]).all()
    assert np.isnan(pose["joint_positions"][0, 1:]).all()


def test_mocap_to_view_coords_uses_y_up_x_left_z_forward():
    points = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)

    viewed = mocap_to_view_coords(points)

    np.testing.assert_allclose(viewed, [[[-1.0, 3.0, 2.0]]])


def test_rotations_to_smpl_params_aligns_root_translation():
    rotations = np.tile(np.eye(3, dtype=np.float32), (1, 3, 1, 1))
    joint_positions = np.array([[[10.0, 20.0, 30.0], [11.0, 20.0, 30.0], [10.0, 21.0, 30.0]]], dtype=np.float32)
    joint_valid = np.ones((1, 3), dtype=bool)
    parents = np.array([-1, 0, 1], dtype=np.int64)
    rest_root = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    global_orient, body_pose, transl = rotations_to_smpl_params(
        rotations,
        joint_positions,
        joint_valid,
        parents,
        rest_root,
    )

    np.testing.assert_allclose(global_orient, [[0.0, 0.0, 0.0]], atol=1e-6)
    np.testing.assert_allclose(body_pose, np.zeros((1, 2 * 3)), atol=1e-6)
    np.testing.assert_allclose(transl, [[9.0, 18.0, 27.0]], atol=1e-6)

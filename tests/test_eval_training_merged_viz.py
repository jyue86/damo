import importlib.util
import sys
from pathlib import Path

import numpy as np


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_training_merged_viz.py"
_SPEC = importlib.util.spec_from_file_location("eval_training_merged_viz", _SCRIPT)
runner = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = runner
_SPEC.loader.exec_module(runner)


def test_list_merged_files_excludes_cmu(tmp_path):
    for dataset in ["ACCAD", "CMU", "HumanEva"]:
        d = tmp_path / "merged" / dataset / "seq"
        d.mkdir(parents=True)
        (d / f"{dataset}_sample_merged.npz").write_bytes(b"placeholder")

    files = runner.list_merged_files(tmp_path, include=None, exclude=["CMU"])

    names = [p.name for p in files]
    assert "CMU_sample_merged.npz" not in names
    assert names == ["ACCAD_sample_merged.npz", "HumanEva_sample_merged.npz"]


def test_filter_files_by_gender_keeps_training_gender(tmp_path):
    male = tmp_path / "male_merged.npz"
    female = tmp_path / "female_merged.npz"
    np.savez(male, smplx_gender=np.array("male"))
    np.savez(female, smplx_gender=np.array("female"))

    files = runner.filter_files_by_gender([female, male], gender="male")

    assert files == [male]


def test_select_per_subset_returns_requested_count_from_each_dataset(tmp_path):
    files = []
    for dataset in ["ACCAD", "BMLmovi", "CMU"]:
        for i in range(3):
            path = tmp_path / "merged" / dataset / "seq" / f"{i}_merged.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"placeholder")
            files.append(path)

    selected = runner.select_per_subset(files, samples_per_subset=2, seed=0, amass_root=tmp_path)

    selected_by_dataset = {}
    for path in selected:
        selected_by_dataset.setdefault(path.relative_to(tmp_path / "merged").parts[0], 0)
        selected_by_dataset[path.relative_to(tmp_path / "merged").parts[0]] += 1
    assert selected_by_dataset == {"ACCAD": 2, "BMLmovi": 2, "CMU": 2}


def test_load_merged_take_pads_variable_marker_observations(tmp_path):
    path = tmp_path / "sample_merged.npz"
    markers = np.empty(2, dtype=object)
    markers[0] = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    markers[1] = np.array([[7, 8, 9]], dtype=np.float32)
    np.savez(
        path,
        smplx_markers_obs=markers,
        smplh_betas=np.zeros(16, dtype=np.float32),
        smplh_poses=np.zeros((2, 156), dtype=np.float32),
        smplh_trans=np.zeros((2, 3), dtype=np.float32),
        smplx_gender=np.array("male"),
    )

    take = runner.load_merged_take(path)

    assert take.points.shape == (2, 2, 3)
    np.testing.assert_array_equal(take.mask, [[True, True], [True, False]])
    np.testing.assert_allclose(take.points[1, 1], [0, 0, 0])


def test_joint_error_summary_reports_mpjpe_and_root_aligned_mpjpe():
    gt = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        ],
        dtype=np.float32,
    )
    pred = gt + np.array([1.0, 0.0, 0.0], dtype=np.float32)

    summary = runner.compute_joint_error_summary(pred, gt)

    assert summary["frames"] == 2
    assert summary["joints"] == 2
    assert summary["mpjpe_mm"] == 1000.0
    assert summary["root_aligned_mpjpe_mm"] == 0.0

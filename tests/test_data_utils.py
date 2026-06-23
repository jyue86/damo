from pathlib import Path

import numpy as np

from damo.utils.data_utils import build_global_index


def _write_merged(path: Path, gender: str, num_frames: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        smplx_gender=np.array(gender),
        smplx_trans=np.zeros((num_frames, 3), dtype=np.float32),
    )


def test_build_global_index_respects_centered_sequence_window(tmp_path):
    path = tmp_path / "CMU" / "82_01_merged.npz"
    _write_merged(path, gender="male", num_frames=10)

    index = build_global_index(
        files=[path],
        sequential_data_key="smplx_trans",
        ratio=(0.1, 0.9),
        requires={"smplx_gender": ["male"]},
        seq_len=7,
    )

    assert index == [(0, 3), (0, 4), (0, 5), (0, 6)]


def test_build_global_index_skips_too_short_centered_windows(tmp_path):
    path = tmp_path / "short_merged.npz"
    _write_merged(path, gender="male", num_frames=5)

    index = build_global_index(
        files=[path],
        sequential_data_key="smplx_trans",
        ratio=(0.0, 1.0),
        requires={"smplx_gender": ["male"]},
        seq_len=7,
    )

    assert index == []

import numpy as np
import pandas as pd

from damo.utils.mocap_utils import (
    clean_mocap_data,
    extract_transform_from_df,
    parse_mocap_data,
)


def _raw_mocap_csv() -> str:
    return "\n".join(
        [
            "metadata",
            "metadata",
            "metadata",
            "Frame,Time (Seconds),Pelvis,,,,,,,Head,,,,,,",
            "metadata",
            "metadata",
            "metadata",
            "metadata",
            "0,0.00,0,0,0,1,1,2,3,0,0,0,1,4,5,6",
            "1,0.01,0,0,0.70710678,0.70710678,7,8,9,0,0,0,1,10,11,12",
            "",
        ]
    )


def test_clean_mocap_data_writes_named_columns_for_selected_frames(tmp_path):
    src = tmp_path / "raw.csv"
    dst = tmp_path / "clean.csv"
    src.write_text(_raw_mocap_csv())

    clean_mocap_data(src, ["Pelvis", "Head"], dst)

    lines = dst.read_text().splitlines()
    assert lines[0] == (
        "Frame,Time (Seconds),"
        "Pelvis:RotationX,Pelvis:RotationY,Pelvis:RotationZ,Pelvis:RotationW,"
        "Pelvis:PositionX,Pelvis:PositionY,Pelvis:PositionZ,"
        "Head:RotationX,Head:RotationY,Head:RotationZ,Head:RotationW,"
        "Head:PositionX,Head:PositionY,Head:PositionZ"
    )
    assert lines[1] == "0,0.00,0,0,0,1,1,2,3,0,0,0,1,4,5,6"
    assert lines[2] == "1,0.01,0,0,0.70710678,0.70710678,7,8,9,0,0,0,1,10,11,12"


def test_parse_mocap_data_extracts_two_frames(tmp_path):
    src = tmp_path / "raw.csv"
    dst = tmp_path / "parsed.csv"
    src.write_text(_raw_mocap_csv())

    parse_mocap_data(src, "Pelvis", "Head", dst)

    lines = dst.read_text().splitlines()
    assert lines[0] == (
        "Frame,Time (seconds),"
        "Pelvis:RotationX,Pelvis:RotationY,Pelvis:RotationZ,Pelvis:RotationW,"
        "Pelvis:PositionX,Pelvis:PositionY,Pelvis:PositionZ,"
        "Head:RotationX,Head:RotationY,Head:RotationZ,Head:RotationW,"
        "Head:PositionX,Head:PositionY,Head:PositionZ"
    )
    assert lines[1] == "0,0.00,0,0,0,1,1,2,3,0,0,0,1,4,5,6"
    assert lines[2] == "1,0.01,0,0,0.70710678,0.70710678,7,8,9,0,0,0,1,10,11,12"


def test_extract_transform_from_df_returns_homogeneous_transform():
    df = pd.DataFrame(
        {
            "Pelvis:RotationX": [0.0],
            "Pelvis:RotationY": [0.0],
            "Pelvis:RotationZ": [0.0],
            "Pelvis:RotationW": [1.0],
            "Pelvis:PositionX": [1.0],
            "Pelvis:PositionY": [2.0],
            "Pelvis:PositionZ": [3.0],
        }
    )

    transform = extract_transform_from_df(df, "Pelvis", 0)

    expected = np.eye(4)
    expected[:3, 3] = [1.0, 2.0, 3.0]
    np.testing.assert_allclose(transform, expected)

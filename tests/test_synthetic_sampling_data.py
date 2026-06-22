import importlib.util
import sys
from pathlib import Path

import numpy as np

from damo.dataset.body_cache import BodyCache


_POOL_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "prepare_data"
    / "build_amass_smplh_body_pool.py"
)
_POOL_SPEC = importlib.util.spec_from_file_location("build_amass_smplh_body_pool", _POOL_SCRIPT)
body_pool = importlib.util.module_from_spec(_POOL_SPEC)
assert _POOL_SPEC.loader is not None
sys.modules[_POOL_SPEC.name] = body_pool
_POOL_SPEC.loader.exec_module(body_pool)

_GEOM_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "prepare_data"
    / "precompute_marker_geom.py"
)
_GEOM_SPEC = importlib.util.spec_from_file_location("precompute_marker_geom", _GEOM_SCRIPT)
marker_geom = importlib.util.module_from_spec(_GEOM_SPEC)
assert _GEOM_SPEC.loader is not None
sys.modules[_GEOM_SPEC.name] = marker_geom
_GEOM_SPEC.loader.exec_module(marker_geom)

_MARKERSETS_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "prepare_data"
    / "build_markersets.py"
)
_MARKERSETS_SPEC = importlib.util.spec_from_file_location("build_markersets", _MARKERSETS_SCRIPT)
markersets = importlib.util.module_from_spec(_MARKERSETS_SPEC)
assert _MARKERSETS_SPEC.loader is not None
sys.modules[_MARKERSETS_SPEC.name] = markersets
_MARKERSETS_SPEC.loader.exec_module(markersets)

_TRAIN_ONE_STEP_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_one_step.py"
_TRAIN_ONE_STEP_SPEC = importlib.util.spec_from_file_location("train_one_step", _TRAIN_ONE_STEP_SCRIPT)
train_one_step = importlib.util.module_from_spec(_TRAIN_ONE_STEP_SPEC)
assert _TRAIN_ONE_STEP_SPEC.loader is not None
sys.modules[_TRAIN_ONE_STEP_SPEC.name] = train_one_step
_TRAIN_ONE_STEP_SPEC.loader.exec_module(train_one_step)


def _write_merged(path, gender, betas):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, smplh_gender=np.array(gender), smplh_betas=np.asarray(betas, dtype=np.float32))


def test_collect_amass_smplh_betas_filters_gender_and_dedupes(tmp_path):
    root = tmp_path / "merged"
    _write_merged(root / "ACCAD" / "male_a_merged.npz", "male", np.arange(16))
    _write_merged(root / "ACCAD" / "male_b_merged.npz", "male", np.arange(16))
    _write_merged(root / "ACCAD" / "male_c_merged.npz", "male", np.arange(16) + 1)
    _write_merged(root / "ACCAD" / "female_merged.npz", "female", np.arange(16) + 2)

    pool = body_pool.collect_amass_smplh_betas(root, gender="male")

    assert pool.betas.shape == (2, 16)
    assert pool.gender == "male"
    assert len(pool.source_paths) == 2
    np.testing.assert_allclose(pool.betas[0], np.arange(16, dtype=np.float32))
    np.testing.assert_allclose(pool.betas[1], np.arange(16, dtype=np.float32) + 1)


def test_save_amass_smplh_body_pool_writes_expected_schema(tmp_path):
    pool = body_pool.BodyPool(
        betas=np.zeros((1, 16), dtype=np.float32),
        gender="male",
        source_paths=["ACCAD/sample_merged.npz"],
    )
    out_path = tmp_path / "amass_smplh_betas_male.npz"

    body_pool.save_body_pool(pool, out_path)

    with np.load(out_path, allow_pickle=True) as data:
        assert set(data.files) == {"betas", "gender", "source_paths", "num_bodies"}
        assert data["betas"].shape == (1, 16)
        assert data["gender"].item() == "male"
        assert data["num_bodies"].item() == 1


def test_body_cache_amass_betas_source_returns_smplh_body_without_caesar(tmp_path, monkeypatch):
    pool_dir = tmp_path / "synthetic_bodies"
    pool_dir.mkdir()
    np.savez(
        pool_dir / "amass_smplh_betas_male.npz",
        betas=np.ones((1, 16), dtype=np.float32),
        gender=np.array("male"),
        source_paths=np.array(["sample_merged.npz"]),
        num_bodies=np.array(1),
    )

    def fail_caesar(self, gender):
        raise AssertionError("CAESAR cache should not be loaded for amass_betas")

    def fake_smpl_body(self, body_type, gender, betas, to_torch=False):
        assert body_type == "smplh"
        assert gender == "male"
        np.testing.assert_allclose(betas, np.ones(16, dtype=np.float32))
        return {
            "vertices": np.zeros((6890, 3), dtype=np.float32),
            "joints": np.zeros((22, 3), dtype=np.float32),
        }

    monkeypatch.setattr(BodyCache, "get_caesar", fail_caesar)
    monkeypatch.setattr(BodyCache, "get_smpl_body", fake_smpl_body)

    cache = BodyCache(
        genders=["male"],
        capacity=4,
        synthetic_body_source="amass_betas",
        synthetic_body_pool_dir=pool_dir,
    )

    body, gender = cache.get_random_synthetic_body(["male"], to_torch=False)

    assert gender == "male"
    assert body["vertices"].shape == (6890, 3)
    assert body["joints"].shape == (22, 3)


def test_body_cache_resolves_relative_pool_dir_from_project_root():
    cache = BodyCache(
        genders=["male"],
        capacity=4,
        synthetic_body_source="amass_betas",
        synthetic_body_pool_dir="data/synthetic_bodies",
    )

    assert cache.synthetic_body_pool_dir == Path(__file__).resolve().parents[1] / "data" / "synthetic_bodies"


def test_save_marker_geom_writes_loader_expected_filename_and_keys(tmp_path):
    geom = {
        "smplh_vertex_normals": np.zeros((4, 3), dtype=np.float32),
        "candidate_mask": np.array([True, False, True, False]),
        "candidate_vids": np.array([0, 2], dtype=np.int64),
        "sample_weights": np.ones(4, dtype=np.float32),
    }

    out_path = marker_geom.save_marker_geom(geom, tmp_path)

    assert out_path == tmp_path / "marker_geom_smplh.npz"
    with np.load(out_path) as data:
        assert set(data.files) == {
            "smplh_vertex_normals",
            "candidate_mask",
            "candidate_vids",
            "sample_weights",
        }


def test_train_one_step_data_type_override_supports_syn_arb(monkeypatch):
    monkeypatch.setenv("DAMO_ONE_STEP_DATA_TYPE", "syn_arb")

    assert train_one_step._one_step_data_type_probs() == {"syn_arb": 1.0}


def test_train_one_step_data_type_override_supports_syn_sup(monkeypatch):
    monkeypatch.setenv("DAMO_ONE_STEP_DATA_TYPE", "syn_sup")

    assert train_one_step._one_step_data_type_probs() == {"syn_sup": 1.0}


def test_build_superset_candidates_groups_slots_filters_and_falls_back():
    marker_idx_set = np.array(
        [
            [0, 5, 8],
            [1, 6, 9],
            [0, 7, 10],
        ],
        dtype=np.int64,
    )
    candidate_vids = np.array([0, 1, 2, 5, 7], dtype=np.int64)
    template_vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [9.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    cands = markersets.build_superset_candidates(marker_idx_set, candidate_vids, template_vertices)

    assert len(cands) == 3
    np.testing.assert_array_equal(cands[0], np.array([0, 1]))
    np.testing.assert_array_equal(cands[1], np.array([5, 7]))
    np.testing.assert_array_equal(cands[2], np.array([7]))


def test_save_markersets_writes_loader_expected_schema(tmp_path):
    marker_idx_set = np.array([[0, 1], [2, 3]], dtype=np.int64)
    superset_cands = [
        np.array([0, 2], dtype=np.int64),
        np.array([1, 3], dtype=np.int64),
    ]

    out_path = markersets.save_markersets(marker_idx_set, superset_cands, tmp_path / "markersets.npz")

    assert out_path == tmp_path / "markersets.npz"
    with np.load(out_path, allow_pickle=True) as data:
        assert set(data.files) == {"soma_superset_smplh_cands", "mocap_solver_smplh"}
        assert data["mocap_solver_smplh"].shape == (2, 2)
        loaded = data["soma_superset_smplh_cands"]
        assert loaded.dtype == object
        np.testing.assert_array_equal(loaded[0], np.array([0, 2]))

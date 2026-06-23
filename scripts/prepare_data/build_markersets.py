import argparse
from pathlib import Path

import numpy as np

import damo.utils as utils
from damo.smpl.body_model import create_smpl_model


MARKER_IDX_SET = np.array(
    [
        [414, 1219, 3495, 2837, 3207, 447, 709, 2911, 1910, 104, 1294, 2920, 3331, 1023, 1718, 1995, 2110, 1068, 1043,
         3232, 3311, 846, 2173, 2032, 1112, 3257, 1238, 1442, 1686, 6604, 3941, 4195, 5244, 5090, 3517, 4778, 6380,
         6730, 4520, 4861, 5594, 5480, 4555, 4529, 6634, 6682, 4332, 5751, 5609, 4598, 6657, 5322, 4915, 5157, 1330,
         751],
        [414, 1301, 3497, 2837, 3207, 447, 2935, 2911, 1910, 104, 650, 2920, 3331, 1023, 1718, 1995, 2110, 1068, 1043,
         3232, 3311, 846, 2173, 2032, 1112, 3257, 1861, 1442, 1686, 6604, 3941, 6396, 5244, 5090, 3517, 4139, 6380,
         6730, 4520, 4861, 5594, 5480, 4555, 4529, 6634, 6682, 4332, 5751, 5609, 4598, 6657, 4124, 4915, 5157, 1330,
         751],
        [414, 453, 3073, 2837, 3207, 447, 1812, 2911, 1910, 104, 1535, 2920, 3331, 1023, 1718, 1995, 2110, 1068, 1043,
         3232, 3311, 846, 2173, 2032, 1112, 3257, 783, 1442, 1686, 6604, 3941, 5273, 5244, 5090, 3517, 4077, 6380, 6730,
         4520, 4861, 5594, 5480, 4555, 4529, 6634, 6682, 4332, 5751, 5609, 4598, 6657, 4721, 4915, 5157, 1330, 751],
        [414, 1219, 3495, 2837, 3207, 517, 709, 2911, 1910, 2786, 1294, 2920, 3331, 1023, 1718, 1995, 2110, 1068, 1043,
         3232, 3311, 846, 2173, 2032, 1135, 3257, 1238, 1442, 1980, 6604, 3973, 4195, 5244, 5090, 3635, 4778, 6380,
         6730, 4520, 4861, 5594, 5480, 4555, 4529, 6634, 6682, 4332, 5751, 5609, 4621, 6657, 5322, 4915, 5441, 1330,
         751],
        [414, 1219, 3495, 2837, 3207, 450, 709, 2911, 1910, 5, 1294, 2920, 3331, 1023, 1718, 1995, 2110, 1068, 1043,
         3232, 3311, 846, 2173, 2032, 1115, 3257, 1238, 1442, 1685, 6604, 3939, 4195, 5244, 5090, 3515, 4778, 6380,
         6730, 4520, 4861, 5594, 5480, 4555, 4529, 6634, 6682, 4332, 5751, 5609, 4599, 6657, 5322, 4915, 5154, 1330,
         751],
        [414, 1301, 3497, 2837, 3207, 447, 2935, 2911, 1910, 104, 650, 2920, 3331, 1023, 1718, 1995, 2110, 1068, 1043,
         3232, 3311, 846, 2173, 2032, 1135, 3257, 1861, 1442, 1980, 6604, 3941, 6396, 5244, 5090, 3517, 4139, 6380,
         6730, 4520, 4861, 5594, 5480, 4555, 4529, 6634, 6682, 4332, 5751, 5609, 4621, 6657, 4124, 4915, 5441, 1330,
         751],
        [414, 453, 3073, 2837, 3207, 447, 1812, 2911, 1910, 104, 1535, 2920, 3331, 1023, 1718, 1995, 2110, 1068, 1043,
         3232, 3311, 846, 2173, 2032, 1135, 3257, 783, 1442, 1980, 6604, 3941, 5273, 5244, 5090, 3517, 4077, 6380, 6730,
         4520, 4861, 5594, 5480, 4555, 4529, 6634, 6682, 4332, 5751, 5609, 4621, 6657, 4721, 4915, 5441, 1330, 751],
        [414, 1219, 3495, 2837, 3207, 447, 709, 2911, 1910, 104, 1294, 2920, 3331, 1023, 1718, 1995, 2110, 1068, 1043,
         3232, 3311, 846, 2173, 2032, 1135, 3257, 1238, 1442, 1980, 6604, 3941, 4195, 5244, 5090, 3517, 4778, 6380,
         6730, 4520, 4861, 5594, 5480, 4555, 4529, 6634, 6682, 4332, 5751, 5609, 4621, 6657, 5322, 4915, 5441, 1330,
         751],
        [414, 1219, 3495, 2837, 3207, 447, 709, 2911, 1910, 104, 1294, 2920, 3331, 1023, 1718, 1995, 2110, 1068, 1043,
         3232, 3311, 846, 2173, 2032, 1115, 3257, 1238, 1442, 1685, 6604, 3941, 4195, 5244, 5090, 3517, 4778, 6380,
         6730, 4520, 4861, 5594, 5480, 4555, 4529, 6634, 6682, 4332, 5751, 5609, 4599, 6657, 5322, 4915, 5154, 1330,
         751],
        [414, 1219, 3495, 2837, 3207, 517, 709, 2911, 1910, 2786, 1294, 2920, 3331, 1023, 1718, 1995, 2110, 1068, 1043,
         3232, 3311, 846, 2173, 2032, 1112, 3257, 1238, 1442, 1686, 6604, 3973, 4195, 5244, 5090, 3635, 4778, 6380,
         6730, 4520, 4861, 5594, 5480, 4555, 4529, 6634, 6682, 4332, 5751, 5609, 4598, 6657, 5322, 4915, 5157, 1330,
         751],
    ],
    dtype=np.int64,
)


def _nearest_candidate_vertex(slot_vids, candidate_vids, template_vertices):
    target = template_vertices[slot_vids].mean(axis=0)
    candidate_points = template_vertices[candidate_vids]
    nearest_idx = np.argmin(np.linalg.norm(candidate_points - target[None, :], axis=1))
    return int(candidate_vids[nearest_idx])


def build_superset_candidates(marker_idx_set, candidate_vids, template_vertices):
    marker_idx_set = np.asarray(marker_idx_set, dtype=np.int64)
    candidate_vids = np.asarray(candidate_vids, dtype=np.int64)
    template_vertices = np.asarray(template_vertices)

    if marker_idx_set.ndim != 2:
        raise ValueError(f"marker_idx_set must be 2D, got shape {marker_idx_set.shape}")
    if candidate_vids.ndim != 1 or candidate_vids.size == 0:
        raise ValueError("candidate_vids must be a non-empty 1D array")

    candidate_set = set(int(v) for v in candidate_vids)
    superset_cands = []
    for slot_idx in range(marker_idx_set.shape[1]):
        slot_vids = marker_idx_set[:, slot_idx]
        valid = sorted({int(v) for v in slot_vids if int(v) in candidate_set})
        if not valid:
            valid = [_nearest_candidate_vertex(slot_vids, candidate_vids, template_vertices)]
        superset_cands.append(np.asarray(valid, dtype=np.int64))

    return superset_cands


def _object_array(arrays):
    out = np.empty(len(arrays), dtype=object)
    for idx, value in enumerate(arrays):
        out[idx] = np.asarray(value, dtype=np.int64)
    return out


def save_markersets(marker_idx_set, superset_cands, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        soma_superset_smplh_cands=_object_array(superset_cands),
        mocap_solver_smplh=np.asarray(marker_idx_set, dtype=np.int64),
    )
    return out_path


def load_marker_geom(path):
    with np.load(path, allow_pickle=True) as data:
        if "candidate_vids" not in data:
            raise KeyError(f"candidate_vids missing from marker geometry: {path}")
        return {"candidate_vids": np.asarray(data["candidate_vids"], dtype=np.int64)}


def load_smplh_template_vertices(gender):
    smplh = create_smpl_model(
        model_dir=utils.Paths.smpl_models,
        model_type="smplh",
        gender=gender,
        num_betas=16,
    ).eval()
    return smplh.v_template.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Build DAMO markersets.npz for synthetic superset sampling.")
    parser.add_argument("--marker-geom", type=Path, default=utils.Paths.data / "marker_geom_smplh.npz")
    parser.add_argument("--out", type=Path, default=utils.Paths.data / "markersets.npz")
    parser.add_argument("--gender", default="male")
    args = parser.parse_args()

    marker_geom = load_marker_geom(args.marker_geom)
    template_vertices = load_smplh_template_vertices(args.gender)
    superset_cands = build_superset_candidates(
        MARKER_IDX_SET,
        marker_geom["candidate_vids"],
        template_vertices,
    )
    out_path = save_markersets(MARKER_IDX_SET, superset_cands, args.out)
    print(f"[ok] wrote {len(superset_cands)} marker candidate groups -> {out_path}")


if __name__ == "__main__":
    main()

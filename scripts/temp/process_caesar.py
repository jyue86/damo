import pickle
import torch
import json

import numpy as np

import damo.utils as utils
from damo.viz.viewer import Viewer
from damo.smpl.body_model import create_smpl_model, vertices2joints


def main():
    legacy_file_path = r"C:\Users\kkm\projects\research\damo_before_refac\data\base\damo_base_20240415.pkl"
    out_path = utils.Paths.smpl_models / "caesar_male.npz"

    geom_path = r"C:\Users\kkm\projects\research\damo\data\marker_geom_smplh.npz"
    geom = np.load(geom_path, allow_pickle=True)

    with open(legacy_file_path, "rb") as f:
        data = pickle.load(f)

    json_path = r"C:\Users\kkm\projects\research\damo_workspace\soma_data\V48_02_SuperSet\marker_dataset\superset.json"
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    superset_smplx_dict = list(json_data["markersets"][0]["indices"].values())
    superset_smplx = np.array(superset_smplx_dict, dtype=np.int64)[:, 0]

    print(superset_smplx.shape)

    valid_supersets = []
    for superset in data["soma_superset_variant"]:
        superset_idx = []
        for idx in superset:
            if idx in geom["candidate_vids"]:
                superset_idx.append(idx)

        superset_idx = np.array(superset_idx, dtype=np.int64)
        valid_supersets.append(superset_idx)

    obj_arr = np.array(valid_supersets, dtype=object)

    marker_idx_set = [
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
         751]]
    marker_idx_set = np.array(marker_idx_set, dtype=int)

    out_data = {
        "soma_superset_smplx": superset_smplx,  # [86]
        "soma_superset_smplh_cands": obj_arr,
        "mocap_solver_smplh": marker_idx_set,
    }

    np.savez(utils.Paths.data / "markersets.npz", **out_data)

    return

    device = utils.torch_utils.get_device(True)
    # legacy_data = np.load(legacy_file_path, allow_pickle=True)

    smplh = create_smpl_model(
        model_dir=utils.Paths.smpl_models,
        model_type="smplh",
        gender="male",
        num_betas=16
    ).to(device).eval()

    weights = smplh.get_weights(to_numpy=True)
    j_regressor = smplh.j_regressor
    parents = smplh.parents.cpu().numpy()

    # v = legacy_data["vertices"]
    v = smplh.v_template.cpu().numpy()
    # v = torch.from_numpy(v).to(device)
    f = smplh.faces

    # joints = vertices2joints(j_regressor, v).cpu().numpy()
    # j_regressor = j_regressor.cpu().numpy()

    v = v[None, ...]
    m = v[:, superset]

    viewer = Viewer(show_axis=False)
    viewer.add_mesh(v, f)  # v: [T, V, 3] / f: [F, 3]
    viewer.add_points(m)
    viewer.run()

    # print(weights.shape, weights.dtype)
    # print(joints.shape, joints.dtype)
    # print(j_regressor.shape, j_regressor.dtype)
    # print(parents.shape, parents.dtype)
    #
    # data = {}
    # data["vertices"] = v.cpu().numpy()
    # data["joints"] = joints
    # data["weights"] = weights
    # data["J_regressor"] = j_regressor
    # data["parents"] = parents
    #
    # np.savez_compressed(out_path, **data)


if __name__ == "__main__":
    main()
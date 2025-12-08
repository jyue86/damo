import pickle
import numpy as np
import torch

import damo.utils as utils
from damo.smpl.body_model import create_smpl_model
from damo.viz.viewer import Viewer


def main():
    caesar_path = r"C:\Users\kkm\projects\research\damo\data\caesar\caesar_male.npz"
    caesar_data = np.load(caesar_path)

    smplh = create_smpl_model(
        model_dir=utils.Paths.smpl_models,
        model_type="smplh",
        gender="male",
        num_betas=16,
    ).eval()

    # v = caesar_data["vertices"][0]
    v = smplh.v_template.detach().cpu().numpy()
    f = smplh.faces
    vn = utils.compute_vertex_normals(v, f)

    valid_mask, sample_weights = utils.build_marker_vertex_candidates(
        v, f
    )

    valid_idx = np.where(valid_mask)[0]
    colors = 1 - np.tile(sample_weights[..., None], (1, 3))
    colors = colors[valid_idx]
    colors = utils.normalize_ndarray(colors)

    mn = vn[valid_idx]
    m = v[valid_idx] + mn * 0.01

    print(m.shape)

    viewer = Viewer(show_axis=False)
    viewer.add_mesh(v[None, ...], smplh.faces)  # v: [T, V, 3] / f: [F, 3]
    viewer.add_points(m[None, ...], color=colors)
    viewer.run()


if __name__ == "__main__":
    main()
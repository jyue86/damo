import pickle
import numpy as np
import torch

import damo.utils as utils
from damo.smpl.body_model import create_smpl_model
from damo.viz.viewer import Viewer


def main():
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

    candidate_mask, sample_weights = utils.build_marker_vertex_candidates(v, f)
    candidate_vids = np.where(candidate_mask)[0]

    out_data = {
        "smplh_vertex_normals": vn,  # [V, 3]
        "candidate_mask": candidate_mask,  # [V]
        "candidate_vids": candidate_vids,  # [<V]
        "sample_weights": sample_weights,  # [V]
    }

    out_path = utils.Paths.data / "marker_geom.npz"
    np.savez_compressed(out_path, **out_data)


if __name__ == "__main__":
    main()
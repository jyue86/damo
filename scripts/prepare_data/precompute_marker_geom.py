import numpy as np

import damo.utils as utils
from damo.smpl.body_model import create_smpl_model


REQUIRED_KEYS = {
    "smplh_vertex_normals",
    "candidate_mask",
    "candidate_vids",
    "sample_weights",
}


def build_marker_geom(smplh_model):
    v = smplh_model.v_template.detach().cpu().numpy()
    f = smplh_model.faces
    vn = utils.compute_vertex_normals(v, f)

    candidate_mask, sample_weights = utils.build_marker_vertex_candidates(v, f)
    candidate_vids = np.where(candidate_mask)[0]

    return {
        "smplh_vertex_normals": vn,  # [V, 3]
        "candidate_mask": candidate_mask,  # [V]
        "candidate_vids": candidate_vids,  # [<V]
        "sample_weights": sample_weights,  # [V]
    }


def save_marker_geom(geom, out_dir=None):
    missing = REQUIRED_KEYS.difference(geom)
    if missing:
        raise ValueError(f"Missing marker geometry keys: {sorted(missing)}")

    out_dir = utils.Paths.data if out_dir is None else out_dir
    out_path = out_dir / "marker_geom_smplh.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **geom)
    return out_path


def main():
    smplh = create_smpl_model(
        model_dir=utils.Paths.smpl_models,
        model_type="smplh",
        gender="male",
        num_betas=16,
    ).eval()

    out_path = save_marker_geom(build_marker_geom(smplh))
    print(f"[ok] wrote marker geometry -> {out_path}")


if __name__ == "__main__":
    main()

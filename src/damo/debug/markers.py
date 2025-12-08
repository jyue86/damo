import torch
import numpy as np

import damo.utils as utils
from damo.smpl.body_model import create_smpl_model
from damo.viz.viewer import Viewer


def inspect_marker_idx(idx, body_type="smplh", gender="male", show_body=True):
    body_model = create_smpl_model(
        model_dir=utils.Paths.smpl_models,
        body_type=body_type,
        gender=gender,
    )
    v = body_model.v_template.numpy()
    f = body_model.faces
    m = v[idx]

    viewer = Viewer(show_axis=False)
    viewer.add_points(m)
    if show_body:
        viewer.add_mesh(v, f)
    viewer.run()
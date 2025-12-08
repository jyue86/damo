import torch
import numpy as np
import pickle

import damo.utils as utils
from damo.viz.viewer import Viewer
from damo.smpl.body_model import create_smpl_model

def main():
    data_smplx = np.load(r"C:\Users\kkm\projects\research\damo\data\amass\SMPLX\CMU\01\01_01_stageii.npz", allow_pickle=True)
    #
    # data_smplx = np.load(r"C:\Users\kkm\projects\research\damo\data\amass\SMPLX\CMU\15\15_03_stageii.npz", allow_pickle=True)
    data_smplh = np.load(r"C:\Users\kkm\projects\research\damo\data\amass\SMPLH\CMU\15\15_03_poses.npz", allow_pickle=True)

    markers_seq = data_smplx["markers_obs"][10:17]
    markers_latent_vids = data_smplx["markers_latent_vids"].item()
    latent_labels = list(markers_latent_vids.keys())
    latent_vids = np.stack([markers_latent_vids[k] for k in latent_labels], axis=0)
    labels_to_idx = {k: i for i, k in enumerate(latent_labels)}
    labels_seq = data_smplx["labels_obs"][10:17]

    S = len(markers_seq)
    M_seq = np.array([len(markers_t) for markers_t in markers_seq])
    M_max = int(M_seq.max())
    markers_mask = np.zeros((S, M_max), dtype=bool)

    if markers_seq.ndim == 3:
        print(1)
        idx_labels = np.vectorize(labels_to_idx.__getitem__)(labels_seq)
        markers_vids_seq = latent_vids[idx_labels]
        markers_mask[:, :] = True
    else:
        print(2)
        vids_shape = latent_vids.shape[1:]
        vid_pad_value = 0
        markers_vids_seq = np.full(
            (S, M_max, *vids_shape),
            vid_pad_value,
            dtype=latent_vids.dtype,
        )
        for i, labels_t in enumerate(labels_seq):
            M = M_seq[i]
            idx_labels = [labels_to_idx[k] for k in labels_t]
            markers_vids_seq[i, :M] = latent_vids[idx_labels]
            markers_mask[i, :M] = True

    model = create_smpl_model(
        model_dir=utils.Paths.smpl_models,
        model_type="smplx",
        gender="male",
        num_betas=16,
    ).eval()

    # markers_vids_seq = torch.from_numpy(markers_vids_seq)
    # markers_mask = torch.from_numpy(markers_mask)

    full_weights = model.get_weights(to_numpy=True)
    full_weights = utils.compress_weights_j55_to_j22(full_weights)
    weights = full_weights[markers_vids_seq]

    max_joint_idx = np.argmax(full_weights, axis=1)
    print(max_joint_idx.shape)
    print(full_weights[0])
    print(max_joint_idx[0])

    return

    rep_idx, rep_weights = utils.topk_weight_joints(weights, markers_mask)
    print(rep_idx.shape)
    print(rep_weights.shape)

    print(weights[0, 0])
    print("-------")
    print(rep_idx[0, 0])
    print("-------")
    print(rep_weights[0, 0])

    a = utils.gather_topk_joints(rep_idx, weights[..., None])
    print(a.shape)
    print(a[0, 0])

    return

    device = utils.get_device(True)
    model = create_smpl_model(
        model_dir=utils.Paths.smpl_models,
        model_type="smplh",
        gender="male",
        num_betas=16,
    ).to(device).eval()

    f = model.faces

    params = utils.get_params_from_amass(data_smplh, device=device)

    v = model(**params).vertices

    print(v.shape)

    markers = data_smplx["markers_obs"]
    offset = np.zeros_like(markers)
    offset[:, :, 1] = 2
    markers_gt = markers + offset
    print(markers.shape)

    viewer = Viewer()
    viewer.add_mesh(v, f)  # v: [T, V, 3] / f: [F, 3]
    viewer.add_points(markers)  # markers: [T, M, 3]
    viewer.add_points(markers_gt)
    viewer.run()

if __name__ == "__main__":
    main()
import numpy as np

import damo.utils as utils


def main():
    target_datasets = [
        "HDM05"
    ]

    root_dir = utils.Paths.amass_data
    for ds in target_datasets:
        merge_data(
            smplx_dir=(root_dir / "smplx" / ds).resolve(),
            smplh_dir=(root_dir / "smplh" / ds).resolve(),
            out_dir=(root_dir / "merged" / ds).resolve()
        )

def merge_data(smplx_dir, smplh_dir, out_dir):
    smplx_suffix = "_stageii"
    smplh_suffix = "_poses"

    frame_length_key = "trans"

    smplx_keys = ["gender", "betas", "trans", "poses",
                  "markers_latent", "latent_labels", "markers_latent_vids",
                  "markers_obs", "markers_sim", "labels_obs"]
    smplh_keys = ["gender", "betas", "trans", "poses"]

    for smplx_path in smplx_dir.rglob("*.npz"):
        if smplx_suffix not in smplx_path.stem:
            continue

        rel = smplx_path.relative_to(smplx_dir)
        stem_x = smplx_path.stem
        base = base_name_from_stem(stem_x, smplx_suffix)

        parent_rel = rel.parent
        smplh_name = base + smplh_suffix + smplx_path.suffix
        smplh_path = smplh_dir / parent_rel / smplh_name

        if not smplh_path.exists():
            print(f"[skip] smplh not found for {rel} (expected: {smplh_path.relative_to(smplx_dir)})")
            continue

        data_x = np.load(smplx_path, allow_pickle=True)
        data_h = np.load(smplh_path, allow_pickle=True)

        if frame_length_key not in data_x or frame_length_key not in data_h:
            print(f"[skip] '{frame_length_key}' missing in {rel}")
            continue

        len_x = data_x[frame_length_key].shape[0]
        len_h = data_h[frame_length_key].shape[0]

        if len_x != len_h:
            print(
                f"[skip] frame length mismatch for {rel}: "
                f"{len_x} (smplx) vs {len_h} (smplh)"
            )
            continue

        out_path = out_dir / parent_rel / (base + "_merged" + smplx_path.suffix)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        out_data = {}

        for k in smplx_keys:
            if k in data_x:
                out_data[f"smplx_{k}"] = data_x[k]

        for k in smplh_keys:
            if k in data_h:
                out_data[f"smplh_{k}"] = data_h[k]

        np.savez(out_path, **out_data)
        print(f"[ok] merged -> {out_path.relative_to(out_dir)}")


def base_name_from_stem(stem: str, suffix: str) -> str:
    return stem[:-len(suffix)] if stem.endswith(suffix) else stem

if __name__ == "__main__":
    main()
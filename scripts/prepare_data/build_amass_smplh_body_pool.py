import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import damo.utils as utils


@dataclass
class BodyPool:
    betas: np.ndarray
    gender: str
    source_paths: list[str]


def _npz_scalar(data, key: str):
    value = data[key]
    return value.item() if value.ndim == 0 else value


def collect_amass_smplh_betas(amass_merged_root: str | Path, gender: str) -> BodyPool:
    root = Path(amass_merged_root)
    if not root.exists():
        raise FileNotFoundError(f"AMASS merged root does not exist: {root}")

    seen = set()
    betas = []
    source_paths = []

    for path in sorted(root.rglob("*merged*.npz")):
        with np.load(path, allow_pickle=True) as data:
            if "smplh_gender" not in data or "smplh_betas" not in data:
                continue
            if _npz_scalar(data, "smplh_gender") != gender:
                continue

            beta = np.asarray(data["smplh_betas"], dtype=np.float32).reshape(-1)
            if beta.shape != (16,):
                raise ValueError(f"Expected smplh_betas shape (16,), got {beta.shape} in {path}")

            key = tuple(np.round(beta.astype(np.float64), decimals=8))
            if key in seen:
                continue

            seen.add(key)
            betas.append(beta)
            source_paths.append(str(path.relative_to(root)))

    if not betas:
        raise ValueError(f"No SMPL-H betas found for gender={gender!r} under {root}")

    return BodyPool(
        betas=np.stack(betas, axis=0).astype(np.float32),
        gender=gender,
        source_paths=source_paths,
    )


def save_body_pool(pool: BodyPool, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        betas=pool.betas.astype(np.float32),
        gender=np.array(pool.gender),
        source_paths=np.asarray(pool.source_paths, dtype=object),
        num_bodies=np.array(pool.betas.shape[0], dtype=np.int64),
    )
    return out_path


def default_out_path(gender: str) -> Path:
    return utils.Paths.data / "synthetic_bodies" / f"amass_smplh_betas_{gender}.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an AMASS SMPL-H beta pool for DAMO synthetic sampling.")
    parser.add_argument(
        "--amass-merged-root",
        type=Path,
        default=utils.Paths.amass_data / "merged",
        help="Root containing merged AMASS npz files.",
    )
    parser.add_argument("--gender", default="male", help="SMPL-H gender to extract.")
    parser.add_argument("--out", type=Path, default=None, help="Output npz path.")
    args = parser.parse_args()

    pool = collect_amass_smplh_betas(args.amass_merged_root, args.gender)
    out_path = save_body_pool(pool, args.out or default_out_path(args.gender))
    print(f"[ok] wrote {pool.betas.shape[0]} {args.gender} bodies -> {out_path}")


if __name__ == "__main__":
    main()

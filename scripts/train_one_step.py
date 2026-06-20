import os
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

import damo.utils as utils
from damo.dataset.amass_dataset import make_dataloader
from damo.trainer import Trainer


AMASS_ROOT = Path("/mnt/bcc-data/proj/ego-exo-collect/amass")
ONE_STEP_AMASS_ROOT = Path("/tmp/damo_train_one_step_amass")
DATA_DIR_NAME = "merged"
DATASET_NAMES = ["ACCAD", "HumanEva", "DFaust", "BMLmovi"]
MAX_FILES_PER_DATASET = 1


def _existing_model_path(model_type: str, gender: str) -> Path | None:
    model_dir = utils.Paths.smpl_models / model_type.lower()
    stem = f"{model_type.upper()}_{gender.upper()}"
    for ext in ("npz", "pkl"):
        path = model_dir / f"{stem}.{ext}"
        if path.exists():
            return path
    return None


def _require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def _npz_scalar(path: Path, key: str):
    with np.load(path, allow_pickle=True) as npz:
        if key not in npz:
            return None
        value = npz[key]
        return value.item() if value.ndim == 0 else value


def _preflight_data(wanted_genders: list[str]) -> tuple[dict[str, int], dict[str, list[Path]]]:
    merged_root = AMASS_ROOT / DATA_DIR_NAME
    _require_path(AMASS_ROOT, "AMASS root")
    _require_path(merged_root, "merged AMASS directory")

    file_counts = {}
    subset_sources = {}
    for dataset_name in DATASET_NAMES:
        dataset_dir = merged_root / dataset_name
        _require_path(dataset_dir, f"{dataset_name} dataset directory")
        files = sorted(p for p in dataset_dir.rglob("*merged*.npz") if p.is_file())
        count = len(files)
        if count == 0:
            raise FileNotFoundError(f"No merged npz files found under: {dataset_dir}")
        file_counts[dataset_name] = count
        matching_files = []
        for path in files:
            gender = _npz_scalar(path, "smplx_gender")
            if gender in wanted_genders:
                matching_files.append(path)
                if len(matching_files) >= MAX_FILES_PER_DATASET:
                    break
        if len(matching_files) < MAX_FILES_PER_DATASET:
            raise FileNotFoundError(
                f"Could not find {MAX_FILES_PER_DATASET} merged files with smplx_gender in "
                f"{wanted_genders} under: {dataset_dir}"
            )
        subset_sources[dataset_name] = matching_files

    if _existing_model_path("smplx", "male") is None:
        model_dir = utils.Paths.smpl_models / "smplx"
        raise FileNotFoundError(
            "Missing SMPLX male model; expected one of: "
            f"{model_dir / 'SMPLX_MALE.npz'} or {model_dir / 'SMPLX_MALE.pkl'}"
        )

    return file_counts, subset_sources


def _prepare_one_step_amass_root(subset_sources: dict[str, list[Path]]) -> Path:
    subset_merged_root = ONE_STEP_AMASS_ROOT / DATA_DIR_NAME
    subset_merged_root.mkdir(parents=True, exist_ok=True)

    for dataset_name, sources in subset_sources.items():
        dataset_dir = subset_merged_root / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for source in sources:
            target = dataset_dir / source.name
            if target.exists() or target.is_symlink():
                if not target.is_symlink() and not target.is_file():
                    raise FileExistsError(f"Refusing to replace non-file subset target: {target}")
                target.unlink()
            target.symlink_to(source)

    return ONE_STEP_AMASS_ROOT


def _apply_sanity_overrides(cfg: DictConfig) -> None:
    cfg.dataset.dataloader.batch_size = 1
    cfg.dataset.dataloader.num_workers = 0
    cfg.dataset.dataloader.pin_memory = False
    cfg.dataset.dataloader.drop_last = False

    cfg.dataset.dataset.cache_capacity = 4
    cfg.dataset.dataset.data_dir_name = DATA_DIR_NAME
    cfg.dataset.dataset.filename_pattern = "merged"
    cfg.dataset.dataset.dataset_names.train = DATASET_NAMES
    cfg.dataset.dataset.dataset_names.val = DATASET_NAMES
    cfg.dataset.dataset.data_type_probs = {"real": 1.0}

    cfg.trainer.amp = False
    cfg.trainer.log_interval = 1


def _format_loss(loss_dict: dict[str, torch.Tensor]) -> str:
    parts = []
    for key, value in loss_dict.items():
        scalar = value.detach().float().cpu().item() if torch.is_tensor(value) else float(value)
        parts.append(f"{key}={scalar:.6f}")
    return ", ".join(parts)


def _tensor_diag(name: str, value: torch.Tensor) -> str:
    detached = value.detach()
    finite = torch.isfinite(detached)
    finite_count = int(finite.sum().item())
    total_count = detached.numel()
    nonfinite_count = total_count - finite_count
    if finite_count:
        finite_values = detached[finite].float()
        min_value = finite_values.min().item()
        max_value = finite_values.max().item()
        mean_value = finite_values.mean().item()
        return (
            f"{name}: shape={tuple(detached.shape)} dtype={detached.dtype} "
            f"nonfinite={nonfinite_count}/{total_count} "
            f"finite_min={min_value:.6g} finite_max={max_value:.6g} finite_mean={mean_value:.6g}"
        )
    return (
        f"{name}: shape={tuple(detached.shape)} dtype={detached.dtype} "
        f"nonfinite={nonfinite_count}/{total_count} finite_min=<none> finite_max=<none> finite_mean=<none>"
    )


def _print_tensor_diags(prefix: str, tensors: dict[str, torch.Tensor]) -> None:
    for name, value in tensors.items():
        if torch.is_tensor(value):
            print(f"{prefix}.{_tensor_diag(name, value)}")


def _grad_diag(model: torch.nn.Module, label: str) -> None:
    total_params = 0
    params_with_grad = 0
    params_with_nonfinite = 0
    first_nonfinite = None
    max_abs_finite = 0.0

    for name, param in model.named_parameters():
        total_params += 1
        grad = param.grad
        if grad is None:
            continue
        params_with_grad += 1
        finite = torch.isfinite(grad)
        finite_count = int(finite.sum().item())
        total_count = grad.numel()
        if finite_count:
            max_abs_finite = max(max_abs_finite, grad.detach()[finite].abs().float().max().item())
        if finite_count != total_count:
            params_with_nonfinite += 1
            if first_nonfinite is None:
                first_nonfinite = (
                    name,
                    total_count - finite_count,
                    total_count,
                    str(grad.dtype),
                    tuple(grad.shape),
                )

    print(
        f"grad_diag[{label}]: total_params={total_params} params_with_grad={params_with_grad} "
        f"params_with_nonfinite={params_with_nonfinite} max_abs_finite_grad={max_abs_finite:.6g}"
    )
    if first_nonfinite is not None:
        name, nonfinite_count, total_count, dtype, shape = first_nonfinite
        print(
            f"grad_diag[{label}].first_nonfinite: name={name} "
            f"nonfinite={nonfinite_count}/{total_count} dtype={dtype} shape={shape}"
        )


@hydra.main(version_base=None, config_path=str(utils.Paths.config), config_name="config")
def run(cfg: DictConfig) -> None:
    utils.set_seed(cfg.seed)
    _apply_sanity_overrides(cfg)

    device = utils.get_device(use_cuda=cfg.use_cuda)

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"torch.cuda.is_available={torch.cuda.is_available()}")
    print(f"torch.cuda.device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"torch.cuda.current_device={torch.cuda.current_device()}")
        print(f"torch.cuda.device_name={torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"device={device}")

    wanted_genders = list(cfg.dataset.dataset.genders)
    file_counts, subset_sources = _preflight_data(wanted_genders)
    one_step_amass_root = _prepare_one_step_amass_root(subset_sources)
    utils.Paths.amass_data = one_step_amass_root
    print(f"amass_source_root={AMASS_ROOT}")
    print(f"amass_one_step_root={utils.Paths.amass_data}")
    print(f"dataset_names={','.join(DATASET_NAMES)}")
    print(f"genders={wanted_genders}")
    print(f"file_counts={file_counts}")
    print(f"subset_files_per_dataset={MAX_FILES_PER_DATASET}")
    print("data_type_probs={'real': 1.0}")
    print(
        "dataloader="
        f"batch_size={cfg.dataset.dataloader.batch_size},"
        f"num_workers={cfg.dataset.dataloader.num_workers},"
        f"pin_memory={cfg.dataset.dataloader.pin_memory}"
    )

    ds_train, train_loader = make_dataloader(cfg.dataset, train=True)
    if len(ds_train) == 0:
        raise RuntimeError(
            "Training dataset produced zero indexed frames after filtering "
            f"root={utils.Paths.amass_data / DATA_DIR_NAME}, datasets={DATASET_NAMES}, genders={cfg.dataset.dataset.genders}"
        )

    model = instantiate(cfg.model).to(device)
    trainer = Trainer(
        cfg=cfg.trainer,
        model=model,
        train_loader=train_loader,
        val_loader=None,
        device=device,
        logger=None,
    )

    batch = next(iter(train_loader))
    batch_shapes = {
        key: tuple(value.shape)
        for key, value in batch.items()
        if torch.is_tensor(value)
    }
    print(f"train_files={len(ds_train.file_paths)}")
    print(f"train_indexed_frames={len(ds_train)}")
    print(f"batch_shapes={batch_shapes}")

    trainer.model.train()
    outs = trainer._forward_loss(batch)
    _print_tensor_diags("batch", batch)
    _print_tensor_diags("model_out", outs["model_out"])
    _print_tensor_diags("loss", outs["loss"])
    print(f"amp_enabled={trainer.scaler.is_enabled()} grad_scaler_scale_before_backward={trainer.scaler.get_scale():.6g}")
    trainer.optim.zero_grad(set_to_none=True)
    trainer.scaler.scale(outs["loss"]["total"]).backward()
    _grad_diag(trainer.model, "after_scaled_backward")

    grad_norm = None
    if trainer.grad_clip_norm:
        trainer.scaler.unscale_(trainer.optim)
        _grad_diag(trainer.model, "after_unscale_before_clip")
        grad_norm = torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.grad_clip_norm)
        _grad_diag(trainer.model, "after_clip")

    trainer.scaler.step(trainer.optim)
    trainer.scaler.update()
    trainer.global_step += 1
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()

    print(f"optimizer_steps={trainer.global_step}")
    print(f"losses={_format_loss(outs['loss'])}")
    if grad_norm is not None:
        print(f"grad_norm_before_clip={grad_norm.detach().float().cpu().item():.6f}")
    print("one_step_status=ok")


if __name__ == "__main__":
    run()

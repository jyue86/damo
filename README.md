# DAMO

DAMO trains a marker-to-body-skinning model from AMASS/SMPL-family motion data. The model takes short sequences of 3D mocap markers and predicts skinning weights, representative joint weights, and representative joint offsets for the markers in the middle frame.

The current codebase supports AMASS preprocessing, dataloader inspection, and Hydra-based training. There is not yet a standalone evaluation or inference CLI, so evaluation and inference are documented below using the existing Python APIs.

## Environment

Use Python 3.11 or 3.12. Python 3.13 is not recommended because `open3d` does not publish Python 3.13 wheels.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

If uv cannot find Python 3.12 locally, install one with uv first:

```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

For CPU-only PyTorch, replace the synced PyTorch wheel after creating the environment:

```bash
uv pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu
```

For a specific CUDA runtime, install the matching PyTorch wheel for your machine. For example:

```bash
uv pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121
```

Dependency note from git history: the deleted `requirements.txt` was a broad Windows environment snapshot from the legacy `modules/` code. It included packages such as `c3d`, `keyboard`, `vpython`, `matplotlib`, `tensorboard`, and Jupyter tooling. The current `src/` and `scripts/` imports do not use `c3d`, `keyboard`, or `vpython`; those were tied to deleted legacy dataset, evaluation, and viewer scripts.

## Docker

The Docker image is an environment image only. It does not copy this repository into the image; mount a local checkout at `/workspace`.

Build:

```bash
docker build -t damo:cuda12.4-py312 .
```

Run with GPU access:

```bash
docker run --rm --gpus all \
  -v "$PWD":/workspace \
  -w /workspace \
  damo:cuda12.4-py312
```

Smoke test:

```bash
docker run --rm --gpus all \
  -v "$PWD":/workspace \
  -w /workspace \
  damo:cuda12.4-py312 \
  python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

For Apptainer, push the image to Docker Hub and pull it from there:

```bash
docker tag damo:cuda12.4-py312 <dockerhub-user>/damo:cuda12.4-py312
docker push <dockerhub-user>/damo:cuda12.4-py312
apptainer pull docker://<dockerhub-user>/damo:cuda12.4-py312
```

Run the resulting SIF with NVIDIA passthrough and bind the repo:

```bash
apptainer shell --nv --bind "$PWD":/workspace damo_cuda12.4-py312.sif
```

## Required Data

Path constants live in `src/damo/utils/paths.py`. The training path expects local assets under:

```text
data/
  amass/
    smplx/<DATASET>/**/*_stageii.npz
    smplh/<DATASET>/**/*_poses.npz
    merged/<DATASET>/**/*_merged.npz
  smpl/models/
    smplh/SMPLH_MALE.npz or .pkl
    smplx/SMPLX_MALE.npz or .pkl
    caesar_male.npz
  markersets.npz
  marker_geom_smplh.npz
```

The active dataset code uses SMPL-H for synthetic samples and SMPL-X marker fields for real samples. It compresses SMPL-H/SMPL-X weights to 22 body joints and adds one ghost-marker class, so model outputs use `J+1 = 23` weight channels by default. Historical `support_data/train_options.json` used `n_joints=24` and `n_max_markers=90`; that belongs to the deleted legacy `modules/` pipeline and should not be copied directly into the current Hydra config. The current defaults are `model.n_joints=22`, `model.n_rep_joints=3`, `model.seq_len=7`, and marker sampling between 44 and 88 markers.

Merged AMASS files must contain the keys written by `scripts/process_amass_dataset.py`, including SMPL-X marker arrays such as `smplx_markers_latent`, `smplx_markers_obs`, `smplx_labels_obs`, and SMPL-H pose arrays such as `smplh_trans` and `smplh_poses`.

No pretrained checkpoints are included. Train a checkpoint first, or provide your own checkpoint in the format saved by `src/damo/utils/io_utils.py`:

```python
{"step": global_step, "model": model.state_dict()}
```

Older README revisions only pointed to a Google Drive trained model, and deleted legacy evaluation scripts hard-coded checkpoints such as `damo_240412022202_epc290`. Treat those as legacy `modules/` checkpoints unless the saved `options.json` and tensor names match the current `src/damo/model/damo.py` state dict.

## Prepare AMASS Data

Merge matching SMPL-X marker files and SMPL-H pose files:

```bash
uv run python scripts/process_amass_dataset.py
```

This script currently hard-codes `HDM05` and reads:

```text
data/amass/smplx/HDM05/**/*_stageii.npz
data/amass/smplh/HDM05/**/*_poses.npz
```

It writes merged files to:

```text
data/amass/merged/HDM05/**/*_merged.npz
```

Precompute marker geometry:

```bash
uv run python scripts/prepare_data/precompute_marker_geom.py
```

Note: the script writes `data/marker_geom.npz`, while `MarkerSampler` expects `data/marker_geom_smplh.npz`. Rename or adjust the path before training:

```bash
mv data/marker_geom.npz data/marker_geom_smplh.npz
```

`data/markersets.npz` is also required. The current generator script is `scripts/temp/process_caesar.py`, but it contains hard-coded local paths and needs cleanup before it is portable.

Historical dataset generation used AMASS subsets `ACCAD`, `PosePrior`, `CMU`, `DanceDB`, `HDM05`, `SFU`, `MoSh`, and `SOMA`, plus CAESAR-derived bind bodies and SOMA marker-set variations. The current portable preprocessing script only merges matching SMPL-X/SMPL-H files; it does not recreate the deleted pickle dataset format from `modules/dataset/generate_train_dataset.py`.

## Inspect Data Loading

Run a dataloader smoke test:

```bash
uv run python scripts/inspect_dataset.py logger.mode=offline dataset.dataloader.num_workers=0
```

The default dataset config uses:

```text
data/amass/merged/HDM05/*merged*.npz
```

Change datasets with Hydra overrides:

```bash
uv run python scripts/inspect_dataset.py \
  dataset.dataset.dataset_names.train='[HDM05]' \
  dataset.dataset.dataset_names.val='[HDM05]'
```

## Train

Train with the default Hydra config:

```bash
uv run python scripts/train.py
```

Useful overrides:

```bash
uv run python scripts/train.py use_cuda=false logger.mode=offline trainer.max_epochs=1
uv run python scripts/train.py dataset.dataloader.batch_size=8 dataset.dataloader.num_workers=0
uv run python scripts/train.py dataset.dataset.dataset_names.train='[HDM05]' dataset.dataset.dataset_names.val='[HDM05]'
```

Hydra changes the working directory to a run directory under:

```text
../outputs/YYYYMMDD_HHMMSS_Damo/
```

Each run saves:

```text
config_resolved.yaml
app.log
ckpt_last.pt
ckpt_best.pt
samples/epc<N>_train_stepXXXXXX.npz
samples/epc<N>_val_stepXXXXXX.npz
wandb/
```

Weights and Biases is enabled by default. Use `logger.mode=offline` for local runs.

Legacy training was driven by `support_data/train_options.json` through `modules/training/train_damo.py`. Useful historical intent from that file maps to current Hydra overrides as follows:

```bash
uv run python scripts/train.py \
  dataset.dataset.dataset_names.train='[ACCAD,PosePrior,CMU]' \
  dataset.dataset.dataset_names.val='[HDM05,SFU,MoSh,SOMA]' \
  dataset.dataloader.batch_size=64 \
  trainer.max_epochs=300 \
  model.seq_len=7 \
  model.marker_attention.d_model=125 \
  model.marker_attention.n_heads=5 \
  model.marker_attention.n_layers=4
```

The old learning rate was `1e-5` with Adam; the current trainer uses AdamW and defaults to `1e-3`, so change `trainer.optimizer.learning_rate` deliberately rather than assuming the legacy value is still optimal.

## Evaluate

There is no dedicated evaluation script yet. Validation runs automatically during training every `trainer.val_interval` epochs and saves validation sample snapshots under the run directory.

The deleted `modules/evaluation/inference.py` and `modules/evaluation/evaluate.py` formed a two-stage pipeline: first save model predictions for each eval dataset/noise type, then solve body pose from predicted weights and offsets and compute joint position/orientation errors. That pipeline depended on legacy pickle files under `datasets/eval` and `test_results`, so it is a reference for future work rather than a current run command.

To evaluate an existing checkpoint against the configured validation loader, use the existing APIs:

```python
import torch
from hydra import compose, initialize
from hydra.utils import instantiate

import damo.utils as utils
from damo.dataset.amass_dataset import make_dataloader
from damo.logger.base_logger import BaseLogger
from damo.trainer import Trainer

with initialize(version_base=None, config_path="conf"):
    cfg = compose(config_name="config", overrides=["logger.mode=offline"])

device = utils.get_device(cfg.use_cuda)
_, train_loader = make_dataloader(cfg.dataset, train=True)
_, val_loader = make_dataloader(cfg.dataset, train=False)

model = instantiate(cfg.model).to(device)
ckpt = torch.load("path/to/ckpt_best.pt", map_location=device)
model.load_state_dict(ckpt["model"])

trainer = Trainer(cfg.trainer, model, train_loader, val_loader, device=device, logger=BaseLogger())
_, val_loss = trainer.val_epoch(epoch=0)
print(val_loss)
```

## Inference

There is no dedicated inference script yet. The model forward pass expects:

```text
markers_seq       [B, S, M, 3]
markers_seq_mask  [B, S, M], optional
```

`S` must match `model.seq_len` from `conf/model/default.yaml`, which defaults to `7`. A minimal checkpoint inference call looks like:

```python
import torch
from hydra import compose, initialize
from hydra.utils import instantiate

import damo.utils as utils

with initialize(version_base=None, config_path="conf"):
    cfg = compose(config_name="config")

device = utils.get_device(cfg.use_cuda)
model = instantiate(cfg.model).to(device).eval()

ckpt = torch.load("path/to/ckpt_best.pt", map_location=device)
model.load_state_dict(ckpt["model"])

markers_seq = torch.zeros(1, cfg.model.seq_len, 64, 3, device=device)
markers_seq_mask = (markers_seq != 0).any(dim=-1)

with torch.no_grad():
    out = model(markers_seq=markers_seq, markers_seq_mask=markers_seq_mask)

print(out["weights"].shape)
print(out["rep_weights"].shape)
print(out["rep_offsets"].shape)
```

The output contains:

```text
weights      [B, M, J+1]
rep_weights  [B, M, Jr]
rep_offsets  [B, M, Jr, 3]
markers_mask [B, M]
```

## Mocap CSV Cleaning

The mocap cleaning helpers are exposed from `damo.utils.mocap_utils`:

```python
from damo.utils.mocap_utils import clean_mocap_data, parse_mocap_data, extract_transform_from_df
```

CLI usage:

```bash
uv run python -m damo.utils.mocap_utils \
  --data path/to/input.csv \
  --frames Pelvis Head \
  --output path/to/output.csv
```

The current cleaner assumes rigid-body mocap CSVs with quaternion rotation and XYZ position columns. The `data/mocap-pcd/*.csv` files are point-cloud marker exports with repeated `X,Y,Z` marker columns and missing values, so they need a dedicated normalization step before they can become model input.

## History Audit Notes

The December 2025 refactor deleted the legacy `modules/` package, `requirements.txt`, `support_data/train_options.json`, `support_data/train_options_processed.json`, and support metadata files. It added the current `src/damo`, Hydra configs in `conf/`, and scripts under `scripts/`.

Deleted files reviewed during README reconstruction:

- `support_data/train_options.json`: legacy run config with train datasets `ACCAD`, `PosePrior`, `CMU`, eval datasets `HDM05`, `SFU`, `MoSh`, `SOMA`, sequence length 7, max 90 markers, batch size 64, 300 epochs, and CUDA.
- `support_data/train_options_processed.json`: saved run metadata for an older SFU-only run, including Windows-local dataset/checkpoint paths.
- `support_data/*_meta.txt`: shape notes showing 22-joint processed data, 52-joint SMPL-H arrays, 87 SOMA superset entries, and 1700 CAESAR bodies.
- `modules/dataset/generate_train_dataset.py`: legacy AMASS/C3D-to-pickle generator using `human_body_prior`, CAESAR bodies, SOMA marker variations, `c3d`, and multiprocessing.
- `modules/training/training_options.py` and `modules/training/train_damo.py`: legacy JSON option loader and incomplete training entry point.
- `modules/evaluation/inference.py` and `modules/evaluation/evaluate.py`: legacy checkpoint-specific inference/evaluation scripts with hard-coded model names, eval dates, and output directories.
- `requirements.txt`: deleted dependency snapshot for the old environment, not a curated dependency list for the refactored package.

## Current Caveats

- `conf/solver/default.yaml` is currently a placeholder. It is included in Hydra defaults, so `cfg.solver` exists, but `scripts/train.py` never reads it and `Trainer` is constructed with `cfg.trainer` instead. Network optimization settings currently live under `trainer`, including optimizer hyperparameters, AMP, gradient clipping, and loss weights. The repository also contains `src/damo/solver/svd_solver.py`, but it is not wired into train, evaluation, or inference. Reasonable next steps are:
  - Remove `solver` from `conf/config.yaml` if it is not part of the current pipeline.
  - Move optimizer/loss settings from `trainer` to `solver` if the intended meaning of solver is "training optimization".
  - Keep `solver` for a future post-processing stage if `svd_solver.py` is intended to convert DAMO predictions into body transforms during evaluation/inference.
- `scripts/temp/*` contains exploratory scripts with hard-coded local paths.
- There is no checkpoint resume path, evaluation CLI, or inference CLI yet.
- Training requires the external AMASS/SMPL assets listed above; this repository does not include them.

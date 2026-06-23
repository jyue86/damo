#!/bin/bash -l

#SBATCH --nodes=1 # Allocate *at least* 1 node to this job.
#SBATCH --ntasks=1 # Allocate *at most* 1 task for job steps in the job
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=0-128:00:00
#SBATCH --mail-user=jyue009@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=damo_training
#SBATCH -p cisl # You could pick other partitions for other jobs
#SBATCH --nodelist=dune1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SIF="${REPO_DIR}/dune1.sif"
AMASS_HOST="/data/HangQiu"

mkdir -p "${REPO_DIR}/logs" "${REPO_DIR}/outputs"
cd "${REPO_DIR}"

if [[ ! -f "${SIF}" ]]; then
  echo "Missing Apptainer image: ${SIF}" >&2
  exit 1
fi

echo "job_id=${SLURM_JOB_ID:-local}"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo "repo_dir=${REPO_DIR}"
echo "sif=${SIF}"
echo "amass_mount=${AMASS_HOST}:/mnt/bcc-data"

# Experiment config controls the train/val split, synthetic mix, validation branch,
# batch size, AMP, and model size. Keep this launcher focused on cluster paths.
apptainer exec --nv \
  --bind "${REPO_DIR}:${REPO_DIR}" \
  --bind "${AMASS_HOST}:/mnt/bcc-data" \
  --env "PYTHONPATH=${REPO_DIR}/src:/workspace/src" \
  --env HYDRA_FULL_ERROR=1 \
  --pwd "${REPO_DIR}" \
  "${SIF}" \
  python scripts/train.py \
    +experiment=mixed_train_real_val \
    out_dir="${REPO_DIR}/outputs" \
    dataset.amass_root=/mnt/bcc-data/proj/ego-exo-collect/amass

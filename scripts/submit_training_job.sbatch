#!/bin/bash -l

#SBATCH --nodes=1 # Allocate *at least* 1 node to this job.
#SBATCH --ntasks=1 # Allocate *at most* 1 task for job steps in the job
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
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
AMASS_HOST="/mnt/bcc-data"

mkdir -p "${REPO_DIR}/logs" "${REPO_DIR}/outputs"

if [[ ! -f "${SIF}" ]]; then
  echo "Missing Apptainer image: ${SIF}" >&2
  exit 1
fi

echo "job_id=${SLURM_JOB_ID:-local}"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo "repo_dir=${REPO_DIR}"
echo "sif=${SIF}"
echo "amass_mount=${AMASS_HOST}:/mnt/bcc-data"

apptainer exec --nv \
  --bind "${REPO_DIR}:/workspace" \
  --bind "${AMASS_HOST}:/mnt/bcc-data" \
  --pwd /workspace \
  "${SIF}" \
  python scripts/train.py \
    dataset.dataset.dataset_names.train='[ACCAD,HumanEva,DFaust,BMLmovi]' \
    dataset.dataset.dataset_names.val='[ACCAD,HumanEva,DFaust,BMLmovi]' \
    dataset.dataset.data_type_probs='{real: 1.0}' \
    dataset.dataloader.batch_size=128 \
    dataset.dataloader.num_workers=4 \
    trainer.amp=true

#!/bin/bash
# Checkpointed, requeueable SFT on ckpt-all. Several copies (e.g. A100 and H200) can be queued:
# the first to start cancels the others; a copy that starts while another is running exits.
#   sbatch --gres=gpu:h200:1 scripts/klone_train_ckpt.sbatch
#SBATCH --job-name=qwen35-pixmo-sft
#SBATCH --account=ckpt-krishna
#SBATCH --partition=ckpt-all
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=9:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err

set -eo pipefail
others_running=$(squeue -h -u "$USER" -n "$SLURM_JOB_NAME" -t RUNNING -o %i | grep -vx "$SLURM_JOB_ID" || true)
if [ -n "$others_running" ]; then
  echo "another copy is running ($others_running); exiting"; exit 0
fi
for j in $(squeue -h -u "$USER" -n "$SLURM_JOB_NAME" -t PENDING -o %i | grep -vx "$SLURM_JOB_ID" || true); do
  echo "cancelling pending sibling $j"; scancel "$j"
done

STORAGE_ROOT="/gscratch/krishna/$USER"
cd "$STORAGE_ROOT/qwen35-pixmo-sft"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
module load cuda/12.8.1
module load coenv/miniconda/3.13.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$STORAGE_ROOT/conda-envs/title-filter"

export PYTHONNOUSERSITE=1
export HF_HOME="$STORAGE_ROOT/hf-cache"
export XDG_CACHE_HOME="$STORAGE_ROOT/xdg-cache"
export TMPDIR="$STORAGE_ROOT/tmp"
export TRITON_CACHE_DIR="$STORAGE_ROOT/triton"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "start $(date) on $(hostname) restart=${SLURM_RESTART_COUNT:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/test_offload.py
python scripts/prepare_data.py
python "${SFT_ENTRY:-train.py}"

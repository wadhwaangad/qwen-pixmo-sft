#!/bin/bash
# Submit from the repo root: sbatch scripts/klone_train.sbatch
#SBATCH --job-name=qwen35-pixmo-sft
#SBATCH --account=krishna
#SBATCH --partition=gpu-a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err

set -eo pipefail

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
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/prepare_data.py
python "${SFT_ENTRY:-train.py}"

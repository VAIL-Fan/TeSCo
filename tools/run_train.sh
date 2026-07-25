#!/usr/bin/env bash
# TeSCo EgoExo4D training (matches paper Sec. 4: 480 res, batch 3/GPU, 60 epochs).
# If other jobs use some GPUs, pin this run, e.g.:
#   CUDA_VISIBLE_DEVICES=0,1 ./tools/run_train.sh 2
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
NUM_GPUS="${1:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
CONDA_ENV="${CONDA_ENV:-tesco}"
exec conda run -n "$CONDA_ENV" python training/train.py \
  -c configs/sam2.1_training/sam2.1_hiera_b+_EgoExo_finetune.yaml \
  --use-cluster 0 \
  --num-gpus "$NUM_GPUS"

#!/usr/bin/env bash
# HANDAL-X "retraining" setting (paper Table 2(b)): train from the official SAM2.1 base+ init
# directly on HANDAL-X (handal_lmeec_i100), matching the paper's "trained and evaluated on
# HANDAL-X" protocol. Saves to training/checkpoints_Handal (kept separate from
# checkpoints_EgoExo -- see comments in the config for why that matters).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
NUM_GPUS="${1:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
CONDA_ENV="${CONDA_ENV:-tesco}"
exec conda run -n "$CONDA_ENV" python training/train.py \
  -c configs/sam2.1_training/sam2.1_hiera_b+_Handal_finetune.yaml \
  --use-cluster 0 \
  --num-gpus "$NUM_GPUS"

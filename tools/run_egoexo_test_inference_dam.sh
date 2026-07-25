#!/usr/bin/env bash
# One-shot: EgoExo4D test split, ego→exo (--swap left default False), text from DAM_FRAME_CAPTIONS_SUBDIR only (predictor init_state).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TEST="${EEO_TEST_DIR:-/home/UNT/bf0191/Documents/bingfan_projects/dataset/EgoExo4D/test}"
CKPT="${CKPT:-$REPO/training/checkpoints_EgoExo/checkpoint.pt}"
PYTHON="${PYTHON:-/opt/miniconda3/bin/python3}"
# Hydra config name is relative to sam2/configs (no "configs/" prefix).
CFG="${SAM2_CFG:-sam2.1/sam2.1_hiera_b+_egoexo_text.yaml}"
TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT_INF:-$REPO/output/egoexo_test_dam_${TAG}}"
LOG="${LOG_DIR:-$REPO/output/logs_egoexo_test_dam_${TAG}}"
NUM_GPUS="${NUM_GPUS:-4}"

mkdir -p "$OUT" "$LOG"
echo "Repo:     $REPO"
echo "Test:     $TEST"
echo "Checkpoint: $CKPT"
echo "SAM2 cfg: $CFG  (ego→exo: default --swap False)"
echo "Output:   $OUT"
echo "Logs:     $LOG"

for ((GPU_ID = 0; GPU_ID < NUM_GPUS; GPU_ID++)); do
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" "$REPO/tools/correspondence_inference.py" \
    --sam2_cfg "$CFG" \
    --sam2_checkpoint "$CKPT" \
    --base_video_dir "$TEST" \
    --output_mask_dir "$OUT" \
    --score_thresh 0.0 \
    --gpu_id "$GPU_ID" \
    --num_gpus "$NUM_GPUS" \
    > "$LOG/gpu${GPU_ID}.log" 2>&1 &
  echo "Started GPU $GPU_ID (pid $!)"
done
wait
echo "Done. Output: $OUT" | tee "$LOG/done.txt"

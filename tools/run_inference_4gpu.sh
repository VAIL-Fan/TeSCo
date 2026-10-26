#!/bin/bash
# 4-GPU parallel inference for TeSCo on the EgoExo4D test set.
# Test videos are distributed in round-robin across 4 GPUs.
# Re-run at any time; already-processed videos are skipped automatically.
set -euo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
DATASET="${EEO_DATA_ROOT:-$PROJ/data/EgoExo4D}/test"
# Default checkpoint: epoch-59 (0-indexed => 60 epochs), matching the paper's stated training
# length and the checkpoint that reproduces this repo's Table 9 TeSCo/Exo->Ego numbers.
# Override with CKPT=$PROJ/training/checkpoints_EgoExo/tesco_egoexo4d_epoch79.pth for the
# the other released checkpoint instead.
CKPT="${CKPT:-$PROJ/training/checkpoints_EgoExo/tesco_egoexo4d_epoch59.pth}"
SAM2_CFG=configs/sam2.1/sam2.1_hiera_b+_egoexo_text.yaml
OUTPUT="${OUTPUT:-$PROJ/output/egoexo_test}"
NUM_GPUS=4
LOG_DIR="$OUTPUT/logs"
PYTHON="${PYTHON:-python3}"

mkdir -p "$LOG_DIR" "$OUTPUT"

echo "Starting 4-GPU inference..."
echo "  Checkpoint : $CKPT"
echo "  SAM2 cfg   : $SAM2_CFG"
echo "  Dataset    : $DATASET"
echo "  Output     : $OUTPUT"
echo ""

for GPU_ID in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$GPU_ID "$PYTHON" "$PROJ/tools/correspondence_inference.py" \
        --sam2_cfg "$SAM2_CFG" \
        --sam2_checkpoint "$CKPT" \
        --base_video_dir  "$DATASET" \
        --output_mask_dir "$OUTPUT" \
        --gpu_id  $GPU_ID \
        --num_gpus $NUM_GPUS \
        > "$LOG_DIR/gpu${GPU_ID}.log" 2>&1 &
    echo "GPU $GPU_ID started (PID $!)"
done

echo ""
echo "All 4 GPU processes launched. Waiting for completion..."
wait
echo "All done. Outputs in: $OUTPUT"
echo "Logs in: $LOG_DIR/gpu{0..3}.log"

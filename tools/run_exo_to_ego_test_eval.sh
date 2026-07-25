#!/usr/bin/env bash
# EgoExo4D **test** eval: exo→ego (matches training: correspondence_direction + dam_frame_captions_v2_exo).
# Inference → merge_pred → merge_results (--swap for exo-ego keys) → compute_metrics (--task-key exo-ego).
#
# Usage:
#   export EEO_DATA_ROOT=/path/to/EgoExo4D
#   bash tools/run_exo_to_ego_test_eval.sh
#
# Only merge + metrics after inference:
#   ONLY_MERGE=1 OUT_INF=/path/to/output bash tools/run_exo_to_ego_test_eval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${REPO_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EEO_DATA_ROOT="${EEO_DATA_ROOT:-/home/UNT/bf0191/Documents/MLLM_EgoExo_Proposal/LM-EEC/data/EgoExo4D}"
TEST_DIR="${EEO_DATA_ROOT}/test"
RELATIONS_TEST="${EEO_DATA_ROOT}/annotations/relations_test.json"
GT_JSON="${EEO_DATA_ROOT}/gt_results/exoego_test_gt.json"

CKPT="${CKPT:-${REPO_ROOT}/training/checkpoints_EgoExo/checkpoint.pt}"
SAM2_CFG="${SAM2_CFG:-sam2.1/sam2.1_hiera_b+_egoexo_text.yaml}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_INF="${OUT_INF:-${REPO_ROOT}/output/exo_to_ego_test_${RUN_TAG}}"
SUBMIT_JSON="${SUBMIT_JSON:-${REPO_ROOT}/output/exo_to_ego_test_results_${RUN_TAG}.json}"
LOG_DIR="${REPO_ROOT}/output/logs_exo_to_ego_test_${RUN_TAG}"
NUM_GPUS="${NUM_GPUS:-4}"
# Optional: pin to a subset, e.g. CUDA_VISIBLE_DEVICES=0,1,2,3 (physical GPUs for workers 0..NUM_GPUS-1)
PHYSICAL_GPUS="${PHYSICAL_GPUS:-$(seq -s, 0 $((NUM_GPUS - 1)))}"
IFS=',' read -r -a _GPU_ARR <<< "${PHYSICAL_GPUS}"

ONLY_MERGE="${ONLY_MERGE:-0}"

[[ -d "${TEST_DIR}" ]] || { echo "Missing test dir: ${TEST_DIR}"; exit 1; }
[[ -f "${RELATIONS_TEST}" ]] || { echo "Missing ${RELATIONS_TEST}"; exit 1; }
[[ -f "${GT_JSON}" ]] || { echo "Missing ${GT_JSON}"; exit 1; }

if [[ "${ONLY_MERGE}" != "1" ]]; then
  [[ -f "${CKPT}" ]] || { echo "Missing checkpoint: ${CKPT}"; exit 1; }
fi

if [[ "${ONLY_MERGE}" != "1" ]]; then
  mkdir -p "${OUT_INF}" "${LOG_DIR}"
  echo "=== Inference exo→ego (${NUM_GPUS} workers, CUDA_VISIBLE_DEVICES per worker from PHYSICAL_GPUS=${PHYSICAL_GPUS}) → ${OUT_INF} ==="
  for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    _PHYS="${_GPU_ARR[$GPU_ID]:-$GPU_ID}"
    CUDA_VISIBLE_DEVICES="${_PHYS}" conda run -n lm-eec-apr13 --no-capture-output python "${REPO_ROOT}/tools/correspondence_inference.py" \
      --sam2_cfg "${SAM2_CFG}" \
      --sam2_checkpoint "${CKPT}" \
      --base_video_dir "${TEST_DIR}" \
      --output_mask_dir "${OUT_INF}" \
      --score_thresh 0.0 \
      --gpu_id "${GPU_ID}" \
      --num_gpus "${NUM_GPUS}" \
      --exo_to_ego \
      > "${LOG_DIR}/gpu${GPU_ID}.log" 2>&1 &
    echo "Started worker ${GPU_ID} on physical GPU ${_PHYS} (log: ${LOG_DIR}/gpu${GPU_ID}.log) pid=$!"
  done
  wait
  echo "=== Inference finished ==="
else
  echo "=== ONLY_MERGE=1: skipping inference, using OUT_INF=${OUT_INF} ==="
  [[ -d "${OUT_INF}" ]] || { echo "OUT_INF not a directory"; exit 1; }
fi

echo "=== merge_pred.py ==="
conda run -n lm-eec-apr13 --no-capture-output python "${REPO_ROOT}/evaluation/merge_pred.py" \
  --input "${OUT_INF}" \
  --gt "${RELATIONS_TEST}" \
  --pred "${OUT_INF}"

echo "=== merge_results.py (exo-ego, --swap) ==="
conda run -n lm-eec-apr13 --no-capture-output python "${REPO_ROOT}/evaluation/merge_results.py" \
  --input_dir "${EEO_DATA_ROOT}" \
  --split test \
  --pred_dir "${OUT_INF}" \
  --json_path "${SUBMIT_JSON}" \
  --swap

echo "=== compute_metrics.py (task exo-ego) ==="
cd "${REPO_ROOT}/evaluation"
conda run -n lm-eec-apr13 --no-capture-output python compute_metrics.py \
  --gt-file "${GT_JSON}" \
  --pred-file "${SUBMIT_JSON}" \
  --task-key exo-ego

cd "${REPO_ROOT}"
echo ""
echo "Done."
echo "  Inference dir:  ${OUT_INF}"
echo "  Submit JSON:    ${SUBMIT_JSON}"
echo "  GT JSON:        ${GT_JSON}"
echo "  Logs:           ${LOG_DIR}/"

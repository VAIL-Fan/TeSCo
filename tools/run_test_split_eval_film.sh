#!/usr/bin/env bash
# Full EgoExo4D **test** eval: FiLM + text checkpoint, ego→exo (official challenge direction).
# Steps match inference_guide_Mar18.md; model config matches docs/VISUALIZATION_EGOEXO.md (FiLM / use_text_cross_attn).
#
# Usage (from repo root):
#   export EEO_DATA_ROOT=/path/to/EgoExo4D   # optional; default below
#   bash tools/run_test_split_eval_film.sh
#
# Or only merge+metrics after inference finished:
#   ONLY_MERGE=1 OUT_INF=/path/to/output/egoexo_test_film bash tools/run_test_split_eval_film.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${REPO_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EEO_DATA_ROOT="${EEO_DATA_ROOT:-${REPO_ROOT}/data/EgoExo4D}"
TEST_DIR="${EEO_DATA_ROOT}/test"
RELATIONS_TEST="${EEO_DATA_ROOT}/annotations/relations_test.json"
GT_JSON="${EEO_DATA_ROOT}/gt_results/egoexo_test_gt.json"

CKPT="${CKPT:-${REPO_ROOT}/training/checkpoints_EgoExo/checkpoint.pt}"
# Hydra config name is relative to sam2/configs (no leading configs/).
SAM2_CFG="${SAM2_CFG:-sam2.1/sam2.1_hiera_b+_egoexo_text.yaml}"
PYTHON="${PYTHON:-/opt/miniconda3/bin/python3}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_INF="${OUT_INF:-${REPO_ROOT}/output/egoexo_test_film_${RUN_TAG}}"
SUBMIT_JSON="${SUBMIT_JSON:-${REPO_ROOT}/output/egoexo_test_film_results_${RUN_TAG}.json}"
LOG_DIR="${REPO_ROOT}/output/logs_test_film_${RUN_TAG}"
NUM_GPUS="${NUM_GPUS:-4}"

ONLY_MERGE="${ONLY_MERGE:-0}"

[[ -d "${TEST_DIR}" ]] || { echo "Missing test dir: ${TEST_DIR}"; exit 1; }
[[ -f "${RELATIONS_TEST}" ]] || { echo "Missing ${RELATIONS_TEST}"; exit 1; }
[[ -f "${GT_JSON}" ]] || { echo "Missing ${GT_JSON}"; exit 1; }

if [[ "${ONLY_MERGE}" != "1" ]]; then
  [[ -f "${CKPT}" ]] || { echo "Missing checkpoint: ${CKPT}"; exit 1; }
fi

if [[ "${ONLY_MERGE}" != "1" ]]; then
  mkdir -p "${OUT_INF}" "${LOG_DIR}"
  echo "=== Inference (${NUM_GPUS} GPUs) → ${OUT_INF} ==="
  for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" "${REPO_ROOT}/tools/correspondence_inference.py" \
      --sam2_cfg "${SAM2_CFG}" \
      --sam2_checkpoint "${CKPT}" \
      --base_video_dir "${TEST_DIR}" \
      --output_mask_dir "${OUT_INF}" \
      --score_thresh 0.0 \
      --gpu_id "${GPU_ID}" \
      --num_gpus "${NUM_GPUS}" \
      > "${LOG_DIR}/gpu${GPU_ID}.log" 2>&1 &
    echo "Started GPU ${GPU_ID} (log: ${LOG_DIR}/gpu${GPU_ID}.log) pid=$!"
  done
  wait
  echo "=== Inference finished ==="
else
  echo "=== ONLY_MERGE=1: skipping inference, using OUT_INF=${OUT_INF} ==="
  [[ -d "${OUT_INF}" ]] || { echo "OUT_INF not a directory"; exit 1; }
fi

echo "=== merge_pred.py ==="
"${PYTHON}" "${REPO_ROOT}/evaluation/merge_pred.py" \
  --input "${OUT_INF}" \
  --gt "${RELATIONS_TEST}" \
  --pred "${OUT_INF}"

echo "=== merge_results.py (ego→exo, no --swap) ==="
"${PYTHON}" "${REPO_ROOT}/evaluation/merge_results.py" \
  --input_dir "${EEO_DATA_ROOT}" \
  --split test \
  --pred_dir "${OUT_INF}" \
  --json_path "${SUBMIT_JSON}"

echo "=== compute_metrics.py ==="
cd "${REPO_ROOT}/evaluation"
"${PYTHON}" compute_metrics.py \
  --gt-file "${GT_JSON}" \
  --pred-file "${SUBMIT_JSON}"

cd "${REPO_ROOT}"
echo ""
echo "Done."
echo "  Inference dir:  ${OUT_INF}"
echo "  Submit JSON:    ${SUBMIT_JSON}"
echo "  GT JSON:        ${GT_JSON}"
echo "  Per-GPU logs:   ${LOG_DIR}/"

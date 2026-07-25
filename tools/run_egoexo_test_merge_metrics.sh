#!/usr/bin/env bash
# After inference on EgoExo4D **test**: merge per-take preds → challenge submit JSON → metrics.
# Direction is **ego → exo** throughout: inference must use default --swap False; merge_results uses
# task name "ego-exo" (swap=False, the default — do not pass --swap True).
#
# Usage (from repo root):
#   OUT_INF=/path/to/output/egoexo_test_dam_YYYYMMDD  bash tools/run_egoexo_test_merge_metrics.sh
#
# Optional:
#   EEO_DATA_ROOT=/path/to/EgoExo4D
#   SUBMIT_JSON=/path/to/results.json   (default: output next to OUT_INF)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${REPO_ROOT}"

EEO_DATA_ROOT="${EEO_DATA_ROOT:-/home/UNT/bf0191/Documents/bingfan_projects/dataset/EgoExo4D}"
RELATIONS_TEST="${EEO_DATA_ROOT}/annotations/relations_test.json"
GT_JSON="${EEO_DATA_ROOT}/gt_results/egoexo_test_gt.json"
PYTHON="${PYTHON:-/opt/miniconda3/bin/python3}"

OUT_INF="${OUT_INF:-}"
if [[ -z "${OUT_INF}" || ! -d "${OUT_INF}" ]]; then
  echo "Set OUT_INF to your inference output directory (containing take UUID subfolders)."
  exit 1
fi

SUBMIT_JSON="${SUBMIT_JSON:-${REPO_ROOT}/output/egoexo_test_submit_$(basename "${OUT_INF}").json}"

[[ -f "${RELATIONS_TEST}" ]] || { echo "Missing ${RELATIONS_TEST}"; exit 1; }
[[ -f "${GT_JSON}" ]] || { echo "Missing ${GT_JSON}"; exit 1; }

echo "=== merge_pred.py (per-take annotations.json under pred dir) ==="
"${PYTHON}" "${REPO_ROOT}/evaluation/merge_pred.py" \
  --input "${OUT_INF}" \
  --gt "${RELATIONS_TEST}" \
  --pred "${OUT_INF}"

echo ""
echo "=== merge_results.py — ego→exo (default swap=False → task 'ego-exo') ==="
"${PYTHON}" "${REPO_ROOT}/evaluation/merge_results.py" \
  --input_dir "${EEO_DATA_ROOT}" \
  --split test \
  --pred_dir "${OUT_INF}" \
  --json_path "${SUBMIT_JSON}"

echo ""
echo "=== compute_metrics.py ==="
cd "${REPO_ROOT}/evaluation"
"${PYTHON}" compute_metrics.py \
  --gt-file "${GT_JSON}" \
  --pred-file "${SUBMIT_JSON}"

cd "${REPO_ROOT}"
echo ""
echo "Done (ego→exo)."
echo "  Pred / merged intermediates: ${OUT_INF}"
echo "  Submitted-style JSON:         ${SUBMIT_JSON}"
echo "  Ground-truth ref:             ${GT_JSON}"

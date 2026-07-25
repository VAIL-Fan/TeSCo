#!/usr/bin/env bash
# End-to-end: inference (text-conditioned) → merge_pred → merge_results → GT JSON → PNGs.
#
#   export EEO_DATA_ROOT=/path/to/EgoExo4D   # parent of val/, annotations/
#   export EEO_SPLIT=val
#   bash tools/run_egoexo_viz_pipeline.sh
#
# Inference output lives under the repo at output/egoexo_viz_inference/ so merge_results
# can use --input_dir <REPO_ROOT> --split output/egoexo_viz_inference.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${REPO_ROOT}"

EEO_DATA_ROOT="${EEO_DATA_ROOT:-/home/UNT/bf0191/Documents/MLLM_EgoExo_Proposal/LM-EEC/data/EgoExo4D}"
EEO_SPLIT="${EEO_SPLIT:-val}"
IMAGE_DIR="${IMAGE_DIR:-${EEO_DATA_ROOT}/${EEO_SPLIT}}"
RELATIONS="${RELATIONS:-${EEO_DATA_ROOT}/annotations/relations_${EEO_SPLIT}.json}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/training/checkpoints_EgoExo/checkpoint.pt}"
# Default matches FiLM + text cross-attn EgoExo fine-tune (see sam2.1_hiera_b+_egoexo_text.yaml)
SAM2_CFG="${SAM2_CFG:-configs/sam2.1/sam2.1_hiera_b+_egoexo_text.yaml}"
CONDA_ENV="${CONDA_ENV:-eel_apr1}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_INF="${OUT_INF:-${REPO_ROOT}/output/egoexo_viz_inference_${RUN_TAG}}"
MERGED_JSON="${MERGED_JSON:-${REPO_ROOT}/output/egoexo_viz_merged_${RUN_TAG}.json}"
MERGED_EGOEXO="${MERGED_EGOEXO:-${REPO_ROOT}/output/egoexo_viz_merged_${RUN_TAG}_ego-exo.json}"
GT_VIZ_JSON="${GT_VIZ_JSON:-${REPO_ROOT}/output/egoexo_viz_gt_${RUN_TAG}.json}"
VIZ_DIR="${VIZ_DIR:-${REPO_ROOT}/output/egoexo_viz_pngs_${RUN_TAG}}"
NUM_VIDEOS="${NUM_VIDEOS:-10}"
SEED="${SEED:-42}"
VIDEO_LIST="${VIDEO_LIST:-${REPO_ROOT}/output/viz_video_list_${RUN_TAG}.txt}"

MERGE_SPLIT_REL="output/$(basename "${OUT_INF}")"

mkdir -p "${REPO_ROOT}/output"
mkdir -p "${OUT_INF}"

if [[ ! -f "${RELATIONS}" ]]; then
  echo "ERROR: relations file not found: ${RELATIONS}"
  echo "Set EEO_DATA_ROOT or RELATIONS to your EgoExo4D annotations."
  exit 1
fi
if [[ ! -d "${IMAGE_DIR}" ]]; then
  echo "ERROR: image dir not found: ${IMAGE_DIR}"
  exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "ERROR: checkpoint not found: ${CHECKPOINT}"
  exit 1
fi

echo "Selecting ${NUM_VIDEOS} takes from ${IMAGE_DIR} ..."
python3 - <<PY
import os, random
root = "${IMAGE_DIR}"
n = int("${NUM_VIDEOS}")
seed = int("${SEED}")
random.seed(seed)
cands = sorted(
    d for d in os.listdir(root)
    if os.path.isdir(os.path.join(root, d)) and os.path.isfile(os.path.join(root, d, "annotation.json"))
)
random.shuffle(cands)
sel = cands[:n]
out = "${VIDEO_LIST}"
with open(out, "w") as f:
    for u in sel:
        f.write(u + "\n")
print(f"Wrote {out} ({len(sel)} ids)")
if len(sel) < n:
    print(f"WARNING: only {len(sel)} takes available (requested {n}).")
PY

echo "=== 1/5 Inference (ego→exo, --swap True; text from object_descriptions.json) ==="
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
conda run -n "${CONDA_ENV}" --no-capture-output python tools/correspondence_inference.py \
  --sam2_cfg "${SAM2_CFG}" \
  --sam2_checkpoint "${CHECKPOINT}" \
  --base_video_dir "${IMAGE_DIR}" \
  --video_list_file "${VIDEO_LIST}" \
  --output_mask_dir "${OUT_INF}" \
  --score_thresh 0.0 \
  --swap True

echo "=== 2/5 merge_pred.py ==="
conda run -n "${CONDA_ENV}" --no-capture-output python evaluation/merge_pred.py \
  --input "${OUT_INF}" \
  --gt "${RELATIONS}" \
  --pred "${OUT_INF}"

echo "=== 3/5 merge_results.py ==="
conda run -n "${CONDA_ENV}" --no-capture-output python evaluation/merge_results.py \
  --input_dir "${REPO_ROOT}" \
  --split "${MERGE_SPLIT_REL}" \
  --pred_dir "${OUT_INF}" \
  --swap True \
  --json_path "${MERGED_JSON}"

echo "=== 4/5 Rename exo-ego → ego-exo + build GT viz JSON ==="
python3 - <<PY
import json
inp, out = "${MERGED_JSON}", "${MERGED_EGOEXO}"
with open(inp) as f:
    d = json.load(f)
if "exo-ego" in d and "ego-exo" not in d:
    d["ego-exo"] = d.pop("exo-ego")
with open(out, "w") as f:
    json.dump(d, f)
print("Wrote", out, "top keys:", list(d.keys()))
PY

conda run -n "${CONDA_ENV}" --no-capture-output python tools/build_gt_viz_json.py \
  --image_dir "${IMAGE_DIR}" \
  --output_json "${GT_VIZ_JSON}" \
  --video_list_file "${VIDEO_LIST}"

echo "=== 5/5 visualize_predictions.py ==="
mkdir -p "${VIZ_DIR}"
conda run -n "${CONDA_ENV}" --no-capture-output python tools/visualize_predictions.py \
  --gt_file "${GT_VIZ_JSON}" \
  --pred_file "${MERGED_EGOEXO}" \
  --image_dir "${IMAGE_DIR}" \
  --output_dir "${VIZ_DIR}" \
  --num_videos "${NUM_VIDEOS}" \
  --seed "${SEED}"

echo "Done."
echo "  PNGs:        ${VIZ_DIR}"
echo "  Pred JSON:   ${MERGED_EGOEXO}"
echo "  GT viz JSON: ${GT_VIZ_JSON}"
echo "  Raw infer:   ${OUT_INF}"

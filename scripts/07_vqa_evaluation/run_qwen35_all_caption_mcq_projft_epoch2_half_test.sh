#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_NAME="qwen35_all_caption_mcq_projft_epoch2_half_test"
RUN_DIR="${ROOT}/results/${RUN_NAME}"

mkdir -p "${RUN_DIR}"
exec > >(tee -a "${RUN_DIR}/eval.log") 2>&1

# Keep the persistent training run on physical GPU 0 uninterrupted.
export CUDA_VISIBLE_DEVICES=1
export KIDNEY_VLM_VQA_GENERATION_CONFIG="07_vqa_evaluation/qwen35_all_caption_mcq_projft_epoch2_half_test.yaml"
cd "${ROOT}"

"${ROOT}/.venv/bin/python" scripts/07_vqa_evaluation/generate_vqa_predictions.py
"${ROOT}/.venv/bin/python" scripts/07_vqa_evaluation/score_vqa_predictions.py \
  "vqa_evaluation.run.name=${RUN_NAME}" \
  "vqa_evaluation.overlap_scoring.enabled=false"

echo "Completed epoch-2 half-test evaluation: results/${RUN_NAME}/metrics.json"

#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_NAME="qwen35_9b_native_base_same_questions"
RUN_DIR="${ROOT}/results/${RUN_NAME}"

mkdir -p "${RUN_DIR}"
exec > >(tee -a "${RUN_DIR}/eval.log") 2>&1

export KIDNEY_VLM_VQA_GENERATION_CONFIG="07_vqa_evaluation/qwen35_base_same_questions.yaml"
cd "${ROOT}"

"${ROOT}/.venv/bin/python" scripts/07_vqa_evaluation/generate_vqa_predictions.py
"${ROOT}/.venv/bin/python" scripts/07_vqa_evaluation/score_vqa_predictions.py \
  "vqa_evaluation.run.name=${RUN_NAME}"

echo "Completed native Qwen3.5-9B base evaluation: results/${RUN_NAME}/metrics.json"

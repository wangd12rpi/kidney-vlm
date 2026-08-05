#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_NAME="qwen35_prefix_grpo_newcot_same_questions"
RUN_DIR="${ROOT}/results/${RUN_NAME}"

mkdir -p "${RUN_DIR}"
exec > >(tee -a "${RUN_DIR}/sft_control.log") 2>&1

export KIDNEY_VLM_VQA_GENERATION_CONFIG="07_vqa_evaluation/qwen35_prefix_grpo_newcot_same_questions.yaml"
cd "${ROOT}"

"${ROOT}/.venv/bin/python" scripts/07_vqa_evaluation/generate_vqa_predictions.py \
  vqa_evaluation.models.oncovlm_qwen35_grpo_newcot_v7_final.enabled=false
"${ROOT}/.venv/bin/python" scripts/07_vqa_evaluation/score_vqa_predictions.py \
  "vqa_evaluation.run.name=${RUN_NAME}"

echo "Completed exact-prompt SFT control: results/${RUN_NAME}/metrics.json"

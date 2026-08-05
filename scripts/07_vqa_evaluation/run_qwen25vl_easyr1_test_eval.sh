#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
EASYR1_ROOT="${ROOT}/external/EasyR1"
PYTHON="${EASYR1_ROOT}/.venv/bin/python"
CHECKPOINT_ROOT="${EASYR1_ROOT}/checkpoints/oncovlm/easyr1_pathology_mcq_qwen25vl3b_lora64_full"
TRACKER="${CHECKPOINT_ROOT}/checkpoint_tracker.json"
FINAL_ADAPTER="${CHECKPOINT_ROOT}/global_step_485/actor/lora_adapter"
RUN_NAME="easyr1_qwen25vl3b_pathology_mcq_test"
RUN_DIR="${ROOT}/results/${RUN_NAME}"

mkdir -p "${RUN_DIR}"
exec > >(tee -a "${RUN_DIR}/eval.log") 2>&1
exec 9>"${RUN_DIR}/eval.lock"
if ! flock -n 9; then
  echo "The paired Qwen2.5-VL evaluation queue is already running."
  exit 0
fi

while pgrep -f '[r]un_pathology_mcq_long.sh' >/dev/null; do
  echo "EasyR1 training/evaluation wrapper is still running; checking again in 30 seconds."
  sleep 30
done

if [[ ! -f "${TRACKER}" ]]; then
  echo "Missing EasyR1 checkpoint tracker: ${TRACKER}" >&2
  exit 1
fi

LAST_STEP="$(${PYTHON} -c 'import json, sys; print(json.load(open(sys.argv[1]))["last_global_step"])' "${TRACKER}")"
if [[ "${LAST_STEP}" != "485" || ! -f "${FINAL_ADAPTER}/adapter_model.safetensors" ]]; then
  echo "Expected completed step-485 adapter, found tracker step ${LAST_STEP}." >&2
  exit 1
fi

export HF_HOME="${EASYR1_ROOT}/.cache/huggingface"
export KIDNEY_VLM_VQA_GENERATION_CONFIG="07_vqa_evaluation/qwen25vl_easyr1_test.yaml"
cd "${ROOT}"

"${PYTHON}" scripts/07_vqa_evaluation/generate_vqa_predictions.py
"${PYTHON}" scripts/07_vqa_evaluation/score_vqa_predictions.py \
  "vqa_evaluation.run.name=${RUN_NAME}"

echo "Completed paired Qwen2.5-VL base versus GRPO evaluation: results/${RUN_NAME}/metrics.json"

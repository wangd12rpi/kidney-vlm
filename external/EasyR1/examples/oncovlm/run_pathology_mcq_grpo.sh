#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EASYR1_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${EASYR1_ROOT}/.venv/bin/python"
CONFIG="${SCRIPT_DIR}/qwen2_5_vl_3b_pathology_mcq_grpo.yaml"
DATA_DIR="${EASYR1_ROOT}/data/oncovlm_pathology_mcq"
DATA_DIR_2ROI="${EASYR1_ROOT}/data/oncovlm_pathology_mcq_2roi"

PROFILE="${1:-full}"
if [[ $# -gt 0 ]]; then
  shift
fi

if [[ ! -x "${PYTHON}" ]]; then
  echo "EasyR1 environment is missing: ${PYTHON}" >&2
  echo "Create external/EasyR1/.venv and install EasyR1 before launching training." >&2
  exit 1
fi

TUNING_MODE="full-bf16"
DATA_PROFILE="${PROFILE}"
if [[ "${PROFILE}" == lora-* ]]; then
  TUNING_MODE="lora64"
  DATA_PROFILE="${PROFILE#lora-}"
fi

case "${DATA_PROFILE}" in
  smoke)
    TRAIN_FILE="${DATA_DIR}/smoke32.parquet"
    VAL_FILE="${DATA_DIR}/val_smoke4.parquet"
    PROFILE_OVERRIDES=(
      data.rollout_batch_size=4
      data.val_batch_size=4
      worker.actor.global_batch_size=4
      trainer.max_steps=1
      trainer.val_freq=-1
      trainer.val_before_train=false
      trainer.val_generations_to_log=0
      trainer.save_freq=1
      trainer.save_limit=1
    )
    ;;
  pilot)
    TRAIN_FILE="${DATA_DIR}/pilot128.parquet"
    VAL_FILE="${DATA_DIR}/val_monitor64.parquet"
    PROFILE_OVERRIDES=(
      trainer.max_steps=8
    )
    ;;
  full)
    TRAIN_FILE="${DATA_DIR}/train_batch8.parquet"
    VAL_FILE="${DATA_DIR}/val_monitor64.parquet"
    PROFILE_OVERRIDES=(
      trainer.max_steps=null
    )
    ;;
  eval)
    if [[ "${TUNING_MODE}" != "lora64" ]]; then
      echo "The evaluation profile is available for the selected lora64 fallback only." >&2
      exit 2
    fi
    TRAIN_FILE="${DATA_DIR}/train_batch8.parquet"
    VAL_FILE="${DATA_DIR}/val.parquet"
    FULL_CHECKPOINT_ROOT="${EASYR1_ROOT}/checkpoints/oncovlm/easyr1_pathology_mcq_qwen25vl3b_lora64_full"
    TRACKER_FILE="${FULL_CHECKPOINT_ROOT}/checkpoint_tracker.json"
    if [[ ! -f "${TRACKER_FILE}" ]]; then
      echo "Full-run checkpoint tracker does not exist: ${TRACKER_FILE}" >&2
      exit 1
    fi
    LAST_STEP="$("${PYTHON}" -c 'import json, sys; print(json.load(open(sys.argv[1]))["last_global_step"])' "${TRACKER_FILE}")"
    LOAD_CHECKPOINT="${FULL_CHECKPOINT_ROOT}/global_step_${LAST_STEP}"
    if [[ ! -d "${LOAD_CHECKPOINT}" ]]; then
      echo "Full-run checkpoint does not exist: ${LOAD_CHECKPOINT}" >&2
      exit 1
    fi
    PROFILE_OVERRIDES=(
      trainer.max_steps=null
      trainer.val_only=true
      trainer.val_before_train=true
      trainer.val_freq=-1
      trainer.save_freq=-1
      trainer.find_last_checkpoint=false
      trainer.load_checkpoint_path="${LOAD_CHECKPOINT}"
    )
    ;;
  *)
    echo "Usage: $0 {smoke|pilot|full|lora-smoke|lora-pilot|lora-full|lora-eval} [OmegaConf overrides ...]" >&2
    exit 2
    ;;
esac

METHOD_OVERRIDES=()
if [[ "${TUNING_MODE}" == "lora64" ]]; then
  METHOD_OVERRIDES=(
    worker.actor.model.lora.rank=64
    worker.actor.model.lora.alpha=64
    worker.actor.optim.lr=1.0e-5
    worker.actor.optim.strategy=adamw
    worker.actor.fsdp.torch_dtype=null
    worker.rollout.gpu_memory_utilization=0.40
    worker.rollout.enforce_eager=true
  )
  if [[ "${DATA_PROFILE}" == "smoke" ]]; then
    TRAIN_FILE="${DATA_DIR_2ROI}/smoke32.parquet"
    VAL_FILE="${DATA_DIR_2ROI}/val_smoke4.parquet"
    METHOD_OVERRIDES+=(worker.rollout.limit_images=2)
  fi
fi

EXPERIMENT_NAME="easyr1_pathology_mcq_qwen25vl3b_${TUNING_MODE//-/_}_${DATA_PROFILE}"

for path in "${CONFIG}" "${TRAIN_FILE}" "${VAL_FILE}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required file does not exist: ${path}" >&2
    exit 1
  fi
done

export PYTHONUNBUFFERED=1
export HF_HOME="${EASYR1_ROOT}/.cache/huggingface"
export UV_CACHE_DIR="${EASYR1_ROOT}/.cache/uv"
export WANDB_ENTITY="d0nnw0n9-rensselaer-polytechnic-institute"
export WANDB_PROJECT="oncovlm"
export WANDB_RUN_GROUP="easyr1_pathology_mcq"
export WANDB_TAGS="EasyR1,external-baseline,GRPO,pathology,Qwen2.5-VL-3B,${TUNING_MODE},${DATA_PROFILE}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-easyr1-pathology-mcq-qwen25vl3b-${TUNING_MODE}-${DATA_PROFILE}}"
export WANDB_NAME="${WANDB_NAME:-${EXPERIMENT_NAME}}"
export WANDB_RESUME="allow"
export WANDB_DIR="${EASYR1_ROOT}/wandb"

mkdir -p "${HF_HOME}" "${UV_CACHE_DIR}" "${WANDB_DIR}"
cd "${EASYR1_ROOT}"

exec "${PYTHON}" -m verl.trainer.main \
  config="${CONFIG}" \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  "${PROFILE_OVERRIDES[@]}" \
  "${METHOD_OVERRIDES[@]}" \
  "$@"

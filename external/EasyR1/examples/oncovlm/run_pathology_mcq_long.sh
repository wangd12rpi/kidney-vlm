#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/run_pathology_mcq_grpo.sh" lora-full "$@"
WANDB_RUN_ID="easyr1-pathology-mcq-qwen25vl3b-lora64-eval" \
WANDB_NAME="easyr1_pathology_mcq_qwen25vl3b_lora64_eval" \
  "${SCRIPT_DIR}/run_pathology_mcq_grpo.sh" lora-eval

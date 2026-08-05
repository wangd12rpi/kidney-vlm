#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Expose only physical GPU 0. Inside the training process it remains cuda:0.
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True

exec "${PROJECT_ROOT}/.venv/bin/python" \
  "${PROJECT_ROOT}/scripts/06_vqa_train/train_vqa_lora.py" \
  method=sft \
  profile=projector_ft \
  "$@"

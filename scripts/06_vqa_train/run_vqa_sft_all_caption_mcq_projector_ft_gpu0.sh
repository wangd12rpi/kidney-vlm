#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_NAME="${1:-qwen35_all_caption_mcq_projft_cont_4ep_b2_ga8_20260729}"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd "${ROOT}"
exec "${ROOT}/.venv/bin/python" \
  scripts/06_vqa_train/train_vqa_lora.py \
  method=sft \
  profile=projector_ft_all_caption_mcq \
  "vqa_train.run_name=${RUN_NAME}"

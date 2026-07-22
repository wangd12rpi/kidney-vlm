# OncoVLM pathology MCQ GRPO

This experiment keeps EasyR1's trainer unchanged. It adds only dataset preparation,
an exact-option reward, and a one-A100 configuration.

The vendored source is from `https://github.com/hiyouga/EasyR1` at commit
`07cae10`.

## Environment

From the kidney-vlm repository root:

```bash
UV_CACHE_DIR=external/EasyR1/.cache/uv uv venv external/EasyR1/.venv --python /usr/bin/python3.12
UV_CACHE_DIR=external/EasyR1/.cache/uv uv pip install --python external/EasyR1/.venv/bin/python torch==2.8.0 setuptools wheel packaging ninja
UV_CACHE_DIR=external/EasyR1/.cache/uv uv pip install --python external/EasyR1/.venv/bin/python 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'
UV_CACHE_DIR=external/EasyR1/.cache/uv uv pip install --python external/EasyR1/.venv/bin/python -e external/EasyR1 torch==2.8.0 vllm==0.11.0
```

The explicit flash-attn wheel is required on this host because CUDA's `nvcc` compiler
is not installed.

## Data and training

```bash
external/EasyR1/.venv/bin/python external/EasyR1/examples/oncovlm/prepare_pathology_mcq.py
external/EasyR1/.venv/bin/python external/EasyR1/examples/oncovlm/prepare_pathology_mcq.py --max-images 2 --output-dir external/EasyR1/data/oncovlm_pathology_mcq_2roi
external/EasyR1/examples/oncovlm/run_pathology_mcq_grpo.sh smoke
external/EasyR1/examples/oncovlm/run_pathology_mcq_grpo.sh pilot
external/EasyR1/examples/oncovlm/run_pathology_mcq_grpo.sh full
external/EasyR1/examples/oncovlm/run_pathology_mcq_grpo.sh lora-smoke
external/EasyR1/examples/oncovlm/run_pathology_mcq_grpo.sh lora-pilot
external/EasyR1/examples/oncovlm/run_pathology_mcq_grpo.sh lora-full
external/EasyR1/examples/oncovlm/run_pathology_mcq_grpo.sh lora-eval
external/EasyR1/examples/oncovlm/run_pathology_mcq_long.sh
```

The launcher accepts additional OmegaConf overrides after the profile name. Generated
data, model caches, W&B files, and checkpoints stay ignored. Set a distinct
`WANDB_RUN_ID` when an override changes the method; reuse the default stable ID only
when resuming the same recipe.

The `lora-eval` profile loads the latest checkpoint from the LoRA full run and evaluates
all 322 validation rows in a separate resumable W&B run. The training profiles monitor
the fixed 64-row validation subset and log all of its generations for qualitative review.
`run_pathology_mcq_long.sh` runs the selected LoRA fallback and then launches that full
validation automatically.

EasyR1's upstream training dataloader drops incomplete batches. The canonical
`train.parquet` therefore remains exactly 3,873 unique questions, while the full profile
uses `train_batch8.parquet`: all 3,873 questions plus seven deterministic repeats with
distinct padding IDs. This preserves every source row in one 485-batch epoch without a
trainer change.

On the available 40 GB A100, the first full-BF16 optimizer update exhausted memory after
the prescribed reductions to 0.40 vLLM utilization, rollout/global batch four, and two
ROIs. The language-only rank-64 LoRA fallback completed its smoke step and checkpoint
resume, so it is the selected long-run profile. The primary full-BF16 profiles remain in
the launcher as the projector-inclusive diagnostic.

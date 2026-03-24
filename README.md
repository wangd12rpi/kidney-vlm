# Kidney VLM 

This repository provides a clean starting point for kidney multimodal research with:
- Hydra hierarchical configs (`conf/`)
- Parquet source-of-truth registry (`pandas`)
- Hugging Face Datasets ingestion for training
- Pathology TRIDENT integration scaffold
- Three-stage training structure: PMC-OA projector pretraining, TCGA caption-stage projector tuning, then Qwen LoRA instruction tuning

## Layout
- `conf/`: runtime, data, and model configs.
- `scripts/`: source-specific dataset build scripts and training/extraction entry scripts.
- `src/`: reusable library modules.

## Script Naming Convention
- Runnable scripts always use a leading verb (`build`, `extract`, `run`, `train`).
- Ordered workflows use numeric prefixes (`01_`, `02_`, `03_`).
- Non-runnable templates live under `scripts/templates/` and do not use runnable naming.

## Quickstart
```bash
uv sync
uv run pytest
```

All runnable scripts set `project.root_dir` to the repository root detected from the nearest `.git` directory.

## Add New Source 
This is the main way to extend the project.

1. Pull the latest unified registry from HF Hub:
```bash
uv run python scripts/data/pull_registry_from_hf.py --repo-id wangd12/kidney_vlm
```
2. Create a new source builder from the template:
```bash
cp scripts/templates/source_template.py scripts/data/01_build_<source>_source.py
```
3. Add source config:
- `conf/data/sources/<source>.yaml`
4. Implement source-specific pull/harmonization in `01_build_<source>_source.py`.
5. Run your source builder:
```bash
uv run python scripts/data/01_build_<source>_source.py
```
6. Verify registry status:
```bash
uv run python scripts/data/print_registry_status.py --source <source> --samples-per-source 2
```
7. Push updated unified registry:
```bash
uv run python scripts/data/push_registry_to_hf.py --repo-id wangd12/kidney_vlm --commit-message "update <source> rows"
```

Note that:
- All path fields in registry must be relative to project root (examples: `data/raw/...`, `data/features/...`).
- Unknown labels/ground truth should be `null`, not placeholder `false`.
- Source builders should use source-slice replacement so only rows for that `source` are replaced.
- Current HF pull script is replace-only (no automatic merge logic).
- Always pull latest HF registry first, then the your source builder, then push.

## Data Construction (Per Source)
Run one source at a time with its builder script:
```bash
uv run python scripts/data/01_build_<source>_source.py
```

Built-in TCGA source:
- Script: `scripts/data/01_build_tcga_source.py`
- Config: `conf/data/sources/tcga.yaml`

Built-in PMC-OA CT-only source:
- Script: `scripts/data/01_build_pmc_oa_source.py`
- Config: `conf/data/sources/pmc_oa.yaml`
- Output registry: `data/registry/pmc_oa_ct.parquet`
- Note: this exception dataset does not write image paths into `data/registry/unified.parquet`; it keeps the CT-only image-caption rows in its own registry parquet instead.
- Feature extraction for this source is also separate from the main unified pipeline and uses `scripts/model/01_extract_pmc_oa_features.py`.

## Parquet Files
Main source registries:
- `data/registry/unified.parquet`
  - Main shared registry for the TCGA and future main-pipeline datasets.
  - Source builders replace only their own `source` slice in this file.
  - Feature extraction writes paths such as `pathology_tile_embedding_paths`, `pathology_slide_embedding_paths`, and future radiology feature paths back into this registry.
- `data/registry/pmc_oa_ct.parquet`
  - Separate exception registry for the PMC-OA CT caption pretraining dataset.
  - This file is intentionally not merged into `data/registry/unified.parquet`.
  - MedSigLIP feature extraction writes `radiology_embedding_paths` back into this registry.

Derived training parquets:
- `data/qa/projector_train_qa.parquet`
  - Specialized caption-training parquet derived from the TCGA rows in `data/registry/unified.parquet`.
  - Used for the caption-stage projector training.
- `data/qa/vlm_instruct_qa.parquet`
  - Specialized instruction-tuning parquet derived from `data/qa/projector_train_qa.parquet`.
  - Used for the final VQA/MCQ-style instruction fine-tuning stage.

## Data Preparation Workflow
Target order:
1. Build the source registry.
2. Extract image features and write feature paths back into the relevant registry parquet.
3. Run segmentation if needed.

Notes:
- Segmentation is optional and is not required for the current pretraining datasets.
- Feature extraction is dataset-specific:
  - PMC-OA radiology pretraining uses MedSigLIP features.
  - TCGA pathology training uses TRIDENT pathology features.

Concrete data-prep commands currently in the repo:

TCGA / main-pipeline data:
```bash
uv run python scripts/data/01_build_tcga_source.py
uv run python scripts/model/01_extract_pathology_features.py embedding_extraction/pathology=trident_slide
# optional / scaffold
uv run python scripts/model/02_run_segmentation.py
```

PMC-OA radiology pretraining data:
```bash
uv run python scripts/data/01_build_pmc_oa_source.py
uv run python scripts/model/01_extract_pmc_oa_features.py
```

## Training Workflow
Target order:
1. Pretrain the radiology projector on PMC-OA.
2. Pretrain the pathology projector on its pathology pretraining dataset.
   - The order of steps 1 and 2 does not matter.
3. Train both projectors jointly on the specialized caption dataset.
4. Run instruction fine-tuning on the specialized VQA / MCQ dataset.

Current repo mapping:
- `conf/projector_train/default.yaml`
  - PMC-OA radiology-projector pretraining on CT captions.
- `conf/projector_train/tcga_caption.yaml`
  - Caption-stage projector tuning on the specialized TCGA caption parquet.
- `conf/vlm_train/qwen_lora.yaml`
  - Final Qwen LoRA instruction-tuning scaffold.

Concrete training commands currently available:

PMC-OA radiology-projector pretraining:
```bash
uv run python scripts/model/03_train_projectors.py
```

TCGA caption-stage projector tuning:
```bash
uv run python scripts/qa/01_gen_projector_train_qa.py qa_genereation=tcga_caption
uv run python scripts/model/03_train_projectors.py projector_train=tcga_caption
```

Qwen LoRA instruction-tuning stage:
```bash
uv run python scripts/qa/01_gen_projector_train_qa.py qa_genereation=tcga_instruct
uv run python scripts/model/04_train_vlm.py
```

Current implementation status:
- The PMC-OA radiology-projector pretraining path is implemented.
- The specialized TCGA caption dataset generation path is implemented.
- The specialized instruct dataset generation path is implemented and is built from the caption parquet.
- A dedicated pathology-only projector pretraining pipeline is not yet a separate first-class entrypoint in the repo.
- A true joint two-projector caption trainer is not yet wired end to end.
- The final Qwen LoRA multimodal trainer is still a scaffold entrypoint.

## Side Dataset-Creation Pipeline
This is the separate dataset-construction path used to create the specialized supervision sets:

1. Create the specialized caption dataset from TCGA:
```bash
uv run python scripts/qa/01_gen_projector_train_qa.py qa_genereation=tcga_caption
```

2. Create the specialized instruct dataset from the TCGA caption dataset:
```bash
uv run python scripts/qa/01_gen_projector_train_qa.py qa_genereation=tcga_instruct
```

Config groups used by these workflows:
- `conf/qa_genereation/`
- `conf/embedding_extraction/`
- `conf/projector_train/`
- `conf/vlm_train/`

Useful configs:
- `conf/embedding_extraction/pathology/trident.yaml`
  - Main pathology extractor preset.
- `conf/embedding_extraction/pathology/trident_slide.yaml`
  - Pathology extraction preset that writes slide embeddings needed by the caption and instruct stages.
- `conf/embedding_extraction/radiology/medsiglip_pmc_oa.yaml`
  - PMC-OA MedSigLIP radiology extractor preset.

Multi-image support:
- Each registry row supports list-valued path columns.
- Collation keeps `pathology_*` and `radiology_*` paths as lists per sample.
- Projectors support tensor inputs shaped `[batch, n_images, dim]` for each modality.

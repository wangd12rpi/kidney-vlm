# Kidney VLM 

This repository provides a clean starting point for kidney multimodal research with:
- Hydra hierarchical configs (`conf/`)
- Parquet source-of-truth registry (`pandas`)
- Hugging Face Datasets ingestion for training
- Pathology TRIDENT integration scaffold
- Two-stage training scaffold: projector training then VLM training

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
- Script: `scripts/data/01_upsert_tcga_registry_rows.py`
- Config: `conf/data/sources/tcga.yaml`

## Model Pipeline Scripts (Ordered)
```bash
uv run python scripts/embeding_extraction/01_extract_pathology_features.py
uv run python scripts/embeding_extraction/02_extract_pathology_features_space_saver.py
uv run python scripts/path_proj_train/02_gen_path_case_captions.py
uv run python scripts/path_proj_train/03_build_path_proj_train_qa.py
uv run python scripts/path_proj_train/04_train_path_projectors.py
uv run python scripts/segmentation/01_run_segmentation.py
uv run python scripts/vlm_train/01_train_vlm.py
```

Stage-scoped configs now live under:
- `conf/qa_genereation/`
- `conf/embeding_extraction/`
- `conf/path_proj_train/`
- `conf/vlm_train/`

Multi-image support:
- Each registry row already supports list-valued path columns.
- Collation keeps `pathology_*` and `radiology_*` paths as lists per sample.
- Projectors support tensor inputs shaped `[batch, n_images, dim]` for each modality.

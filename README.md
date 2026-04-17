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
2. Create a new source builder:
- `scripts/data/01_build_<source>_source.py`
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

Mutation panel provenance for TCGA:
- The current shared pan-cancer mutation panel in `conf/data/sources/tcga.yaml` is derived from the official GDC PanCanAtlas driver publication: https://gdc.cancer.gov/about-data/publications/pancan-driver
- Source supplement tables used:
  `2020plus.tar.gz` https://api.gdc.cancer.gov/data/5fa7121d-2f4b-40e5-822d-7b5c037a7641
  `CompositeDriver.tar.gz` https://api.gdc.cancer.gov/data/bee64811-952a-4fac-b011-9cd28c0bd8d1
- Selection rule:
  for each TCGA cancer type, keep genes with `q <= 0.05` in both official per-cancer tables; when that overlap is empty or the second table is unavailable, fall back to `2020plus` genes with `q <= 0.01`; the final registry panel is the union across cancer types.
- The reusable per-cancer driver-gene mapping is saved at `src/kidney_vlm/data/sources/tcga_project_driver_genes.json`.

## Model Pipeline Scripts (Ordered)
```bash
uv run python scripts/01_pathology_proj/02_gen_path_case_captions.py
uv run python scripts/01_pathology_proj/03_build_path_proj_train_qa.py
uv run python scripts/01_pathology_proj/04_train_path_projectors.py
uv run python scripts/01_pathology_segmentation/01_run_pathology_segmentation.py
uv run python scripts/02_radiology_features/02_prepare_radiology_series_manifest.py
uv run python scripts/02_radiology_features/03_extract_radiology_pngs.py
uv run python scripts/02_radiology_features/04_extract_radiology_features.py
uv run python scripts/02_radiology_segmentation/05_extract_radiology_segmentation.py
uv run python scripts/03_dnam_proj/02_gen_dnam_case_captions.py
uv run python scripts/03_dnam_proj/03_build_dnam_proj_train_qa.py
uv run python scripts/03_dnam_proj/04_train_dnam_projectors.py
uv run python scripts/vlm_train/01_train_vlm.py
```

Stage-scoped configs now live under:
- `conf/01_pathology_proj/`
- `conf/01_pathology_segmentation/`
- `conf/02_radiology_features/`
- `conf/02_radiology_segmentation/`
- `conf/03_dnam_proj/`
- `conf/vlm_train/`

Multi-image support:
- Each registry row already supports list-valued path columns.
- Collation keeps `pathology_*` and `radiology_*` paths as lists per sample.
- Projectors support tensor inputs shaped `[batch, n_images, dim]` for each modality.

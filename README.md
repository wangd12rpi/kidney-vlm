# Kidney VLM Research Scaffold

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
- `tests/`: smoke and contract tests.

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

## Data Construction (Simple Per-Source Scripts)
Run one source at a time.

```bash
uv run python scripts/data/01_build_tcga_source.py
```

TCGA project selection is config-driven in:
- `conf/data/sources/tcga.yaml`
- Default projects: `TCGA-KIRC`, `TCGA-KIRP`, `TCGA-KICH`

`01_build_tcga_source.py`:
1. Loads Hydra base config + source config.
2. Pulls case metadata + clinical fields from GDC (including survival/task labels and extended pathology staging fields).
3. Pulls pathology slide metadata and PDF report metadata from GDC.
4. Pulls radiology study + series metadata from TCIA and joins by TCGA patient ID.
5. Pulls targeted GDC mutation calls for common kidney genes and emits per-case mutation labels/features.
6. Optionally downloads payloads when `data.source.download.enabled=true`:
   - pathology SVS files from GDC `/data/<file_id>`
   - TCIA radiology series zip files from `getImage`
   - GDC clinical/pathology PDF reports
7. Rebuilds the TCGA slice in `data/registry/unified.parquet` and emits a manifest.

Split policy for TCGA default config:
- `train=0.9`
- `test=0.1`

## Add a New Source
1. Copy `scripts/templates/source_template.py` to `scripts/data/01_build_<source>_source.py`.
2. Add `conf/data/sources/<source>.yaml`.
3. Implement source-specific pull + normalization logic.

## Model Pipeline Scripts (Ordered)
```bash
uv run python scripts/model/01_extract_pathology_features.py
uv run python scripts/model/02_run_segmentation.py
uv run python scripts/model/03_train_projectors.py
uv run python scripts/model/04_train_vlm.py
```

Stage-scoped configs now live under:
- `conf/qa_genereation/`
- `conf/embeding_extraction/`
- `conf/projector_train/`
- `conf/vlm_train/`

Multi-image support:
- Each registry row already supports list-valued path columns.
- Collation keeps `pathology_*` and `radiology_*` paths as lists per sample.
- Projectors support tensor inputs shaped `[batch, n_images, dim]` for each modality.

## TRIDENT
Expected local path:
- `${project.root_dir}/external/trident`

The adapter in `src/kidney_vlm/pathology/trident_adapter.py` adds this vendored directory to `sys.path` and imports TRIDENT modules if present.

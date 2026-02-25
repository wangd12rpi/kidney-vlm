# Scripts Guide

## Runnable Scripts
- `scripts/data/01_build_tcga_source.py`
  - Builds/refreshes TCGA source rows (GDC+TCIA metadata join) and replaces `source='tcga'` in unified registry.
  - Pulls TCIA study + series metadata by default even when payload downloads are disabled.
  - Pulls targeted GDC mutation metadata for a kidney-focused gene panel by default.
  - Downloads pathology SVS, TCIA radiology series zips, and GDC PDF reports when enabled.
  - Example metadata-only run:
    - `uv run python scripts/data/01_build_tcga_source.py data.source.download.enabled=false`
  - Example full download run:
    - `uv run python scripts/data/01_build_tcga_source.py data.source.download.enabled=true`
- `scripts/data/02_print_registry_status.py`
  - Prints per-source database status and checks local existence of referenced binaries in path columns (`*_path`, `*_paths`).
  - Reports missing reference counts and prints one sampled row per source by default.
  - Sample output is printed as `field: value` lines (one field per line), not table format.
  - Example:
    - `uv run python scripts/data/02_print_registry_status.py --samples-per-source 1 --missing-examples 5`
- `scripts/data/print_registry_debug.py`
  - Standalone parquet viewer for debugging (`no yaml` required).
  - Example:
    - `uv run python scripts/data/print_registry_debug.py --source tcga --rows 20 --head`
- `scripts/data/generate_stage1_projector_caption_examples.py`
  - Standalone caption-demo generator for stage-1 projector training.
  - Pulls paired TCGA+TCIA cases from parquet, fetches matching GDC PDF reports, and prints caption strings.
  - Example:
    - `uv run python scripts/data/generate_stage1_projector_caption_examples.py --source tcga --examples 3`
- `scripts/model/01_extract_pathology_features.py`
  - Checks TRIDENT integration and feature extraction scaffold entrypoint.
- `scripts/model/02_run_segmentation.py`
  - Segmentation scaffold entrypoint.
- `scripts/model/03_train_projectors.py`
  - Stage 1: projector training scaffold entrypoint.
- `scripts/model/04_train_vlm.py`
  - Stage 2: VLM training scaffold entrypoint.

## Non-Runnable Template
- `scripts/templates/source_template.py`
  - Copy this to create a new source script. Not intended to run directly.

## Naming Rules
- Runnable scripts must start with a verb.
- Ordered steps use `NN_` prefixes (`01_`, `02_`, ...).

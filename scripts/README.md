# Scripts Guide

## Runnable Scripts
- `scripts/data/01_upsert_tcga_registry_rows.py`
  - Fetches/refreshes TCGA source rows (GDC+TCIA metadata join) and upserts the `source='tcga'` slice in the unified registry.
  - Resolves all `TCGA-*` projects by default and lets you remove projects with `data.source.tcga.exclude_project_ids`.
  - Pulls TCIA study + series metadata by default even when payload downloads are disabled.
  - Pulls targeted GDC mutation metadata for a kidney-focused gene panel by default.
  - Builds TCGA genomics as text-first derived artifacts by default.
  - Downloads PanCancer source files into a temporary cache, writes per-patient `genomics_text` plus JSONL/text sidecars, then deletes the temporary cache when `data.source.tcga.genomics.cleanup_temp_cache=true`.
  - Does not persist RNA-seq matrices, methylation arrays, raw CN segments, or other high-dimensional genomics payloads on registry rows.
  - Downloads pathology SVS, TCIA radiology series zips, and GDC PDF reports when enabled.
  - Example metadata-only run:
    - `uv run python scripts/data/01_upsert_tcga_registry_rows.py data.source.download.enabled=false`
  - Example full download run:
    - `uv run python scripts/data/01_upsert_tcga_registry_rows.py data.source.download.enabled=true`
- `scripts/data/print_registry_status.py`
  - Prints per-source database status and checks local existence of referenced binaries in path columns (`*_path`, `*_paths`).
  - Reports missing reference counts and prints one sampled row per source by default.
  - Sample output is printed as `field: value` lines (one field per line), not table format.
  - Example:
    - `uv run python scripts/data/print_registry_status.py --samples-per-source 1 --missing-examples 5`
- `scripts/data/print_registry_debug.py`
  - Standalone parquet viewer for debugging (`no yaml` required).
  - Example:
    - `uv run python scripts/data/print_registry_debug.py --source tcga --rows 20 --head`
- `scripts/data/02_register_existing_pathology_features.py`
  - Registers already-extracted pathology patch feature files back into the unified registry.
- `scripts/embeding_extraction/01_extract_pathology_features.py`
  - Regular TRIDENT pathology feature extraction entrypoint for already-downloaded WSIs.
- `scripts/embeding_extraction/02_extract_pathology_features_space_saver.py`
  - Downloads only missing TCGA pathology SVS files into a temp directory, extracts embeddings, updates the registry, and deletes the raw SVS afterward.
- `scripts/path_proj_train/02_gen_path_case_captions.py`
  - Generates case-level pathology captions from registry metadata plus PDF pathology reports.
- `scripts/path_proj_train/03_build_path_proj_train_qa.py`
  - Builds slide-caption pathology projector training rows by matching available slide embeddings with case captions.
- `scripts/path_proj_train/04_train_path_projectors.py`
  - Stage 1: pathology projector training entrypoint.
- `scripts/segmentation/01_run_segmentation.py`
  - Segmentation scaffold entrypoint.
- `scripts/vlm_train/01_train_vlm.py`
  - Stage 2: VLM training scaffold entrypoint.

## Non-Runnable Template
- `scripts/templates/source_template.py`
  - Copy this to create a new source script. Not intended to run directly.

## Naming Rules
- Runnable scripts must start with a verb.
- Ordered steps use `NN_` prefixes (`01_`, `02_`, ...).

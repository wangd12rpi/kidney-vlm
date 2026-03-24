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
- `scripts/data/01_build_pmc_oa_source.py`
  - Builds a CT-only PMC-OA image-caption registry from `data/raw/pmc_oa/{train,valid,test}.jsonl`.
  - Uses caption-based CT matching and drops captions that also mention clearly different modalities (for example microscopy, MRI, PET, ultrasound, or x-ray) to stay conservative.
  - Keeps image paths repo-relative (for example `data/raw/pmc_oa/images/<file>.jpg`) instead of resolving the `images` symlink target.
  - Writes the CT slice to a separate registry parquet (`data/registry/pmc_oa_ct.parquet`) and intentionally does not modify `data/registry/unified.parquet`.
  - Example:
    - `uv run python scripts/data/01_build_pmc_oa_source.py`
  - Example small debug run:
    - `uv run python scripts/data/01_build_pmc_oa_source.py data.source.pmc_oa.max_rows_total=200`
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
- `scripts/qa/01_gen_projector_train_qa.py`
  - Generates derived supervision parquet files from the TCGA registry.
  - Default config builds the TCGA caption dataset used for stage-2 projector tuning.
  - The same entrypoint can also build the stage-3 instruct dataset with `qa_genereation=tcga_instruct`.
  - Example:
    - `python scripts/qa/01_gen_projector_train_qa.py qa_genereation=tcga_caption`
  - Example instruct-data run:
    - `python scripts/qa/01_gen_projector_train_qa.py qa_genereation=tcga_instruct`
- `scripts/model/01_extract_pathology_features.py`
  - Extracts TRIDENT pathology features for the main unified registry (`data/registry/unified.parquet`).
  - Reads `pathology_wsi_paths` and writes slide/tile outputs back into the unified registry.
  - This is the main-dataset pathology extractor and does not process the separate PMC-OA pretraining registry.
  - Use `embedding_extraction/pathology=trident_slide` when you need slide embeddings for the TCGA caption-stage projector run.
- `scripts/model/01_extract_pmc_oa_features.py`
  - Extracts MedSigLIP CT image features for the PMC-OA pretraining registry (`data/registry/pmc_oa_ct.parquet`).
  - Saves one HDF5 feature file per image under `data/raw/pmc_oa/features/`, mirroring the image filename/path, and writes repo-relative paths into `radiology_embedding_paths`.
  - Supports limiting how many new images to extract and skipping images whose canonical feature file already exists.
  - Example:
    - `uv run python scripts/model/01_extract_pmc_oa_features.py`
  - Example small resumable run:
    - `uv run python scripts/model/01_extract_pmc_oa_features.py embedding_extraction.radiology.max_images=128`
- `scripts/model/02_run_segmentation.py`
  - Segmentation scaffold entrypoint.
- `scripts/model/03_train_projectors.py`
  - Shared projector-training entrypoint.
  - Default config (`projector_train=default`) runs PMC-OA CT projector pretraining.
  - TCGA caption-stage tuning uses `projector_train=tcga_caption`.
- `scripts/model/04_train_vlm.py`
  - Stage 3: Qwen LoRA VLM training scaffold entrypoint.

## Non-Runnable Template
- `scripts/templates/source_template.py`
  - Copy this to create a new source script. Not intended to run directly.

## Naming Rules
- Runnable scripts must start with a verb.
- Ordered steps use `NN_` prefixes (`01_`, `02_`, ...).

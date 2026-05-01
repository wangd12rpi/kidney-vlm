# Scripts Guide

## Canonical Layout
- Shared data / registry / import-export flows live under `scripts/data/` and `scripts/hf_integration/`.
- Modality-owned processing and projector scripts use ordered modality-first folders:
  - `scripts/01_pathology_features/`
  - `scripts/01_pathology_png/`
  - `scripts/01_pathology_proj/`
  - `scripts/02_radiology_features/`
  - `scripts/02_radiology_proj/`
  - `scripts/02_radiology_segmentation/`
  - `scripts/03_dnam_features/`
  - `scripts/03_dnam_proj/`
  - `scripts/04_rna_features/`
  - `scripts/04_rna_proj/`
  - `scripts/05_text_genomics/`
  - `scripts/05_vqa_question_generation/`
  - `scripts/06_vqa_train/`
  - `scripts/07_vqa_evaluation/`

## Runnable Scripts
- `scripts/data/01_upsert_tcga_registry_rows.py`
  - Fetches/refreshes TCGA source rows (GDC+TCIA metadata join) and upserts the `source='tcga'` slice in the unified registry.
  - Resolves all `TCGA-*` projects by default and lets you remove projects with `data.source.tcga.exclude_project_ids`.
  - Pulls TCIA study + series metadata by default even when payload downloads are disabled.
  - Pulls targeted GDC mutation metadata for a kidney-focused gene panel by default.
  - Downloads pathology SVS, TCIA radiology series zips, and GDC PDF reports when enabled.
  - Example metadata-only run:
    - `uv run python scripts/data/01_upsert_tcga_registry_rows.py data.source.download.enabled=false`
  - Example full download run:
    - `uv run python scripts/data/01_upsert_tcga_registry_rows.py data.source.download.enabled=true`
- `scripts/data/01_build_pmc_oa_source.py`
  - Builds a registry-backed `source='pmc_oa'` slice from PMC-OA caption JSONL splits plus a PMC-OA MedSigLIP feature store.
  - Keeps PMC-OA radiology supervision aligned with the same unified split and source-slice replacement flow used elsewhere in the repo.
  - Example:
    - `uv run python scripts/data/01_build_pmc_oa_source.py`
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
- `scripts/data/05_register_uni_paths_into_registry.py`
  - Clears old pathology patch-embedding registry fields and writes existing UNI paths back into the unified registry.
  - Uses fixed variables at the top of the script instead of CLI args.
- `scripts/data/07_register_cpgpt_dnam_features_into_registry.py`
  - Aggregates the imported CpGPT DNAm manifest to the TCGA case level, writes all raw methylation beta paths into `genomics_dna_methylation_paths`, and fills one canonical `genomics_dna_methylation_feature_path` per case in the unified registry.
  - Uses fixed variables at the top of the script instead of CLI args.
- `scripts/01_pathology_features/03_download_uni_tcga_archives.py`
  - Downloads gated UNI2 TCGA tar archives from Hugging Face into a local staging folder.
  - Uses fixed variables at the top of the script instead of CLI args.
- `scripts/01_pathology_features/04_prepare_uni_tcga_features.py`
  - Extracts UNI2 TCGA tar archives one at a time, converts them into the flatter CONCH-like H5 layout, writes them into `data/features/features_uni`, updates the unified registry, and deletes processed archives.
  - Uses fixed variables at the top of the script instead of CLI args.
- `scripts/01_pathology_png/01_extract_pathology_pngs.py`
  - Downloads one TCGA pathology SVS at a time from GDC, renders a whole-slide thumbnail, writes portable thumbnail paths into the unified registry, then deletes the staged SVS.
  - Defaults to diagnostic `DX` slides only.
  - Saves outputs under `data/pathology_png/<TCGA barcode>/`; no manifest JSON is written.
  - Example quick smoke run:
    - `uv run python scripts/01_pathology_png/01_extract_pathology_pngs.py pathology_png.max_slides=2`
- `scripts/01_pathology_png/02_import_uniform_tumor_rois.py`
  - Registers already-downloaded UniformTumor ROI PNGs from `data/pathology_png` into `pathology_png_roi_paths`.
  - Does not download from Hugging Face or write new PNG files; the ROI folder is treated as the input artifact.
  - Fails if ROI filenames cannot be matched back to unified registry slide IDs.
  - Example:
    - `uv run python scripts/01_pathology_png/02_import_uniform_tumor_rois.py`
- `scripts/03_dnam_features/06_import_cpgpt_tcga_dnam_features.py`
  - Maps hashed CpGPT DNAm cache files from the external `hescapedna` repo back to TCGA methylation files using the original JSONL indexes, renames them into readable TCGA-linked feature filenames, copies them into `data/features/features_cpgpt_dnam`, and writes a manifest parquet/csv.
  - Uses fixed variables at the top of the script instead of CLI args.
- `scripts/01_pathology_proj/02_gen_path_case_captions.py`
  - Generates case-level pathology captions from registry metadata plus PDF pathology reports.
- `scripts/01_pathology_proj/03_build_path_proj_train_qa.py`
  - Builds slide-caption pathology projector training rows by matching available slide embeddings with case captions.
- `scripts/01_pathology_proj/04_train_path_projectors.py`
  - Stage 1: pathology projector training entrypoint.
- `scripts/03_dnam_proj/02_gen_dnam_case_captions.py`
  - Procedurally generates case-level DNAm captions from unified registry metadata, raw methylation beta-value summary statistics, and tracked cancer-prioritized driver mutations.
- `scripts/03_dnam_proj/03_build_dnam_proj_train_qa.py`
  - Builds case-caption DNAm projector training rows by matching available DNAm feature paths with the generated DNAm captions.
- `scripts/03_dnam_proj/04_train_dnam_projectors.py`
  - Stage 1: DNAm projector training entrypoint that follows the unified registry train/val/test split.
- `scripts/02_radiology_features/02_prepare_radiology_series_manifest.py`
  - Expands `radiology_download_paths` from the unified registry into a one-series-per-row processing table, optionally extracts the zip files, and runs radiology QC.
- `scripts/02_radiology_features/03_extract_radiology_pngs.py`
  - Renders accepted CT/MR DICOM slices into PNGs for downstream radiology processing.
- `scripts/02_radiology_features/04_extract_radiology_features.py`
  - Extracts MedSigLIP radiology embeddings from the rendered PNG slices into a shared H5 feature store.
- `scripts/02_radiology_segmentation/05_extract_radiology_segmentation.py`
  - Runs radiology segmentation on the rendered PNG slices and writes per-series mask artifacts plus a manifest.
- `scripts/02_radiology_features/06_register_radiology_artifacts_into_registry.py`
  - Writes radiology embedding refs, PNG dirs, mask paths, and mask manifests back into the unified registry.
- `scripts/02_radiology_proj/01_import_pmc_oa_captions.py`
  - Normalizes PMC-OA caption JSONL splits into `data/proj_train/radiology/pmc_oa_radiology_captions.parquet`.
- `scripts/02_radiology_proj/02_build_radiology_proj_train_qa.py`
  - Builds split-aware radiology projector QA rows by joining registry-backed radiology embeddings with imported caption rows.
  - Preserves the unified registry split as the source of truth even if imported caption metadata uses a different split label.
- `scripts/02_radiology_proj/03_train_radiology_projectors.py`
  - Stage 1: radiology projector training entrypoint that follows the unified registry train/val/test split.
- `scripts/05_text_genomics/01_download_tcga_extra_genomics.py`
  - Queries GDC for TCGA extra genomics files such as masked MAFs and gene/segment CNA tables, optionally downloads them, writes a manifest, and updates the unified registry.
  - Example metadata-only run:
    - `uv run python scripts/05_text_genomics/01_download_tcga_extra_genomics.py data.source.download.enabled=false`
- `scripts/05_text_genomics/02_build_genomics_text_blocks.py`
  - Builds per-case teacher/student genomics blocks from registered DNAm, RNA, mutation, and CNA artifacts. Teacher blocks include encoder-derivable DNAm/RNA summaries; student blocks keep only the discrete text-channel genomics.
  - Example:
    - `uv run python scripts/05_text_genomics/02_build_genomics_text_blocks.py`
- `scripts/05_text_genomics/02_build_llm_input_contexts.py`
  - Builds per-case clinical plus discrete-genomics text context files from registry MAF/CNA paths; DNAm and RNA remain embedding-backed and are not serialized into this prompt.
  - Example:
    - `uv run python scripts/05_text_genomics/02_build_llm_input_contexts.py --require-text-genomics`
- `scripts/05_vqa_question_generation/generate_gt_mcq.py`
  - Generates procedural ground-truth MCQ rows from `unified.parquet` into the shared VQA schema.
  - Writes `data/vqa/gt_mcq_questions.parquet`; caption-derived questions are generated into a separate file.
  - Task definitions live in `conf/05_vqa_question_generation/generate_gt_mcq.yaml`.
  - Example:
    - `uv run python scripts/05_vqa_question_generation/generate_gt_mcq.py`
- `scripts/06_vqa_train/train_vqa_lora.py`
  - Trains the projector-backed VLM on the single split-aware VQA parquet using PEFT LoRA SFT.
  - Injects MCQ choices into the prompt; open-ended rows use the same schema with empty option columns.
  - Saves LoRA adapters and projector checkpoints under `outputs/oncovlm/<run_name>/`.
  - Training config lives in `conf/06_vqa_train/vqa_lora_sft.yaml`.
  - Example:
    - `uv run python scripts/06_vqa_train/train_vqa_lora.py projectors.pathology.checkpoint_path=/path/to/path.ckpt`
- `scripts/07_vqa_evaluation/generate_vqa_predictions.py`
  - Multi-model VQA generation runner. Every model with `enabled: true` in the YAML is evaluated sequentially.
  - Supports Azure GPT, HF image-text VLMs, and the projector-only `oncovlm_qwen_no_finetune` baseline.
  - Writes per-model raw predictions under `results/<run.name>/<model.display_name>/predictions.parquet`.
  - Generation config lives in `conf/07_vqa_evaluation/generate_vqa_predictions.yaml`.
  - Example:
    - `uv run python scripts/07_vqa_evaluation/generate_vqa_predictions.py`
- `scripts/07_vqa_evaluation/score_vqa_predictions.py`
  - Reparses saved predictions and computes metrics without rerunning any model.
  - Supports MCQ semantic option-text scoring and open-ended QA BERTScore scoring.
  - Discovers every `predictions.parquet` under `results/<run.name>/*/` and reads model identity from the parquet.
  - Writes `metrics.json` beside each model's prediction parquet.
  - Scoring config lives in `conf/07_vqa_evaluation/score_vqa_predictions.yaml`.
  - Example:
    - `uv run python scripts/07_vqa_evaluation/score_vqa_predictions.py`

## VQA Parquet Schema
`modality_combination_name` is the named input recipe that produced the row, while the `use_*` columns are the actual modality booleans for that row. For example, `all_available` means "use every enabled artifact that exists for this case", so a case without radiology can still have `modality_combination_name=all_available` and `use_radiology=false`. In contrast, `radiology_only` requires radiology and sets the other `use_*` columns false. Current GT MCQ defaults are `all_available`, `path_only`, and `radiology_only`, but task configs can override these names.

| Column | Type | Notes |
| --- | --- | --- |
| `case_id` | string | TCGA case ID. |
| `project_id` | string | TCGA project name, e.g. `TCGA-LUAD`. |
| `question_id` | int64 | Unique row/question ID. |
| `base_question_id` | int64 | Same semantic question before modality expansion. |
| `split` | string | Registry split, usually `train`, `val`, or `test`. |
| `question_type` | string | `mcq` or `qa`. |
| `generation_type` | string | `from_ground_truth` or `from_caption`. |
| `task_category` | string | Broad task family, e.g. `mutation`, `stage`, `grade`, `subtype`. |
| `task_id` | string | Specific task, e.g. `mutation_tp53`, `pathologic_stage`. |
| `modality_combination_name` | string | Config-defined modality recipe name. |
| `use_pathology` | bool | Whether pathology evidence is used in this row. |
| `use_radiology` | bool | Whether radiology evidence is used in this row. |
| `use_dnam` | bool | Whether DNAm evidence is used in this row. |
| `use_rna` | bool | Whether RNA evidence is used in this row. |
| `question` | string | User-facing question text. |
| `option_a` | string | Nullable/empty for open-ended rows. |
| `option_b` | string | Nullable/empty for open-ended rows. |
| `option_c` | string | Nullable/empty for open-ended rows. |
| `option_d` | string | Nullable/empty for open-ended rows. |
| `answer` | string | Semantic answer text, not the option letter. |
| `answer_label` | string | `A`/`B`/`C`/`D` for MCQ rows, empty for open-ended rows. |
| `caption_id` | string | Empty unless the row is caption-generated. |
| `ground_truth_source` | string | Unified registry column(s), pipe-separated when multiple. |
| `radiology_biomarker` | string | Optional radiology biomarker text for radiology rows. |
| `pathology_feature_paths` | list[string] | Pathology feature refs, one-layer list. |
| `radiology_feature_paths` | list[string] | Radiology feature refs, one-layer list. |
| `dnam_feature_path` | string | DNAm feature path. |
| `rna_feature_path` | string | RNA feature path. |
| `pathology_roi_png_dir` | string | Test-row fallback image directory, empty otherwise. |
| `radiology_view_png_dir` | string | Test-row fallback image directory, empty otherwise. |
| `dnam_text_summary` | string | Test-row text fallback, empty otherwise. |
| `rna_text_summary` | string | Test-row text fallback, empty otherwise. |

- `scripts/hf_integration/01_upload_projector_train_to_hf.py`
  - Uploads projector-train parquet datasets to HF Hub using split-aware `DatasetDict` payloads.
  - Uses its own config file at `conf/hf_integration/projector_train_upload.yaml`.
- `scripts/hf_integration/02_upload_unified_parquet_to_hf.py`
  - Uploads the unified registry parquet to HF Hub as a split-aware dataset so the viewer exposes split selection.
  - Uses its own config file at `conf/hf_integration/unified_registry_upload.yaml`.
- `scripts/vlm_train/01_train_vlm.py`
  - Legacy VLM training scaffold. The active VQA SFT path is `scripts/06_vqa_train/train_vqa_lora.py`.

## Naming Rules
- Runnable scripts must start with a verb.
- Ordered steps use `NN_` prefixes (`01_`, `02_`, ...).

## Data Layout Notes
- Projector-train parquet artifacts live under `data/proj_train/<modality>/`.
- Active modalities currently reserved there are:
  - `pathology`
  - `radiology`
  - `dnam`
  - `rna`
- External supervision corpora should be normalized through a source builder plus modality projector parquet steps before training; do not point projector trainers at raw JSONL files directly.
- Radiology segmentation artifacts live under `data/segmentation/radiology/`.
- Pathology thumbnail/ROI PNG artifacts live under `data/pathology_png/`.

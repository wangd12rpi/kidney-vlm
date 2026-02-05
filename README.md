# kidney-vlm

Minimal, clean starter repo to:
- Build a TCGA pathology index (TCGA-KICH/KIRC/KIRP) from the GDC API
- Download 1 SVS per case (resumable)
- Tile SVS into a small set of tissue-rich patches (resumable)
- Evaluate base MedGemma on:
  - kidney stone dataset (binary)
  - TCGA kidney subtype (KICH/KIRC/KIRP) using WSI tiles

## Setup

1) Create env and install deps:

```bash
pip install -r requirements.txt
```

2) Install OpenSlide system dependency (needed for openslide-python).

- Ubuntu:
```bash
sudo apt-get update
sudo apt-get install -y openslide-tools
```

3) Make sure `transformers>=4.50.0` and `accelerate` are installed.

## Data layout

- Kidney stone images expected at:
  - `data/raw/kidney_stone/Original/Stone/*.jpg`
  - `data/raw/kidney_stone/Original/Non-Stone/*.jpg`

- TCGA pathology downloads go to:
  - `data/raw/tcga/path/<PROJECT>/<CASE_SUBMITTER_ID>/<SVS_FILE>.svs`

- TCGA index lives at:
  - `data/tcga_path_index.jsonl`

- Tiles go to:
  - `data/raw/tcga/path/<PROJECT>/<CASE_SUBMITTER_ID>/tiles/`

## TCGA: build index (1 SVS per case)

```bash
python data/tcga_build_path_index.py
```

## TCGA: download SVS (resumable)

```bash
python data/tcga_download_path_from_index.py
```

You can safely re-run it. Existing complete files are skipped.

## TCGA: tile SVS (resumable)

```bash
python data/tcga_tile_svs.py
```

This extracts a fixed number of tissue-rich tiles per case.

## Evaluate base MedGemma

These are prompt-based baselines.

Kidney stone:
```bash
python scripts/eval_medgemma_kidney_stone.py
```

TCGA pathology (KICH/KIRC/KIRP):
```bash
python scripts/eval_medgemma_tcga_path.py
```


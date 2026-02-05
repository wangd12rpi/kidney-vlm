# kidney-vlm

Minimal, clean starter repo to:
- Build a TCGA pathology index (TCGA-KICH/KIRC/KIRP) from the GDC API
- Download 1 SVS per case (resumable)
- Tile SVS into a small set of tissue-rich patches (resumable)
- Evaluate base MedGemma on:
  - kidney stone dataset (binary)
  - TCGA pathology tasks using slide-level labels from the JSONL (grade/stage/subtype/etc.)

## Setup

1) Create env and install deps:

```bash
pip install -r requirements.txt

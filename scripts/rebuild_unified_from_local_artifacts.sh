#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD=("${PYTHON_BIN}")
elif [[ -x ".venv/bin/python" ]]; then
  PYTHON_CMD=(".venv/bin/python")
else
  PYTHON_CMD=("uv" "run" "python")
fi

run_python() {
  echo
  echo "+ ${PYTHON_CMD[*]} $*"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    return 0
  fi
  "${PYTHON_CMD[@]}" "$@"
}

echo "Rebuilding unified registry from metadata plus local artifacts."
echo "Repo root: ${REPO_ROOT}"
echo "Python: ${PYTHON_CMD[*]}"
echo "Dry run: ${DRY_RUN:-0}"
echo
echo "Do not run this while another registry-writing job is running."
echo "Expected local artifacts:"
echo "  - data/features/features_uni/"
echo "  - data/features/features_cpgpt_dnam_manifest.parquet"
echo "  - data/pathology_png/**/__uniform_tumor_8k__*.png"




run_python scripts/data/05_register_uni_paths_into_registry.py

run_python scripts/data/07_register_cpgpt_dnam_features_into_registry.py

run_python scripts/01_pathology_png/02_import_uniform_tumor_rois.py

run_python - <<'PY'
from pathlib import Path
import pandas as pd

registry_path = Path("data/registry/unified.parquet")
df = pd.read_parquet(registry_path)

def has_value(value):
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, (list, tuple)):
        return any(str(item).strip() for item in value)
    if hasattr(value, "tolist") and not isinstance(value, str):
        converted = value.tolist()
        if isinstance(converted, list):
            return any(str(item).strip() for item in converted)
    return str(value).strip().lower() not in {"", "nan", "none", "<na>", "[]"}

print()
print("Unified rebuild summary")
print(f"Registry: {registry_path}")
print(f"Rows: {len(df)}")
print("Sources:")
print(df["source"].value_counts(dropna=False).to_string())
for column in [
    "pathology_tile_embedding_paths",
    "genomics_dna_methylation_feature_path",
    "pathology_png_roi_paths",
    "radiology_download_paths",
]:
    if column in df.columns:
        print(f"{column}: {int(df[column].map(has_value).sum())}")
PY

echo
echo "Unified registry rebuild complete."

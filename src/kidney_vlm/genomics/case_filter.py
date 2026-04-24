from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PATIENT_ID_KEYS = (
    "patient_id",
    "case_submitter_id",
    "submitter_id",
    "case_id",
    "id",
)


def _collect_patient_ids(value: Any) -> set[str]:
    patient_ids: set[str] = set()
    if value is None:
        return patient_ids
    if isinstance(value, str):
        text = value.strip()
        if text:
            patient_ids.add(text)
        return patient_ids
    if isinstance(value, dict):
        for key in PATIENT_ID_KEYS:
            raw = value.get(key)
            if isinstance(raw, str) and raw.strip():
                patient_ids.add(raw.strip())
        for nested_key in ("cases", "patient_ids", "case_ids", "patients"):
            if nested_key in value:
                patient_ids.update(_collect_patient_ids(value.get(nested_key)))
        return patient_ids
    if isinstance(value, (list, tuple, set)):
        for item in value:
            patient_ids.update(_collect_patient_ids(item))
        return patient_ids
    return patient_ids


def load_patient_ids_from_json(path: str | Path | None) -> list[str]:
    """Load TCGA patient/case submitter IDs from a flexible JSON file.

    The repository's `pathology_cases.json` is a simple list of TCGA patient
    IDs. This parser also accepts dict wrappers and list-of-dict records so the
    option is not brittle if the file gains metadata later.
    """
    if path is None or not str(path).strip():
        return []
    json_path = Path(path).expanduser()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    return sorted(_collect_patient_ids(payload))


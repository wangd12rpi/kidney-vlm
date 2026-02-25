from __future__ import annotations

import hashlib


def make_sample_id(
    source: str,
    patient_id: str,
    study_id: str,
    modality_scope: str = "multimodal",
    hash_len: int = 16,
) -> str:
    payload = "||".join([str(source), str(patient_id), str(study_id), str(modality_scope)])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:hash_len]
    return f"{source}-{digest}"

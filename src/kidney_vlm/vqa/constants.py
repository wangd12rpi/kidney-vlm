from __future__ import annotations

MODALITIES = ("pathology", "radiology", "dnam", "rna")

MODALITY_FLAG_COLUMNS = {
    "pathology": "use_pathology",
    "radiology": "use_radiology",
    "dnam": "use_dnam",
    "rna": "use_rna",
}

MODALITY_FEATURE_COLUMNS = {
    "pathology": "pathology_feature_paths",
    "radiology": "radiology_feature_paths",
    "dnam": "dnam_feature_path",
    "rna": "rna_feature_path",
}

OPTION_COLUMNS = ("option_a", "option_b", "option_c", "option_d")

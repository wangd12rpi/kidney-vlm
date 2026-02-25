from __future__ import annotations

import pandas as pd
import pytest

from kidney_vlm.data.registry_schema import CORE_COLUMNS, empty_registry_frame, normalize_registry_df, validate_registry_df


def test_missing_core_column_raises() -> None:
    frame = empty_registry_frame().drop(columns=["sample_id"])
    with pytest.raises(ValueError):
        validate_registry_df(frame)


def test_normalize_fills_core_columns() -> None:
    frame = pd.DataFrame([{"sample_id": "a", "source": "tcga"}])
    normalized = normalize_registry_df(frame)
    assert all(column in normalized.columns for column in CORE_COLUMNS)
    validate_registry_df(normalized)

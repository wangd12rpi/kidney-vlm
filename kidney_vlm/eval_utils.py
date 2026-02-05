from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def extract_label(text: str, labels: Sequence[str]) -> Optional[str]:
    """Pick the first matching label from free-form model output.

    Matching is case-insensitive and word-boundary based.
    """
    t = normalize_text(text).upper()
    for lab in labels:
        pat = r"\b" + re.escape(lab.upper()) + r"\b"
        if re.search(pat, t):
            return lab
    return None


def extract_label_with_synonyms(
    text: str,
    label_to_synonyms: Dict[str, List[str]],
) -> Optional[str]:
    t = normalize_text(text).upper()
    for label, syns in label_to_synonyms.items():
        for s in [label] + syns:
            pat = r"\b" + re.escape(s.upper()) + r"\b"
            if re.search(pat, t):
                return label
    return None


def compute_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, float]:
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro"))
    return {"accuracy": acc, "macro_f1": f1}


def print_report(y_true: List[str], y_pred: List[str], labels: List[str]) -> None:
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=labels, digits=4))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred, labels=labels))

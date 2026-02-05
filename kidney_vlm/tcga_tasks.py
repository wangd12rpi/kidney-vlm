from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from kidney_vlm.eval_utils import extract_label, extract_label_with_synonyms


@dataclass(frozen=True)
class TCGATask:
    name: str
    label_key: str
    classes: List[str]
    question: str
    synonyms: Dict[str, List[str]] | None = None

    def build_prompt(self) -> str:
        labels = ", ".join(self.classes)
        return (
            "You are a pathology assistant. "
            "This is a patch from a kidney whole-slide image (H&E). "
            f"Task: {self.question} "
            f"Answer with exactly one label: {labels}."
        )

    def parse_prediction(self, text: str) -> Optional[str]:
        if self.synonyms:
            return extract_label_with_synonyms(text, self.synonyms)
        return extract_label(text, self.classes)


TCGA_TASKS: Dict[str, TCGATask] = {
    # Always available from our index builder
    "kidney_subtype": TCGATask(
        name="kidney_subtype",
        label_key="tcga_kidney_subtype",
        classes=["KICH", "KIRC", "KIRP"],
        question="Classify the TCGA kidney subtype.",
        synonyms={
            "KICH": ["CHROMOPHOBE", "CHROMOPHOBE RCC", "CHROMOPHOBE RENAL CELL CARCINOMA"],
            "KIRC": ["CLEAR CELL", "CLEAR CELL RCC", "CLEAR CELL RENAL CELL CARCINOMA", "CCRCC", "CC RCC"],
            "KIRP": ["PAPILLARY", "PAPILLARY RCC", "PAPILLARY RENAL CELL CARCINOMA", "PRCC", "P RCC"],
        },
    ),
    # Common WSI tasks in TCGA papers
    "tumor_grade": TCGATask(
        name="tumor_grade",
        label_key="tumor_grade",
        classes=["G1", "G2", "G3", "G4"],
        question="Predict tumor grade.",
        synonyms={
            "G1": ["GRADE 1", "GRADE I", "G 1"],
            "G2": ["GRADE 2", "GRADE II", "G 2"],
            "G3": ["GRADE 3", "GRADE III", "G 3"],
            "G4": ["GRADE 4", "GRADE IV", "G 4"],
        },
    ),
    "tumor_stage": TCGATask(
        name="tumor_stage",
        label_key="tumor_stage_coarse",
        classes=["STAGE I", "STAGE II", "STAGE III", "STAGE IV"],
        question="Predict tumor stage (coarse).",
        synonyms={
            "STAGE I": ["STAGE 1", "STAGE I"],
            "STAGE II": ["STAGE 2", "STAGE II"],
            "STAGE III": ["STAGE 3", "STAGE III"],
            "STAGE IV": ["STAGE 4", "STAGE IV"],
        },
    ),
    "vital_status": TCGATask(
        name="vital_status",
        label_key="vital_status",
        classes=["Alive", "Dead"],
        question="Predict patient vital status from pathology (hard).",
        synonyms={
            "Alive": ["ALIVE", "LIVING"],
            "Dead": ["DEAD", "DECEASED"],
        },
    ),
    "age_bin": TCGATask(
        name="age_bin",
        label_key="age_bin",
        classes=["<50", "50-59", "60-69", "70+"],
        question="Predict age bin (hard, likely weak signal).",
        synonyms={
            "<50": ["<50", "UNDER 50"],
            "50-59": ["50-59"],
            "60-69": ["60-69"],
            "70+": ["70+", "70 PLUS", ">=70"],
        },
    ),
    "progression_or_recurrence": TCGATask(
        name="progression_or_recurrence",
        label_key="progression_or_recurrence",
        classes=["Yes", "No"],
        question="Predict whether there was progression or recurrence (hard).",
        synonyms={
            "Yes": ["YES", "TRUE"],
            "No": ["NO", "FALSE"],
        },
    ),
}
#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results" / "sft_rationale_comparison_20260715"

OLD_TEACHER_PATH = (
    ROOT
    / "data/21_cot_rationale_gen/caption_mcq_all_available_pathology_findings_cot.parquet"
)
NEW_TEACHER_PATH = (
    ROOT
    / "data/21_cot_rationale_gen/caption_mcq_all_available_pathology_findings_image_step_cot.parquet"
)
OLD_TEST_PATH = ROOT / "results/cot_vs_nocot_train_split_eval/predictions.parquet"
NEW_TEST_PATH = (
    ROOT / "results/image_step_cot_vs_nocot_pathology_findings_test/predictions.parquet"
)

OLD_CHECKPOINT = (
    "outputs/oncovlm/kidneyvlm_cot_newrationale_qwen35_9b_2ep/best/lora_adapter"
)
NEW_CHECKPOINT = (
    "outputs/oncovlm/sft/"
    "qwen3_5_9b_sft_caption_mcq_all_available_pathology_findings_"
    "image_step_cot_n2673_r8_projfrozen_20260715_020332_est/best/lora_adapter"
)
OLD_TEST_MODEL = "oncovlm_qwen_lora_cot_new_rationale"
NEW_TEST_MODEL = "oncovlm_qwen_lora_image_step_cot"

TRAIN_EXAMPLES = {
    "8582030154938399198": (
        "Concrete papilliform architecture, pushing nests, and laminated keratinization "
        "replace a mechanical four-choice walkthrough."
    ),
    "8727423578361554004": (
        "The new rationale scopes claims to sampled ROIs and separates spindle-cell "
        "architecture from its closest alternative."
    ),
    "7977529583008979548": (
        "The new rationale discriminates diffuse tissue-dropout pallor from cavitary "
        "necrosis and glomeruloid vascular proliferation."
    ),
}

TEST_EXAMPLES = {
    "59001304306035156": (
        "Both SFTs answer correctly; the new SFT separates gland geometry and dirty "
        "necrosis from the inference instead of traversing all choices."
    ),
    "6332281543206486651": (
        "Both SFTs answer correctly; the new SFT uses center-to-periphery tumor-island "
        "morphology and contrasts only the closest alternative."
    ),
    "6826264933257959015": (
        "Both SFTs answer correctly; the new SFT grounds the decision in fascicles, "
        "nodules, nuclei, clefts, and stromal quality."
    ),
}

MATCH_FIELDS = [
    "case_id",
    "project_id",
    "question",
    "option_a",
    "option_b",
    "option_c",
    "option_d",
    "answer",
    "answer_label",
]


def _index(frame: pd.DataFrame) -> pd.DataFrame:
    indexed = frame.copy()
    indexed["question_id"] = indexed["question_id"].astype(str)
    if indexed["question_id"].duplicated().any():
        duplicate = indexed.loc[
            indexed["question_id"].duplicated(), "question_id"
        ].iloc[0]
        raise ValueError(f"Duplicate question_id: {duplicate}")
    return indexed.set_index("question_id", drop=False)


def _assert_same_question(old_row: pd.Series, new_row: pd.Series) -> None:
    mismatched = [field for field in MATCH_FIELDS if old_row[field] != new_row[field]]
    if mismatched:
        raise ValueError(
            f"Question {old_row['question_id']} differs across artifacts: {mismatched}"
        )


def _think_words(text: str) -> int:
    before_close = str(text).split("</think>", maxsplit=1)[0]
    before_close = re.sub(r"^\s*<think>\s*", "", before_close, flags=re.IGNORECASE)
    return len(before_close.split())


def _choice_lines(row: pd.Series) -> str:
    return "\n".join(f"- {label}. {row[f'option_{label.lower()}']}" for label in "ABCD")


def _code_block(text: str) -> str:
    return f"```text\n{str(text).strip()}\n```"


def _build_train_examples() -> tuple[list[dict], dict[str, float]]:
    old = _index(pd.read_parquet(OLD_TEACHER_PATH))
    new = _index(pd.read_parquet(NEW_TEACHER_PATH))
    old_train = old.loc[old["split"].eq("train")]
    new_train = new.loc[new["split"].eq("train")]
    shared_ids = old_train.index.intersection(new_train.index)
    shared_old = old_train.loc[shared_ids]
    shared_new = new_train.loc[shared_ids]
    for question_id in shared_ids:
        _assert_same_question(shared_old.loc[question_id], shared_new.loc[question_id])

    records: list[dict] = []
    for question_id, takeaway in TRAIN_EXAMPLES.items():
        old_row = old_train.loc[question_id]
        new_row = new_train.loc[question_id]
        _assert_same_question(old_row, new_row)
        records.append(
            {
                "comparison": "training_teacher_rationale",
                "question_id": question_id,
                "case_id": old_row["case_id"],
                "project_id": old_row["project_id"],
                "question": old_row["question"],
                **{
                    f"option_{label.lower()}": old_row[f"option_{label.lower()}"]
                    for label in "ABCD"
                },
                "answer": old_row["answer"],
                "answer_label": old_row["answer_label"],
                "old_prediction": old_row["answer"],
                "new_prediction": new_row["answer"],
                "old_correct": True,
                "new_correct": True,
                "old_text": old_row["rationale"],
                "new_text": new_row["rationale"],
                "old_words": _think_words(old_row["rationale"]),
                "new_words": _think_words(new_row["rationale"]),
                "takeaway": takeaway,
                "old_source": str(OLD_TEACHER_PATH.relative_to(ROOT)),
                "new_source": str(NEW_TEACHER_PATH.relative_to(ROOT)),
            }
        )

    old_walkthrough = (
        shared_old["rationale"]
        .str.lower()
        .apply(
            lambda value: all(
                f"{ordinal} choice" in value
                for ordinal in ("first", "second", "third", "fourth")
            )
        )
    )
    new_steps = shared_new["rationale"].apply(
        lambda value: (
            "Step 1 — Observation:" in value and "Step 2 — Reasoning:" in value
        )
    )
    stats = {
        "shared_train_questions": float(len(shared_ids)),
        "old_mean_words": float(shared_old["rationale"].map(_think_words).mean()),
        "new_mean_words": float(shared_new["rationale"].map(_think_words).mean()),
        "old_four_choice_walkthrough_rate": float(old_walkthrough.mean()),
        "new_two_step_rate": float(new_steps.mean()),
    }
    return records, stats


def _build_test_examples() -> tuple[list[dict], dict[str, float]]:
    old_all = pd.read_parquet(OLD_TEST_PATH)
    new_all = pd.read_parquet(NEW_TEST_PATH)
    old = _index(old_all.loc[old_all["model_display_name"].eq(OLD_TEST_MODEL)])
    new = _index(new_all.loc[new_all["model_display_name"].eq(NEW_TEST_MODEL)])
    shared_ids = old.index.intersection(new.index)
    shared_old = old.loc[shared_ids]
    shared_new = new.loc[shared_ids]
    for question_id in shared_ids:
        _assert_same_question(shared_old.loc[question_id], shared_new.loc[question_id])

    jointly_correct = shared_old["correct"].astype(bool) & shared_new["correct"].astype(
        bool
    )
    records: list[dict] = []
    for question_id, takeaway in TEST_EXAMPLES.items():
        old_row = old.loc[question_id]
        new_row = new.loc[question_id]
        _assert_same_question(old_row, new_row)
        if not bool(old_row["correct"]) or not bool(new_row["correct"]):
            raise ValueError(f"Test example {question_id} is not jointly correct.")
        if old_row["predicted_answer"] != new_row["predicted_answer"]:
            raise ValueError(
                f"Test example {question_id} does not have the same prediction."
            )
        records.append(
            {
                "comparison": "sft_test_rationale",
                "question_id": question_id,
                "case_id": old_row["case_id"],
                "project_id": old_row["project_id"],
                "question": old_row["question"],
                **{
                    f"option_{label.lower()}": old_row[f"option_{label.lower()}"]
                    for label in "ABCD"
                },
                "answer": old_row["answer"],
                "answer_label": old_row["answer_label"],
                "old_prediction": old_row["predicted_answer"],
                "new_prediction": new_row["predicted_answer"],
                "old_correct": bool(old_row["correct"]),
                "new_correct": bool(new_row["correct"]),
                "old_text": old_row["raw_response"],
                "new_text": new_row["raw_response"],
                "old_words": _think_words(old_row["raw_response"]),
                "new_words": _think_words(new_row["raw_response"]),
                "takeaway": takeaway,
                "old_source": f"{OLD_TEST_PATH.relative_to(ROOT)}::{OLD_TEST_MODEL}",
                "new_source": f"{NEW_TEST_PATH.relative_to(ROOT)}::{NEW_TEST_MODEL}",
                "old_user_prompt": old_row["user_prompt"],
                "new_user_prompt": new_row["user_prompt"],
            }
        )

    eligible_old = shared_old.loc[jointly_correct]
    eligible_new = shared_new.loc[jointly_correct]
    old_walkthrough = (
        eligible_old["raw_response"]
        .str.lower()
        .apply(
            lambda value: all(
                f"{ordinal} choice" in value
                for ordinal in ("first", "second", "third", "fourth")
            )
        )
    )
    new_steps = eligible_new["raw_response"].apply(
        lambda value: (
            "Step 1 — Observation:" in value and "Step 2 — Reasoning:" in value
        )
    )
    stats = {
        "shared_test_questions": float(len(shared_ids)),
        "jointly_correct_questions": float(jointly_correct.sum()),
        "old_accuracy_shared": float(shared_old["correct"].mean()),
        "new_accuracy_shared": float(shared_new["correct"].mean()),
        "old_mean_words_jointly_correct": float(
            eligible_old["raw_response"].map(_think_words).mean()
        ),
        "new_mean_words_jointly_correct": float(
            eligible_new["raw_response"].map(_think_words).mean()
        ),
        "old_four_choice_walkthrough_rate_jointly_correct": float(
            old_walkthrough.mean()
        ),
        "new_two_step_rate_jointly_correct": float(new_steps.mean()),
    }
    return records, stats


def _render_markdown(
    train_records: list[dict],
    train_stats: dict[str, float],
    test_records: list[dict],
    test_stats: dict[str, float],
) -> str:
    lines = [
        "# Old vs new SFT rationale comparison",
        "",
        "Prepared 2026-07-15. These are matched qualitative examples for presentation use.",
        "",
        "## Slide-level summary",
        "",
        f"- Training teachers: {int(train_stats['shared_train_questions']):,} matched train questions. "
        f"Old rationales average {train_stats['old_mean_words']:.1f} words; new rationales average "
        f"{train_stats['new_mean_words']:.1f} words.",
        f"- Old teacher rationales walk through all four choices in "
        f"{train_stats['old_four_choice_walkthrough_rate']:.0%} of matched rows. New rationales use "
        f"Observation → Reasoning steps in {train_stats['new_two_step_rate']:.0%}.",
        f"- Test SFTs: {int(test_stats['shared_test_questions'])} shared test questions; "
        f"{int(test_stats['jointly_correct_questions'])} are jointly correct. On the shared set, old SFT "
        f"accuracy is {test_stats['old_accuracy_shared']:.1%} and today's SFT accuracy is "
        f"{test_stats['new_accuracy_shared']:.1%}.",
        f"- Within the jointly-correct pool, old rationale sections average "
        f"{test_stats['old_mean_words_jointly_correct']:.1f} words and use a four-choice walkthrough in "
        f"{test_stats['old_four_choice_walkthrough_rate_jointly_correct']:.0%}; today's rationale sections average "
        f"{test_stats['new_mean_words_jointly_correct']:.1f} words and use the two-step format in "
        f"{test_stats['new_two_step_rate_jointly_correct']:.0%}.",
        "",
        "## Checkpoint provenance",
        "",
        f"- Old SFT: `{OLD_CHECKPOINT}`",
        f"- Today's SFT: `{NEW_CHECKPOINT}`",
        "- Both are Qwen3.5-9B LoRA SFT checkpoints; no GRPO checkpoint is used here.",
        "",
        "## A. Training-teacher rationales (same question and gold answer)",
        "",
    ]

    for index, record in enumerate(train_records, start=1):
        row = pd.Series(record)
        lines.extend(
            [
                f"### A{index}. {record['project_id']} / {record['case_id']} — `{record['question_id']}`",
                "",
                f"**Question:** {record['question']}",
                "",
                _choice_lines(row),
                "",
                f"**Gold answer:** {record['answer']}",
                "",
                f"**Old GPT rationale ({record['old_words']} words)**",
                "",
                _code_block(record["old_text"]),
                "",
                f"**New image-derived GPT rationale ({record['new_words']} words)**",
                "",
                _code_block(record["new_text"]),
                "",
                f"**Slide takeaway:** {record['takeaway']}",
                "",
            ]
        )

    lines.extend(
        [
            "## B. SFT test outputs (same question, same correct prediction)",
            "",
            "These three cases are jointly correct so the comparison focuses on rationale form and content.",
            "",
        ]
    )
    for index, record in enumerate(test_records, start=1):
        row = pd.Series(record)
        lines.extend(
            [
                f"### B{index}. {record['project_id']} / {record['case_id']} — `{record['question_id']}`",
                "",
                f"**Question:** {record['question']}",
                "",
                _choice_lines(row),
                "",
                f"**Gold and both predicted answers:** {record['answer']}",
                "",
                f"**Old SFT output ({record['old_words']} words)**",
                "",
                _code_block(record["old_text"]),
                "",
                f"**Today's SFT output ({record['new_words']} words)**",
                "",
                _code_block(record["new_text"]),
                "",
                f"**Slide takeaway:** {record['takeaway']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation boundaries",
            "",
            "- This is a legacy-versus-updated CoT pipeline comparison, not a format-only ablation. "
            "The SFT datasets, initialization, and learning rates also differ.",
            "- Existing test outputs use each checkpoint's native inference instruction: the old prompt "
            "requests a generic choice comparison, while today's prompt explicitly requests Observation → Reasoning.",
            "- The new teacher dataset is image-conditioned and answer-validated, not independently pathologist-verified.",
            "- The six cases are selected qualitative examples. The aggregate matched-set statistics above "
            "should accompany them to avoid implying that three examples establish overall efficacy.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    train_records, train_stats = _build_train_examples()
    test_records, test_stats = _build_test_examples()
    records = train_records + test_records
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_parquet(OUTPUT_DIR / "comparison_examples.parquet", index=False)
    frame.to_csv(OUTPUT_DIR / "comparison_examples.csv", index=False)
    (OUTPUT_DIR / "comparison.md").write_text(
        _render_markdown(train_records, train_stats, test_records, test_stats),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(
            {"training": train_stats, "test": test_stats}, indent=2, sort_keys=True
        ),
        encoding="utf-8",
    )
    print(f"Saved {len(records)} matched examples to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

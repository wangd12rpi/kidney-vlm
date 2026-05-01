#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from openai import AzureOpenAI

ROOT = Path(__file__).resolve().parents[2]
VQA_PATH = ROOT / "data/vqa/captions_mcq_oe_questions.parquet"

N = 30
SEED = 13
SPLIT = "test"

AZURE_ENDPOINT = "https://azurefoundry-dannong.cognitiveservices.azure.com/"
AZURE_DEPLOYMENT = "gpt-5.4"
AZURE_API_VERSION = "2024-12-01-preview"
AZURE_API_KEY_ENV = "AZURE_OPENAI_API_KEY"


def env_value(name: str) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    env_path = ROOT / ".env"
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith(f"{name}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def norm(text: str) -> str:
    return " ".join(str(text).casefold().strip().split())


def prompt_for(row: pd.Series) -> str:
    options = [str(row[f"option_{letter}"]).strip() for letter in "abcd"]
    return (
        "Select the correct answer. Return only the exact text of one option.\n\n"
        f"Question: {str(row['question']).strip()}\n\n"
        "Options:\n"
        + "\n".join(f"- {option}" for option in options)
    )


def parse_label(row: pd.Series, response: str) -> str:
    response_norm = norm(response)
    options = [(letter.upper(), str(row[f"option_{letter}"]).strip()) for letter in "abcd"]
    for label, option in options:
        if norm(option) == response_norm:
            return label
    for label, option in options:
        if norm(option) and norm(option) in response_norm:
            return label
    return ""


def main() -> None:
    df = pd.read_parquet(VQA_PATH)
    df = df[(df["question_type"] == "mcq") & (df["generation_type"] == "from_caption")]
    if SPLIT:
        df = df[df["split"].astype(str) == SPLIT]
    sample = df.sample(n=min(N, len(df)), random_state=SEED).reset_index(drop=True)

    client = AzureOpenAI(
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=env_value(AZURE_API_KEY_ENV),
    )

    correct = 0
    parsed = 0
    for i, (_, row) in enumerate(sample.iterrows(), start=1):
        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "Return only the exact text of one provided option."},
                {"role": "user", "content": prompt_for(row)},
            ],
            max_completion_tokens=256,
            reasoning_effort="none",
            verbosity="low",
        )
        raw = str(response.choices[0].message.content or "").strip()
        pred = parse_label(row, raw)
        gt = str(row["answer_label"]).strip().upper()
        parsed += int(bool(pred))
        correct += int(pred == gt)
        print(
            f"[{i:03d}/{len(sample)}] qid={row['question_id']} "
            f"task={row['task_category']} combo={row['modality_combination_name']} "
            f"gt={gt} pred={pred or '-'} correct={pred == gt}\n"
            f"  response: {raw}\n"
            f"  accuracy_so_far: {correct}/{i} ({correct / i:.1%})"
        )

    print(f"\nFinal parsed: {parsed}/{len(sample)} ({parsed / len(sample):.1%})")
    print(f"Final accuracy: {correct}/{len(sample)} ({correct / len(sample):.1%})")


if __name__ == "__main__":
    main()

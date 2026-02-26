#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tqdm.auto import tqdm

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.data.registry_io import read_parquet_or_empty
from kidney_vlm.repo_root import find_repo_root

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


def load_cfg():
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "conf")):
        cfg = compose(config_name="config")
    OmegaConf.set_struct(cfg, False)
    return cfg


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, float) and pd.isna(value):
        return []
    if hasattr(value, "tolist") and not isinstance(value, str):
        converted = value.tolist()
        if isinstance(converted, list):
            return [str(item).strip() for item in converted if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _normalize_local_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _to_prompt_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value if str(item).strip())
    return str(value).strip()


def _extract_pdf_text(pdf_path: Path, max_chars: int, max_pages: int | None) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is required. Install it with: uv add pypdf") from exc

    reader = PdfReader(str(pdf_path))
    chunks: list[str] = []
    running_chars = 0
    page_iter = reader.pages if max_pages is None else reader.pages[:max_pages]
    for page in page_iter:
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        chunks.append(text)
        running_chars += len(text)
        if running_chars >= max_chars:
            break

    joined = "\n\n".join(chunks).strip()
    return joined[:max_chars]


def _build_client(azure_cfg: Any):
    try:
        from openai import AzureOpenAI
    except ImportError as exc:
        raise RuntimeError("openai is required. Install it with: uv add openai") from exc

    api_key_env = str(azure_cfg.api_key_env).strip()
    api_key = os.getenv(api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing Azure OpenAI key in env var: {api_key_env}")

    return AzureOpenAI(
        api_version=str(azure_cfg.api_version),
        azure_endpoint=str(azure_cfg.endpoint),
        api_key=api_key,
    )


def _extract_text_content(raw_content: Any) -> str:
    if isinstance(raw_content, str):
        return raw_content.strip()
    if isinstance(raw_content, list):
        chunks: list[str] = []
        for item in raw_content:
            if isinstance(item, str):
                if item.strip():
                    chunks.append(item.strip())
                continue
            text_attr = getattr(item, "text", None)
            if isinstance(text_attr, str) and text_attr.strip():
                chunks.append(text_attr.strip())
                continue
            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    chunks.append(text_value.strip())
        return "\n".join(chunks).strip()
    return str(raw_content or "").strip()


def _generate_caption(
    client: Any,
    azure_cfg: Any,
    *,
    system_prompt: str,
    instruction: str,
    metadata_lines: list[str],
    report_text: str,
) -> str:
    deployment = str(azure_cfg.deployment)
    max_tokens = int(azure_cfg.max_completion_tokens)
    retries = int(azure_cfg.max_retries)
    retry_sleep_seconds = float(azure_cfg.retry_sleep_seconds)

    metadata_block = "\n".join(metadata_lines).strip()
    report_block = report_text.strip()
    user_prompt = (
        f"Instruction: {instruction}\n\n"
        "Patient metadata:\n"
        f"{metadata_block if metadata_block else '[none]'}\n\n"
        "Pathology report text:\n"
        f"{report_block if report_block else '[none]'}\n\n"
        "Generate exactly one caption for the pathology image."
    )

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=max_tokens,
                model=deployment,
            )
            caption = _extract_text_content(response.choices[0].message.content)
            caption = caption.strip()
            if not caption:
                raise RuntimeError("Model returned empty caption.")
            return caption
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_sleep_seconds)

    raise RuntimeError(f"Caption generation failed after {retries} attempts: {last_error}")


def main() -> None:
    cfg = load_cfg()
    qa_cfg = cfg.qa_genereation

    registry_path = Path(str(qa_cfg.source_registry_path)).expanduser()
    if not registry_path.is_absolute():
        registry_path = (ROOT / registry_path).resolve()
    else:
        registry_path = registry_path.resolve()

    output_path = Path(str(qa_cfg.output_parquet_path)).expanduser()
    if not output_path.is_absolute():
        output_path = (ROOT / output_path).resolve()
    else:
        output_path = output_path.resolve()

    registry_df = read_parquet_or_empty(registry_path)
    if registry_df.empty:
        raise RuntimeError(f"Registry is empty: {registry_path}")

    if bool(qa_cfg.require_pathology) and "pathology_wsi_paths" in registry_df.columns:
        registry_df = registry_df[registry_df["pathology_wsi_paths"].map(lambda v: len(_as_list(v)) > 0)]

    first_n = qa_cfg.get("first_n")
    if first_n is not None and str(first_n).strip():
        registry_df = registry_df.head(int(first_n))

    if registry_df.empty:
        print("No rows selected for QA generation.")
        return

    existing_output = pd.DataFrame()
    done_sample_ids: set[str] = set()
    overwrite_output = bool(qa_cfg.overwrite_output)
    if output_path.exists() and not overwrite_output:
        existing_output = pd.read_parquet(output_path)
        if "sample_id" in existing_output.columns:
            done_sample_ids = set(existing_output["sample_id"].map(str).tolist())

    rows_to_process: list[dict[str, Any]] = []
    for _, row in registry_df.iterrows():
        row_dict = row.to_dict()
        sample_id = str(row_dict.get("sample_id", "")).strip()
        if done_sample_ids and sample_id in done_sample_ids:
            continue
        rows_to_process.append(row_dict)

    if not rows_to_process:
        print("All selected rows already generated in output parquet.")
        return

    azure_cfg = qa_cfg.azure_openai
    client = _build_client(azure_cfg)

    instruction_text = str(qa_cfg.instruction)
    system_prompt = str(qa_cfg.system_prompt)
    metadata_fields = list(qa_cfg.metadata_fields)
    report_max_chars = int(qa_cfg.report_text_max_chars)
    report_max_pages = qa_cfg.get("report_max_pages")
    report_max_pages = None if report_max_pages in (None, "", "null") else int(report_max_pages)
    report_max_files = int(qa_cfg.max_reports_per_sample)
    print_first_n = int(qa_cfg.print_first_n)

    generated_rows: list[dict[str, Any]] = []
    loop = tqdm(rows_to_process, total=len(rows_to_process), desc="Generating projector QA captions")
    for idx, row in enumerate(loop, start=1):
        sample_id = str(row.get("sample_id", "")).strip()
        report_paths_raw = _as_list(row.get("report_pdf_paths"))
        report_paths: list[Path] = []
        for raw_path in report_paths_raw[:report_max_files]:
            local_path = _normalize_local_path(raw_path)
            if local_path.exists():
                report_paths.append(local_path)

        report_text_parts: list[str] = []
        for report_path in report_paths:
            try:
                extracted = _extract_pdf_text(report_path, max_chars=report_max_chars, max_pages=report_max_pages)
            except Exception as exc:
                extracted = f"[failed to read {report_path.name}: {exc}]"
            if extracted:
                report_text_parts.append(f"Report file: {report_path.name}\n{extracted}")
            if sum(len(chunk) for chunk in report_text_parts) >= report_max_chars:
                break

        report_text = "\n\n".join(report_text_parts)[:report_max_chars]
        metadata_lines: list[str] = []
        for field_name in metadata_fields:
            value = _to_prompt_value(row.get(field_name))
            if value:
                metadata_lines.append(f"{field_name}: {value}")

        caption = _generate_caption(
            client,
            azure_cfg,
            system_prompt=system_prompt,
            instruction=instruction_text,
            metadata_lines=metadata_lines,
            report_text=report_text,
        )

        qa_row = {
            "sample_id": sample_id,
            "source": str(row.get("source", "")),
            "patient_id": str(row.get("patient_id", "")),
            "study_id": str(row.get("study_id", "")),
            "split": str(row.get("split", "")),
            "pathology_wsi_paths": _as_list(row.get("pathology_wsi_paths")),
            "pathology_tile_embedding_paths": _as_list(row.get("pathology_tile_embedding_paths")),
            "pathology_slide_embedding_paths": _as_list(row.get("pathology_slide_embedding_paths")),
            "report_pdf_paths": [str(path) for path in report_paths],
            "instruction": instruction_text,
            "question": instruction_text,
            "caption": caption,
            "answer": caption,
            "caption_model": str(azure_cfg.deployment),
        }
        generated_rows.append(qa_row)

        if idx <= print_first_n:
            print("-" * 80)
            print(f"sample_id: {sample_id}")
            print(f"instruction: {instruction_text}")
            print(f"caption: {caption}")

    generated_df = pd.DataFrame(generated_rows)
    if not existing_output.empty and not overwrite_output:
        final_df = pd.concat([existing_output, generated_df], ignore_index=True)
        if "sample_id" in final_df.columns:
            final_df = final_df.drop_duplicates(subset=["sample_id"], keep="last").reset_index(drop=True)
    else:
        final_df = generated_df

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(output_path, index=False)
    print(f"Saved QA parquet: {output_path}")
    print(f"Rows written: {len(final_df)}")


if __name__ == "__main__":
    main()

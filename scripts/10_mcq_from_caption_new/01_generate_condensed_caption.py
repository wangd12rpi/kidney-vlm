#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf
from tqdm.auto import tqdm

try:
    from openai import AzureOpenAI
except ModuleNotFoundError as exc:
    AzureOpenAI = Any  # type: ignore[misc,assignment]
    OPENAI_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    OPENAI_IMPORT_ERROR = None


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "conf" / "10_mcq_from_caption_new" / "generate_condensed_caption.yaml"

SECTION_KEYS = [
    "radiology_findings",
    "pathology_findings",
    "genomic_findings",
    "integrated_interpretation",
]
OUTPUT_COLUMNS = [
    "case_id",
    "project_id",
    *SECTION_KEYS,
]


class EmptyFingerprintResponse(RuntimeError):
    pass


class InvalidFingerprintJson(ValueError):
    pass


class MissingFingerprintKeys(ValueError):
    pass


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _read_repo_env() -> dict[str, str]:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _env_value(name: str, repo_env: dict[str, str]) -> str:
    return os.environ.get(name, "").strip() or repo_env.get(name, "").strip()


def _load_cfg(config_path: Path) -> Any:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    return OmegaConf.load(config_path)


def _load_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input caption parquet not found: {path}")
    df = pd.read_parquet(path)
    required = {"case_id", "project_id", "split", "caption"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input caption parquet is missing columns: {missing}")
    if df["case_id"].isna().any():
        raise ValueError("Input caption parquet has null case_id values.")
    if df["project_id"].isna().any():
        raise ValueError("Input caption parquet has null project_id values.")
    return df


def _apply_filters(df: pd.DataFrame, cfg: Any) -> pd.DataFrame:
    return _apply_filters_impl(df, cfg, apply_first_n=True)


def _apply_filters_impl(df: pd.DataFrame, cfg: Any, *, apply_first_n: bool) -> pd.DataFrame:
    out = df.copy()

    split_cfg = cfg.filters.split
    if bool(split_cfg.enabled):
        allowed = {str(value).strip() for value in split_cfg["values"] if str(value).strip()}
        if not allowed:
            raise ValueError("filters.split.enabled=true but filters.split.values is empty.")
        out = out[out["split"].astype(str).str.strip().isin(allowed)].copy()

    project_cfg = cfg.filters.project_id
    if bool(project_cfg.enabled):
        allowed = {str(value).strip() for value in project_cfg["values"] if str(value).strip()}
        if not allowed:
            raise ValueError("filters.project_id.enabled=true but filters.project_id.values is empty.")
        out = out[out["project_id"].astype(str).str.strip().isin(allowed)].copy()

    caption_contains_cfg = cfg.filters.caption_contains
    if bool(caption_contains_cfg.enabled):
        values = [str(value).strip() for value in caption_contains_cfg["values"] if str(value).strip()]
        if not values:
            raise ValueError("filters.caption_contains.enabled=true but filters.caption_contains.values is empty.")
        mask = pd.Series(False, index=out.index)
        caption_text = out["caption"].fillna("").astype(str)
        for value in values:
            mask |= caption_text.str.contains(value, case=False, regex=False)
        out = out[mask].copy()

    first_n_cfg = cfg.filters.first_n
    if apply_first_n and bool(first_n_cfg.enabled):
        first_n = int(first_n_cfg.value)
        if first_n <= 0:
            raise ValueError("filters.first_n.value must be positive when enabled.")
        out = out.head(first_n).copy()

    if out.empty:
        raise ValueError("No caption rows remain after filters.")
    return out.reset_index(drop=True)


def _format_contrast_examples(examples: list[dict[str, str]]) -> str:
    blocks = []
    for index, example in enumerate(examples, start=1):
        blocks.append(f"Other case {index} (case_id={example['case_id']}):\n```text\n{example['caption']}\n```")
    return "\n\n".join(blocks)


def _build_contrast_caption_map(df: pd.DataFrame, *, count: int) -> dict[str, list[dict[str, str]]]:
    if count <= 0:
        raise ValueError("prompt.contrast_examples.count must be positive.")
    if df["case_id"].astype(str).duplicated().any():
        duplicated = df.loc[df["case_id"].astype(str).duplicated(), "case_id"].head(5).tolist()
        raise ValueError(f"Contrast caption pool has duplicated case_id values: {duplicated}")

    contrast_map: dict[str, list[dict[str, str]]] = {}
    for project_id, project_df in df.groupby("project_id", sort=True):
        project_rows = (
            project_df[["case_id", "caption"]]
            .assign(case_id=lambda frame: frame["case_id"].astype(str).str.strip())
            .assign(caption=lambda frame: frame["caption"].astype(str).str.strip())
            .sort_values("case_id")
            .to_dict(orient="records")
        )
        if len(project_rows) <= count:
            raise ValueError(
                f"Project {project_id} has {len(project_rows)} caption rows, but {count} contrast examples are required."
            )
        for index, row in enumerate(project_rows):
            examples = [project_rows[(index + offset) % len(project_rows)] for offset in range(1, count + 1)]
            contrast_map[row["case_id"]] = examples
    return contrast_map


def _create_client(cfg: Any) -> tuple[AzureOpenAI, str]:
    if OPENAI_IMPORT_ERROR is not None:
        raise RuntimeError("Missing Python package 'openai'. Run uv sync before using this script.") from OPENAI_IMPORT_ERROR

    repo_env = _read_repo_env()
    endpoint = str(cfg.azure.endpoint).strip()
    api_key = _env_value(str(cfg.azure.api_key_env), repo_env)
    deployment = str(cfg.azure.deployment).strip()

    if not endpoint:
        raise ValueError("Missing azure.endpoint in config.")
    if not api_key:
        raise EnvironmentError(f"Missing Azure API key env var: {cfg.azure.api_key_env}")
    if not deployment:
        raise ValueError("Missing azure.deployment in config.")

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=str(cfg.azure.api_version),
    )
    return client, deployment


def _extract_caption_fingerprint(
    *,
    client: AzureOpenAI,
    deployment: str,
    cfg: Any,
    case_id: str,
    project_id: str,
    caption: str,
    contrast_examples: list[dict[str, str]],
) -> dict[str, str]:
    user_prompt = str(cfg.prompt.user_template).format(
        max_words=int(cfg.prompt.max_words_per_section),
        case_id=case_id,
        project_id=project_id,
        target_caption=caption,
        other_case_captions=_format_contrast_examples(contrast_examples),
    )
    response = client.chat.completions.create(
        **{
            key: value
            for key, value in {
                "model": deployment,
                "messages": [
                    {"role": "system", "content": str(cfg.prompt.system_prompt)},
                    {"role": "user", "content": user_prompt},
                ],
                "max_completion_tokens": int(cfg.azure.max_completion_tokens),
                "temperature": float(cfg.azure.temperature),
                "reasoning_effort": str(cfg.azure.get("reasoning_effort", "")).strip() or None,
            }.items()
            if value is not None
        }
    )
    raw = response.choices[0].message.content
    if raw is None or not raw.strip():
        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", "")
        content_filter = getattr(choice, "content_filter_results", "")
        raise EmptyFingerprintResponse(
            "Model returned a completely empty response "
            f"(finish_reason={finish_reason!r}, content_filter_results={content_filter!r})."
        )

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raw_preview = raw.replace("\n", "\\n")[:500]
        raise InvalidFingerprintJson(
            f"Model response is not valid JSON: {exc}. Raw response preview: {raw_preview!r}"
        ) from exc
    if not isinstance(payload, dict):
        raise InvalidFingerprintJson(
            f"Fingerprint JSON must be an object, got {type(payload).__name__}: {payload!r}"
        )
    missing = sorted(set(SECTION_KEYS) - set(payload))
    if missing:
        raise MissingFingerprintKeys(
            f"Fingerprint JSON is missing required keys: {missing}. Raw response: {raw}"
        )

    return {key: str(payload[key]).strip() for key in SECTION_KEYS}


def _extract_caption_fingerprint_with_retries(
    *,
    client: AzureOpenAI,
    deployment: str,
    cfg: Any,
    caption: str,
    case_id: str,
    project_id: str,
    contrast_examples: list[dict[str, str]],
) -> dict[str, str] | None:
    max_retries = int(cfg.azure.max_retries)
    retry_sleep_seconds = float(cfg.azure.retry_sleep_seconds)
    if max_retries <= 0:
        raise ValueError("azure.max_retries must be positive.")
    if retry_sleep_seconds < 0:
        raise ValueError("azure.retry_sleep_seconds must be non-negative.")

    for attempt in range(1, max_retries + 1):
        try:
            return _extract_caption_fingerprint(
                client=client,
                deployment=deployment,
                cfg=cfg,
                case_id=case_id,
                project_id=project_id,
                caption=caption,
                contrast_examples=contrast_examples,
            )
        except Exception as exc:
            print(
                f"case_id={case_id} fingerprint extraction failed "
                f"attempt {attempt}/{max_retries}: {type(exc).__name__}: {exc}",
                flush=True,
            )
            if attempt < max_retries:
                time.sleep(retry_sleep_seconds)

    print(f"Skipping case_id={case_id} after {max_retries} failed fingerprint extraction attempts.", flush=True)
    return None


def _load_existing_output(path: Path, *, resume: bool, overwrite: bool) -> pd.DataFrame:
    if overwrite:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    if not path.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    if not resume:
        raise FileExistsError(f"Output already exists and overwrite=false: {path}")

    existing = pd.read_parquet(path)
    missing = sorted(set(OUTPUT_COLUMNS) - set(existing.columns))
    if missing:
        raise ValueError(f"Existing output is missing columns: {missing}")
    if existing["case_id"].duplicated().any():
        duplicated = existing.loc[existing["case_id"].duplicated(), "case_id"].head(5).tolist()
        raise ValueError(f"Existing output has duplicated case_id values: {duplicated}")
    return existing[OUTPUT_COLUMNS].copy()


def _write_output(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_parquet(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract short case fingerprints from caption_all sections for caption-derived MCQ generation."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("overrides", nargs="*", help="Optional OmegaConf dotlist overrides, e.g. run.overwrite=true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_cfg(_resolve_repo_path(args.config))
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    input_path = _resolve_repo_path(cfg.data.input_parquet_path)
    output_path = _resolve_repo_path(cfg.data.output_parquet_path)

    full_df = _load_input(input_path)
    input_df = _apply_filters_impl(full_df, cfg, apply_first_n=True)
    contrast_pool_df = _apply_filters_impl(full_df, cfg, apply_first_n=False)
    contrast_map = _build_contrast_caption_map(
        contrast_pool_df,
        count=int(cfg.prompt.contrast_examples.count),
    )
    existing_df = _load_existing_output(
        output_path,
        resume=bool(cfg.run.resume),
        overwrite=bool(cfg.run.overwrite),
    )
    completed_case_ids = set(existing_df["case_id"].astype(str).tolist())
    rows_to_process = input_df[~input_df["case_id"].astype(str).isin(completed_case_ids)].copy()

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Selected rows: {len(input_df)}")
    print(f"Existing fingerprint rows: {len(existing_df)}")
    print(f"Rows to process: {len(rows_to_process)}")

    output_rows = existing_df.to_dict(orient="records")
    if rows_to_process.empty:
        print("All selected captions already have extracted fingerprints.")
        return

    client, deployment = _create_client(cfg)
    print(f"Azure deployment: {deployment}")

    save_every = int(cfg.run.save_every)
    if save_every <= 0:
        raise ValueError("run.save_every must be positive.")

    pending_since_save = 0
    for _, row in tqdm(rows_to_process.iterrows(), total=len(rows_to_process), desc="Condensing captions"):
        case_id = str(row["case_id"]).strip()
        project_id = str(row["project_id"]).strip()
        caption = str(row["caption"]).strip()
        if not caption:
            raise ValueError(f"Caption is empty for case_id={case_id}")
        contrast_examples = contrast_map[case_id]

        condensed = _extract_caption_fingerprint_with_retries(
            client=client,
            deployment=deployment,
            cfg=cfg,
            caption=caption,
            case_id=case_id,
            project_id=project_id,
            contrast_examples=contrast_examples,
        )
        if condensed is None:
            continue
        output_rows.append(
            {
                "case_id": case_id,
                "project_id": project_id,
                **condensed,
            }
        )
        pending_since_save += 1
        if pending_since_save >= save_every:
            _write_output(output_path, output_rows)
            pending_since_save = 0

    _write_output(output_path, output_rows)
    print(f"Saved condensed captions: {output_path}")
    print(f"Rows written: {len(output_rows)}")


if __name__ == "__main__":
    main()

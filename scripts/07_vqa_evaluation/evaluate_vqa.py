#!/usr/bin/env python3
from __future__ import annotations

# ruff: noqa: E402

import json
import base64
import mimetypes
import os
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.modeling.path_projectors import resolve_language_model_hidden_size
from kidney_vlm.repo_root import find_repo_root
from kidney_vlm.script_config import load_script_cfg
from kidney_vlm.vqa.constants import MODALITIES
from kidney_vlm.vqa.data import (
    _coerce_token_ids,
    _find_placeholder_span,
    load_modality_feature_tensor,
    pad_optional_feature_tensors,
)
from kidney_vlm.vqa.eval_gpt import (
    add_bertscore_columns,
    apply_group_sampling,
    build_eval_prompt,
    build_flat_metric_records,
    collect_required_image_paths,
    enabled_model_configs,
    parse_model_response,
    question_type_key,
    select_eval_rows,
)
from kidney_vlm.vqa.modeling import (
    OncoVLMVQASFTModel,
    build_language_model,
    build_tokenizer,
    load_projectors,
    move_batch_to_device,
)
from kidney_vlm.vqa.prompts import (
    build_vqa_prompt,
    prefix_placeholder_for_modality,
    row_modalities,
    row_uses_modality,
)
from kidney_vlm.vqa.schema import normalize_vqa_df, validate_vqa_df
from kidney_vlm.vqa.stage_config import as_bool, cfg_get, clean_text, resolve_torch_dtype

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


def load_cfg():
    return load_script_cfg(
        repo_root=ROOT,
        config_relative_path="07_vqa_evaluation/evaluate_vqa_gpt.yaml",
        overrides=sys.argv[1:],
    )


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _read_repo_env_value(name: str) -> str:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return ""
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == name:
            return value.strip().strip('"').strip("'")
    return ""


def _build_azure_client(azure_cfg: Any):
    try:
        from openai import AzureOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "openai is required. Install it with: uv add openai"
        ) from exc

    api_key_env = clean_text(cfg_get(azure_cfg, "api_key_env"))
    if not api_key_env:
        raise ValueError("Azure VQA model config must define azure_openai.api_key_env.")
    api_key = os.getenv(api_key_env, "").strip() or _read_repo_env_value(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing Azure OpenAI key in env var: {api_key_env}")

    return AzureOpenAI(
        api_version=clean_text(cfg_get(azure_cfg, "api_version")),
        azure_endpoint=clean_text(cfg_get(azure_cfg, "endpoint")),
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


def _image_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _build_user_content(
    user_prompt: str, image_paths: list[Path]
) -> str | list[dict[str, Any]]:
    if not image_paths:
        return user_prompt

    content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for image_path in image_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": _image_data_url(image_path),
                },
            }
        )
    return content


def _call_azure_gpt(
    *,
    client: Any,
    azure_cfg: Any,
    system_prompt: str,
    user_prompt: str,
    image_paths: list[Path],
    generation_kwargs: Mapping[str, Any],
) -> str:
    deployment = clean_text(cfg_get(azure_cfg, "deployment"))
    if not deployment:
        raise ValueError("Azure VQA model config must define azure_openai.deployment.")
    retries = int(cfg_get(azure_cfg, "max_retries", 3))
    retry_sleep_seconds = float(cfg_get(azure_cfg, "retry_sleep_seconds", 2.0))
    reasoning_effort = clean_text(cfg_get(azure_cfg, "reasoning_effort"))
    verbosity = clean_text(cfg_get(azure_cfg, "verbosity"))
    max_tokens = int(
        cfg_get(azure_cfg, "max_completion_tokens", generation_kwargs.get("max_new_tokens", 256))
    )

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request_kwargs: dict[str, Any] = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": _build_user_content(user_prompt, image_paths),
                    },
                ],
                "max_completion_tokens": max_tokens,
                "model": deployment,
            }
            if reasoning_effort:
                request_kwargs["reasoning_effort"] = reasoning_effort
            if verbosity:
                request_kwargs["verbosity"] = verbosity
            response = client.chat.completions.create(**request_kwargs)
            text = _extract_text_content(response.choices[0].message.content)
            if not text:
                raise RuntimeError("GPT returned an empty response.")
            return text
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_sleep_seconds)
    raise RuntimeError(
        f"GPT evaluation call failed after {retries} attempts: {last_error}"
    )


def _existing_prediction_ids(predictions_path: Path) -> set[int]:
    if not predictions_path.exists():
        return set()
    existing = pd.read_parquet(predictions_path)
    if "question_id" not in existing.columns:
        return set()
    return {int(value) for value in existing["question_id"].dropna().tolist()}


def _write_predictions(
    predictions_path: Path, rows: list[dict[str, Any]], *, resume_existing: bool
) -> pd.DataFrame:
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    generated = pd.DataFrame(rows)
    if generated.empty:
        if predictions_path.exists() and resume_existing:
            return pd.read_parquet(predictions_path)
        generated.to_parquet(predictions_path, index=False)
        return generated

    if predictions_path.exists() and resume_existing:
        existing = pd.read_parquet(predictions_path)
        if not existing.empty:
            existing = existing[~existing["question_id"].isin(generated["question_id"])]
            final = pd.concat([existing, generated], ignore_index=True)
        else:
            final = generated
    else:
        final = generated
    final = final.sort_values(
        ["project_id", "case_id", "task_id", "question_id"]
    ).reset_index(drop=True)
    final.to_parquet(predictions_path, index=False)
    return final


def _write_metrics(metrics_path: Path, metrics: dict[str, Any]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )


def _resolve_device(device_value: Any) -> torch.device:
    requested = clean_text(device_value) or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {requested!r}, but CUDA is unavailable.")
    return torch.device(requested)


def _model_output_dir(eval_cfg: Mapping[str, Any], model_cfg: Mapping[str, Any]) -> Path:
    run_cfg = dict(eval_cfg.get("run") or {})
    run_name = clean_text(run_cfg.get("name"))
    if not run_name:
        raise ValueError("vqa_evaluation.run.name must be populated.")
    output_root = _resolve_path(run_cfg.get("output_root", "results/vqa_eval"))
    display_name = clean_text(model_cfg.get("display_name"))
    if not display_name:
        raise ValueError("Enabled VQA evaluation model is missing display_name.")
    return output_root / run_name / display_name


def _model_name_or_path(model_cfg: Mapping[str, Any]) -> str:
    if clean_text(model_cfg.get("model_name_or_path")):
        return clean_text(model_cfg.get("model_name_or_path"))
    azure_cfg = dict(model_cfg.get("azure_openai") or {})
    if clean_text(azure_cfg.get("deployment")):
        return clean_text(azure_cfg.get("deployment"))
    return clean_text(model_cfg.get("display_name"))


def _generation_kwargs(eval_cfg: Mapping[str, Any], model_cfg: Mapping[str, Any]) -> dict[str, Any]:
    generation_cfg = dict(eval_cfg.get("generation") or {})
    model_generation_cfg = dict(model_cfg.get("generation") or {})
    generation_cfg.update(model_generation_cfg)
    kwargs: dict[str, Any] = {
        "max_new_tokens": int(generation_cfg.get("max_new_tokens", 128)),
        "do_sample": bool(generation_cfg.get("do_sample", False)),
    }
    temperature = generation_cfg.get("temperature")
    if temperature not in (None, "", "null"):
        kwargs["temperature"] = float(temperature)
    top_p = generation_cfg.get("top_p")
    if top_p not in (None, "", "null"):
        kwargs["top_p"] = float(top_p)
    return kwargs


def _model_batch_size(model_cfg: Mapping[str, Any]) -> int:
    raw_batch_size = model_cfg.get("batch_size", 1)
    if raw_batch_size in (None, "", "null"):
        raw_batch_size = 1
    batch_size = int(raw_batch_size)
    if batch_size < 1:
        raise ValueError(f"Model {model_cfg.get('display_name')} has invalid batch_size={batch_size}.")
    return batch_size


def _batched_records(records: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [records[start : start + batch_size] for start in range(0, len(records), batch_size)]


def _print_first_n_outputs(run_cfg: Mapping[str, Any]) -> int:
    raw_value = run_cfg.get("print_first_n_outputs", 0)
    if raw_value in (None, "", "null"):
        return 0
    count = int(raw_value)
    if count < 0:
        raise ValueError(f"run.print_first_n_outputs must be non-negative, got {count}.")
    return count


def _print_output_preview(
    *,
    preview_index: int,
    preview_limit: int,
    row: Mapping[str, Any],
    raw_response: str,
    parsed: Mapping[str, str],
    model_cfg: Mapping[str, Any],
) -> None:
    answer_label = clean_text(row.get("answer_label"))
    predicted_label = clean_text(parsed.get("predicted_answer_label"))
    label_text = ""
    if answer_label or predicted_label:
        label_text = f" | labels gt={answer_label or '-'} pred={predicted_label or '-'}"
    print(
        f"\n[VQA preview {preview_index}/{preview_limit}] "
        f"model={clean_text(model_cfg.get('display_name'))} "
        f"qid={row.get('question_id')}"
    )
    print(f"Q: {clean_text(row.get('question'))}")
    print(
        f"GT: {clean_text(row.get('answer'))} | "
        f"OUT: {clean_text(raw_response)} | "
        f"parsed: {clean_text(parsed.get('predicted_answer'))}"
        f"{label_text}"
    )


def _prompt_block_for_projector(row: Mapping[str, Any], eval_cfg: Mapping[str, Any]) -> dict[str, str]:
    key = question_type_key(row.get("question_type", ""))
    prompts = eval_cfg.get("prompts") or {}
    block = prompts.get(key)
    if not isinstance(block, Mapping):
        raise ValueError(f"vqa_evaluation.prompts.{key} must be defined.")
    system_prompt = clean_text(block.get("system_prompt"))
    response_instruction = clean_text(block.get("response_instruction"))
    if not system_prompt or not response_instruction:
        raise ValueError(f"vqa_evaluation.prompts.{key} must define system_prompt and response_instruction.")
    return {
        "system_prompt": system_prompt,
        "mcq_response_instruction": response_instruction,
        "open_response_instruction": response_instruction,
    }


def _prompt_token_ids(tokenizer: Any, prompt_text: str, *, enable_thinking: bool = False) -> list[int]:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt_text}]
        kwargs = {"tokenize": True, "add_generation_prompt": True}
        try:
            token_ids = tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
        except TypeError:
            try:
                token_ids = tokenizer.apply_chat_template(
                    messages,
                    chat_template_kwargs={"enable_thinking": enable_thinking},
                    **kwargs,
                )
            except TypeError:
                token_ids = tokenizer.apply_chat_template(messages, **kwargs)
        return _coerce_token_ids(token_ids)
    return _coerce_token_ids(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])


def _hf_message_payload(system_prompt: str, user_prompt: str, image_paths: list[Path]) -> list[dict[str, Any]]:
    from PIL import Image

    content: list[dict[str, Any]] = []
    for image_path in image_paths:
        content.append({"type": "image", "image": Image.open(image_path).convert("RGB")})
    content.append({"type": "text", "text": user_prompt})
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": content},
    ]


class AzureGPTBackend:
    def __init__(self, model_cfg: Mapping[str, Any], *, dry_run: bool):
        self.model_cfg = dict(model_cfg)
        self.azure_cfg = self.model_cfg.get("azure_openai")
        if self.azure_cfg is None:
            raise ValueError(f"Model {self.model_cfg.get('display_name')} is missing azure_openai config.")
        self.client = None if dry_run else _build_azure_client(self.azure_cfg)
        self.dry_run = dry_run

    def generate(self, *, system_prompt: str, user_prompt: str, image_paths: list[Path], generation_kwargs: Mapping[str, Any]) -> str:
        return self.generate_batch(
            requests=[
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "image_paths": image_paths,
                }
            ],
            generation_kwargs=generation_kwargs,
        )[0]

    def generate_batch(self, *, requests: list[Mapping[str, Any]], generation_kwargs: Mapping[str, Any]) -> list[str]:
        if self.dry_run:
            return ['{"answer": "", "rationale": "dry run"}' for _ in requests]
        return [
            _call_azure_gpt(
                client=self.client,
                azure_cfg=self.azure_cfg,
                system_prompt=clean_text(request.get("system_prompt")),
                user_prompt=clean_text(request.get("user_prompt")),
                image_paths=list(request.get("image_paths", [])),
                generation_kwargs=generation_kwargs,
            )
            for request in requests
        ]


class HFImageTextBackend:
    def __init__(self, model_cfg: Mapping[str, Any], *, dry_run: bool):
        self.model_cfg = dict(model_cfg)
        self.dry_run = dry_run
        self.device = _resolve_device(self.model_cfg.get("device"))
        self.processor = None
        self.model = None
        if dry_run:
            return
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("transformers is required for HF image-text VQA evaluation.") from exc

        model_name = _model_name_or_path(self.model_cfg)
        dtype = resolve_torch_dtype(self.model_cfg.get("torch_dtype"))
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": bool(self.model_cfg.get("trust_remote_code", True)),
        }
        if dtype is not None:
            model_kwargs["dtype"] = dtype
        if clean_text(self.model_cfg.get("attn_implementation")):
            model_kwargs["attn_implementation"] = clean_text(self.model_cfg.get("attn_implementation"))
        if self.model_cfg.get("device_map") not in (None, "", "null"):
            model_kwargs["device_map"] = self.model_cfg.get("device_map")
        load_in_8bit = bool(self.model_cfg.get("load_in_8bit", False))
        if load_in_8bit:
            if self.device.type != "cuda":
                raise RuntimeError("HF 8-bit image-text VLM evaluation requires a CUDA device.")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            if "device_map" not in model_kwargs:
                model_kwargs["device_map"] = {"": self.device.index or 0}
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=bool(self.model_cfg.get("trust_remote_code", True)),
        )
        self.model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)
        if "device_map" not in model_kwargs and not load_in_8bit:
            self.model.to(self.device)
        self.model.eval()

    def generate(self, *, system_prompt: str, user_prompt: str, image_paths: list[Path], generation_kwargs: Mapping[str, Any]) -> str:
        return self.generate_batch(
            requests=[
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "image_paths": image_paths,
                }
            ],
            generation_kwargs=generation_kwargs,
        )[0]

    def generate_batch(self, *, requests: list[Mapping[str, Any]], generation_kwargs: Mapping[str, Any]) -> list[str]:
        if self.dry_run:
            return ['{"answer": "", "rationale": "dry run"}' for _ in requests]
        messages = [
            _hf_message_payload(
                clean_text(request.get("system_prompt")),
                clean_text(request.get("user_prompt")),
                list(request.get("image_paths", [])),
            )
            for request in requests
        ]
        message_payload = messages[0] if len(messages) == 1 else messages
        inputs = self.processor.apply_chat_template(
            message_payload,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(getattr(self.model, "device", self.device))
        input_len = int(inputs["input_ids"].shape[1]) if "input_ids" in inputs else 0
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **dict(generation_kwargs))
        generated = outputs[:, input_len:] if input_len and outputs.shape[1] > input_len else outputs
        return [text.strip() for text in self.processor.batch_decode(generated, skip_special_tokens=True)]


def _oncovlm_stage_cfg(model_cfg: Mapping[str, Any]) -> dict[str, Any]:
    stage_cfg = dict(model_cfg)
    stage_cfg["enable_thinking"] = as_bool(stage_cfg.get("enable_thinking", False))
    projectors = dict(stage_cfg.get("projectors") or {})
    missing = [modality for modality in MODALITIES if modality not in projectors]
    if missing:
        raise ValueError(f"oncovlm_projector model config must define all projectors. Missing: {missing}")
    normalized_projectors: dict[str, dict[str, Any]] = {}
    for modality in MODALITIES:
        block = dict(projectors[modality] or {})
        if not clean_text(block.get("checkpoint_path")):
            raise ValueError(f"oncovlm_projector projectors.{modality}.checkpoint_path is required.")
        block["enabled"] = True
        block["trainable"] = False
        normalized_projectors[modality] = block
    stage_cfg["projectors"] = normalized_projectors
    return stage_cfg


def _pad_prompt_batch(tokenizer: Any, token_id_rows: list[list[int]]) -> dict[str, torch.Tensor]:
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id for VQA evaluation.")
    if not token_id_rows:
        raise ValueError("Cannot pad an empty VQA prompt batch.")
    max_tokens = max(len(token_ids) for token_ids in token_id_rows)
    input_ids = torch.full((len(token_id_rows), max_tokens), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(token_id_rows), max_tokens), dtype=torch.long)
    for row_index, token_ids in enumerate(token_id_rows):
        token_count = len(token_ids)
        input_ids[row_index, :token_count] = torch.tensor(token_ids, dtype=torch.long)
        attention_mask[row_index, :token_count] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _projector_rows_batch(
    *,
    rows: list[Mapping[str, Any]],
    tokenizer: Any,
    eval_cfg: Mapping[str, Any],
    stage_cfg: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    if not rows:
        raise ValueError("OncoVLM projector evaluation received an empty row batch.")

    token_id_rows: list[list[int]] = []
    prefix_spans: list[list[dict[str, int | str]]] = []
    prompt_texts: list[str] = []
    modality_tensors: dict[str, list[torch.Tensor | None]] = {modality: [] for modality in MODALITIES}
    projectors_cfg = stage_cfg["projectors"]
    enable_thinking = as_bool(stage_cfg.get("enable_thinking", False))

    for row in rows:
        prompt_text = build_vqa_prompt(row, _prompt_block_for_projector(row, eval_cfg))
        token_ids = _prompt_token_ids(tokenizer, prompt_text, enable_thinking=enable_thinking)
        spans: list[dict[str, int | str]] = []
        for modality in row_modalities(row):
            start, end = _find_placeholder_span(tokenizer, token_ids, prefix_placeholder_for_modality(modality))
            spans.append({"modality": modality, "start": start, "end": end})
        token_id_rows.append(token_ids)
        prefix_spans.append(spans)
        prompt_texts.append(prompt_text)

        for modality in MODALITIES:
            if row_uses_modality(row, modality):
                modality_tensors[modality].append(load_modality_feature_tensor(ROOT, row, modality, projectors_cfg[modality]))
            else:
                modality_tensors[modality].append(None)

    batch = _pad_prompt_batch(tokenizer, token_id_rows)
    batch["prefix_spans"] = prefix_spans
    for modality, tensors in modality_tensors.items():
        batch.update(pad_optional_feature_tensors(modality, tensors))
    return batch, prompt_texts


class OncoVLMProjectorBackend:
    def __init__(self, model_cfg: Mapping[str, Any], *, eval_cfg: Mapping[str, Any], dry_run: bool):
        self.model_cfg = dict(model_cfg)
        self.eval_cfg = eval_cfg
        self.stage_cfg = _oncovlm_stage_cfg(model_cfg)
        self.dry_run = dry_run
        self.device = _resolve_device(self.stage_cfg.get("device"))
        self.tokenizer = None
        self.model = None
        self.projector_dtype = resolve_torch_dtype(self.stage_cfg.get("projector_dtype", "float32")) or torch.float32
        self.autocast_dtype = resolve_torch_dtype(self.stage_cfg.get("autocast_dtype", "bfloat16")) or torch.bfloat16
        if dry_run:
            return
        self.tokenizer = build_tokenizer(
            _model_name_or_path(self.stage_cfg),
            trust_remote_code=bool(self.stage_cfg.get("trust_remote_code", True)),
        )
        language_model = build_language_model(self.stage_cfg, device=self.device)
        hidden_size = resolve_language_model_hidden_size(language_model)
        projectors, projector_metadata = load_projectors(self.stage_cfg, repo_root=ROOT, hidden_size=hidden_size)
        self.model = OncoVLMVQASFTModel(
            language_model=language_model,
            projectors=projectors,
            projector_metadata=projector_metadata,
        )
        if not bool(self.stage_cfg.get("load_in_8bit", False)):
            self.model.to(self.device)
        self.model.move_projectors_to(self.device, dtype=self.projector_dtype)
        self.model.eval()
        self.model.set_frozen_projectors_eval()

    def generate(self, *, row: Mapping[str, Any], generation_kwargs: Mapping[str, Any]) -> tuple[str, str]:
        return self.generate_batch(rows=[row], generation_kwargs=generation_kwargs)[0]

    def generate_batch(self, *, rows: list[Mapping[str, Any]], generation_kwargs: Mapping[str, Any]) -> list[tuple[str, str]]:
        if self.dry_run:
            return [('{"answer": "", "rationale": "dry run"}', "") for _ in rows]
        batch, prompt_texts = _projector_rows_batch(
            rows=rows,
            tokenizer=self.tokenizer,
            eval_cfg=self.eval_cfg,
            stage_cfg=self.stage_cfg,
        )
        batch = move_batch_to_device(batch, self.device, floating_dtype=self.projector_dtype)
        use_autocast = self.device.type == "cuda" and self.autocast_dtype != torch.float32
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=use_autocast):
            inputs = self.model.prepare_interleaved_generation_inputs(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pathology_features=batch.get("pathology_features"),
                pathology_feature_mask=batch.get("pathology_feature_mask"),
                radiology_features=batch.get("radiology_features"),
                radiology_feature_mask=batch.get("radiology_feature_mask"),
                dnam_features=batch.get("dnam_features"),
                dnam_feature_mask=batch.get("dnam_feature_mask"),
                rna_features=batch.get("rna_features"),
                rna_feature_mask=batch.get("rna_feature_mask"),
                prefix_spans=batch["prefix_spans"],
            )
            generate_kwargs = dict(generation_kwargs)
            generate_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
            if self.tokenizer.eos_token_id is not None:
                generate_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)
            prompt_len = int(inputs["input_ids"].shape[1])
            try:
                outputs = self.model.language_model.generate(**inputs, **generate_kwargs)
                generated = outputs[:, prompt_len:] if outputs.shape[1] > prompt_len else outputs
            except (TypeError, ValueError):
                inputs_without_ids = {key: value for key, value in inputs.items() if key != "input_ids"}
                outputs = self.model.language_model.generate(**inputs_without_ids, **generate_kwargs)
                generated = outputs
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [(text.strip(), prompt_text) for text, prompt_text in zip(decoded, prompt_texts, strict=True)]


def _build_backend(model_cfg: Mapping[str, Any], *, eval_cfg: Mapping[str, Any], dry_run: bool):
    backend = clean_text(model_cfg.get("backend"))
    if backend == "azure_openai_gpt":
        return AzureGPTBackend(model_cfg, dry_run=dry_run)
    if backend == "hf_image_text_to_text":
        return HFImageTextBackend(model_cfg, dry_run=dry_run)
    if backend == "oncovlm_projector":
        return OncoVLMProjectorBackend(model_cfg, eval_cfg=eval_cfg, dry_run=dry_run)
    if backend in {"hf_causal_lm", "oncovlm_lora"}:
        raise NotImplementedError(f"VQA evaluation backend {backend!r} is intentionally not supported yet.")
    raise ValueError(f"Unsupported VQA evaluation backend: {backend!r}")


def _prediction_row(
    *,
    row: Mapping[str, Any],
    parsed: Mapping[str, str],
    raw_response: str,
    image_paths: list[Path],
    model_cfg: Mapping[str, Any],
    evaluated_at: str,
    system_prompt: str,
    user_prompt: str,
    include_prompt: bool,
) -> dict[str, Any]:
    predicted_answer = clean_text(parsed.get("predicted_answer"))
    correct = bool(predicted_answer and predicted_answer == clean_text(row.get("answer")))
    output = {
        "question_id": int(row["question_id"]),
        "base_question_id": int(row["base_question_id"]),
        "case_id": clean_text(row.get("case_id")),
        "project_id": clean_text(row.get("project_id")),
        "split": clean_text(row.get("split")),
        "question_type": clean_text(row.get("question_type")),
        "generation_type": clean_text(row.get("generation_type")),
        "task_category": clean_text(row.get("task_category")),
        "task_id": clean_text(row.get("task_id")),
        "modality_combination_name": clean_text(row.get("modality_combination_name")),
        "use_pathology": bool(row.get("use_pathology")),
        "use_radiology": bool(row.get("use_radiology")),
        "use_dnam": bool(row.get("use_dnam")),
        "use_rna": bool(row.get("use_rna")),
        "question": clean_text(row.get("question")),
        "option_a": clean_text(row.get("option_a")),
        "option_b": clean_text(row.get("option_b")),
        "option_c": clean_text(row.get("option_c")),
        "option_d": clean_text(row.get("option_d")),
        "answer": clean_text(row.get("answer")),
        "answer_label": clean_text(row.get("answer_label")),
        "predicted_answer": predicted_answer,
        "predicted_answer_label": clean_text(parsed.get("predicted_answer_label")),
        "parse_status": clean_text(parsed.get("parse_status")),
        "correct": correct,
        "raw_response": raw_response,
        "image_paths": [str(path) for path in image_paths],
        "backend": clean_text(model_cfg.get("backend")),
        "model_display_name": clean_text(model_cfg.get("display_name")),
        "model_name_or_path": _model_name_or_path(model_cfg),
        "evaluated_at": evaluated_at,
    }
    if include_prompt:
        output["system_prompt"] = system_prompt
        output["user_prompt"] = user_prompt
    return output


def _run_payload(
    *,
    eval_cfg: Mapping[str, Any],
    model_cfg: Mapping[str, Any],
    vqa_path: Path,
    predictions_path: Path,
    evaluated_at: str,
) -> dict[str, Any]:
    run_cfg = dict(eval_cfg.get("run") or {})
    return {
        "name": clean_text(run_cfg.get("name")),
        "model_display_name": clean_text(model_cfg.get("display_name")),
        "backend": clean_text(model_cfg.get("backend")),
        "model_name_or_path": _model_name_or_path(model_cfg),
        "vqa_parquet_path": str(vqa_path),
        "predictions_path": str(predictions_path),
        "evaluated_at": evaluated_at,
        "filters": eval_cfg.get("filters", {}),
        "sampling": eval_cfg.get("sampling", {}),
    }


def main() -> None:
    cfg = load_cfg()
    eval_cfg = cfg.vqa_evaluation
    eval_dict = OmegaConf.to_container(eval_cfg, resolve=True)
    if not isinstance(eval_dict, dict):
        raise TypeError("Resolved VQA evaluation config must be a mapping.")

    run_cfg = dict(eval_dict.get("run") or {})
    data_cfg = dict(eval_dict.get("data") or {})
    if not run_cfg or not data_cfg:
        raise ValueError("VQA evaluation config must use the v2 run/data/models/prompts sections.")

    vqa_path_value = clean_text(data_cfg.get("vqa_parquet_path"))
    if not vqa_path_value:
        raise ValueError("vqa_evaluation.data.vqa_parquet_path must be populated.")
    vqa_path = _resolve_path(vqa_path_value)

    vqa_df = normalize_vqa_df(pd.read_parquet(vqa_path))
    validate_vqa_df(vqa_df)
    selected_df = select_eval_rows(vqa_df, eval_dict)
    selected_df = apply_group_sampling(selected_df, eval_dict.get("sampling", {}))
    if selected_df.empty:
        raise RuntimeError(
            "No VQA rows selected for evaluation. Check filters in the YAML."
        )

    print(f"VQA parquet: {vqa_path}")
    print(f"Rows selected: {len(selected_df)}")
    enabled_models = enabled_model_configs(eval_dict.get("models") or {})
    print(f"Enabled models: {', '.join(model['display_name'] for model in enabled_models)}")

    resume_existing = bool(run_cfg.get("resume_existing", True))
    dry_run = bool(run_cfg.get("dry_run", False))
    include_prompt = bool(run_cfg.get("include_prompt_in_predictions", False))
    print_first_n_outputs = _print_first_n_outputs(run_cfg)

    for model_cfg in enabled_models:
        model_dir = _model_output_dir(eval_dict, model_cfg)
        predictions_path = model_dir / "predictions.parquet"
        metrics_path = model_dir / "metrics.json"
        existing_ids = (
            _existing_prediction_ids(predictions_path)
            if resume_existing
            else set()
        )
        eval_df = selected_df[
            ~selected_df["question_id"].astype(int).isin(existing_ids)
        ].reset_index(drop=True)
        evaluated_at = datetime.now(timezone.utc).isoformat()
        print("\nVQA evaluation model")
        print(f"  Model: {model_cfg['display_name']}")
        print(f"  Backend: {model_cfg['backend']}")
        print(f"  Rows skipped from existing predictions: {len(selected_df) - len(eval_df)}")
        print(f"  Rows to evaluate: {len(eval_df)}")
        print(f"  Predictions path: {predictions_path}")
        print(f"  Metrics path: {metrics_path}")

        backend = _build_backend(model_cfg, eval_cfg=eval_dict, dry_run=dry_run)
        prediction_rows: list[dict[str, Any]] = []
        generation_kwargs = _generation_kwargs(eval_dict, model_cfg)
        batch_size = _model_batch_size(model_cfg)
        print(f"  Batch size: {batch_size}")
        row_records = eval_df.to_dict(orient="records")
        previews_printed = 0

        progress = tqdm(
            total=len(row_records),
            desc=f"Evaluating {model_cfg['display_name']}",
            unit="question",
        )
        for row_batch in _batched_records(row_records, batch_size):
            if clean_text(model_cfg["backend"]) == "oncovlm_projector":
                raw_outputs = backend.generate_batch(
                    rows=row_batch,
                    generation_kwargs=generation_kwargs,
                )
                for row, (raw_response, projector_prompt) in zip(row_batch, raw_outputs, strict=True):
                    system_prompt, user_prompt = build_eval_prompt(row, eval_dict)
                    if projector_prompt:
                        user_prompt = projector_prompt
                    parsed = parse_model_response(row, raw_response)
                    if dry_run:
                        parsed = {"predicted_answer": "", "predicted_answer_label": "", "parse_status": "dry_run"}
                    if previews_printed < print_first_n_outputs:
                        previews_printed += 1
                        _print_output_preview(
                            preview_index=previews_printed,
                            preview_limit=print_first_n_outputs,
                            row=row,
                            raw_response=raw_response,
                            parsed=parsed,
                            model_cfg=model_cfg,
                        )
                    prediction_rows.append(
                        _prediction_row(
                            row=row,
                            parsed=parsed,
                            raw_response=raw_response,
                            image_paths=[],
                            model_cfg=model_cfg,
                            evaluated_at=evaluated_at,
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            include_prompt=include_prompt,
                        )
                    )
            else:
                requests: list[dict[str, Any]] = []
                for row in row_batch:
                    system_prompt, user_prompt = build_eval_prompt(row, eval_dict)
                    image_paths = collect_required_image_paths(row, eval_dict, repo_root=ROOT)
                    requests.append(
                        {
                            "row": row,
                            "system_prompt": system_prompt,
                            "user_prompt": user_prompt,
                            "image_paths": image_paths,
                        }
                    )
                raw_responses = backend.generate_batch(
                    requests=requests,
                    generation_kwargs=generation_kwargs,
                )
                for request, raw_response in zip(requests, raw_responses, strict=True):
                    row = request["row"]
                    parsed = parse_model_response(row, raw_response)
                    if dry_run:
                        parsed = {"predicted_answer": "", "predicted_answer_label": "", "parse_status": "dry_run"}
                    if previews_printed < print_first_n_outputs:
                        previews_printed += 1
                        _print_output_preview(
                            preview_index=previews_printed,
                            preview_limit=print_first_n_outputs,
                            row=row,
                            raw_response=raw_response,
                            parsed=parsed,
                            model_cfg=model_cfg,
                        )
                    prediction_rows.append(
                        _prediction_row(
                            row=row,
                            parsed=parsed,
                            raw_response=raw_response,
                            image_paths=request["image_paths"],
                            model_cfg=model_cfg,
                            evaluated_at=evaluated_at,
                            system_prompt=request["system_prompt"],
                            user_prompt=request["user_prompt"],
                            include_prompt=include_prompt,
                        )
                    )
            progress.update(len(row_batch))
        progress.close()

        final_predictions = _write_predictions(
            predictions_path, prediction_rows, resume_existing=resume_existing
        )
        final_predictions = add_bertscore_columns(
            final_predictions,
            dict(dict(eval_dict.get("metrics") or {}).get("bert_score") or {}),
        )
        final_predictions.to_parquet(predictions_path, index=False)
        metric_records = build_flat_metric_records(
            final_predictions,
            model_display_name=clean_text(model_cfg.get("display_name")),
            backend=clean_text(model_cfg.get("backend")),
            model_name_or_path=_model_name_or_path(model_cfg),
        )
        payload = {
            "run": _run_payload(
                eval_cfg=eval_dict,
                model_cfg=model_cfg,
                vqa_path=vqa_path,
                predictions_path=predictions_path,
                evaluated_at=evaluated_at,
            ),
            "metrics": metric_records,
            "errors": [],
        }
        _write_metrics(metrics_path, payload)
        overall = next((item for item in metric_records if item["metric_group"] == "overall"), {})
        print(f"  Final prediction rows: {len(final_predictions)}")
        print(f"  Overall metrics: {overall}")

        del backend
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

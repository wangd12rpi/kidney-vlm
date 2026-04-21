#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.data.registry_io import read_parquet_or_empty, write_registry_parquet
from kidney_vlm.repo_root import find_repo_root
from kidney_vlm.script_config import load_script_cfg

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


@dataclass(frozen=True)
class DnamExtractionTask:
    row_index: int
    sample_id: str
    source: str
    project_id: str
    patient_id: str
    raw_beta_path: Path
    raw_beta_path_value: str
    feature_path: Path
    feature_path_value: str


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if OmegaConf.is_config(value):
        return _as_list(OmegaConf.to_container(value, resolve=True))
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


def _string_list(values: Any) -> list[str]:
    items: list[str] = []
    for value in _as_list(values):
        text = str(value).strip()
        if text and text not in items:
            items.append(text)
    return items


def _is_blank_cell(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {"", "nan", "none", "<na>"}


def _normalize_local_path(path_value: str | Path, *, root_dir: Path) -> Path:
    path = Path(str(path_value).strip()).expanduser()
    if path.is_absolute():
        return path
    return (root_dir / path).resolve()


def _to_repo_relative(path: str | Path, *, root_dir: Path) -> str:
    path_obj = Path(path).expanduser()
    if not path_obj.is_absolute():
        return path_obj.as_posix().lstrip("/")
    resolved = path_obj.resolve()
    try:
        return resolved.relative_to(root_dir).as_posix()
    except ValueError:
        return resolved.as_posix().lstrip("/")


def _sanitize_filename_component(value: str, *, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    cleaned = cleaned.strip("._")
    return cleaned or fallback


def _feature_output_path(
    *,
    output_root: Path,
    source: str,
    project_id: str,
    sample_id: str,
    patient_id: str,
    raw_beta_path: Path,
) -> Path:
    source_token = _sanitize_filename_component(source or "unknown_source", fallback="unknown_source")
    project_token = _sanitize_filename_component(project_id or "unknown_project", fallback="unknown_project")
    sample_token = _sanitize_filename_component(patient_id or sample_id, fallback="unknown_sample")
    raw_token = _sanitize_filename_component(raw_beta_path.stem, fallback="dnam_beta")
    return output_root / source_token / project_token / f"{sample_token}__{raw_token}.pt"


def build_extraction_tasks(
    registry_df: pd.DataFrame,
    *,
    root_dir: Path,
    output_root: Path,
    raw_paths_column: str,
    feature_path_column: str,
    sources: list[str],
    project_ids: list[str],
    splits: list[str],
    selected_raw_path_index: int,
    overwrite_existing_registry_paths: bool,
    fail_on_missing_raw_file: bool,
    max_rows: int | None,
) -> list[DnamExtractionTask]:
    required_columns = {"sample_id", "source", "project_id", "patient_id", raw_paths_column, feature_path_column}
    missing = sorted(required_columns.difference(registry_df.columns))
    if missing:
        raise ValueError(f"Unified registry is missing required DNAm extraction columns: {missing}")

    source_set = {source.lower() for source in sources if source.strip()}
    project_set = {project for project in project_ids if project.strip()}
    split_set = {split for split in splits if split.strip()}
    tasks: list[DnamExtractionTask] = []

    for row_index, row in registry_df.iterrows():
        source = str(row.get("source", "")).strip()
        if source_set and source.lower() not in source_set:
            continue
        project_id = str(row.get("project_id", "")).strip()
        if project_set and project_id not in project_set:
            continue
        split = str(row.get("split", "")).strip()
        if split_set and split not in split_set:
            continue
        current_feature_path = (
            ""
            if _is_blank_cell(row.get(feature_path_column, ""))
            else str(row.get(feature_path_column)).strip()
        )
        if current_feature_path and not overwrite_existing_registry_paths:
            continue

        raw_path_values = _as_list(row.get(raw_paths_column))
        if not raw_path_values:
            continue
        if selected_raw_path_index >= len(raw_path_values):
            raise IndexError(
                f"selected_raw_path_index={selected_raw_path_index} is out of range for "
                f"sample_id={row.get('sample_id')!r} with {len(raw_path_values)} DNAm raw paths."
            )

        raw_path_value = raw_path_values[selected_raw_path_index]
        raw_beta_path = _normalize_local_path(raw_path_value, root_dir=root_dir)
        if not raw_beta_path.exists():
            if fail_on_missing_raw_file:
                raise FileNotFoundError(f"DNAm raw beta file not found for sample {row.get('sample_id')}: {raw_beta_path}")
            continue

        sample_id = str(row.get("sample_id", "")).strip()
        patient_id = str(row.get("patient_id", "")).strip()
        feature_path = _feature_output_path(
            output_root=output_root,
            source=source,
            project_id=project_id,
            sample_id=sample_id,
            patient_id=patient_id,
            raw_beta_path=raw_beta_path,
        )
        tasks.append(
            DnamExtractionTask(
                row_index=int(row_index),
                sample_id=sample_id,
                source=source,
                project_id=project_id,
                patient_id=patient_id,
                raw_beta_path=raw_beta_path,
                raw_beta_path_value=raw_path_value,
                feature_path=feature_path,
                feature_path_value=_to_repo_relative(feature_path, root_dir=root_dir),
            )
        )
        if max_rows is not None and len(tasks) >= max_rows:
            break

    return tasks


def _load_vocab_sites(vocab_path: Path) -> list[str]:
    import json

    data = json.loads(vocab_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [str(value) for value in data]
    for key in ("input", "sites", "var_names", "features"):
        values = data.get(key)
        if isinstance(values, list):
            return [str(value) for value in values]
    raise ValueError(f"Unrecognized CpGPT vocabulary format in {vocab_path}")


def _write_feather_chunk(
    tasks: list[DnamExtractionTask],
    *,
    feather_path: Path,
    vocab_sites: list[str],
    dropna: bool,
) -> None:
    vocab_set = set(vocab_sites)
    rows: list[dict[str, Any]] = []
    for task in tqdm(tasks, desc=f"Reading DNAm beta files for {feather_path.stem}", unit="file", leave=False):
        beta_df = pd.read_csv(
            task.raw_beta_path,
            sep="\t",
            header=None,
            names=["CpG_Site", "Beta_Value"],
            na_values=["NA", "NaN", "nan", ""],
            dtype={"CpG_Site": str},
        )
        beta_df["Beta_Value"] = pd.to_numeric(beta_df["Beta_Value"], errors="coerce")
        if dropna:
            beta_df = beta_df.dropna(subset=["Beta_Value"])
        beta_df = beta_df[beta_df["CpG_Site"].isin(vocab_set)]

        row: dict[str, Any] = {"CASE_ID": task.sample_id or str(task.row_index)}
        if not beta_df.empty:
            row.update(beta_df.set_index("CpG_Site")["Beta_Value"].to_dict())
        rows.append(row)

    combined = pd.DataFrame(rows)
    present_cols = [cpg for cpg in vocab_sites if cpg in combined.columns]
    combined = combined[["CASE_ID"] + present_cols]
    feather_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_feather(feather_path)


def _ensure_cpgpt_imports(package_root: Path | None) -> dict[str, Any]:
    if package_root is not None and str(package_root).strip():
        resolved_package_root = package_root.expanduser().resolve()
        if not resolved_package_root.exists():
            raise FileNotFoundError(f"Configured CpGPT package_root does not exist: {resolved_package_root}")
        if str(resolved_package_root) not in sys.path:
            sys.path.insert(0, str(resolved_package_root))
    try:
        from cpgpt.data.components.cpgpt_datasaver import CpGPTDataSaver
        from cpgpt.data.components.cpgpt_dataset import CpGPTDataset, cpgpt_data_collate
        from cpgpt.data.components.dna_llm_embedder import DNALLMEmbedder
        from cpgpt.data.components.illumina_methylation_prober import IlluminaMethylationProber
        from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "CpGPT runtime import failed. Run `uv sync` for the added runtime dependencies, "
            "and keep cpgpt.package_root pointed at external/CpGPT unless CpGPT is installed normally. "
            f"Missing module: {exc.name}"
        ) from exc
    return {
        "CpGPTDataSaver": CpGPTDataSaver,
        "CpGPTDataset": CpGPTDataset,
        "cpgpt_data_collate": cpgpt_data_collate,
        "DNALLMEmbedder": DNALLMEmbedder,
        "IlluminaMethylationProber": IlluminaMethylationProber,
        "CpGPTInferencer": CpGPTInferencer,
    }


class CpGPTEmbeddingRunner:
    def __init__(self, *, cpgpt_root: Path, model_name: str, device: str, runtime: dict[str, Any]) -> None:
        self.cpgpt_root = cpgpt_root.expanduser().resolve()
        self.dependencies_dir = self.cpgpt_root / "dependencies"
        self.data_dir = self.cpgpt_root / "data"
        model_root = self.dependencies_dir / "model"
        model_ckpt = model_root / "weights" / f"{model_name}.ckpt"
        model_cfg = model_root / "config" / f"{model_name}.yaml"
        model_vocab = model_root / "vocab" / f"{model_name}.json"
        for path in (model_ckpt, model_cfg, model_vocab):
            if not path.exists():
                raise FileNotFoundError(f"Missing CpGPT model dependency: {path}")

        inferencer = runtime["CpGPTInferencer"](
            dependencies_dir=str(self.dependencies_dir),
            data_dir=str(self.data_dir),
        )
        self.config = inferencer.load_cpgpt_config(str(model_cfg))
        self.model = inferencer.load_cpgpt_model(self.config, model_ckpt_path=str(model_ckpt), strict_load=True)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.embedding_dim = int(getattr(self.config.model.net, "d_embedding", 128))

    def encode_batch(self, batch: dict[str, Any]) -> torch.Tensor:
        for key, value in list(batch.items()):
            if torch.is_tensor(value):
                batch[key] = value.to(self.device)

        prev_predict_mode = getattr(self.model, "predict_mode_predict", None)
        prev_return_keys = getattr(self.model, "return_keys_predict", None)
        prev_training = self.model.training
        setattr(self.model, "predict_mode_predict", "forward")
        setattr(self.model, "return_keys_predict", ["sample_embedding"])
        try:
            self.model.eval()
            with torch.no_grad():
                pred = self.model.predict_step(batch, batch_idx=0)
        finally:
            if prev_predict_mode is None:
                try:
                    delattr(self.model, "predict_mode_predict")
                except Exception:
                    pass
            else:
                setattr(self.model, "predict_mode_predict", prev_predict_mode)
            if prev_return_keys is None:
                try:
                    delattr(self.model, "return_keys_predict")
                except Exception:
                    pass
            else:
                setattr(self.model, "return_keys_predict", prev_return_keys)
            self.model.train(prev_training)

        embedding = pred.get("sample_embedding", pred.get("sample_embeddings"))
        if embedding is None:
            raise KeyError(f"Unexpected CpGPT predict_step output keys: {list(pred.keys())}")
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.as_tensor(embedding, device=self.device)
        return embedding.float()


def _torch_dtype(name: str) -> torch.dtype:
    normalized = str(name).strip().lower()
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32", "single"}:
        return torch.float32
    raise ValueError(f"Unsupported save dtype: {name}")


def _process_chunk(
    chunk_tasks: list[DnamExtractionTask],
    *,
    chunk_index: int,
    cfg: Any,
    runtime: dict[str, Any],
    runner: CpGPTEmbeddingRunner,
    vocab_sites: list[str],
    save_dtype: torch.dtype,
) -> list[dict[str, Any]]:
    cpgpt_cfg = cfg.dnam_features.cpgpt
    feather_root = Path(str(cpgpt_cfg.feather_root))
    processed_root = Path(str(cpgpt_cfg.processed_root))
    feather_path = feather_root / f"chunk_{chunk_index:05d}.feather"
    processed_dir = processed_root / f"chunk_{chunk_index:05d}"

    if bool(cpgpt_cfg.force_reprocess):
        if feather_path.exists():
            feather_path.unlink()
        if processed_dir.exists():
            shutil.rmtree(processed_dir)

    if not feather_path.exists():
        _write_feather_chunk(
            chunk_tasks,
            feather_path=feather_path,
            vocab_sites=vocab_sites,
            dropna=bool(cpgpt_cfg.dropna),
        )

    human_dir = Path(str(cpgpt_cfg.root_dir)) / "dependencies" / "human"
    for path in (human_dir / "illumina_metadata.db", human_dir / "ensembl_metadata.db"):
        if not path.exists():
            raise FileNotFoundError(f"Missing CpGPT human dependency required for preprocessing: {path}")

    embedder = runtime["DNALLMEmbedder"](dependencies_dir=str(human_dir))
    prober = runtime["IlluminaMethylationProber"](dependencies_dir=str(human_dir), embedder=embedder)
    datasaver = runtime["CpGPTDataSaver"](data_paths=[str(feather_path)], processed_dir=str(processed_dir))
    datasaver.process_files(prober, embedder)

    dataset = runtime["CpGPTDataset"](
        embedder,
        processed_dir=str(processed_dir),
        max_length=int(cpgpt_cfg.max_length),
        sorting_strategy=str(cpgpt_cfg.sorting_strategy),
        dna_context_len=int(cpgpt_cfg.dna_context_len),
        dna_llm=str(cpgpt_cfg.dna_llm),
        seed=int(cpgpt_cfg.seed),
    )
    if len(dataset) != len(chunk_tasks):
        raise RuntimeError(
            f"CpGPT processed chunk length mismatch for chunk {chunk_index}: "
            f"dataset={len(dataset)} tasks={len(chunk_tasks)}"
        )

    loader = DataLoader(
        dataset,
        batch_size=int(cpgpt_cfg.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=runtime["cpgpt_data_collate"],
    )
    manifest_rows: list[dict[str, Any]] = []
    task_offset = 0
    for batch in tqdm(loader, desc=f"Embedding DNAm chunk {chunk_index:05d}", unit="batch", leave=False):
        embeddings = runner.encode_batch(batch)
        if bool(cpgpt_cfg.normalize_output):
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        embeddings = embeddings.detach().cpu()
        batch_tasks = chunk_tasks[task_offset : task_offset + embeddings.shape[0]]
        task_offset += embeddings.shape[0]

        for task, embedding in zip(batch_tasks, embeddings, strict=True):
            task.feature_path.parent.mkdir(parents=True, exist_ok=True)
            tensor_to_save = embedding.unsqueeze(0).to(dtype=save_dtype)
            torch.save(tensor_to_save, task.feature_path)
            manifest_rows.append(_manifest_row(task, tensor_to_save))
    return manifest_rows


def _manifest_row(task: DnamExtractionTask, saved_tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "sample_id": task.sample_id,
        "source": task.source,
        "project_id": task.project_id,
        "patient_id": task.patient_id,
        "raw_beta_path": _to_repo_relative(task.raw_beta_path, root_dir=ROOT),
        "raw_beta_path_value": task.raw_beta_path_value,
        "feature_path": task.feature_path_value,
        "feature_shape": list(saved_tensor.shape),
        "feature_dtype": str(saved_tensor.dtype).replace("torch.", ""),
    }


def update_registry_with_manifest(
    registry_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    *,
    feature_path_column: str,
    overwrite_existing_registry_paths: bool,
) -> pd.DataFrame:
    out = registry_df.copy()
    required_columns = {"sample_id", "feature_path"}
    missing = sorted(required_columns.difference(manifest_df.columns))
    if missing:
        raise ValueError(f"DNAm manifest is missing required columns for registry update: {missing}")
    feature_by_sample_id = {
        str(row.sample_id).strip(): str(row.feature_path).strip()
        for row in manifest_df.itertuples(index=False)
        if str(row.sample_id).strip() and str(row.feature_path).strip()
    }
    for row_index, row in out.iterrows():
        sample_id = str(row.get("sample_id", "")).strip()
        feature_path = feature_by_sample_id.get(sample_id)
        if not feature_path:
            continue
        current = (
            ""
            if _is_blank_cell(row.get(feature_path_column, ""))
            else str(row.get(feature_path_column)).strip()
        )
        if current and not overwrite_existing_registry_paths:
            continue
        out.at[row_index, feature_path_column] = feature_path
    return out


def _write_manifest(rows: list[dict[str, Any]], *, manifest_path: Path, csv_manifest_path: Path, write_csv: bool) -> pd.DataFrame:
    manifest_df = pd.DataFrame(rows)
    if not manifest_df.empty:
        manifest_df = manifest_df.sort_values(by=["source", "project_id", "patient_id", "sample_id"], kind="stable")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_parquet(manifest_path, index=False)
    if write_csv:
        csv_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_df.to_csv(csv_manifest_path, index=False)
    return manifest_df


def _existing_feature_manifest_rows(tasks: list[DnamExtractionTask]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        tensor = torch.load(task.feature_path, map_location="cpu")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Existing DNAm feature file does not contain a tensor: {task.feature_path}")
        rows.append(_manifest_row(task, tensor))
    return rows


def _chunks(values: list[DnamExtractionTask], chunk_size: int) -> list[list[DnamExtractionTask]]:
    if chunk_size <= 0:
        raise ValueError("preprocess_chunk_size must be positive.")
    return [values[start : start + chunk_size] for start in range(0, len(values), chunk_size)]


def main() -> None:
    cfg = load_script_cfg(
        repo_root=ROOT,
        config_relative_path="03_dnam_features/01_extract_cpgpt_dnam_features.yaml",
        overrides=sys.argv[1:],
    )
    feature_cfg = cfg.dnam_features
    cpgpt_cfg = feature_cfg.cpgpt
    input_cfg = feature_cfg.input
    output_cfg = feature_cfg.output

    registry_path = Path(str(feature_cfg.registry_path))
    if not registry_path.exists():
        raise FileNotFoundError(f"Unified registry not found: {registry_path}")
    registry_df = read_parquet_or_empty(registry_path)
    if registry_df.empty:
        raise RuntimeError(f"Unified registry is empty: {registry_path}")

    max_rows = None if feature_cfg.max_rows is None else int(feature_cfg.max_rows)
    tasks = build_extraction_tasks(
        registry_df,
        root_dir=ROOT,
        output_root=Path(str(output_cfg.features_root)),
        raw_paths_column=str(input_cfg.raw_paths_column),
        feature_path_column=str(input_cfg.feature_path_column),
        sources=_string_list(feature_cfg.sources),
        project_ids=_string_list(feature_cfg.project_ids),
        splits=_string_list(feature_cfg.splits),
        selected_raw_path_index=int(input_cfg.selected_raw_path_index),
        overwrite_existing_registry_paths=bool(feature_cfg.overwrite_existing_registry_paths),
        fail_on_missing_raw_file=bool(input_cfg.fail_on_missing_raw_file),
        max_rows=max_rows,
    )
    if not tasks:
        print("No DNAm rows need CpGPT embedding extraction.")
        return

    print(f"Registry path: {registry_path}")
    print(f"Selected DNAm rows: {len(tasks)}")
    print(f"Sources: {_string_list(feature_cfg.sources)}")
    print(f"CpGPT root: {Path(str(cpgpt_cfg.root_dir))}")
    print(f"CpGPT model: {cpgpt_cfg.model_name}")
    print(f"Output features root: {Path(str(output_cfg.features_root))}")

    existing_tasks: list[DnamExtractionTask] = []
    pending_tasks: list[DnamExtractionTask] = []
    for task in tasks:
        if task.feature_path.exists() and bool(feature_cfg.skip_existing_feature_files):
            existing_tasks.append(task)
        else:
            pending_tasks.append(task)

    manifest_rows = _existing_feature_manifest_rows(existing_tasks)
    if pending_tasks:
        runtime = _ensure_cpgpt_imports(Path(str(cpgpt_cfg.package_root)) if str(cpgpt_cfg.package_root).strip() else None)
        vocab_path = Path(str(cpgpt_cfg.root_dir)) / "dependencies" / "model" / "vocab" / f"{cpgpt_cfg.model_name}.json"
        vocab_sites = _load_vocab_sites(vocab_path)
        runner = CpGPTEmbeddingRunner(
            cpgpt_root=Path(str(cpgpt_cfg.root_dir)),
            model_name=str(cpgpt_cfg.model_name),
            device=str(cpgpt_cfg.device),
            runtime=runtime,
        )
        save_dtype = _torch_dtype(str(cpgpt_cfg.save_dtype))
        task_chunks = _chunks(pending_tasks, int(cpgpt_cfg.preprocess_chunk_size))
        for chunk_index, chunk_tasks in enumerate(tqdm(task_chunks, desc="CpGPT DNAm extraction chunks", unit="chunk")):
            manifest_rows.extend(
                _process_chunk(
                    chunk_tasks,
                    chunk_index=chunk_index,
                    cfg=cfg,
                    runtime=runtime,
                    runner=runner,
                    vocab_sites=vocab_sites,
                    save_dtype=save_dtype,
                )
            )

    manifest_df = _write_manifest(
        manifest_rows,
        manifest_path=Path(str(output_cfg.manifest_path)),
        csv_manifest_path=Path(str(output_cfg.csv_manifest_path)),
        write_csv=bool(output_cfg.write_csv_manifest),
    )

    if bool(feature_cfg.update_registry):
        updated_df = update_registry_with_manifest(
            registry_df,
            manifest_df,
            feature_path_column=str(input_cfg.feature_path_column),
            overwrite_existing_registry_paths=bool(feature_cfg.overwrite_existing_registry_paths),
        )
        write_registry_parquet(updated_df, registry_path, validate=True)

    print("CpGPT DNAm embedding extraction complete.")
    print(f"Existing feature files reused: {len(existing_tasks)}")
    print(f"New feature files written: {len(pending_tasks)}")
    print(f"Manifest rows: {len(manifest_df)}")
    print(f"Manifest parquet: {Path(str(output_cfg.manifest_path))}")
    if bool(output_cfg.write_csv_manifest):
        print(f"Manifest csv: {Path(str(output_cfg.csv_manifest_path))}")
    if bool(feature_cfg.update_registry):
        print(f"Updated registry: {registry_path}")


if __name__ == "__main__":
    main()

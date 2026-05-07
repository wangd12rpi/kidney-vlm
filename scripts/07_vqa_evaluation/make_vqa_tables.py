#!/usr/bin/env python3
from __future__ import annotations

# ruff: noqa: E402

import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.repo_root import find_repo_root
from kidney_vlm.script_config import load_script_cfg
from kidney_vlm.vqa.stage_config import clean_text

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)

VRULE_TOKEN = "_vrule"

MAIN_GROUPS = [
    {
        "id": "mcq_from_ground_truth",
        "default_label": "MCQ from ground truth",
        "question_type": "mcq",
        "generation_type": "from_ground_truth",
        "metric": "mcq",
    },
    {
        "id": "mcq_from_caption",
        "default_label": "MCQ from caption",
        "question_type": "mcq",
        "generation_type": "from_caption",
        "metric": "mcq",
    },
    {
        "id": "qa_from_caption",
        "default_label": "Open-ended from caption",
        "question_type": "qa",
        "generation_type": "from_caption",
        "metric": "qa",
    },
]

TABLE2_GROUPS = [
    {
        "id": "mcq_all",
        "default_label": "MCQ",
        "question_type": "mcq",
        "generation_type": None,
        "metric": "mcq",
    },
    {
        "id": "qa_all",
        "default_label": "Open-ended",
        "question_type": "qa",
        "generation_type": None,
        "metric": "qa",
    },
]


def load_cfg():
    return load_script_cfg(
        repo_root=ROOT,
        config_relative_path="07_vqa_evaluation/make_vqa_tables.yaml",
        overrides=sys.argv[1:],
    )


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _run_root(table_cfg: Mapping[str, Any]) -> Path:
    run_cfg = dict(table_cfg.get("run") or {})
    run_name = clean_text(run_cfg.get("name"))
    if not run_name:
        raise ValueError("vqa_tables.run.name must be populated.")
    output_root = _resolve_path(run_cfg.get("output_root", "results"))
    return output_root / run_name


def _run_filename(table_cfg: Mapping[str, Any], key: str, default: str) -> str:
    run_cfg = dict(table_cfg.get("run") or {})
    value = clean_text(run_cfg.get(key)) or default
    if "/" in value or "\\" in value:
        raise ValueError(f"vqa_tables.run.{key} must be a file name, got {value!r}.")
    return value


def _tables_dir(table_cfg: Mapping[str, Any]) -> Path:
    dirname = _run_filename(table_cfg, "tables_dirname", "tables")
    return _run_root(table_cfg) / dirname


def _predictions_path(table_cfg: Mapping[str, Any]) -> Path:
    return _run_root(table_cfg) / _run_filename(table_cfg, "prediction_filename", "predictions.parquet")


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _latex_escape(value: Any) -> str:
    text = clean_text(value)
    return _latex_escape_raw(text)


def _latex_escape_raw(value: Any) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _bool_icon(value: Any) -> str:
    if isinstance(value, bool):
        return r"\faCheckCircle" if value else r"\faMinus"
    return clean_text(value) or "--"


def _model_metadata(model_display: Mapping[str, Any], model: str) -> dict[str, str]:
    raw = model_display.get(model)
    if raw is None:
        return {"name": model, "backbone": "--", "finetuned": "--", "use_projector": "--"}
    if not isinstance(raw, Mapping):
        return {"name": clean_text(raw) or model, "backbone": "--", "finetuned": "--", "use_projector": "--"}

    name = clean_text(raw.get("name")) or model
    backbone = clean_text(raw.get("backbone")) or "--"
    return {
        "name": name,
        "backbone": backbone,
        "finetuned": _bool_icon(raw.get("finetuned")),
        "use_projector": _bool_icon(raw.get("use_projector")),
    }


def _selected_models(cfg: Mapping[str, Any]) -> set[str]:
    layout_cfg = dict(cfg.get("layout") or {})
    configured = [
        clean_text(model)
        for model in layout_cfg.get("selected_models", ["oncovlm_qwen_lora"])
        if clean_text(model)
    ]
    return set(configured)


def _selected_cell(content: str, *, model: str, selected_models: set[str], colors: Mapping[str, str]) -> str:
    if model not in selected_models:
        return content
    return r"\cellcolor[HTML]{" + clean_text(colors.get("selected_model_bg", "F6F9FF")) + r"}" + content


def _model_is_finetuned(model_display: Mapping[str, Any], model: str) -> bool:
    raw = model_display.get(model)
    return isinstance(raw, Mapping) and bool(raw.get("finetuned", False))


def _model_is_gpt(model_display: Mapping[str, Any], model: str) -> bool:
    raw = model_display.get(model)
    name = clean_text(raw.get("name")) if isinstance(raw, Mapping) else ""
    model_text = f"{model} {name}".lower()
    return "gpt" in model_text


def _model_header_cell(
    content: str,
    *,
    model: str,
    model_display: Mapping[str, Any],
    selected_models: set[str],
    colors: Mapping[str, str],
) -> str:
    if model in selected_models or _model_is_finetuned(model_display, model):
        color = clean_text(colors.get("selected_model_bg", "F6F9FF"))
    elif not _model_is_gpt(model_display, model):
        color = clean_text(colors.get("os_baseline_model_bg", "EAF7F1"))
    else:
        color = ""
    return (r"\cellcolor[HTML]{" + color + r"}" if color else "") + content


def _model_metadata_row(
    label: str,
    key: str,
    models: list[str],
    model_display: Mapping[str, Any],
    *,
    selected_models: set[str],
    colors: Mapping[str, str],
) -> str:
    label_text = _latex_escape(label)
    cells = [r"\textbf{" + label_text + r"}" if label_text else ""]
    for model in models:
        value = _model_metadata(model_display, model)[key]
        content = value if key in {"finetuned", "use_projector"} else _latex_escape(value)
        cells.append(
            _model_header_cell(
                content,
                model=model,
                model_display=model_display,
                selected_models=selected_models,
                colors=colors,
            )
        )
    return " & ".join(cells) + r" \\"


def _model_metadata_row_two_label_columns(
    label: str,
    key: str,
    models: list[str],
    model_display: Mapping[str, Any],
    *,
    selected_models: set[str],
    colors: Mapping[str, str],
) -> str:
    label_text = _latex_escape(label)
    cells = [r"\multicolumn{2}{l}{\textbf{" + label_text + r"}}" if label_text else r"\multicolumn{2}{l}{}"]
    for model in models:
        value = _model_metadata(model_display, model)[key]
        content = value if key in {"finetuned", "use_projector"} else _latex_escape(value)
        cells.append(
            _model_header_cell(
                content,
                model=model,
                model_display=model_display,
                selected_models=selected_models,
                colors=colors,
            )
        )
    return " & ".join(cells) + r" \\"


def _group_label(group: Mapping[str, str], display_cfg: Mapping[str, Any]) -> str:
    group_labels = dict(display_cfg.get("question_groups") or {})
    return clean_text(group_labels.get(clean_text(group.get("id")))) or clean_text(group.get("default_label"))


def _global_colors(cfg: Mapping[str, Any]) -> dict[str, str]:
    return dict(cfg.get("colors") or {})


def _table_caption(table_cfg: Mapping[str, Any], default: str, **replacements: str) -> str:
    caption = clean_text(table_cfg.get("caption")) or default
    for key, value in replacements.items():
        caption = caption.replace("{" + key + "}", value)
    return caption


def _result_column_width(table_cfg: Mapping[str, Any]) -> str:
    return clean_text(table_cfg.get("result_column_width")) or "1.85cm"


def _centered_p_column(width: str) -> str:
    return r">{\centering\arraybackslash}p{" + width + r"}"


def _fmt_metric(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{100.0 * value:.1f}"


def _fmt_unit_metric(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.3f}"


def _fmt_unit_stdev_metric(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.2f}"


def _fmt_metric_summary(
    mean: float | None,
    std: float | None,
    std_color: str,
    *,
    bold: bool = False,
    underline: bool = False,
) -> str:
    value_text = _fmt_metric(mean)
    if bold and value_text != "--":
        value_text = r"\textbf{" + value_text + r"}"
    if underline and value_text != "--":
        value_text = r"\underline{" + value_text + r"}"
    if std is None or value_text == "--":
        return value_text
    return (
        value_text
        + r"\,{\scriptsize \textcolor[HTML]{"
        + std_color
        + r"}{$\pm$ "
        + _fmt_metric(std)
        + r"}}"
    )


def _emphasize_metric_text(text: str, *, bold: bool = False, underline: bool = False) -> str:
    if bold and text != "--":
        text = r"\textbf{" + text + r"}"
    if underline and text != "--":
        text = r"\underline{" + text + r"}"
    return text


def _fmt_mcq_value_row(
    *,
    accuracy: float | None,
    f1_macro: float | None,
    f1_color: str,
    bold_accuracy: bool = False,
    bold_f1: bool = False,
    underline_accuracy: bool = False,
    underline_f1: bool = False,
) -> str:
    accuracy_text = _emphasize_metric_text(
        _fmt_metric(accuracy),
        bold=bold_accuracy,
        underline=underline_accuracy,
    )
    f1_text = _emphasize_metric_text(
        _fmt_unit_metric(f1_macro),
        bold=bold_f1,
        underline=underline_f1,
    )
    if f1_text != "--":
        f1_text = r"\textcolor[HTML]{" + f1_color + r"}{" + f1_text + r"}"
    return accuracy_text + r" / " + f1_text


def _fmt_mcq_stacked_rows(
    *,
    accuracy: float | None,
    f1_macro: float | None,
    bold_accuracy: bool = False,
    bold_f1: bool = False,
    underline_accuracy: bool = False,
    underline_f1: bool = False,
) -> tuple[str, str]:
    return (
        _emphasize_metric_text(
            _fmt_metric(accuracy),
            bold=bold_accuracy,
            underline=underline_accuracy,
        ),
        _emphasize_metric_text(
            _fmt_unit_metric(f1_macro),
            bold=bold_f1,
            underline=underline_f1,
        ),
    )


def _fmt_qa_value_row(
    *,
    bertscore_f1: float | None,
    rouge_l_f1: float | None,
    rouge_color: str,
    bold_bert: bool = False,
    bold_rouge: bool = False,
    underline_bert: bool = False,
    underline_rouge: bool = False,
) -> str:
    bert_text = _emphasize_metric_text(
        _fmt_unit_metric(bertscore_f1),
        bold=bold_bert,
        underline=underline_bert,
    )
    rouge_text = _emphasize_metric_text(
        _fmt_unit_metric(rouge_l_f1),
        bold=bold_rouge,
        underline=underline_rouge,
    )
    if rouge_text != "--":
        rouge_text = r"\textcolor[HTML]{" + rouge_color + r"}{" + rouge_text + r"}"
    return bert_text + r" / " + rouge_text


def _fmt_qa_stacked_rows(
    *,
    bertscore_f1: float | None,
    rouge_l_f1: float | None,
    bold_bert: bool = False,
    bold_rouge: bool = False,
    underline_bert: bool = False,
    underline_rouge: bool = False,
) -> tuple[str, str]:
    return (
        _emphasize_metric_text(
            _fmt_unit_metric(bertscore_f1),
            bold=bold_bert,
            underline=underline_bert,
        ),
        _emphasize_metric_text(
            _fmt_unit_metric(rouge_l_f1),
            bold=bold_rouge,
            underline=underline_rouge,
        ),
    )


def _fmt_mcq_stdev_row(metrics: dict[str, Any], *, show_stdev: bool) -> str:
    if not show_stdev:
        return ""
    return (
        r"$\pm$ "
        + _fmt_metric(metrics.get("accuracy_stdev"))
        + r" / "
        + r"$\pm$ "
        + _fmt_unit_stdev_metric(metrics.get("f1_macro_stdev"))
    )


def _fmt_qa_stdev_row(metrics: dict[str, Any], *, show_stdev: bool) -> str:
    if not show_stdev:
        return ""
    return (
        r"$\pm$ "
        + _fmt_unit_stdev_metric(metrics.get("bertscore_f1_mean_stdev"))
        + r" / "
        + r"$\pm$ "
        + _fmt_unit_stdev_metric(metrics.get("rouge_l_f1_mean_stdev"))
    )


def _weighted_mean(records: list[Mapping[str, Any]], key: str) -> float | None:
    weighted_sum = 0.0
    weight_total = 0
    for record in records:
        value = record.get(key)
        n = int(record.get("n", 0) or 0)
        if value is None or n <= 0:
            continue
        weighted_sum += float(value) * n
        weight_total += n
    if weight_total <= 0:
        return None
    return weighted_sum / weight_total


def _aggregate_records(records: list[Mapping[str, Any]], metric_kind: str) -> dict[str, Any] | None:
    if not records:
        return None
    n = sum(int(record.get("n", 0) or 0) for record in records)
    if n <= 0:
        return None
    if metric_kind == "mcq":
        correct = sum(int(record.get("correct", 0) or 0) for record in records)
        return {
            "n": n,
            "accuracy": correct / n,
            "accuracy_stdev": _weighted_mean(records, "accuracy_stdev"),
            "f1_macro": _weighted_mean(records, "f1_macro"),
            "f1_macro_stdev": _weighted_mean(records, "f1_macro_stdev"),
        }
    if metric_kind == "qa":
        return {
            "n": n,
            "bertscore_f1_mean": _weighted_mean(records, "bertscore_f1_mean"),
            "bertscore_f1_mean_stdev": _weighted_mean(records, "bertscore_f1_mean_stdev"),
            "rouge_l_f1_mean": _weighted_mean(records, "rouge_l_f1_mean"),
            "rouge_l_f1_mean_stdev": _weighted_mean(records, "rouge_l_f1_mean_stdev"),
        }
    raise ValueError(f"Unknown metric kind: {metric_kind}")


def _metric_value(metrics: dict[str, Any] | None, key: str) -> float | None:
    if metrics is None or metrics.get(key) is None:
        return None
    return float(metrics[key])


def _is_row_best(value: float | None, values: list[float | None]) -> bool:
    present = [item for item in values if item is not None]
    return value is not None and bool(present) and value == max(present)


def _is_row_second_best(value: float | None, values: list[float | None]) -> bool:
    present = sorted({item for item in values if item is not None}, reverse=True)
    return value is not None and len(present) >= 2 and value == present[1]


def _metric_cell(
    metrics: dict[str, Any] | None,
    metric_kind: str,
    colors: Mapping[str, str],
    *,
    bold_first: bool = False,
    bold_second: bool = False,
    underline_first: bool = False,
    underline_second: bool = False,
    show_stdev: bool = True,
    mcq_layout: str = "slash",
) -> str:
    if metrics is None:
        return r"\makebox[\linewidth][c]{--}"
    std_color = clean_text(colors.get("stdev", "64748B"))
    if metric_kind == "mcq":
        f1_color = clean_text(colors.get("f1", "2563EB"))
        if mcq_layout == "stacked":
            accuracy_text, f1_text = _fmt_mcq_stacked_rows(
                accuracy=metrics.get("accuracy"),
                f1_macro=metrics.get("f1_macro"),
                bold_accuracy=bold_first,
                bold_f1=bold_second,
                underline_accuracy=underline_first,
                underline_f1=underline_second,
            )
            return r"\cellmetric{" + accuracy_text + r"}{" + f1_text + r"}{" + f1_color + r"}"
        if mcq_layout != "slash":
            raise ValueError(f"Unknown MCQ metric cell layout: {mcq_layout}")
        return (
            r"\cellmetric{"
            + _fmt_mcq_value_row(
                accuracy=metrics.get("accuracy"),
                f1_macro=metrics.get("f1_macro"),
                f1_color=f1_color,
                bold_accuracy=bold_first,
                bold_f1=bold_second,
                underline_accuracy=underline_first,
                underline_f1=underline_second,
            )
            + r"}{"
            + _fmt_mcq_stdev_row(metrics, show_stdev=show_stdev)
            + r"}{"
            + std_color
            + r"}"
        )
    f1_color = clean_text(colors.get("f1", "2563EB"))
    if mcq_layout == "stacked":
        bert_text, rouge_text = _fmt_qa_stacked_rows(
            bertscore_f1=metrics.get("bertscore_f1_mean"),
            rouge_l_f1=metrics.get("rouge_l_f1_mean"),
            bold_bert=bold_first,
            bold_rouge=bold_second,
            underline_bert=underline_first,
            underline_rouge=underline_second,
        )
        return r"\cellmetric{" + bert_text + r"}{" + rouge_text + r"}{" + f1_color + r"}"
    if mcq_layout != "slash":
        raise ValueError(f"Unknown QA metric cell layout: {mcq_layout}")
    return (
        r"\cellmetric{"
        + _fmt_qa_value_row(
            bertscore_f1=metrics.get("bertscore_f1_mean"),
            rouge_l_f1=metrics.get("rouge_l_f1_mean"),
            rouge_color=f1_color,
            bold_bert=bold_first,
            bold_rouge=bold_second,
            underline_bert=underline_first,
            underline_rouge=underline_second,
        )
        + r"}{"
        + _fmt_qa_stdev_row(metrics, show_stdev=show_stdev)
        + r"}{"
        + std_color
        + r"}"
    )


def _model_order_and_vrules(metrics: list[Mapping[str, Any]], table_cfg: Mapping[str, Any]) -> tuple[list[str], set[int]]:
    configured = [clean_text(item) for item in table_cfg.get("model_order", []) if clean_text(item)]
    present = sorted({clean_text(record.get("model_display_name")) for record in metrics if clean_text(record.get("model_display_name"))})
    ordered: list[str] = []
    vrules_after: set[int] = set()
    for item in configured:
        if item == VRULE_TOKEN:
            continue
        if item in present and item not in ordered:
            ordered.append(item)
    ordered.extend(model for model in present if model not in ordered)
    return ordered, set()


def _fill_column_width(*, fixed_widths: list[str], column_count: int) -> str:
    fixed_terms = "".join("-" + clean_text(width) for width in fixed_widths if clean_text(width))
    return r"\dimexpr\linewidth" + fixed_terms + "-" + str(2 * column_count) + r"\tabcolsep\relax"


def _resolved_first_column_width(raw_width: Any, *, fixed_widths: list[str], column_count: int, default: str) -> str:
    width = clean_text(raw_width) or default
    if width == "fill":
        return _fill_column_width(fixed_widths=fixed_widths, column_count=column_count)
    return width


def _column_spec(model_count: int, result_column_width: str, *, label_column_width: str | None = None) -> str:
    label_width = _resolved_first_column_width(
        label_column_width,
        fixed_widths=[result_column_width] * model_count,
        column_count=model_count + 1,
        default="",
    )
    label_column = _ragged_p_column(label_width) if label_width else "l"
    return label_column + (_centered_p_column(result_column_width) * model_count)


def _two_label_column_spec(
    model_count: int,
    result_column_width: str,
    *,
    first_label_column_width: str | None = None,
    second_label_column_width: str | None = None,
) -> str:
    first_label_width = _resolved_first_column_width(
        first_label_column_width,
        fixed_widths=([second_label_column_width] if second_label_column_width else []) + ([result_column_width] * model_count),
        column_count=model_count + 2,
        default="",
    )
    first_column = _ragged_p_column(first_label_width) if first_label_width else "l"
    second_column = _ragged_p_column(second_label_column_width) if second_label_column_width else "l"
    return first_column + second_column + (_centered_p_column(result_column_width) * model_count)


def _ragged_p_column(width: str) -> str:
    return r">{\raggedright\arraybackslash}p{" + width + r"}"


def _ragged_m_column(width: str) -> str:
    return r">{\raggedright\arraybackslash}m{" + width + r"}"


def _centered_m_column(width: str) -> str:
    return r">{\centering\arraybackslash}m{" + width + r"}"


def _ragged_small_p_column(width: str) -> str:
    return r">{\raggedright\arraybackslash\small}p{" + width + r"}"


def _justified_small_p_column(width: str) -> str:
    return r">{\arraybackslash\small}p{" + width + r"}"


def _cellmetric_macro(*, stdev_scriptsize: bool) -> list[str]:
    second_row = (
        r"{\scriptsize\textcolor[HTML]{#3}{#2}}"
        if stdev_scriptsize
        else r"\textcolor[HTML]{#3}{#2}"
    )
    return [
        r"\providecommand{\cellmetric}[3]{}",
        r"\renewcommand{\cellmetric}[3]{\makebox[\linewidth][c]{\begin{tabular}[t]{@{}c@{}}#1\\[-0.30em]"
        + second_row
        + r"\end{tabular}}}",
    ]


def _colored_hline(colors: Mapping[str, str]) -> str:
    return (
        r"\arrayrulecolor[HTML]{"
        + clean_text(colors.get("rule", "CBD5E1"))
        + r"}\hline\arrayrulecolor{black}"
    )


def _spaced_colored_hline(colors: Mapping[str, str], *, before: str = "0.24em", after: str = "0.24em") -> list[str]:
    return [
        r"\noalign{\vskip " + before + r"}",
        _colored_hline(colors),
        r"\noalign{\vskip " + after + r"}",
    ]


def _qa_metric_label(colors: Mapping[str, str]) -> str:
    return (
        r"BERT-F1 / "
        + r"\textcolor[HTML]{"
        + clean_text(colors.get("f1", "2563EB"))
        + r"}{ROUGE-L}"
    )


def _task_order(metrics: list[Mapping[str, Any]], display_names: Mapping[str, Any]) -> list[str]:
    configured = [clean_text(key) for key in display_names.keys() if clean_text(key)]
    present = sorted({clean_text(record.get("task_category")) for record in metrics if clean_text(record.get("task_category")) not in {"", "ALL"}})
    ordered = [task for task in configured if task in present]
    ordered.extend(task for task in present if task not in ordered)
    return ordered


def _slice_records(
    metrics: list[Mapping[str, Any]],
    *,
    model: str,
    task_category: str,
    question_type: str,
    generation_type: str,
    modality_combination_name: str,
) -> list[Mapping[str, Any]]:
    return [
        record
        for record in metrics
        if record.get("metric_group") == "main_table"
        and clean_text(record.get("model_display_name")) == model
        and clean_text(record.get("task_category")) == task_category
        and clean_text(record.get("question_type")) == question_type
        and clean_text(record.get("generation_type")) == generation_type
        and clean_text(record.get("modality_combination_name")) == modality_combination_name
    ]


def _group_task_categories(
    metrics: list[Mapping[str, Any]],
    *,
    group: Mapping[str, str],
    task_order: list[str],
    modality_combination_name: str,
) -> list[str]:
    out: list[str] = []
    for task in task_order:
        if any(
            _slice_records(
                metrics,
                model=clean_text(record.get("model_display_name")),
                task_category=task,
                question_type=group["question_type"],
                generation_type=group["generation_type"],
                modality_combination_name=modality_combination_name,
            )
            for record in metrics
        ):
            out.append(task)
    return out


def _cancer_records(
    metrics: list[Mapping[str, Any]],
    *,
    model: str,
    project_id: str,
    question_type: str,
    generation_type: str | None,
    modality_combination_name: str,
) -> list[Mapping[str, Any]]:
    return [
        record
        for record in metrics
        if record.get("metric_group") == "cancer_table"
        and clean_text(record.get("model_display_name")) == model
        and clean_text(record.get("project_id")) == project_id
        and clean_text(record.get("question_type")) == question_type
        and (generation_type is None or clean_text(record.get("generation_type")) == generation_type)
        and clean_text(record.get("modality_combination_name")) == modality_combination_name
    ]


def _task_cancer_records(
    metrics: list[Mapping[str, Any]],
    *,
    model: str,
    task_category: str,
    project_id: str,
    question_type: str,
    generation_type: str,
    modality_combination_name: str,
) -> list[Mapping[str, Any]]:
    return [
        record
        for record in metrics
        if record.get("metric_group") == "task_cancer_table"
        and clean_text(record.get("model_display_name")) == model
        and clean_text(record.get("task_category")) == task_category
        and clean_text(record.get("project_id")) == project_id
        and clean_text(record.get("question_type")) == question_type
        and clean_text(record.get("generation_type")) == generation_type
        and clean_text(record.get("modality_combination_name")) == modality_combination_name
    ]


def _group_project_ids(
    metrics: list[Mapping[str, Any]],
    *,
    group: Mapping[str, str],
    project_order: list[str],
    modality_combination_name: str,
) -> list[str]:
    out: list[str] = []
    for project_id in project_order:
        if any(
            _cancer_records(
                metrics,
                model=clean_text(record.get("model_display_name")),
                project_id=project_id,
                question_type=group["question_type"],
                generation_type=group["generation_type"],
                modality_combination_name=modality_combination_name,
            )
            for record in metrics
        ):
            out.append(project_id)
    return out


def _main_result_table(metrics_blob: Mapping[str, Any], cfg: Mapping[str, Any]) -> str:
    metrics = list(metrics_blob.get("metrics") or [])
    if any("repeat_id" in record for record in metrics):
        raise ValueError("metrics.json is in the old repeat-level format. Rerun score_vqa_predictions.py.")
    table_cfg = dict(dict(cfg.get("tables") or {}).get("main_result") or {})
    colors = _global_colors(cfg)
    modality_combination_name = clean_text(table_cfg.get("modality_combination_name")) or "all_available"
    display_cfg = dict(cfg.get("display_names") or {})
    model_display = dict(display_cfg.get("models") or {})
    task_display = dict(display_cfg.get("task_categories") or {})
    models, vrules_after = _model_order_and_vrules(metrics, table_cfg)
    selected_models = _selected_models(cfg)
    if not models:
        raise RuntimeError("No models found in metrics.json.")
    task_order = _task_order(metrics, task_display)

    num_columns = len(models) + 1
    task_column_width = clean_text(table_cfg.get("task_column_width")) or "2.25cm"
    column_spec = _column_spec(
        len(models),
        _result_column_width(table_cfg),
        label_column_width=task_column_width,
    )
    lines = [
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.18}",
        *_cellmetric_macro(stdev_scriptsize=True),
        r"\providecommand{\bertcell}[1]{#1}",
        r"\begin{tabular}{"
        + column_spec
        + r"}",
        r"\toprule",
        r"\rowcolor[HTML]{"
        + clean_text(colors.get("model_header_bg", "F3F6FA"))
        + r"}",
        _model_metadata_row("Name", "name", models, model_display, selected_models=selected_models, colors=colors),
        r"\rowcolor[HTML]{"
        + clean_text(colors.get("model_header_bg", "F3F6FA"))
        + r"}",
        _model_metadata_row("", "backbone", models, model_display, selected_models=selected_models, colors=colors),
        r"\rowcolor[HTML]{"
        + clean_text(colors.get("model_header_bg", "F3F6FA"))
        + r"}",
        _model_metadata_row("Finetuned", "finetuned", models, model_display, selected_models=selected_models, colors=colors),
        r"\rowcolor[HTML]{"
        + clean_text(colors.get("model_header_bg", "F3F6FA"))
        + r"}",
        _model_metadata_row("Use projector", "use_projector", models, model_display, selected_models=selected_models, colors=colors),
    ]

    first_group = True
    for group in MAIN_GROUPS:
        group_tasks = _group_task_categories(
            metrics,
            group=group,
            task_order=task_order,
            modality_combination_name=modality_combination_name,
        )
        if not group_tasks:
            continue
        if not first_group:
            lines.append(r"\addlinespace[0.35em]")
        first_group = False
        lines.extend(
            [
                _colored_hline(colors),
                r"\rowcolor[HTML]{"
                + clean_text(colors.get("subheading_bg", "EFF6FF"))
                + r"}",
                r"\multicolumn{"
                + str(num_columns)
                + r"}{l}{\rule[-0.8ex]{0pt}{3.2ex}\textbf{"
                + _latex_escape(_group_label(group, display_cfg))
                + r"}"
                + (
                    r" \textnormal{("
                    + r"Accuracy\% / "
                    + r"\textcolor[HTML]{"
                    + clean_text(colors.get("f1", "2563EB"))
                    + r"}{F1}"
                    + r")}"
                    if group["metric"] == "mcq"
                    else r" \textnormal{(" + _qa_metric_label(colors) + r")}"
                )
                + r"} \\",
                _colored_hline(colors),
                r"\noalign{\vskip 0.28em}",
            ]
        )
        group_model_records: dict[str, list[Mapping[str, Any]]] = {model: [] for model in models}
        for task in group_tasks:
            cells = [_latex_escape(task_display.get(task, task))]
            row_aggregates: dict[str, dict[str, Any] | None] = {}
            for model in models:
                records = _slice_records(
                    metrics,
                    model=model,
                    task_category=task,
                    question_type=group["question_type"],
                    generation_type=group["generation_type"],
                    modality_combination_name=modality_combination_name,
                )
                group_model_records[model].extend(records)
                row_aggregates[model] = _aggregate_records(records, group["metric"])
            if group["metric"] == "mcq":
                acc_values = [_metric_value(row_aggregates[model], "accuracy") for model in models]
                f1_values = [_metric_value(row_aggregates[model], "f1_macro") for model in models]
                for model in models:
                    aggregate = row_aggregates[model]
                    cells.append(
                        _selected_cell(
                            _metric_cell(
                                aggregate,
                                group["metric"],
                                colors,
                                bold_first=_is_row_best(_metric_value(aggregate, "accuracy"), acc_values),
                                bold_second=_is_row_best(_metric_value(aggregate, "f1_macro"), f1_values),
                                underline_first=_is_row_second_best(_metric_value(aggregate, "accuracy"), acc_values),
                                underline_second=_is_row_second_best(_metric_value(aggregate, "f1_macro"), f1_values),
                            ),
                            model=model,
                            selected_models=selected_models,
                            colors=colors,
                        )
                    )
            else:
                bert_values = [_metric_value(row_aggregates[model], "bertscore_f1_mean") for model in models]
                rouge_values = [_metric_value(row_aggregates[model], "rouge_l_f1_mean") for model in models]
                for model in models:
                    aggregate = row_aggregates[model]
                    cells.append(
                        _selected_cell(
                            _metric_cell(
                                aggregate,
                                group["metric"],
                                colors,
                                bold_first=_is_row_best(_metric_value(aggregate, "bertscore_f1_mean"), bert_values),
                                bold_second=_is_row_best(_metric_value(aggregate, "rouge_l_f1_mean"), rouge_values),
                                underline_first=_is_row_second_best(_metric_value(aggregate, "bertscore_f1_mean"), bert_values),
                                underline_second=_is_row_second_best(_metric_value(aggregate, "rouge_l_f1_mean"), rouge_values),
                            ),
                            model=model,
                            selected_models=selected_models,
                            colors=colors,
                        )
                    )
            lines.append(" & ".join(cells) + r" \\")

        mean_cells = [r"\textbf{Mean}"]
        mean_aggregates = {
            model: _aggregate_records(group_model_records[model], group["metric"])
            for model in models
        }
        if group["metric"] == "mcq":
            mean_acc_values = [_metric_value(mean_aggregates[model], "accuracy") for model in models]
            mean_f1_values = [_metric_value(mean_aggregates[model], "f1_macro") for model in models]
            for model in models:
                aggregate = mean_aggregates[model]
                mean_cells.append(
                    _selected_cell(
                        _metric_cell(
                            aggregate,
                            group["metric"],
                            colors,
                            bold_first=_is_row_best(_metric_value(aggregate, "accuracy"), mean_acc_values),
                            bold_second=_is_row_best(_metric_value(aggregate, "f1_macro"), mean_f1_values),
                            underline_first=_is_row_second_best(_metric_value(aggregate, "accuracy"), mean_acc_values),
                            underline_second=_is_row_second_best(_metric_value(aggregate, "f1_macro"), mean_f1_values),
                        ),
                        model=model,
                        selected_models=selected_models,
                        colors=colors,
                    )
                )
        else:
            mean_bert_values = [_metric_value(mean_aggregates[model], "bertscore_f1_mean") for model in models]
            mean_rouge_values = [_metric_value(mean_aggregates[model], "rouge_l_f1_mean") for model in models]
            for model in models:
                aggregate = mean_aggregates[model]
                mean_cells.append(
                    _selected_cell(
                        _metric_cell(
                            aggregate,
                            group["metric"],
                            colors,
                            bold_first=_is_row_best(_metric_value(aggregate, "bertscore_f1_mean"), mean_bert_values),
                            bold_second=_is_row_best(_metric_value(aggregate, "rouge_l_f1_mean"), mean_rouge_values),
                            underline_first=_is_row_second_best(_metric_value(aggregate, "bertscore_f1_mean"), mean_bert_values),
                            underline_second=_is_row_second_best(_metric_value(aggregate, "rouge_l_f1_mean"), mean_rouge_values),
                        ),
                        model=model,
                        selected_models=selected_models,
                        colors=colors,
                    )
                )
        lines.extend(
            [
                " & ".join(mean_cells) + r" \\",
            ]
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    return "\n".join(lines)


def _cancer_result_table(metrics_blob: Mapping[str, Any], cfg: Mapping[str, Any]) -> str:
    metrics = list(metrics_blob.get("metrics") or [])
    if any("repeat_id" in record for record in metrics):
        raise ValueError("metrics.json is in the old repeat-level format. Rerun score_vqa_predictions.py.")
    table_cfg = dict(dict(cfg.get("tables") or {}).get("cancer_result") or {})
    if not table_cfg:
        return ""
    colors = _global_colors(cfg)
    modality_combination_name = clean_text(table_cfg.get("modality_combination_name")) or "all_available"
    display_cfg = dict(cfg.get("display_names") or {})
    model_display = dict(display_cfg.get("models") or {})
    project_display = dict(display_cfg.get("projects") or {})
    models, vrules_after = _model_order_and_vrules(metrics, table_cfg)
    selected_models = _selected_models(cfg)
    if not models:
        raise RuntimeError("No models found in metrics.json.")
    project_order = [clean_text(key) for key in table_cfg.get("projects_to_show", []) if clean_text(key)]
    if not project_order:
        project_order = sorted(
            {
                clean_text(record.get("project_id"))
                for record in metrics
                if record.get("metric_group") == "cancer_table" and clean_text(record.get("project_id")) not in {"", "ALL"}
            }
        )

    group_column_width = clean_text(table_cfg.get("group_column_width")) or "1.35cm"
    result_column_width = _result_column_width(table_cfg)
    cancer_column_width = _resolved_first_column_width(
        table_cfg.get("cancer_column_width"),
        fixed_widths=[group_column_width] + ([result_column_width] * len(models)),
        column_count=len(models) + 2,
        default="2.15cm",
    )
    column_spec = _two_label_column_spec(
        len(models),
        result_column_width,
        first_label_column_width=cancer_column_width,
        second_label_column_width=group_column_width,
    )
    lines = [
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.18}",
        *_cellmetric_macro(stdev_scriptsize=True),
        r"\providecommand{\bertcell}[1]{#1}",
        r"\begin{tabular}{"
        + column_spec
        + r"}",
        r"\toprule",
        r"\rowcolor[HTML]{"
        + clean_text(colors.get("model_header_bg", "F3F6FA"))
        + r"}",
        _model_metadata_row_two_label_columns("Name", "name", models, model_display, selected_models=selected_models, colors=colors),
        r"\rowcolor[HTML]{"
        + clean_text(colors.get("model_header_bg", "F3F6FA"))
        + r"}",
        _model_metadata_row_two_label_columns("", "backbone", models, model_display, selected_models=selected_models, colors=colors),
        r"\rowcolor[HTML]{"
        + clean_text(colors.get("model_header_bg", "F3F6FA"))
        + r"}",
        _model_metadata_row_two_label_columns("Finetuned", "finetuned", models, model_display, selected_models=selected_models, colors=colors),
        r"\rowcolor[HTML]{"
        + clean_text(colors.get("model_header_bg", "F3F6FA"))
        + r"}",
        _model_metadata_row_two_label_columns("Use projector", "use_projector", models, model_display, selected_models=selected_models, colors=colors),
        *_spaced_colored_hline(colors, before="0.12em", after="0.20em"),
        r"\rowcolor[HTML]{"
        + clean_text(colors.get("subheading_bg", "EFF6FF"))
        + r"}",
        r"\textbf{Cancer} & \textbf{Group} & \multicolumn{"
        + str(len(models))
        + r"}{c}{\textbf{Performance by model}} \\",
        *_spaced_colored_hline(colors, before="0.20em", after="0.28em"),
    ]

    rendered_project_count = 0
    cancer_label_is_fill = clean_text(table_cfg.get("cancer_column_width")) == "fill"
    cancer_multirow_width = "=" if cancer_label_is_fill else cancer_column_width
    cancer_parbox_width = r"\linewidth" if cancer_label_is_fill else cancer_column_width
    for project_id in project_order:
        project_records = [
            record
            for record in metrics
            if record.get("metric_group") == "cancer_table"
            and clean_text(record.get("project_id")) == project_id
            and clean_text(record.get("modality_combination_name")) == modality_combination_name
        ]
        if not project_records:
            continue

        project_rows: list[tuple[str, str, dict[str, dict[str, Any] | None]]] = []
        for group in TABLE2_GROUPS:
            metric_kind = group["metric"]
            row_aggregates: dict[str, dict[str, Any] | None] = {}
            for model in models:
                records = _cancer_records(
                    metrics,
                    model=model,
                    project_id=project_id,
                    question_type=group["question_type"],
                    generation_type=group["generation_type"],
                    modality_combination_name=modality_combination_name,
                )
                row_aggregates[model] = _aggregate_records(records, metric_kind)
            if all(aggregate is None for aggregate in row_aggregates.values()):
                continue
            project_rows.append((_group_label(group, display_cfg), metric_kind, row_aggregates))
        if not project_rows:
            continue

        if rendered_project_count > 0:
            lines.extend(_spaced_colored_hline(colors, before="0.30em", after="0.30em"))
        rendered_project_count += 1

        row_count = len(project_rows)
        project_label = _latex_escape(project_display.get(project_id, project_id))
        for row_index, (row_label, metric_kind, row_aggregates) in enumerate(project_rows):
            cells = [
                (
                    r"\multirow[c]{"
                    + str(row_count)
                    + r"}{"
                    + cancer_multirow_width
                    + r"}{\raisebox{-0.55\baselineskip}{\parbox[c]{"
                    + cancer_parbox_width
                    + r"}{\raggedright\textbf{"
                    + project_label
                    + r"}}}}"
                    if row_index == 0 and row_count > 1
                    else r"\textbf{" + project_label + r"}"
                    if row_index == 0
                    else ""
                ),
                _latex_escape(row_label),
            ]
            if metric_kind == "mcq":
                acc_values = [_metric_value(row_aggregates[model], "accuracy") for model in models]
                f1_values = [_metric_value(row_aggregates[model], "f1_macro") for model in models]
                for model in models:
                    aggregate = row_aggregates[model]
                    cells.append(
                        _selected_cell(
                            _metric_cell(
                                aggregate,
                                metric_kind,
                                colors,
                                bold_first=_is_row_best(_metric_value(aggregate, "accuracy"), acc_values),
                                bold_second=_is_row_best(_metric_value(aggregate, "f1_macro"), f1_values),
                                underline_first=_is_row_second_best(_metric_value(aggregate, "accuracy"), acc_values),
                                underline_second=_is_row_second_best(_metric_value(aggregate, "f1_macro"), f1_values),
                            ),
                            model=model,
                            selected_models=selected_models,
                            colors=colors,
                        )
                    )
            else:
                bert_values = [_metric_value(row_aggregates[model], "bertscore_f1_mean") for model in models]
                rouge_values = [_metric_value(row_aggregates[model], "rouge_l_f1_mean") for model in models]
                for model in models:
                    aggregate = row_aggregates[model]
                    cells.append(
                        _selected_cell(
                            _metric_cell(
                                aggregate,
                                metric_kind,
                                colors,
                                bold_first=_is_row_best(_metric_value(aggregate, "bertscore_f1_mean"), bert_values),
                                bold_second=_is_row_best(_metric_value(aggregate, "rouge_l_f1_mean"), rouge_values),
                                underline_first=_is_row_second_best(_metric_value(aggregate, "bertscore_f1_mean"), bert_values),
                                underline_second=_is_row_second_best(_metric_value(aggregate, "rouge_l_f1_mean"), rouge_values),
                            ),
                            model=model,
                            selected_models=selected_models,
                            colors=colors,
                        )
                    )
            lines.append(" & ".join(cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    return "\n".join(lines)


def _group_modality_records(
    metrics: list[Mapping[str, Any]],
    *,
    model: str,
    task_category: str,
    question_type: str,
    generation_type: str,
    modality_combination_name: str,
) -> list[Mapping[str, Any]]:
    return [
        record
        for record in metrics
        if record.get("metric_group") == "modality_ablation"
        and clean_text(record.get("model_display_name")) == model
        and clean_text(record.get("task_category")) == task_category
        and clean_text(record.get("question_type")) == question_type
        and clean_text(record.get("generation_type")) == generation_type
        and clean_text(record.get("modality_combination_name")) == modality_combination_name
    ]


def _group_modality_task_categories(
    metrics: list[Mapping[str, Any]],
    *,
    model: str,
    group: Mapping[str, str],
    task_order: list[str],
    modality_names: list[str],
) -> list[str]:
    out: list[str] = []
    for task in task_order:
        if any(
            _group_modality_records(
                metrics,
                model=model,
                task_category=task,
                question_type=group["question_type"],
                generation_type=group["generation_type"],
                modality_combination_name=modality_name,
            )
            for modality_name in modality_names
        ):
            out.append(task)
    return out


def _modality_ablation_table(metrics_blob: Mapping[str, Any], cfg: Mapping[str, Any]) -> str:
    metrics = list(metrics_blob.get("metrics") or [])
    if any("repeat_id" in record for record in metrics):
        raise ValueError("metrics.json is in the old repeat-level format. Rerun score_vqa_predictions.py.")
    all_tables_cfg = dict(cfg.get("tables") or {})
    table_cfg = dict(all_tables_cfg.get("modality_ablation") or {})
    if not table_cfg:
        return ""
    colors = _global_colors(cfg)
    display_cfg = dict(cfg.get("display_names") or {})
    model_display = dict(display_cfg.get("models") or {})
    task_display = dict(display_cfg.get("task_categories") or {})
    modality_display = dict(display_cfg.get("modality_combinations") or {})
    modality_label_overrides = dict(table_cfg.get("modality_label_overrides") or {})
    model = clean_text(table_cfg.get("model_display_name"))
    if not model:
        raise ValueError("tables.modality_ablation.model_display_name must be populated.")
    model_label = _model_metadata(model_display, model)["name"]
    modality_names = [clean_text(key) for key in table_cfg.get("modality_combination_names", []) if clean_text(key)]
    if not modality_names:
        modality_names = [clean_text(key) for key in modality_display.keys() if clean_text(key)]
    task_label_overrides = dict(table_cfg.get("task_label_overrides") or {})
    task_order = [clean_text(key) for key in table_cfg.get("task_categories", []) if clean_text(key)]
    if not task_order:
        task_order = [clean_text(key) for key in task_label_overrides.keys() if clean_text(key)]
    if not task_order:
        return ""

    group_blocks: list[tuple[Mapping[str, str], list[str]]] = []
    for group in MAIN_GROUPS:
        group_tasks = _group_modality_task_categories(
            metrics,
            model=model,
            group=group,
            task_order=task_order,
            modality_names=modality_names,
        )
        if group_tasks:
            group_blocks.append((group, group_tasks))
    if not group_blocks:
        return ""

    result_column = _centered_m_column(_result_column_width(table_cfg))
    modality_column_width = _resolved_first_column_width(
        table_cfg.get("modality_column_width"),
        fixed_widths=[_result_column_width(table_cfg)] * sum(len(group_tasks) for _, group_tasks in group_blocks),
        column_count=1 + sum(len(group_tasks) for _, group_tasks in group_blocks),
        default="1.60cm",
    )
    column_spec = _ragged_m_column(modality_column_width)
    for _, group_tasks in group_blocks:
        column_spec += result_column * len(group_tasks)

    group_header_cells: list[str] = [r"\multicolumn{1}{c}{}"]
    modality_header_label = clean_text(table_cfg.get("modality_header_label")) or "Modality"
    task_header_cells: list[str] = [r"\textbf{" + _latex_escape(modality_header_label) + r"}"]
    for group, group_tasks in group_blocks:
        metric_label = (
            r"Accuracy\% / "
            + r"\textcolor[HTML]{"
            + clean_text(colors.get("f1", "2563EB"))
            + r"}{F1}"
            if group["metric"] == "mcq"
            else _qa_metric_label(colors)
        )
        group_header_cells.append(
            r"\multicolumn{"
            + str(len(group_tasks))
            + r"}{c}{\begin{tabular}[c]{@{}c@{}}\rule[-0.55ex]{0pt}{3.0ex}\textbf{"
            + _latex_escape(_group_label(group, display_cfg))
            + r"}\\[-0.18em]{\small "
            + metric_label
            + r"}\end{tabular}}"
        )
        task_header_cells.extend(
            r"{\small\textbf{" + _latex_escape(task_label_overrides.get(task, task_display.get(task, task))) + r"}}"
            for task in group_tasks
        )

    lines = [
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\renewcommand{\arraystretch}{1.14}",
        *_cellmetric_macro(stdev_scriptsize=False),
        r"\providecommand{\bertcell}[1]{#1}",
        r"\begin{tabular}{"
        + column_spec
        + r"}",
        r"\toprule",
        r"\rowcolor[HTML]{"
        + clean_text(colors.get("subheading_bg", "EFF6FF"))
        + r"}",
        " & ".join(group_header_cells) + r" \\",
        _colored_hline(colors),
        r"\rowcolor[HTML]{"
        + clean_text(colors.get("subheading_bg", "EFF6FF"))
        + r"}",
        " & ".join(task_header_cells) + r" \\",
        _colored_hline(colors),
        r"\noalign{\vskip 0.28em}",
    ]

    for modality_index, modality_name in enumerate(modality_names):
        if modality_index > 0:
            lines.extend(_spaced_colored_hline(colors, before="0.16em", after="0.16em"))
        row_cells = [
            r"\textbf{"
            + _latex_escape(modality_label_overrides.get(modality_name, modality_display.get(modality_name, modality_name)))
            + r"}",
        ]
        for block_index, (group, group_tasks) in enumerate(group_blocks):
            if group["metric"] == "mcq":
                for task in group_tasks:
                    modality_aggregates = {
                        candidate_modality: _aggregate_records(
                            _group_modality_records(
                                metrics,
                                model=model,
                                task_category=task,
                                question_type=group["question_type"],
                                generation_type=group["generation_type"],
                                modality_combination_name=candidate_modality,
                            ),
                            group["metric"],
                        )
                        for candidate_modality in modality_names
                    }
                    acc_values = [_metric_value(aggregate, "accuracy") for aggregate in modality_aggregates.values()]
                    f1_values = [_metric_value(aggregate, "f1_macro") for aggregate in modality_aggregates.values()]
                    aggregate = modality_aggregates[modality_name]
                    row_cells.append(
                        _metric_cell(
                            aggregate,
                            group["metric"],
                            colors,
                            bold_first=_is_row_best(_metric_value(aggregate, "accuracy"), acc_values),
                            bold_second=_is_row_best(_metric_value(aggregate, "f1_macro"), f1_values),
                            underline_first=_is_row_second_best(_metric_value(aggregate, "accuracy"), acc_values),
                            underline_second=_is_row_second_best(_metric_value(aggregate, "f1_macro"), f1_values),
                            show_stdev=False,
                            mcq_layout="stacked",
                        )
                    )
            else:
                for task in group_tasks:
                    modality_aggregates = {
                        candidate_modality: _aggregate_records(
                            _group_modality_records(
                                metrics,
                                model=model,
                                task_category=task,
                                question_type=group["question_type"],
                                generation_type=group["generation_type"],
                                modality_combination_name=candidate_modality,
                            ),
                            group["metric"],
                        )
                        for candidate_modality in modality_names
                    }
                    bert_values = [_metric_value(aggregate, "bertscore_f1_mean") for aggregate in modality_aggregates.values()]
                    rouge_values = [_metric_value(aggregate, "rouge_l_f1_mean") for aggregate in modality_aggregates.values()]
                    aggregate = modality_aggregates[modality_name]
                    row_cells.append(
                        _metric_cell(
                            aggregate,
                            group["metric"],
                            colors,
                            bold_first=_is_row_best(_metric_value(aggregate, "bertscore_f1_mean"), bert_values),
                            bold_second=_is_row_best(_metric_value(aggregate, "rouge_l_f1_mean"), rouge_values),
                            underline_first=_is_row_second_best(_metric_value(aggregate, "bertscore_f1_mean"), bert_values),
                            underline_second=_is_row_second_best(_metric_value(aggregate, "rouge_l_f1_mean"), rouge_values),
                            show_stdev=False,
                            mcq_layout="stacked",
                        )
                    )
        lines.append(" & ".join(row_cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    return "\n".join(lines)


def _clip_table_text(value: Any, max_chars: int) -> str:
    text = " ".join(clean_text(value).split())
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:")
    return clipped + "..."


def _span_table_text(value: Any, *, max_chars: int, span: Any) -> str:
    text = " ".join(clean_text(value).split())
    if span is None:
        return _clip_table_text(text, max_chars)
    if not isinstance(span, Mapping):
        raise TypeError(f"Qualitative text span must be a mapping with start/end substrings, got {span!r}.")
    start_marker = clean_text(span.get("start"))
    end_marker = clean_text(span.get("end"))
    start = 0
    if start_marker:
        start = text.find(start_marker)
        if start < 0:
            raise ValueError(f"Qualitative text span start {start_marker!r} was not found in text: {text!r}")
    end = len(text)
    if end_marker:
        end = text.find(end_marker, start)
        if end < 0:
            raise ValueError(f"Qualitative text span end {end_marker!r} was not found after start in text: {text!r}")
    if end <= start:
        raise ValueError(f"Qualitative text span end must occur after start: {span!r}")
    clipped = text[start:end].strip(" ,;:")
    if start > 0:
        clipped = "..." + clipped
    if end < len(text):
        clipped = clipped.rstrip(" ,;:") + "..."
    return clipped


def _candidate_text_span(candidate: Mapping[str, Any], key: str) -> Any:
    spans = dict(candidate.get("text_spans") or {})
    return spans.get(key)


def _highlight_text(
    value: Any,
    *,
    highlights: list[str],
    color: str,
    max_chars: int,
    span: Any = None,
    style: str = "fill",
) -> str:
    text = _span_table_text(value, max_chars=max_chars, span=span)

    def highlighted_phrase_latex(phrase: str) -> str:
        parts = re.split(r"(\s+)", phrase)
        latex_parts: list[str] = []
        part_index = 0
        command = r"\vqagtul" if style == "underline" else r"\vqahl"
        while part_index < len(parts):
            part = parts[part_index]
            if not part:
                part_index += 1
                continue
            if part.isspace():
                latex_parts.append(_latex_escape_raw(part))
                part_index += 1
                continue
            if part_index + 1 < len(parts) and parts[part_index + 1].isspace():
                part = part + parts[part_index + 1]
                part_index += 2
            else:
                part_index += 1
            latex_parts.append(command + r"{" + color + r"}{" + _latex_escape_raw(part) + r"}")
        return "".join(latex_parts)

    matches: list[tuple[int, int, str]] = []
    for highlight in highlights:
        phrase = clean_text(highlight)
        if not phrase:
            continue
        start = text.find(phrase)
        if start < 0:
            raise ValueError(f"Qualitative table highlight {phrase!r} was not found in clipped text: {text!r}")
        matches.append((start, start + len(phrase), phrase))
    matches.sort(key=lambda item: item[0])

    pieces: list[str] = []
    cursor = 0
    for start, end, _phrase in matches:
        if start < cursor:
            raise ValueError(f"Qualitative table highlights overlap in clipped text: {text!r}")
        pieces.append(_latex_escape_raw(text[cursor:start]))
        pieces.append(highlighted_phrase_latex(text[start:end]))
        cursor = end
    pieces.append(_latex_escape_raw(text[cursor:]))
    return "".join(pieces)


def _qualitative_prediction_row(
    predictions: pd.DataFrame,
    *,
    candidate: Mapping[str, Any],
    model: str,
) -> Mapping[str, Any]:
    if "question_id" not in candidate or "repeat_id" not in candidate:
        raise ValueError("Each qualitative_open_ended candidate must define question_id and repeat_id.")
    mask = (
        (predictions["question_id"].astype("int64") == int(candidate["question_id"]))
        & (predictions["repeat_id"].astype("int64") == int(candidate["repeat_id"]))
        & (predictions["model_display_name"].astype(str) == model)
    )
    for column in ["task_category", "case_id", "project_id", "modality_combination_name"]:
        value = clean_text(candidate.get(column))
        if value:
            mask &= predictions[column].astype(str) == value
    rows = predictions.loc[mask]
    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one prediction row for model={model!r}, candidate={dict(candidate)!r}; got {len(rows)}."
        )
    return rows.iloc[0].to_dict()


def _qualitative_open_ended_table(predictions: pd.DataFrame, cfg: Mapping[str, Any]) -> str:
    all_tables_cfg = dict(cfg.get("tables") or {})
    table_cfg = dict(all_tables_cfg.get("qualitative_open_ended") or {})
    if not table_cfg or not bool(table_cfg.get("enabled", False)):
        return ""

    colors = _global_colors(cfg)
    table_colors = dict(table_cfg.get("colors") or {})
    header_bg = clean_text(table_colors.get("header_bg")) or clean_text(colors.get("subheading_bg", "F3F5FC"))
    question_bg = clean_text(table_colors.get("question_bg")) or "F8FAFC"
    gpt_highlight = clean_text(table_colors.get("gpt_highlight", "FEE2E2"))
    our_highlight = clean_text(table_colors.get("our_highlight", "DCFCE7"))
    gt_highlight = clean_text(table_colors.get("gt_highlight")) or our_highlight
    display_cfg = dict(cfg.get("display_names") or {})
    model_display = dict(display_cfg.get("models") or {})
    gpt_model = clean_text(table_cfg.get("gpt_model")) or "gpt_5_4"
    our_model = clean_text(table_cfg.get("our_model")) or "oncovlm_qwen_lora"
    gpt_label = clean_text(table_cfg.get("gpt_column_label")) or _model_metadata(model_display, gpt_model)["name"]
    our_label = clean_text(table_cfg.get("our_column_label")) or _model_metadata(model_display, our_model)["name"]

    width_cfg = dict(table_cfg.get("column_widths") or {})
    question_width = clean_text(width_cfg.get("question")) or "3.0cm"
    gpt_width = clean_text(width_cfg.get("gpt")) or "4.4cm"
    our_width = clean_text(width_cfg.get("our")) or "4.4cm"
    gt_width = clean_text(width_cfg.get("gt")) or "4.4cm"
    max_cfg = dict(table_cfg.get("max_chars") or {})
    question_max_chars = int(max_cfg.get("question", 220))
    output_max_chars = int(max_cfg.get("output", 380))
    gt_max_chars = int(max_cfg.get("gt", 380))
    tabcolsep = clean_text(table_cfg.get("tabcolsep")) or "3pt"
    arraystretch = clean_text(table_cfg.get("arraystretch")) or "1.08"
    row_gap = clean_text(table_cfg.get("row_gap")) or "0.28em"

    candidates = list(table_cfg.get("candidates") or [])
    if not candidates:
        return ""

    column_spec = (
        _ragged_small_p_column(question_width)
        + _justified_small_p_column(gpt_width)
        + _justified_small_p_column(our_width)
        + _justified_small_p_column(gt_width)
    )
    lines = [
        r"\begingroup",
        r"\setlength{\tabcolsep}{" + tabcolsep + r"}",
        r"\renewcommand{\arraystretch}{" + arraystretch + r"}",
        r"\pretolerance=0",
        r"\tolerance=200",
        r"\hyphenpenalty=0",
        r"\exhyphenpenalty=0",
        r"\doublehyphendemerits=0",
        r"\finalhyphendemerits=0",
        r"\spaceskip=0.33em plus 0.04em minus 0.04em",
        r"\xspaceskip=0.45em plus 0.05em minus 0.04em",
        r"\emergencystretch=0pt",
        r"\makeatletter",
        r"\@ifundefined{hl}{\providecommand{\vqahl}[2]{\begingroup\setlength{\fboxsep}{1pt}\colorbox[HTML]{#1}{\strut #2}\endgroup{}}}{\providecommand{\vqahl}[2]{\begingroup\definecolor{vqahlcolor}{HTML}{#1}\sethlcolor{vqahlcolor}\hl{#2}\endgroup{}}}",
        r"\@ifundefined{ul}{\providecommand{\vqagtul}[2]{\textcolor[HTML]{#1}{#2}}}{\providecommand{\vqagtul}[2]{\begingroup\definecolor{vqagtulcolor}{HTML}{#1}\setulcolor{vqagtulcolor}\setul{0.42ex}{0.12em}\ul{#2}\endgroup{}}}",
        r"\makeatother",
        r"\begin{tabular}{"
        + column_spec
        + r"}",
        r"\toprule",
        r"\rowcolor[HTML]{"
        + header_bg
        + r"}",
        r"\textbf{Question} & \textbf{"
        + _latex_escape(gpt_label)
        + r"} & \textbf{"
        + _latex_escape(our_label)
        + r"} & \textbf{Ground truth} \\",
        r"\midrule",
    ]
    for candidate_index, candidate in enumerate(candidates):
        gpt_row = _qualitative_prediction_row(predictions, candidate=candidate, model=gpt_model)
        our_row = _qualitative_prediction_row(predictions, candidate=candidate, model=our_model)
        cells = [
            r"\cellcolor[HTML]{"
            + question_bg
            + r"}"
            + _highlight_text(
                candidate.get("question_text") or gpt_row.get("question"),
                highlights=[],
                color=header_bg,
                max_chars=question_max_chars,
                span=_candidate_text_span(candidate, "question"),
            ),
            _highlight_text(
                gpt_row.get("predicted_answer"),
                highlights=[clean_text(item) for item in candidate.get("gpt_highlights", [])],
                color=gpt_highlight,
                max_chars=output_max_chars,
                span=_candidate_text_span(candidate, "gpt"),
            ),
            _highlight_text(
                our_row.get("predicted_answer"),
                highlights=[clean_text(item) for item in candidate.get("our_highlights", [])],
                color=our_highlight,
                max_chars=output_max_chars,
                span=_candidate_text_span(candidate, "our"),
            ),
            _highlight_text(
                our_row.get("answer"),
                highlights=[clean_text(item) for item in candidate.get("gt_highlights", [])],
                color=gt_highlight,
                max_chars=gt_max_chars,
                span=_candidate_text_span(candidate, "gt"),
                style="underline",
            ),
        ]
        if candidate_index > 0:
            lines.append(r"\addlinespace[" + row_gap + r"]")
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\endgroup",
            "",
        ]
    )
    return "\n".join(lines)


def _appendix_task_cancer_projects(
    metrics: list[Mapping[str, Any]],
    *,
    group: Mapping[str, Any],
    task_category: str,
    project_order: list[str],
    modality_combination_name: str,
) -> list[str]:
    out: list[str] = []
    for project_id in project_order:
        if any(
            record.get("metric_group") == "task_cancer_table"
            and clean_text(record.get("task_category")) == task_category
            and clean_text(record.get("project_id")) == project_id
            and clean_text(record.get("question_type")) == group["question_type"]
            and clean_text(record.get("generation_type")) == group["generation_type"]
            and clean_text(record.get("modality_combination_name")) == modality_combination_name
            for record in metrics
        ):
            out.append(project_id)
    return out


def _appendix_task_cancer_project_n(
    metrics: list[Mapping[str, Any]],
    *,
    group: Mapping[str, Any],
    task_category: str,
    project_id: str,
    modality_combination_name: str,
) -> int:
    values = [
        int(record.get("n", 0) or 0)
        for record in metrics
        if record.get("metric_group") == "task_cancer_table"
        and clean_text(record.get("task_category")) == task_category
        and clean_text(record.get("project_id")) == project_id
        and clean_text(record.get("question_type")) == group["question_type"]
        and clean_text(record.get("generation_type")) == group["generation_type"]
        and clean_text(record.get("modality_combination_name")) == modality_combination_name
    ]
    return max(values) if values else 0


def _appendix_task_cancer_project_rows(
    metrics: list[Mapping[str, Any]],
    *,
    group: Mapping[str, Any],
    task_category: str,
    project_order: list[str],
    modality_combination_name: str,
    min_questions_per_cancer: int,
    top_n_projects: int,
) -> tuple[list[str], list[str]]:
    displayed: list[str] = []
    other: list[str] = []
    for project_id in project_order:
        n = _appendix_task_cancer_project_n(
            metrics,
            group=group,
            task_category=task_category,
            project_id=project_id,
            modality_combination_name=modality_combination_name,
        )
        if n <= 0:
            continue
        if n >= min_questions_per_cancer:
            displayed.append(project_id)
        else:
            other.append(project_id)
    if top_n_projects > 0 and len(displayed) > top_n_projects:
        other.extend(displayed[top_n_projects:])
        displayed = displayed[:top_n_projects]
    return displayed, other


def _short_project_code(project_id: str) -> str:
    return clean_text(project_id).removeprefix("TCGA-")


def _appendix_other_project_label(project_ids: list[str], *, other_label: str) -> str:
    codes = ", ".join(_short_project_code(project_id) for project_id in project_ids)
    return (
        r"\parbox[c]{\linewidth}{\raggedright\textbf{"
        + _latex_escape(other_label)
        + r"}\\[-0.28em]{\fontsize{5.6}{3.4}\selectfont "
        + _latex_escape(codes)
        + r"}}"
    )


def _appendix_task_cancer_tasks(
    metrics: list[Mapping[str, Any]],
    *,
    group: Mapping[str, Any],
    task_order: list[str],
    project_order: list[str],
    modality_combination_name: str,
    min_questions_per_cancer: int,
    top_n_projects: int,
) -> list[str]:
    out: list[str] = []
    for task_category in task_order:
        displayed, other = _appendix_task_cancer_project_rows(
            metrics,
            group=group,
            task_category=task_category,
            project_order=project_order,
            modality_combination_name=modality_combination_name,
            min_questions_per_cancer=min_questions_per_cancer,
            top_n_projects=top_n_projects,
        )
        if displayed or other:
            out.append(task_category)
    return out


def _appendix_task_cancer_caption_label(
    cfg: Mapping[str, Any],
    *,
    group: Mapping[str, Any],
) -> tuple[str, str]:
    all_tables_cfg = dict(cfg.get("tables") or {})
    table_cfg = dict(all_tables_cfg.get("appendix_task_cancer_results") or {})
    colors = _global_colors(cfg)
    display_cfg = dict(cfg.get("display_names") or {})
    group_label = _group_label(group, display_cfg)
    top_n_projects = int(table_cfg.get("top_n_projects", 10))
    min_questions_per_cancer = int(table_cfg.get("min_questions_per_cancer", 4))
    metric_label = (
        r"Accuracy\% / "
        + r"\textcolor[HTML]{"
        + clean_text(colors.get("f1", "2563EB"))
        + r"}{F1}"
        if group["metric"] == "mcq"
        else _qa_metric_label(colors)
    )
    caption = _table_caption(
        table_cfg,
        (
            r"Comprehensive {group_label} results by task and cancer on the all-available modality setting. "
            r"Each task includes up to the top {top_n_projects} cancer projects with at least {min_questions_per_cancer} questions; remaining projects are pooled as Other. "
            r"Cells report {metric_label}. Values are mean and standard deviation over inference repeats when available."
        ),
        group_label=_latex_escape(group_label),
        top_n_projects=str(top_n_projects),
        min_questions_per_cancer=str(min_questions_per_cancer),
        metric_label=metric_label,
    )
    label_id = clean_text(group.get("id")).replace("_", "-")
    return caption, f"tab:appendix-{label_id}"


def _appendix_task_cancer_table(
    metrics_blob: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    group: Mapping[str, Any],
) -> str:
    metrics = list(metrics_blob.get("metrics") or [])
    if any("repeat_id" in record for record in metrics):
        raise ValueError("metrics.json is in the old repeat-level format. Rerun score_vqa_predictions.py.")
    all_tables_cfg = dict(cfg.get("tables") or {})
    table_cfg = dict(all_tables_cfg.get("appendix_task_cancer_results") or {})
    if not table_cfg or not bool(table_cfg.get("enabled", False)):
        return ""

    colors = _global_colors(cfg)
    display_cfg = dict(cfg.get("display_names") or {})
    model_display = dict(display_cfg.get("models") or {})
    task_display = dict(display_cfg.get("task_categories") or {})
    project_display = dict(display_cfg.get("projects") or {})
    models, _ = _model_order_and_vrules(metrics, table_cfg)
    selected_models = _selected_models(cfg)
    if not models:
        raise RuntimeError("No models found in metrics.json.")

    modality_combination_name = clean_text(table_cfg.get("modality_combination_name")) or "all_available"
    top_n_projects = int(table_cfg.get("top_n_projects", 10))
    min_questions_per_cancer = int(table_cfg.get("min_questions_per_cancer", 4))
    other_label = clean_text(table_cfg.get("other_label")) or "Other"
    project_order = [clean_text(key) for key in table_cfg.get("projects_by_case_count", []) if clean_text(key)]
    if not project_order:
        project_order = sorted(
            {
                clean_text(record.get("project_id"))
                for record in metrics
                if record.get("metric_group") == "task_cancer_table" and clean_text(record.get("project_id")) not in {"", "ALL"}
            }
        )
    if not project_order:
        return ""

    task_order = [clean_text(key) for key in table_cfg.get("task_categories", []) if clean_text(key)]
    if not task_order:
        task_order = _task_order(metrics, task_display)
    task_order = _appendix_task_cancer_tasks(
        metrics,
        group=group,
        task_order=task_order,
        project_order=project_order,
        modality_combination_name=modality_combination_name,
        min_questions_per_cancer=min_questions_per_cancer,
        top_n_projects=top_n_projects,
    )
    if not task_order:
        return ""

    main_table_cfg = dict(all_tables_cfg.get("main_result") or {})
    result_column_width = _result_column_width(main_table_cfg)
    project_column_width = _resolved_first_column_width(
        table_cfg.get("project_column_width"),
        fixed_widths=[result_column_width] * len(models),
        column_count=len(models) + 1,
        default="2.25cm",
    )
    column_spec = _ragged_m_column(project_column_width)
    column_spec += _centered_m_column(result_column_width) * len(models)
    header_cells = [r"\textbf{Cancer}"]
    header_cells.extend(r"\textbf{" + _latex_escape(_model_metadata(model_display, model)["name"]) + r"}" for model in models)
    num_columns = len(models) + 1

    metadata_header_lines = [
        r"\rowcolor[HTML]{"
        + clean_text(colors.get("model_header_bg", "F3F6FA"))
        + r"}",
        _model_metadata_row("Name", "name", models, model_display, selected_models=selected_models, colors=colors),
        r"\rowcolor[HTML]{"
        + clean_text(colors.get("model_header_bg", "F3F6FA"))
        + r"}",
        _model_metadata_row("", "backbone", models, model_display, selected_models=selected_models, colors=colors),
        r"\rowcolor[HTML]{"
        + clean_text(colors.get("model_header_bg", "F3F6FA"))
        + r"}",
        _model_metadata_row("Finetuned", "finetuned", models, model_display, selected_models=selected_models, colors=colors),
        r"\rowcolor[HTML]{"
        + clean_text(colors.get("model_header_bg", "F3F6FA"))
        + r"}",
        _model_metadata_row("Use projector", "use_projector", models, model_display, selected_models=selected_models, colors=colors),
        *_spaced_colored_hline(colors, before="0.12em", after="0.20em"),
        r"\rowcolor[HTML]{"
        + clean_text(colors.get("subheading_bg", "EFF6FF"))
        + r"}",
        " & ".join(header_cells) + r" \\",
    ]

    lines = [
        r"\begingroup",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.18}",
        *_cellmetric_macro(stdev_scriptsize=True),
        r"\providecommand{\bertcell}[1]{#1}",
        r"\begin{longtable}{"
        + column_spec
        + r"}",
        r"\toprule",
        *metadata_header_lines,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        *metadata_header_lines,
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]

    first_task = True
    for task_category in task_order:
        project_ids, other_project_ids = _appendix_task_cancer_project_rows(
            metrics,
            group=group,
            task_category=task_category,
            project_order=project_order,
            modality_combination_name=modality_combination_name,
            min_questions_per_cancer=min_questions_per_cancer,
            top_n_projects=top_n_projects,
        )
        if not project_ids and not other_project_ids:
            continue
        if not first_task:
            lines.append(r"\addlinespace[0.35em]")
        first_task = False
        lines.extend(
            [
                _colored_hline(colors),
                r"\rowcolor[HTML]{"
                + clean_text(colors.get("subheading_bg", "EFF6FF"))
                + r"}",
                r"\multicolumn{"
                + str(num_columns)
                + r"}{l}{\rule[-0.8ex]{0pt}{3.2ex}\textbf{"
                + _latex_escape(task_display.get(task_category, task_category))
                + r"}} \\",
                _colored_hline(colors),
                r"\noalign{\vskip 0.24em}",
            ]
        )

        task_records_by_model: dict[str, list[Mapping[str, Any]]] = {model: [] for model in models}
        row_project_groups = [
            (_latex_escape(project_display.get(project_id, project_id)), [project_id])
            for project_id in project_ids
        ]
        if other_project_ids:
            row_project_groups.append(
                (
                    _appendix_other_project_label(other_project_ids, other_label=other_label),
                    other_project_ids,
                )
            )

        for project_label, row_project_ids in row_project_groups:
            row_aggregates: dict[str, dict[str, Any] | None] = {}
            for model in models:
                records: list[Mapping[str, Any]] = []
                for project_id in row_project_ids:
                    records.extend(
                        _task_cancer_records(
                            metrics,
                            model=model,
                            task_category=task_category,
                            project_id=project_id,
                            question_type=group["question_type"],
                            generation_type=group["generation_type"],
                            modality_combination_name=modality_combination_name,
                        )
                    )
                task_records_by_model[model].extend(records)
                row_aggregates[model] = _aggregate_records(records, group["metric"])

            cells = [
                project_label,
            ]
            if group["metric"] == "mcq":
                acc_values = [_metric_value(row_aggregates[model], "accuracy") for model in models]
                f1_values = [_metric_value(row_aggregates[model], "f1_macro") for model in models]
                for model in models:
                    aggregate = row_aggregates[model]
                    cells.append(
                        _selected_cell(
                            _metric_cell(
                                aggregate,
                                group["metric"],
                                colors,
                                bold_first=_is_row_best(_metric_value(aggregate, "accuracy"), acc_values),
                                bold_second=_is_row_best(_metric_value(aggregate, "f1_macro"), f1_values),
                                underline_first=_is_row_second_best(_metric_value(aggregate, "accuracy"), acc_values),
                                underline_second=_is_row_second_best(_metric_value(aggregate, "f1_macro"), f1_values),
                            ),
                            model=model,
                            selected_models=selected_models,
                            colors=colors,
                        )
                    )
            else:
                bert_values = [_metric_value(row_aggregates[model], "bertscore_f1_mean") for model in models]
                rouge_values = [_metric_value(row_aggregates[model], "rouge_l_f1_mean") for model in models]
                for model in models:
                    aggregate = row_aggregates[model]
                    cells.append(
                        _selected_cell(
                            _metric_cell(
                                aggregate,
                                group["metric"],
                                colors,
                                bold_first=_is_row_best(_metric_value(aggregate, "bertscore_f1_mean"), bert_values),
                                bold_second=_is_row_best(_metric_value(aggregate, "rouge_l_f1_mean"), rouge_values),
                                underline_first=_is_row_second_best(_metric_value(aggregate, "bertscore_f1_mean"), bert_values),
                                underline_second=_is_row_second_best(_metric_value(aggregate, "rouge_l_f1_mean"), rouge_values),
                            ),
                            model=model,
                            selected_models=selected_models,
                            colors=colors,
                        )
                    )
            lines.append(" & ".join(cells) + r" \\")

        avg_aggregates = {
            model: _aggregate_records(task_records_by_model[model], group["metric"])
            for model in models
        }
        cells = [r"\textbf{Avg.}"]
        if group["metric"] == "mcq":
            acc_values = [_metric_value(avg_aggregates[model], "accuracy") for model in models]
            f1_values = [_metric_value(avg_aggregates[model], "f1_macro") for model in models]
            for model in models:
                aggregate = avg_aggregates[model]
                cells.append(
                    _selected_cell(
                        _metric_cell(
                            aggregate,
                            group["metric"],
                            colors,
                            bold_first=_is_row_best(_metric_value(aggregate, "accuracy"), acc_values),
                            bold_second=_is_row_best(_metric_value(aggregate, "f1_macro"), f1_values),
                            underline_first=_is_row_second_best(_metric_value(aggregate, "accuracy"), acc_values),
                            underline_second=_is_row_second_best(_metric_value(aggregate, "f1_macro"), f1_values),
                        ),
                        model=model,
                        selected_models=selected_models,
                        colors=colors,
                    )
                )
        else:
            bert_values = [_metric_value(avg_aggregates[model], "bertscore_f1_mean") for model in models]
            rouge_values = [_metric_value(avg_aggregates[model], "rouge_l_f1_mean") for model in models]
            for model in models:
                aggregate = avg_aggregates[model]
                cells.append(
                    _selected_cell(
                        _metric_cell(
                            aggregate,
                            group["metric"],
                            colors,
                            bold_first=_is_row_best(_metric_value(aggregate, "bertscore_f1_mean"), bert_values),
                            bold_second=_is_row_best(_metric_value(aggregate, "rouge_l_f1_mean"), rouge_values),
                            underline_first=_is_row_second_best(_metric_value(aggregate, "bertscore_f1_mean"), bert_values),
                            underline_second=_is_row_second_best(_metric_value(aggregate, "rouge_l_f1_mean"), rouge_values),
                        ),
                        model=model,
                        selected_models=selected_models,
                        colors=colors,
                    )
                )
        lines.append(" & ".join(cells) + r" \\")

    lines.extend(
        [
            r"\end{longtable}",
            r"\endgroup",
            "",
        ]
    )
    return "\n".join(lines)


def _appendix_task_cancer_tables(metrics_blob: Mapping[str, Any], cfg: Mapping[str, Any]) -> list[tuple[str, str, str, str]]:
    tables: list[tuple[str, str, str, str]] = []
    for group in MAIN_GROUPS:
        table = _appendix_task_cancer_table(metrics_blob, cfg, group=group)
        if table:
            caption, label = _appendix_task_cancer_caption_label(cfg, group=group)
            tables.append((f"appendix_{clean_text(group.get('id'))}.tex", table, caption, label))
    return tables


def _preview_table_block(*, table_file: str, caption: str, label: str) -> str:
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\caption{" + caption + r"}",
            r"\label{" + label + r"}",
            r"\centering",
            r"\input{" + table_file + r"}",
            r"\end{table}",
            r"\FloatBarrier",
        ]
    )


def _preview_longtable_block(*, table_file: str, caption: str, label: str) -> str:
    return "\n".join(
        [
            r"\begingroup",
            r"\captionsetup{type=table}",
            r"\captionof{table}{"
            + caption
            + r"}",
            r"\label{"
            + label
            + r"}",
            r"\input{"
            + table_file
            + r"}",
            r"% The caption lives here; undo longtable's internal table-counter step.",
            r"\addtocounter{table}{-1}",
            r"\endgroup",
            r"\FloatBarrier",
        ]
    )


def _preview_document(blocks: list[str]) -> str:
    preamble = [
        r"\documentclass{article}",
        r"\usepackage[letterpaper,left=1.5in,textwidth=5.5in,top=0.75in,bottom=0.75in]{geometry}",
        r"\usepackage{iftex}",
        r"\ifPDFTeX",
        r"\usepackage{mathptmx}",
        r"\else",
        r"\usepackage{fontspec}",
        r"\setmainfont{Times New Roman}",
        r"\fi",
        r"\usepackage{booktabs}",
        r"\usepackage[table]{xcolor}",
        r"\usepackage{eso-pic}",
        r"\AddToShipoutPictureFG{\AtPageLowerLeft{\color[HTML]{FF3366}\hspace*{1.5in}\rule{0.4pt}{\paperheight}\hspace*{5.5in}\rule{0.4pt}{\paperheight}}}",
        r"\usepackage{soul}",
        r"\usepackage{array}",
        r"\usepackage{fontawesome5}",
        r"\usepackage{graphicx}",
        r"\usepackage{multirow}",
        r"\usepackage{longtable}",
        r"\usepackage{caption}",
        r"\usepackage{placeins}",
        r"\begin{document}",
    ]
    return "\n".join(preamble + ["\n\n\n".join(blocks), r"\end{document}", ""])


def _render_preview_pdf(preview_path: Path) -> None:
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", preview_path.name],
        cwd=preview_path.parent,
        check=True,
    )


def _reset_tables_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for tex_path in path.glob("*.tex"):
        tex_path.unlink()


def main() -> None:
    cfg = load_cfg()
    table_cfg = cfg.vqa_evaluation
    table_dict = OmegaConf.to_container(table_cfg, resolve=True)
    if not isinstance(table_dict, dict):
        raise TypeError("Resolved VQA table config must be a mapping.")

    metrics_path = _run_root(table_dict) / _run_filename(table_dict, "metrics_filename", "metrics.json")
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing VQA metrics JSON: {metrics_path}")
    predictions_path = _predictions_path(table_dict)
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Missing VQA predictions parquet: {predictions_path}")
    tables_dir = _tables_dir(table_dict)
    _reset_tables_dir(tables_dir)

    metrics_blob = json.loads(metrics_path.read_text(encoding="utf-8"))
    predictions = pd.read_parquet(predictions_path)
    main_table = _main_result_table(metrics_blob, table_dict)
    cancer_table = _cancer_result_table(metrics_blob, table_dict)
    modality_ablation_table = _modality_ablation_table(metrics_blob, table_dict)
    qualitative_table = _qualitative_open_ended_table(predictions, table_dict)
    appendix_tables = _appendix_task_cancer_tables(metrics_blob, table_dict)
    main_table_path = tables_dir / "main_result.tex"
    cancer_table_path = tables_dir / "cancer_result.tex"
    modality_ablation_path = tables_dir / "modality_ablation.tex"
    qualitative_table_path = tables_dir / "qualitative_open_ended.tex"
    preview_path = tables_dir / "preview_tables.tex"
    _write_text_atomic(main_table_path, main_table)
    all_tables_cfg = dict(table_dict.get("tables") or {})
    main_table_cfg = dict(all_tables_cfg.get("main_result") or {})
    cancer_table_cfg = dict(all_tables_cfg.get("cancer_result") or {})
    modality_table_cfg = dict(all_tables_cfg.get("modality_ablation") or {})
    model_display = dict(dict(table_dict.get("display_names") or {}).get("models") or {})

    preview_blocks = [
        _preview_table_block(
            table_file=main_table_path.name,
            caption=_table_caption(
                main_table_cfg,
                r"Main VQA benchmark results by task category on the all-available modality setting. MCQ cells report Accuracy\% and macro-F1; open-ended cells report BERTScore F1 and ROUGE-L F1. Values are mean and standard deviation over inference repeats when available.",
            ),
            label="tab:main_result",
        )
    ]
    if cancer_table:
        _write_text_atomic(cancer_table_path, cancer_table)
        preview_blocks.append(
            _preview_table_block(
                table_file=cancer_table_path.name,
                caption=_table_caption(
                    cancer_table_cfg,
                    r"VQA benchmark results by cancer type on the all-available modality setting. Each cancer is split into MCQ and open-ended rows. MCQ cells report Accuracy\% and macro-F1; open-ended cells report BERTScore F1 and ROUGE-L F1. Values are mean and standard deviation over inference repeats when available.",
                ),
                label="tab:cancer_result",
            )
        )
    if modality_ablation_table:
        _write_text_atomic(modality_ablation_path, modality_ablation_table)
        modality_model = clean_text(modality_table_cfg.get("model_display_name"))
        modality_model_label = _model_metadata(model_display, modality_model)["name"] if modality_model else ""
        preview_blocks.append(
            _preview_table_block(
                table_file=modality_ablation_path.name,
                caption=_table_caption(
                    modality_table_cfg,
                    r"VQA modality ablation for {model_label} on cases with pathology and radiology features. MCQ cells report Accuracy\% and macro-F1; open-ended cells report BERTScore F1 and ROUGE-L F1.",
                    model_label=_latex_escape(modality_model_label),
                ),
                label="tab:modality_ablation",
            )
        )
    qualitative_table_cfg = dict(all_tables_cfg.get("qualitative_open_ended") or {})
    if qualitative_table:
        _write_text_atomic(qualitative_table_path, qualitative_table)
        preview_blocks.append(
            _preview_table_block(
                table_file=qualitative_table_path.name,
                caption=_table_caption(
                    qualitative_table_cfg,
                    r"Representative open-ended VQA examples comparing GPT-5.4 and finetuned OncoVLM. Colored spans mark incorrect GPT evidence, matching OncoVLM evidence, and key ground-truth evidence.",
                ),
                label=clean_text(qualitative_table_cfg.get("label")) or "tab:qualitative_open_ended",
            )
        )
    appendix_paths: list[Path] = []
    for appendix_filename, appendix_table, appendix_caption, appendix_label in appendix_tables:
        appendix_path = tables_dir / appendix_filename
        _write_text_atomic(appendix_path, appendix_table)
        appendix_paths.append(appendix_path)
        preview_blocks.append(
            _preview_longtable_block(
                table_file=appendix_path.name,
                caption=appendix_caption,
                label=appendix_label,
            )
        )
    _write_text_atomic(preview_path, _preview_document(preview_blocks))
    _render_preview_pdf(preview_path)

    print(f"Metrics path: {metrics_path}")
    print(f"Predictions path: {predictions_path}")
    print(f"Tables dir: {tables_dir}")
    print(f"Wrote: {main_table_path}")
    if cancer_table:
        print(f"Wrote: {cancer_table_path}")
    if modality_ablation_table:
        print(f"Wrote: {modality_ablation_path}")
    if qualitative_table:
        print(f"Wrote: {qualitative_table_path}")
    for appendix_path in appendix_paths:
        print(f"Wrote: {appendix_path}")
    print(f"Wrote: {preview_path}")
    print(f"Rendered: {preview_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

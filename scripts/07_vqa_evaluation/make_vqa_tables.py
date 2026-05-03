#!/usr/bin/env python3
from __future__ import annotations

# ruff: noqa: E402

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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


def _model_metadata(model_display: Mapping[str, Any], model: str) -> dict[str, str]:
    raw = model_display.get(model)
    if raw is None:
        return {"name": model, "backbone": "--", "finetuned": "--"}
    if not isinstance(raw, Mapping):
        return {"name": clean_text(raw) or model, "backbone": "--", "finetuned": "--"}

    name = clean_text(raw.get("name")) or model
    backbone = clean_text(raw.get("backbone")) or "--"
    finetuned_raw = raw.get("finetuned")
    if isinstance(finetuned_raw, bool):
        finetuned = r"\faCheckCircle" if finetuned_raw else r"\faMinusCircle"
    else:
        finetuned = clean_text(finetuned_raw) or "--"
    return {"name": name, "backbone": backbone, "finetuned": finetuned}


def _model_metadata_row(label: str, key: str, models: list[str], model_display: Mapping[str, Any]) -> str:
    cells = [r"\textbf{" + _latex_escape(label) + r"}"]
    for model in models:
        value = _model_metadata(model_display, model)[key]
        cells.append(value if key == "finetuned" else _latex_escape(value))
    return " & ".join(cells) + r" \\"


def _model_metadata_row_two_label_columns(label: str, key: str, models: list[str], model_display: Mapping[str, Any]) -> str:
    cells = [r"\multicolumn{2}{l}{\textbf{" + _latex_escape(label) + r"}}"]
    for model in models:
        value = _model_metadata(model_display, model)[key]
        cells.append(value if key == "finetuned" else _latex_escape(value))
    return " & ".join(cells) + r" \\"


def _group_label(group: Mapping[str, str], display_cfg: Mapping[str, Any]) -> str:
    group_labels = dict(display_cfg.get("question_groups") or {})
    return clean_text(group_labels.get(clean_text(group.get("id")))) or clean_text(group.get("default_label"))


def _global_colors(cfg: Mapping[str, Any]) -> dict[str, str]:
    return dict(cfg.get("colors") or {})


def _fmt_metric(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{100.0 * value:.1f}"


def _fmt_metric_summary(mean: float | None, std: float | None, std_color: str, *, bold: bool = False) -> str:
    value_text = _fmt_metric(mean)
    if bold and value_text != "--":
        value_text = r"\textbf{" + value_text + r"}"
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
        }
    raise ValueError(f"Unknown metric kind: {metric_kind}")


def _metric_value(metrics: dict[str, Any] | None, key: str) -> float | None:
    if metrics is None or metrics.get(key) is None:
        return None
    return float(metrics[key])


def _is_row_best(value: float | None, values: list[float | None]) -> bool:
    present = [item for item in values if item is not None]
    return value is not None and bool(present) and value == max(present)


def _metric_cell(
    metrics: dict[str, Any] | None,
    metric_kind: str,
    colors: Mapping[str, str],
    *,
    bold_first: bool = False,
    bold_second: bool = False,
    show_stdev: bool = True,
) -> str:
    if metrics is None:
        return r"\multicolumn{1}{c}{--}"
    std_color = clean_text(colors.get("stdev", "64748B"))
    if metric_kind == "mcq":
        acc = _fmt_metric_summary(
            metrics.get("accuracy"),
            metrics.get("accuracy_stdev") if show_stdev else None,
            std_color,
            bold=bold_first,
        )
        f1 = _fmt_metric_summary(
            metrics.get("f1_macro"),
            metrics.get("f1_macro_stdev") if show_stdev else None,
            std_color,
            bold=bold_second,
        )
        return (
            r"\cellmetric{"
            + f"{acc}"
            + r"}{"
            + f"{f1}"
            + r"}{"
            + clean_text(colors.get("f1", "2563EB"))
            + r"}"
        )
    bert = _fmt_metric_summary(
        metrics.get("bertscore_f1_mean"),
        metrics.get("bertscore_f1_mean_stdev") if show_stdev else None,
        std_color,
        bold=bold_first,
    )
    return r"\bertcell{" + f"{bert}" + r"}"


def _model_order_and_vrules(metrics: list[Mapping[str, Any]], table_cfg: Mapping[str, Any]) -> tuple[list[str], set[int]]:
    configured = [clean_text(item) for item in table_cfg.get("model_order", []) if clean_text(item)]
    present = sorted({clean_text(record.get("model_display_name")) for record in metrics if clean_text(record.get("model_display_name"))})
    ordered: list[str] = []
    vrules_after: set[int] = set()
    for item in configured:
        if item == VRULE_TOKEN:
            vrules_after.add(len(ordered))
            continue
        if item in present and item not in ordered:
            ordered.append(item)
    ordered.extend(model for model in present if model not in ordered)
    vrules_after = {index for index in vrules_after if 0 < index < len(ordered)}
    return ordered, vrules_after


def _column_spec(model_count: int, vrules_after: set[int]) -> str:
    parts = ["l", "|"]
    for model_index in range(1, model_count + 1):
        parts.append("c")
        if model_index in vrules_after:
            parts.append("|")
    return "".join(parts)


def _two_label_column_spec(model_count: int, vrules_after: set[int]) -> str:
    parts = ["l", "l", "|"]
    for model_index in range(1, model_count + 1):
        parts.append("c")
        if model_index in vrules_after:
            parts.append("|")
    return "".join(parts)


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
    if not models:
        raise RuntimeError("No models found in metrics.json.")
    task_order = _task_order(metrics, task_display)

    num_columns = len(models) + 1
    column_spec = _column_spec(len(models), vrules_after)
    lines = [
        r"\begin{table}[t]",
        r"\caption{Main VQA benchmark results by task category on the all-available modality setting. MCQ cells report accuracy and macro-F1; open-ended cells report BERTScore F1. Values are mean $\pm$ standard deviation over inference repeats when available.}",
        r"\label{tab:main_result}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.18}",
        r"\providecommand{\cellmetric}[3]{\begin{tabular}{@{}c@{}}#1\\[-0.12em]\textcolor[HTML]{#3}{#2}\end{tabular}}",
        r"\providecommand{\bertcell}[1]{#1}",
        r"\begin{tabular}{"
        + column_spec
        + r"}",
        r"\toprule",
        r"\rowcolor[HTML]{"
        + clean_text(colors.get("header_bg", "F3F6FA"))
        + r"}",
        _model_metadata_row("Name", "name", models, model_display),
        r"\rowcolor[HTML]{"
        + clean_text(colors.get("header_bg", "F3F6FA"))
        + r"}",
        _model_metadata_row("Backbone", "backbone", models, model_display),
        r"\rowcolor[HTML]{"
        + clean_text(colors.get("header_bg", "F3F6FA"))
        + r"}",
        _model_metadata_row("Finetuned", "finetuned", models, model_display),
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
                + clean_text(colors.get("group_bg", "EFF6FF"))
                + r"}",
                r"\multicolumn{"
                + str(num_columns)
                + r"}{l}{\rule[-0.8ex]{0pt}{3.2ex}\textbf{"
                + _latex_escape(_group_label(group, display_cfg))
                + r"}"
                + (
                    r" \textnormal{("
                    + r"Accuracy/"
                    + r"\textcolor[HTML]{"
                    + clean_text(colors.get("f1", "2563EB"))
                    + r"}{F1}"
                    + r")}"
                    if group["metric"] == "mcq"
                    else r" \textnormal{(BERT-F1)}"
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
                        _metric_cell(
                            aggregate,
                            group["metric"],
                            colors,
                            bold_first=_is_row_best(_metric_value(aggregate, "accuracy"), acc_values),
                            bold_second=_is_row_best(_metric_value(aggregate, "f1_macro"), f1_values),
                        )
                    )
            else:
                bert_values = [_metric_value(row_aggregates[model], "bertscore_f1_mean") for model in models]
                for model in models:
                    aggregate = row_aggregates[model]
                    cells.append(
                        _metric_cell(
                            aggregate,
                            group["metric"],
                            colors,
                            bold_first=_is_row_best(_metric_value(aggregate, "bertscore_f1_mean"), bert_values),
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
                    _metric_cell(
                        aggregate,
                        group["metric"],
                        colors,
                        bold_first=_is_row_best(_metric_value(aggregate, "accuracy"), mean_acc_values),
                        bold_second=_is_row_best(_metric_value(aggregate, "f1_macro"), mean_f1_values),
                    )
                )
        else:
            mean_bert_values = [_metric_value(mean_aggregates[model], "bertscore_f1_mean") for model in models]
            for model in models:
                aggregate = mean_aggregates[model]
                mean_cells.append(
                    _metric_cell(
                        aggregate,
                        group["metric"],
                        colors,
                        bold_first=_is_row_best(_metric_value(aggregate, "bertscore_f1_mean"), mean_bert_values),
                    )
                )
        lines.extend(
            [
                r"\rowcolor[HTML]{"
                + clean_text(colors.get("mean_bg", "F8FAFC"))
                + r"}",
                " & ".join(mean_cells) + r" \\",
            ]
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
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
    project_display = dict(display_cfg.get("project_ids") or {})
    models, vrules_after = _model_order_and_vrules(metrics, table_cfg)
    if not models:
        raise RuntimeError("No models found in metrics.json.")
    project_order = [clean_text(key) for key in project_display.keys() if clean_text(key)]
    if not project_order:
        project_order = sorted(
            {
                clean_text(record.get("project_id"))
                for record in metrics
                if record.get("metric_group") == "cancer_table" and clean_text(record.get("project_id")) not in {"", "ALL"}
            }
        )

    column_spec = _two_label_column_spec(len(models), vrules_after)
    lines = [
        r"\begin{table}[t]",
        r"\caption{VQA benchmark results by cancer type on the all-available modality setting. Each cancer is split into MCQ and open-ended QA rows. MCQ cells report accuracy and macro-F1; QA cells report BERTScore F1. Values are mean $\pm$ standard deviation over inference repeats when available.}",
        r"\label{tab:cancer_result}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.18}",
        r"\providecommand{\cellmetric}[3]{\begin{tabular}{@{}c@{}}#1\\[-0.12em]\textcolor[HTML]{#3}{#2}\end{tabular}}",
        r"\providecommand{\bertcell}[1]{#1}",
        r"\begin{tabular}{"
        + column_spec
        + r"}",
        r"\toprule",
        r"\rowcolor[HTML]{"
        + clean_text(colors.get("header_bg", "F3F6FA"))
        + r"}",
        _model_metadata_row_two_label_columns("Name", "name", models, model_display),
        r"\rowcolor[HTML]{"
        + clean_text(colors.get("header_bg", "F3F6FA"))
        + r"}",
        _model_metadata_row_two_label_columns("Backbone", "backbone", models, model_display),
        r"\rowcolor[HTML]{"
        + clean_text(colors.get("header_bg", "F3F6FA"))
        + r"}",
        _model_metadata_row_two_label_columns("Finetuned", "finetuned", models, model_display),
        *_spaced_colored_hline(colors, before="0.12em", after="0.20em"),
        r"\rowcolor[HTML]{"
        + clean_text(colors.get("group_bg", "EFF6FF"))
        + r"}",
        r"\textbf{Cancer} & \textbf{Question} & \multicolumn{"
        + str(len(models))
        + r"}{c}{\textbf{Performance by model}} \\",
        *_spaced_colored_hline(colors, before="0.20em", after="0.28em"),
    ]

    question_rows = [("mcq", "MCQ", "mcq"), ("qa", "QA", "qa")]
    rendered_project_count = 0
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
        for row_index, (question_type, row_label, metric_kind) in enumerate(question_rows):
            row_aggregates: dict[str, dict[str, Any] | None] = {}
            for model in models:
                records = _cancer_records(
                    metrics,
                    model=model,
                    project_id=project_id,
                    question_type=question_type,
                    generation_type=None,
                    modality_combination_name=modality_combination_name,
                )
                row_aggregates[model] = _aggregate_records(records, metric_kind)
            if all(aggregate is None for aggregate in row_aggregates.values()):
                continue
            project_rows.append((row_label, metric_kind, row_aggregates))
        if not project_rows:
            continue

        if rendered_project_count > 0:
            lines.extend(_spaced_colored_hline(colors, before="0.30em", after="0.30em"))
        rendered_project_count += 1

        row_count = len(project_rows)
        for row_index, (row_label, metric_kind, row_aggregates) in enumerate(project_rows):
            cells = [
                (
                    r"\multirow[c]{"
                    + str(row_count)
                    + r"}{*}{\textbf{"
                    + _latex_escape(project_display.get(project_id, project_id))
                    + r"}}"
                    if row_index == 0 and row_count > 1
                    else r"\textbf{" + _latex_escape(project_display.get(project_id, project_id)) + r"}"
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
                        _metric_cell(
                            aggregate,
                            metric_kind,
                            colors,
                            bold_first=_is_row_best(_metric_value(aggregate, "accuracy"), acc_values),
                            bold_second=_is_row_best(_metric_value(aggregate, "f1_macro"), f1_values),
                        )
                    )
            else:
                bert_values = [_metric_value(row_aggregates[model], "bertscore_f1_mean") for model in models]
                for model in models:
                    aggregate = row_aggregates[model]
                    cells.append(
                        _metric_cell(
                            aggregate,
                            metric_kind,
                            colors,
                            bold_first=_is_row_best(_metric_value(aggregate, "bertscore_f1_mean"), bert_values),
                        )
                    )
            lines.append(" & ".join(cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
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
    model = clean_text(table_cfg.get("model_display_name"))
    if not model:
        raise ValueError("tables.modality_ablation.model_display_name must be populated.")
    model_label = _model_metadata(model_display, model)["name"]
    modality_names = [clean_text(key) for key in table_cfg.get("modality_combination_names", []) if clean_text(key)]
    if not modality_names:
        modality_names = [clean_text(key) for key in modality_display.keys() if clean_text(key)]
    task_order = [clean_text(key) for key in table_cfg.get("task_categories", []) if clean_text(key)]
    if not task_order:
        task_order = _task_order(metrics, task_display)

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

    column_spec = "l|" + "|".join("c" * len(tasks) for _, tasks in group_blocks)
    group_header_cells: list[str] = [r"\multicolumn{1}{l}{}"]
    task_header_cells: list[str] = [r"\textbf{Modality}"]
    for block_index, (group, tasks) in enumerate(group_blocks):
        metric_label = (
            r"Accuracy/"
            + r"\textcolor[HTML]{"
            + clean_text(colors.get("f1", "2563EB"))
            + r"}{F1}"
            if group["metric"] == "mcq"
            else r"BERT-F1"
        )
        group_header_cells.append(
            r"\multicolumn{"
            + str(len(tasks))
            + r"}{c}{\textbf{"
            + _latex_escape(_group_label(group, display_cfg))
            + r"} \textnormal{("
            + metric_label
            + r")}}"
        )
        task_header_cells.extend(r"\textbf{" + _latex_escape(task_display.get(task, task)) + r"}" for task in tasks)

    lines = [
        r"\begin{table}[t]",
        r"\caption{VQA modality ablation for "
        + _latex_escape(model_label)
        + r" on matched base questions from cases with pathology and radiology features. MCQ cells report accuracy and macro-F1; open-ended cells report BERTScore F1.}",
        r"\label{tab:modality_ablation}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.18}",
        r"\providecommand{\cellmetric}[3]{\begin{tabular}{@{}c@{}}#1\\[-0.12em]\textcolor[HTML]{#3}{#2}\end{tabular}}",
        r"\providecommand{\bertcell}[1]{#1}",
        r"\begin{tabular}{"
        + column_spec
        + r"}",
        r"\toprule",
        r"\rowcolor[HTML]{"
        + clean_text(colors.get("header_bg", "F3F6FA"))
        + r"}",
        " & ".join(group_header_cells) + r" \\",
        r"\rowcolor[HTML]{"
        + clean_text(colors.get("header_bg", "F3F6FA"))
        + r"}",
        " & ".join(task_header_cells) + r" \\",
        _colored_hline(colors),
        r"\noalign{\vskip 0.28em}",
    ]

    for modality_name in modality_names:
        cells = [r"\textbf{" + _latex_escape(modality_display.get(modality_name, modality_name)) + r"}"]
        for group, group_tasks in group_blocks:
            task_aggregates: dict[str, dict[str, Any] | None] = {}
            for task in group_tasks:
                records = _group_modality_records(
                    metrics,
                    model=model,
                    task_category=task,
                    question_type=group["question_type"],
                    generation_type=group["generation_type"],
                    modality_combination_name=modality_name,
                )
                task_aggregates[task] = _aggregate_records(records, group["metric"])
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
                    aggregate = task_aggregates[task]
                    cells.append(
                        _metric_cell(
                            aggregate,
                            group["metric"],
                            colors,
                            bold_first=_is_row_best(_metric_value(aggregate, "accuracy"), acc_values),
                            bold_second=_is_row_best(_metric_value(aggregate, "f1_macro"), f1_values),
                            show_stdev=False,
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
                    aggregate = task_aggregates[task]
                    cells.append(
                        _metric_cell(
                            aggregate,
                            group["metric"],
                            colors,
                            bold_first=_is_row_best(_metric_value(aggregate, "bertscore_f1_mean"), bert_values),
                            show_stdev=False,
                        )
                    )
        lines.append(" & ".join(cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def _preview_document(table_files: list[str]) -> str:
    imports = [r"\input{" + table_file + r"}" for table_file in table_files]
    return "\n".join(
        [
            r"\documentclass{article}",
            r"\usepackage[margin=0.45in]{geometry}",
            r"\usepackage{iftex}",
            r"\ifPDFTeX",
            r"\usepackage{mathptmx}",
            r"\else",
            r"\usepackage{fontspec}",
            r"\setmainfont{Times New Roman}",
            r"\fi",
            r"\usepackage{booktabs}",
            r"\usepackage[table]{xcolor}",
            r"\usepackage{array}",
            r"\usepackage{fontawesome5}",
            r"\usepackage{graphicx}",
            r"\usepackage{multirow}",
            r"\usepackage{caption}",
            r"\begin{document}",
            *imports,
            r"\end{document}",
            "",
        ]
    )


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
    tables_dir = _tables_dir(table_dict)
    _reset_tables_dir(tables_dir)

    metrics_blob = json.loads(metrics_path.read_text(encoding="utf-8"))
    main_table = _main_result_table(metrics_blob, table_dict)
    cancer_table = _cancer_result_table(metrics_blob, table_dict)
    modality_ablation_table = _modality_ablation_table(metrics_blob, table_dict)
    main_table_path = tables_dir / "main_result.tex"
    cancer_table_path = tables_dir / "cancer_result.tex"
    modality_ablation_path = tables_dir / "modality_ablation.tex"
    preview_path = tables_dir / "preview_tables.tex"
    _write_text_atomic(main_table_path, main_table)
    table_files = [main_table_path.name]
    if cancer_table:
        _write_text_atomic(cancer_table_path, cancer_table)
        table_files.append(cancer_table_path.name)
    if modality_ablation_table:
        _write_text_atomic(modality_ablation_path, modality_ablation_table)
        table_files.append(modality_ablation_path.name)
    _write_text_atomic(preview_path, _preview_document(table_files))
    _render_preview_pdf(preview_path)

    print(f"Metrics path: {metrics_path}")
    print(f"Tables dir: {tables_dir}")
    print(f"Wrote: {main_table_path}")
    if cancer_table:
        print(f"Wrote: {cancer_table_path}")
    if modality_ablation_table:
        print(f"Wrote: {modality_ablation_path}")
    print(f"Wrote: {preview_path}")
    print(f"Rendered: {preview_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

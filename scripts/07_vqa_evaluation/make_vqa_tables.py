#!/usr/bin/env python3
from __future__ import annotations

# ruff: noqa: E402

import json
import math
import os
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

MAIN_GROUPS = [
    {
        "label": "MCQ from ground truth",
        "question_type": "mcq",
        "generation_type": "from_ground_truth",
        "metric": "mcq",
    },
    {
        "label": "MCQ from caption",
        "question_type": "mcq",
        "generation_type": "from_caption",
        "metric": "mcq",
    },
    {
        "label": "Open-ended from caption",
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


def _fmt_metric(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{100.0 * value:.1f}"


def _safe_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _fmt_metric_summary(mean: float | None, std: float | None) -> str:
    value_text = _fmt_metric(mean)
    if std is None or value_text == "--":
        return value_text
    return value_text + r"\,{\scriptsize $\pm$ " + _fmt_metric(std) + r"}"


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
            "f1_macro": _weighted_mean(records, "f1_macro"),
        }
    if metric_kind == "qa":
        return {
            "n": n,
            "bertscore_f1_mean": _weighted_mean(records, "bertscore_f1_mean"),
        }
    raise ValueError(f"Unknown metric kind: {metric_kind}")


def _metric_value(metrics: dict[str, Any] | None, key: str) -> float | None:
    if metrics is None or metrics.get(key) is None:
        return None
    return float(metrics[key])


def _repeat_summary_values(records: list[Mapping[str, Any]], metric_kind: str) -> dict[str, Any] | None:
    if not records:
        return None
    repeat_ids = sorted({int(record["repeat_id"]) for record in records})
    repeat_aggregates = [
        _aggregate_records(
            [record for record in records if int(record["repeat_id"]) == repeat_id],
            metric_kind,
        )
        for repeat_id in repeat_ids
    ]
    repeat_aggregates = [aggregate for aggregate in repeat_aggregates if aggregate is not None]
    if not repeat_aggregates:
        return None

    out: dict[str, Any] = {"repeat_count": len(repeat_aggregates)}
    keys = ["accuracy", "f1_macro"] if metric_kind == "mcq" else ["bertscore_f1_mean"]
    for key in keys:
        values = [float(aggregate[key]) for aggregate in repeat_aggregates if aggregate.get(key) is not None]
        out[f"{key}_mean"] = sum(values) / len(values) if values else None
        out[f"{key}_std"] = _safe_std(values)
    return out


def _is_row_best(value: float | None, values: list[float | None]) -> bool:
    present = [item for item in values if item is not None]
    return value is not None and bool(present) and value == max(present)


def _bold_if(value: str, enabled: bool) -> str:
    if not enabled or value == "--":
        return value
    return r"\textbf{" + value + r"}"


def _metric_cell(
    metrics: dict[str, Any] | None,
    metric_kind: str,
    colors: Mapping[str, str],
    *,
    bold_first: bool = False,
    bold_second: bool = False,
) -> str:
    if metrics is None:
        return r"\multicolumn{1}{c}{--}"
    if metric_kind == "mcq":
        acc = _bold_if(_fmt_metric_summary(metrics.get("accuracy_mean"), metrics.get("accuracy_std")), bold_first)
        f1 = _bold_if(_fmt_metric_summary(metrics.get("f1_macro_mean"), metrics.get("f1_macro_std")), bold_second)
        return (
            r"\cellmetric{"
            + f"{acc}"
            + r"}{"
            + f"{f1}"
            + r"}{"
            + clean_text(colors.get("f1", "2563EB"))
            + r"}"
        )
    bert = _bold_if(_fmt_metric_summary(metrics.get("bertscore_f1_mean_mean"), metrics.get("bertscore_f1_mean_std")), bold_first)
    return r"\bertcell{" + f"{bert}" + r"}"


def _model_order(metrics: list[Mapping[str, Any]], table_cfg: Mapping[str, Any]) -> list[str]:
    configured = [clean_text(item) for item in table_cfg.get("model_order", []) if clean_text(item)]
    present = sorted({clean_text(record.get("model_display_name")) for record in metrics if clean_text(record.get("model_display_name"))})
    ordered = [model for model in configured if model in present]
    ordered.extend(model for model in present if model not in ordered)
    return ordered


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
        if record.get("metric_group") == "core_slice"
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


def _main_result_table(metrics_blob: Mapping[str, Any], cfg: Mapping[str, Any]) -> str:
    metrics = list(metrics_blob.get("metrics") or [])
    if metrics and any("repeat_id" not in record for record in metrics):
        raise ValueError("metrics.json records must contain repeat_id. Rerun score_vqa_predictions.py.")
    table_cfg = dict(dict(cfg.get("tables") or {}).get("main_result") or {})
    colors = dict(table_cfg.get("colors") or {})
    modality_combination_name = clean_text(table_cfg.get("modality_combination_name")) or "all_available"
    display_cfg = dict(cfg.get("display_names") or {})
    model_display = dict(display_cfg.get("models") or {})
    task_display = dict(display_cfg.get("task_categories") or {})
    models = _model_order(metrics, table_cfg)
    if not models:
        raise RuntimeError("No models found in metrics.json.")
    task_order = _task_order(metrics, task_display)

    num_columns = len(models) + 1
    column_spec = "l" + ("c" * len(models))
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.18}",
        r"\newcommand{\cellmetric}[3]{\begin{tabular}{@{}c@{}}#1\\[-0.12em]\textcolor[HTML]{#3}{#2}\end{tabular}}",
        r"\newcommand{\bertcell}[1]{#1}",
        r"\begin{tabular}{"
        + column_spec
        + r"}",
        r"\toprule",
        r"\rowcolor[HTML]{"
        + clean_text(colors.get("header_bg", "F3F6FA"))
        + r"}",
        r"\textbf{Task} & "
        + " & ".join(r"\textbf{" + _latex_escape(model_display.get(model, model)) + r"}" for model in models)
        + r" \\",
        r"\midrule",
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
                r"\rowcolor[HTML]{"
                + clean_text(colors.get("group_bg", "EFF6FF"))
                + r"}",
                r"\multicolumn{"
                + str(num_columns)
                + r"}{l}{\textbf{"
                + _latex_escape(group["label"])
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
                row_aggregates[model] = _repeat_summary_values(records, group["metric"])
            if group["metric"] == "mcq":
                acc_values = [_metric_value(row_aggregates[model], "accuracy_mean") for model in models]
                f1_values = [_metric_value(row_aggregates[model], "f1_macro_mean") for model in models]
                for model in models:
                    aggregate = row_aggregates[model]
                    cells.append(
                        _metric_cell(
                            aggregate,
                            group["metric"],
                            colors,
                            bold_first=_is_row_best(_metric_value(aggregate, "accuracy_mean"), acc_values),
                            bold_second=_is_row_best(_metric_value(aggregate, "f1_macro_mean"), f1_values),
                        )
                    )
            else:
                bert_values = [_metric_value(row_aggregates[model], "bertscore_f1_mean_mean") for model in models]
                for model in models:
                    aggregate = row_aggregates[model]
                    cells.append(
                        _metric_cell(
                            aggregate,
                            group["metric"],
                            colors,
                            bold_first=_is_row_best(_metric_value(aggregate, "bertscore_f1_mean_mean"), bert_values),
                        )
                    )
            lines.append(" & ".join(cells) + r" \\")

        mean_cells = [r"\textbf{Mean}"]
        mean_aggregates = {
            model: _repeat_summary_values(group_model_records[model], group["metric"])
            for model in models
        }
        if group["metric"] == "mcq":
            mean_acc_values = [_metric_value(mean_aggregates[model], "accuracy_mean") for model in models]
            mean_f1_values = [_metric_value(mean_aggregates[model], "f1_macro_mean") for model in models]
            for model in models:
                aggregate = mean_aggregates[model]
                mean_cells.append(
                    _metric_cell(
                        aggregate,
                        group["metric"],
                        colors,
                        bold_first=_is_row_best(_metric_value(aggregate, "accuracy_mean"), mean_acc_values),
                        bold_second=_is_row_best(_metric_value(aggregate, "f1_macro_mean"), mean_f1_values),
                    )
                )
        else:
            mean_bert_values = [_metric_value(mean_aggregates[model], "bertscore_f1_mean_mean") for model in models]
            for model in models:
                aggregate = mean_aggregates[model]
                mean_cells.append(
                    _metric_cell(
                        aggregate,
                        group["metric"],
                        colors,
                        bold_first=_is_row_best(_metric_value(aggregate, "bertscore_f1_mean_mean"), mean_bert_values),
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
            r"\caption{Main VQA benchmark results by task category on the all-available modality setting. MCQ cells report accuracy and macro-F1; open-ended cells report BERTScore F1. Values are mean $\pm$ standard deviation over inference repeats when available.}",
            r"\label{tab:main_result}",
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
            r"\usepackage{booktabs}",
            r"\usepackage[table]{xcolor}",
            r"\usepackage{array}",
            r"\usepackage{graphicx}",
            r"\usepackage{caption}",
            r"\begin{document}",
            *imports,
            r"\end{document}",
            "",
        ]
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
    main_table_path = tables_dir / "main_result.tex"
    preview_path = tables_dir / "preview_tables.tex"
    _write_text_atomic(main_table_path, main_table)
    _write_text_atomic(preview_path, _preview_document([main_table_path.name]))

    print(f"Metrics path: {metrics_path}")
    print(f"Tables dir: {tables_dir}")
    print(f"Wrote: {main_table_path}")
    print(f"Wrote: {preview_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

# ruff: noqa: E402

import json
import math
import os
import sys
import textwrap
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
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


def load_cfg():
    return load_script_cfg(
        repo_root=ROOT,
        config_relative_path="07_vqa_evaluation/create_figures.yaml",
        overrides=sys.argv[1:],
    )


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _run_root(cfg: Mapping[str, Any]) -> Path:
    run_cfg = dict(cfg.get("run") or {})
    run_name = clean_text(run_cfg.get("name"))
    if not run_name:
        raise ValueError("figures.run.name must be populated.")
    return _resolve_path(run_cfg.get("output_root", "results")) / run_name


def _run_filename(cfg: Mapping[str, Any], key: str, default: str) -> str:
    value = clean_text(dict(cfg.get("run") or {}).get(key)) or default
    if "/" in value or "\\" in value:
        raise ValueError(f"figures.run.{key} must be a file name, got {value!r}.")
    return value


def _metrics_path(cfg: Mapping[str, Any]) -> Path:
    path = _run_root(cfg) / _run_filename(cfg, "metrics_filename", "metrics.json")
    if not path.is_file():
        raise FileNotFoundError(f"Missing metrics JSON: {path}")
    return path


def _tables_dir(cfg: Mapping[str, Any]) -> Path:
    path = _run_root(cfg) / _run_filename(cfg, "tables_dirname", "tables")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _hex(value: str) -> str:
    text = clean_text(value).lstrip("#")
    return "#" + text


def _set_font(font_family: str) -> None:
    matches = [font.name for font in font_manager.fontManager.ttflist]
    if font_family not in matches:
        print(f"Font family {font_family!r} was not found by matplotlib; using the default font.")
    plt.rcParams.update(
        {
            "font.family": font_family,
            "svg.fonttype": "path",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
            "font.style": "normal",
        }
    )


def _metric_for_record(record: Mapping[str, Any], selected_metrics: Mapping[str, str]) -> float:
    question_type = clean_text(record.get("question_type"))
    metric_key = clean_text(selected_metrics.get(question_type))
    if not metric_key:
        raise ValueError(f"No radar metric key configured for question_type={question_type!r}.")
    value = record.get(metric_key)
    if value is None:
        raise ValueError(f"Metric record is missing {metric_key!r}: {record}")
    return float(value)


def _radar_values(metrics: list[Mapping[str, Any]], radar_cfg: Mapping[str, Any]) -> tuple[list[str], list[str], dict[str, list[float]]]:
    model_cfg = dict(radar_cfg.get("models") or {})
    task_cfg = dict(radar_cfg.get("task_categories") or {})
    selected_metrics = dict(radar_cfg.get("selected_metrics") or {})
    metric_group = clean_text(radar_cfg.get("metric_group")) or "main_table"
    modality = clean_text(radar_cfg.get("modality_combination_name")) or "all_available"

    model_ids = [clean_text(model_id) for model_id in model_cfg.keys() if clean_text(model_id)]
    task_ids = [clean_text(task_id) for task_id in task_cfg.keys() if clean_text(task_id)]
    values: dict[str, list[float]] = {model_id: [] for model_id in model_ids}

    for task_id in task_ids:
        for model_id in model_ids:
            records = [
                record
                for record in metrics
                if clean_text(record.get("metric_group")) == metric_group
                and clean_text(record.get("model_display_name")) == model_id
                and clean_text(record.get("task_category")) == task_id
                and clean_text(record.get("modality_combination_name")) == modality
                and clean_text(record.get("task_id")) == "ALL"
                and clean_text(record.get("project_id")) == "ALL"
            ]
            if len(records) != 1:
                raise ValueError(f"Expected one metric record for model={model_id}, task={task_id}, found {len(records)}.")
            metric_value = _metric_for_record(records[0], selected_metrics)
            values[model_id].append(max(0.0, min(1.0, metric_value)))
    return model_ids, task_ids, values


def _wrap_label(label: str, width: int) -> str:
    return "\n".join(textwrap.wrap(label, width=width, break_long_words=False))


def _plot_radar(metrics: list[Mapping[str, Any]], cfg: Mapping[str, Any]) -> tuple[Path, Path]:
    radar_cfg = dict(dict(cfg.get("figures") or {}).get("radar") or {})
    colors = dict(cfg.get("colors") or {})
    if not radar_cfg:
        raise ValueError("figures.radar must be populated.")
    _set_font(clean_text(radar_cfg.get("font_family")) or "Inter")

    model_ids, task_ids, values = _radar_values(metrics, radar_cfg)
    model_cfg = dict(radar_cfg.get("models") or {})
    task_cfg = dict(radar_cfg.get("task_categories") or {})
    task_labels = [_wrap_label(clean_text(task_cfg[task_id]), int(radar_cfg.get("label_wrap_chars", 15))) for task_id in task_ids]

    count = len(task_ids)
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    closed_angles = np.concatenate([angles, angles[:1]])
    radial_min = float(radar_cfg.get("radial_min", 0.0))
    radial_max = float(radar_cfg.get("radial_max", 1.0))
    if radial_min >= radial_max:
        raise ValueError(f"radial_min must be smaller than radial_max, got {radial_min} and {radial_max}.")

    width = float(radar_cfg.get("width_inches"))
    height = float(radar_cfg.get("height_inches"))
    transparent_background = bool(radar_cfg.get("transparent_background", True))
    figure_facecolor = "none" if transparent_background else _hex(colors.get("background", "F7F9FD"))
    fig = plt.figure(figsize=(width, height), facecolor=figure_facecolor)
    ax = fig.add_subplot(111, projection="polar")
    ax.set_facecolor(_hex(colors.get("panel", "FFFFFF")))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(radial_min, radial_max)
    ax.spines["polar"].set_visible(False)
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    ring_color = _hex(colors.get("grid", "D8E0EC"))
    emphasis_color = _hex(colors.get("grid_emphasis", "ADBBD1"))
    for tick in radar_cfg.get("radial_ticks", [0.2, 0.4, 0.6, 0.8, 1.0]):
        tick_value = float(tick)
        ax.plot(closed_angles, [tick_value] * len(closed_angles), color=ring_color, linewidth=0.85, zorder=0)
    ax.plot(closed_angles, [radial_max] * len(closed_angles), color=emphasis_color, linewidth=1.25, zorder=1)
    for angle in angles:
        ax.plot([angle, angle], [radial_min, radial_max], color=ring_color, linewidth=0.75, alpha=0.68, zorder=0)

    accent = _hex(colors.get("accent_ring", "7C3AED"))
    ax.add_artist(Circle((0.5, 0.5), 0.505, transform=ax.transAxes, fill=False, color=accent, linewidth=1.0, alpha=0.10))

    label_radius = float(radar_cfg.get("label_radius", 0.615))
    for angle, label in zip(angles, task_labels, strict=True):
        x = 0.5 + label_radius * math.sin(angle)
        y = 0.5 + label_radius * math.cos(angle)
        horizontal = math.sin(angle)
        vertical = math.cos(angle)
        align = "left" if horizontal > 0.16 else "right" if horizontal < -0.16 else "center"
        valign = "bottom" if vertical > 0.34 else "top" if vertical < -0.34 else "center"
        ax.text(
            x,
            y,
            label,
            ha=align,
            va=valign,
            fontsize=float(radar_cfg.get("task_label_font_size", 14.0)),
            fontweight="semibold",
            fontstyle="normal",
            color=_hex(colors.get("label", "111827")),
            linespacing=1.05,
            transform=ax.transAxes,
            clip_on=False,
        )

    for tick in radar_cfg.get("radial_ticks", [0.2, 0.4, 0.6, 0.8, 1.0]):
        tick_value = float(tick)
        ax.text(
            np.deg2rad(88),
            tick_value,
            f"{tick_value:.1f}",
            ha="left",
            va="center",
            fontsize=float(radar_cfg.get("tick_label_font_size", 11.0)),
            color=_hex(colors.get("muted_label", "64748B")),
        )

    handles = []
    for model_id in model_ids:
        style = dict(model_cfg[model_id])
        series = np.array(values[model_id], dtype=float)
        closed = np.concatenate([series, series[:1]])
        color = _hex(style.get("color", "111827"))
        line = ax.plot(
            closed_angles,
            closed,
            color=color,
            linewidth=float(style.get("linewidth", 2.0)),
            alpha=float(style.get("alpha", 0.85)),
            solid_capstyle="round",
            zorder=int(style.get("zorder", 5)),
        )[0]
        ax.scatter(
            angles,
            series,
            s=18 if float(style.get("linewidth", 2.0)) < 3 else 28,
            color=color,
            edgecolor=_hex(colors.get("marker_edge", "FFFFFF")),
            linewidth=0.9,
            alpha=float(style.get("alpha", 0.85)),
            zorder=int(style.get("zorder", 5)) + 1,
        )
        fill_alpha = float(style.get("fill_alpha", 0.0))
        if fill_alpha > 0:
            ax.fill(closed_angles, closed, color=color, alpha=fill_alpha, zorder=int(style.get("zorder", 5)) - 1)
        legend_line = Line2D(
            [0],
            [0],
            color=color,
            linewidth=float(radar_cfg.get("legend_linewidth", 5.0)),
            alpha=float(style.get("alpha", 0.85)),
            solid_capstyle="round",
        )
        handles.append((legend_line, clean_text(style.get("name")) or model_id))

    fig.legend(
        [handle for handle, _ in handles],
        [label for _, label in handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=int(radar_cfg.get("legend_columns", min(len(handles), 5))),
        frameon=False,
        fontsize=float(radar_cfg.get("legend_font_size", 14.0)),
        handlelength=1.20,
        handletextpad=0.7,
        columnspacing=1.4,
    )

    fig.subplots_adjust(left=0.150, right=0.850, top=0.850, bottom=0.285)
    out_dir = _tables_dir(cfg)
    output_name = clean_text(radar_cfg.get("output_name")) or "task_radar"
    png_path = out_dir / f"{output_name}.png"
    svg_path = out_dir / f"{output_name}.svg"
    fig.savefig(
        png_path,
        dpi=int(radar_cfg.get("dpi", 320)),
        facecolor=figure_facecolor,
        transparent=transparent_background,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    fig.savefig(
        svg_path,
        facecolor=figure_facecolor,
        transparent=transparent_background,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(fig)
    return png_path, svg_path


def main() -> None:
    cfg = load_cfg()
    figure_cfg = cfg.vqa_evaluation
    figure_dict = OmegaConf.to_container(figure_cfg, resolve=True)
    if not isinstance(figure_dict, dict):
        raise TypeError("Resolved VQA figure config must be a mapping.")

    metrics = json.loads(_metrics_path(figure_dict).read_text(encoding="utf-8"))["metrics"]
    png_path, svg_path = _plot_radar(metrics, figure_dict)
    print(f"Wrote: {png_path}")
    print(f"Wrote: {svg_path}")


if __name__ == "__main__":
    main()

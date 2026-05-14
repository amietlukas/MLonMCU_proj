from __future__ import annotations

import csv
import math
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Dict, Iterable

import torch
import yaml

from ball_detection.src.config import config_to_serializable


def timestamp_now() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def make_run_dir(output_dir: Path, run_name: str) -> Path:
    run_id = f"{timestamp_now()}-{run_name}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_yaml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config_to_serializable(data), f, sort_keys=False)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def init_metrics_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "lr",
                "train_total_loss",
                "train_obj_loss",
                "train_box_loss",
                "train_num_pos",
                "val_total_loss",
                "val_obj_loss",
                "val_box_loss",
                "val_num_pos",
                "val_map50",
                "val_map5095",
                "val_precision",
                "val_recall",
                "val_f1",
                "val_mean_iou",
                "val_center_error_px",
                "val_center_error_px_median",
                "val_center_error_px_p95",
                "val_center_error_norm_diag",
                "val_center_error_norm_diag_median",
                "val_center_error_norm_diag_p95",
                "val_fp_per_image",
                "val_recall_small",
                "val_recall_medium",
                "val_recall_large",
                "val_count_small",
                "val_count_medium",
                "val_count_large",
            ]
        )


def append_metrics_csv(path: Path, row: Dict[str, float | int]) -> None:
    keys: Iterable[str] = [
        "epoch",
        "lr",
        "train_total_loss",
        "train_obj_loss",
        "train_box_loss",
        "train_num_pos",
        "val_total_loss",
        "val_obj_loss",
        "val_box_loss",
        "val_num_pos",
        "val_map50",
        "val_map5095",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_mean_iou",
        "val_center_error_px",
        "val_center_error_px_median",
        "val_center_error_px_p95",
        "val_center_error_norm_diag",
        "val_center_error_norm_diag_median",
        "val_center_error_norm_diag_p95",
        "val_fp_per_image",
        "val_recall_small",
        "val_recall_medium",
        "val_recall_large",
        "val_count_small",
        "val_count_medium",
        "val_count_large",
    ]
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([row.get(k, "") for k in keys])


def _safe_float(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _read_metrics_rows(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        return []

    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict[str, float] = {}
            for k, v in row.items():
                fv = _safe_float(v)
                if fv is not None:
                    parsed[k] = fv
            if parsed:
                rows.append(parsed)
    return rows


def plot_metrics_svg(metrics_csv_path: Path, out_svg_path: Path) -> bool:
    """Render a dependency-free SVG dashboard from metrics.csv."""
    rows = _read_metrics_rows(metrics_csv_path)
    if not rows:
        return False

    epochs = [r["epoch"] for r in rows if "epoch" in r]
    if not epochs:
        return False

    x_min = min(epochs)
    x_max = max(epochs)
    if abs(x_max - x_min) < 1e-9:
        x_max = x_min + 1.0

    # title, [(metric_key, legend_label, color), ...]
    panels = [
        (
            "Losses",
            [
                ("train_total_loss", "train_total", "#1f77b4"),
                ("val_total_loss", "val_total", "#ff7f0e"),
                ("train_obj_loss", "train_obj", "#2ca02c"),
                ("val_obj_loss", "val_obj", "#d62728"),
                ("train_box_loss", "train_box", "#9467bd"),
                ("val_box_loss", "val_box", "#8c564b"),
            ],
        ),
        (
            "Detection Metrics",
            [
                ("val_map50", "mAP@0.5", "#1f77b4"),
                ("val_map5095", "mAP@0.5:0.95", "#ff7f0e"),
                ("val_precision", "precision", "#2ca02c"),
                ("val_recall", "recall", "#d62728"),
                ("val_f1", "f1", "#9467bd"),
                ("val_mean_iou", "mean_iou", "#8c564b"),
            ],
        ),
        (
            "Diagnostics",
            [
                ("train_num_pos", "train_num_pos", "#1f77b4"),
                ("val_num_pos", "val_num_pos", "#ff7f0e"),
                ("val_center_error_px", "center_err_px", "#2ca02c"),
                ("val_center_error_norm_diag", "center_err_norm_diag", "#9467bd"),
                ("val_fp_per_image", "fp_per_image", "#d62728"),
            ],
        ),
    ]

    fig_w = 1400.0
    fig_h = 960.0
    margin_left = 95.0
    margin_right = 30.0
    margin_top = 25.0
    margin_bottom = 30.0
    panel_gap = 26.0
    panel_count = len(panels)
    panel_w = fig_w - margin_left - margin_right
    panel_h = (fig_h - margin_top - margin_bottom - panel_gap * (panel_count - 1)) / panel_count

    def sx(epoch: float) -> float:
        return margin_left + (epoch - x_min) / (x_max - x_min) * panel_w

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(fig_w)}" height="{int(fig_h)}" viewBox="0 0 {int(fig_w)} {int(fig_h)}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
        '<text x="20" y="20" font-size="16" font-family="Arial, Helvetica, sans-serif" fill="#111111">ball_detection training dashboard</text>',
    ]

    for panel_idx, (panel_title, series) in enumerate(panels):
        panel_x = margin_left
        panel_y = margin_top + panel_idx * (panel_h + panel_gap)

        values: list[float] = []
        for metric_key, _label, _color in series:
            for r in rows:
                if metric_key in r:
                    values.append(r[metric_key])

        if not values:
            continue

        y_min = min(values)
        y_max = max(values)
        if abs(y_max - y_min) < 1e-12:
            pad = 1.0 if abs(y_max) < 1e-9 else abs(y_max) * 0.1
            y_min -= pad
            y_max += pad
        else:
            pad = 0.05 * (y_max - y_min)
            y_min -= pad
            y_max += pad

        def sy(v: float) -> float:
            return panel_y + panel_h - (v - y_min) / (y_max - y_min) * panel_h

        svg.append(
            f'<rect x="{panel_x:.2f}" y="{panel_y:.2f}" width="{panel_w:.2f}" height="{panel_h:.2f}" fill="#ffffff" stroke="#cfcfcf" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{panel_x + 8:.2f}" y="{panel_y + 18:.2f}" font-size="14" font-family="Arial, Helvetica, sans-serif" fill="#111111">{escape(panel_title)}</text>'
        )

        y_ticks = 5
        for i in range(y_ticks):
            t = i / max(y_ticks - 1, 1)
            y = panel_y + panel_h - t * panel_h
            val = y_min + t * (y_max - y_min)
            svg.append(
                f'<line x1="{panel_x:.2f}" y1="{y:.2f}" x2="{panel_x + panel_w:.2f}" y2="{y:.2f}" stroke="#efefef" stroke-width="1"/>'
            )
            svg.append(
                f'<text x="{panel_x - 8:.2f}" y="{y + 4:.2f}" text-anchor="end" font-size="10" font-family="Arial, Helvetica, sans-serif" fill="#666666">{val:.4g}</text>'
            )

        x_ticks = 6
        for i in range(x_ticks):
            t = i / max(x_ticks - 1, 1)
            ep = x_min + t * (x_max - x_min)
            x = sx(ep)
            svg.append(
                f'<line x1="{x:.2f}" y1="{panel_y:.2f}" x2="{x:.2f}" y2="{panel_y + panel_h:.2f}" stroke="#f4f4f4" stroke-width="1"/>'
            )
            svg.append(
                f'<text x="{x:.2f}" y="{panel_y + panel_h + 14:.2f}" text-anchor="middle" font-size="10" font-family="Arial, Helvetica, sans-serif" fill="#666666">{ep:.3g}</text>'
            )

        legend_x = panel_x + 12.0
        legend_y = panel_y + 32.0
        for s_idx, (metric_key, label, color) in enumerate(series):
            pts: list[tuple[float, float]] = []
            for r in rows:
                if "epoch" not in r or metric_key not in r:
                    continue
                pts.append((sx(r["epoch"]), sy(r[metric_key])))

            if not pts:
                continue

            if len(pts) == 1:
                x0, y0 = pts[0]
                svg.append(f'<circle cx="{x0:.2f}" cy="{y0:.2f}" r="2.5" fill="{color}"/>')
            else:
                poly = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
                svg.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="1.8"/>')

            ly = legend_y + 14.0 * s_idx
            svg.append(f'<line x1="{legend_x:.2f}" y1="{ly:.2f}" x2="{legend_x + 12:.2f}" y2="{ly:.2f}" stroke="{color}" stroke-width="2"/>')
            svg.append(
                f'<text x="{legend_x + 16:.2f}" y="{ly + 3:.2f}" font-size="10" font-family="Arial, Helvetica, sans-serif" fill="#333333">{escape(label)}</text>'
            )

    svg.append("</svg>")

    out_svg_path.parent.mkdir(parents=True, exist_ok=True)
    out_svg_path.write_text("\n".join(svg), encoding="utf-8")
    return True

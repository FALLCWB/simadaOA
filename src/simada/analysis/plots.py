# simada -- Simulation of AID Adherence
# Copyright (C) 2026 Dr. Filipe Augusto da Luz Lemos, MSc. Ph.D.
# Contact: filipellemos@gmail.com | filipe@falleng.com.br | fadaluzl@syr.edu
#
# "Omnis enim res, quae dando non deficit, dum habetur et non datur,
#  nondum habetur quomodo habenda est." -- St. Augustine, De Doctrina Christiana
#  (A thing not diminished by being shared is not yet rightly possessed if only possessed and not shared.)
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

"""Visualization functions for simulation results.

Standard output set (always generated after a simulation run):
    1. agp_comparison.png       — side-by-side AGP for all archetypes
    2. agp_<archetype>.png      — individual AGP per archetype
    3. metrics_comparison.png   — bar chart TIR/TBR/TAR with clinical targets
    4. metrics_table.png        — full metrics table with color coding
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simada.analysis.style import (
    COLORS, FIGSIZE, TITLE_SIZE, SUBTITLE_SIZE, LABEL_SIZE, ANNOTATION_SIZE,
    TICK_SIZE, LINE_WIDTH_PRIMARY, LINE_WIDTH_REFERENCE,
    TARGET_RANGE_COLOR, SEVERE_HYPO_COLOR,
    CELL_GREEN, CELL_YELLOW, CELL_RED, CELL_WHITE, CELL_GRAY, HEADER_BG, ROW_LABEL_BG,
    TIR_COLORS, REFERENCE_LINE_COLOR,
    simada_figure, finalize, add_target_range, add_hypo_line, format_bg_axis,
)
from simada.analysis.utils import extract_archetype


def plot_metrics_comparison(
    summary: pd.DataFrame,
    output_path: Path | None = None,
) -> plt.Figure:
    """Bar chart comparing TIR/TBR/TAR across archetypes.

    Expects a DataFrame from compute_cohort_summary() with a 'patient'
    column containing archetype labels (extracted from filename).
    """
    # Extract archetype from patient name (e.g. "adult001_adherent_000" → "adherent")
    summary = summary.copy()
    summary["archetype"] = summary["patient"].apply(extract_archetype)

    metrics = ["tir", "tbr_l1", "tbr_l2", "tar_l1", "tar_l2"]
    labels = ["TIR\n70-180", "TBR L1\n54-69", "TBR L2\n<54", "TAR L1\n181-250", "TAR L2\n>250"]
    archetypes = sorted(summary["archetype"].unique())

    fig, ax = simada_figure(figsize=FIGSIZE.double_col)
    x = np.arange(len(metrics))
    width = 0.8 / len(archetypes)

    for i, arch in enumerate(archetypes):
        arch_data = summary[summary["archetype"] == arch]
        means = [arch_data[m].mean() for m in metrics]
        offset = (i - len(archetypes) / 2 + 0.5) * width
        ax.bar(
            x + offset, means, width,
            label=arch.capitalize(),
            color=COLORS.get(arch),
            alpha=0.85,
        )

    # Target lines (dashed reference for clinical targets)
    targets = [70, 4, 1, 25, 5]
    for xi, target in enumerate(targets):
        ax.plot(
            [xi - 0.45, xi + 0.45], [target, target],
            color=REFERENCE_LINE_COLOR, linewidth=LINE_WIDTH_REFERENCE,
            linestyle="--", alpha=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Glycemic Metrics Comparison by Archetype")
    ax.legend()

    finalize(fig, output_path)
    return fig


def plot_bg_trace(
    bg_series: pd.Series,
    title: str = "Blood Glucose Trace",
    output_path: Path | None = None,
    color: str | None = None,
) -> plt.Figure:
    """Plot a single patient's BG time series."""
    if color is None:
        color = COLORS.adherent

    fig, ax = simada_figure(figsize=FIGSIZE.double_col)

    hours = np.arange(len(bg_series)) * 3 / 60  # assuming 3-min intervals
    ax.plot(hours, bg_series.values, color=color, linewidth=LINE_WIDTH_PRIMARY, alpha=0.8)

    format_bg_axis(ax)
    ax.set_xlabel("Hours")
    ax.set_title(title)

    finalize(fig, output_path)
    return fig


_METRIC_FORMAT = {
    "TIR (70-180)": ("tir", "%"),
    "TBR L1 (54-69)": ("tbr_l1", "%"),
    "TBR L2 (<54)": ("tbr_l2", "%"),
    "TAR L1 (181-250)": ("tar_l1", "%"),
    "TAR L2 (>250)": ("tar_l2", "%"),
    "GMI": ("gmi", ""),
    "CV": ("cv", "%"),
    "Mean BG": ("mean_bg", " mg/dL"),
    "Std BG": ("std_bg", " mg/dL"),
    "LBGI": ("lbgi", ""),
    "HBGI": ("hbgi", ""),
    "MAGE": ("mage", " mg/dL"),
}

_CLINICAL_TARGETS = {
    "TIR (70-180)": ">70%",
    "TBR L1 (54-69)": "<4%",
    "TBR L2 (<54)": "<1%",
    "TAR L1 (181-250)": "<25%",
    "TAR L2 (>250)": "<5%",
    "GMI": "<7.0",
    "CV": "<36%",
    "Mean BG": "\u2014",
    "Std BG": "\u2014",
    "LBGI": "\u2014",
    "HBGI": "\u2014",
    "MAGE": "\u2014",
    "Min BG": ">54 mg/dL",
    "Max BG": "<250 mg/dL",
    "Severe Hypo (#)": "0",
    "Readings": "\u2014",
}


def _get_metric(m: dict, *keys: str, default: float = 0.0) -> float:
    """Return the first matching key from m, accepting multiple aliases.

    GlycemicMetricsResult._asdict() uses ``min_bg`` / ``max_bg``, but some
    callers historically passed ``bg_min`` / ``bg_max``. Accepting both keys
    prevents silent zeros that produce wrong colors and values — H7#2.
    """
    for key in keys:
        if key in m:
            return float(m[key])
    return default


def _color_for_metric(row_label: str, m: dict) -> str:
    """Return cell background color based on clinical target adherence."""
    if row_label == "TIR (70-180)":
        v = m["tir"]
        return CELL_GREEN if v >= 70 else CELL_YELLOW if v >= 50 else CELL_RED
    # Clinical targets are strict inequalities (e.g. TBR L1 target is "<4%"),
    # so GREEN must use the same strict comparison. Using `<=` would falsely
    # classify exactly-at-threshold values as on-target.
    if row_label.startswith("TBR L1"):
        v = m["tbr_l1"]
        return CELL_GREEN if v < 4 else CELL_YELLOW if v < 8 else CELL_RED
    if row_label.startswith("TBR L2"):
        v = m["tbr_l2"]
        return CELL_GREEN if v < 1 else CELL_YELLOW if v < 3 else CELL_RED
    if row_label.startswith("TAR L1"):
        v = m["tar_l1"]
        return CELL_GREEN if v < 25 else CELL_YELLOW if v < 35 else CELL_RED
    if row_label.startswith("TAR L2"):
        v = m["tar_l2"]
        return CELL_GREEN if v < 5 else CELL_YELLOW if v < 10 else CELL_RED
    if row_label == "GMI":
        v = m["gmi"]
        return CELL_GREEN if v < 7.0 else CELL_YELLOW if v < 8.0 else CELL_RED
    if row_label == "CV":
        v = m["cv"]
        return CELL_GREEN if v < 36 else CELL_YELLOW if v < 50 else CELL_RED
    if row_label.startswith("Severe Hypo"):
        v = m.get("severe_hypo_episodes", 0)
        # Absolute event counts. Calibrated for typical simada periods (30-90 days):
        # 0 events = excellent, 1-3 = concerning, >3 = clinically dangerous.
        return CELL_GREEN if v < 1 else CELL_YELLOW if v <= 3 else CELL_RED
    if row_label == "Min BG":
        # Accept both key forms: GlycemicMetricsResult uses min_bg; legacy callers
        # may pass bg_min. _get_metric tries both to prevent a silent 0 default. H7#2.
        v = _get_metric(m, "min_bg", "bg_min", default=0.0)
        # Clinical thresholds (mg/dL): >70 healthy, 54-70 borderline hypo, <54 severe hypo.
        return CELL_GREEN if v > 70 else CELL_YELLOW if v >= 54 else CELL_RED
    if row_label == "Max BG":
        # Accept both key forms: GlycemicMetricsResult uses max_bg; legacy callers
        # may pass bg_max. H7#2.
        v = _get_metric(m, "max_bg", "bg_max", default=0.0)
        # Clinical thresholds (mg/dL): <180 in-range, 180-250 hyper L1, >250 hyper L2.
        return CELL_GREEN if v < 180 else CELL_YELLOW if v <= 250 else CELL_RED
    return CELL_WHITE


def _fmt_value(metric_key: str, m: dict) -> str:
    """Format a single metric value for display."""
    if metric_key == "Min BG":
        # Accept both min_bg (GlycemicMetricsResult) and legacy bg_min. H7#2.
        return f"{_get_metric(m, 'min_bg', 'bg_min'):.0f} mg/dL"
    if metric_key == "Max BG":
        # Accept both max_bg (GlycemicMetricsResult) and legacy bg_max. H7#2.
        return f"{_get_metric(m, 'max_bg', 'bg_max'):.0f} mg/dL"
    if metric_key == "Readings":
        return f"{m['readings']:,}"
    if metric_key.startswith("Severe Hypo"):
        return f"{m.get('severe_hypo_episodes', 0):.0f}"
    key, suffix = _METRIC_FORMAT[metric_key]
    val = m[key]
    if suffix == "%":
        return f"{val:.1f}%"
    if suffix:
        return f"{val:.1f}{suffix}"
    return f"{val:.2f}"


def render_metrics_table(
    all_metrics: dict[str, dict],
    output_path: Path,
    *,
    subtitle: str = "",
) -> None:
    """Render a clinical metrics table as a color-coded PNG image.

    Args:
        all_metrics: Mapping archetype_name -> dict of metric values.
                     Each dict must include keys from GlycemicMetricsResult
                     plus 'bg_min' and 'bg_max'.
        output_path: Where to save the PNG.
        subtitle: Extra subtitle line (e.g. patient name, seed info).
    """
    row_labels = list(_CLINICAL_TARGETS.keys())

    cell_text = []
    cell_colors = []
    for row_label in row_labels:
        row_data = [_CLINICAL_TARGETS[row_label]]
        row_colors = [CELL_GRAY]
        for m in all_metrics.values():
            row_data.append(_fmt_value(row_label, m))
            row_colors.append(_color_for_metric(row_label, m))
        cell_text.append(row_data)
        cell_colors.append(row_colors)

    n_arch = len(all_metrics)
    col_labels = ["Target"] + [a.capitalize() for a in all_metrics]

    fig, ax = simada_figure(figsize=FIGSIZE.metrics_table)
    ax.axis("off")

    title_lines = "simada \u2014 Glycemic Metrics Comparison"
    if subtitle:
        title_lines += f"\n{subtitle}"
    ax.set_title(title_lines, fontsize=TITLE_SIZE, fontweight="bold", pad=20)

    # Target column needs more width than before — clinical targets like
    # "<250 mg/dL" or ">54 mg/dL" do not fit in 0.10.
    col_widths = [0.20] + [0.18] * n_arch

    # Defensive: col_widths must match the number of data columns (colLabels).
    # rowLabels is handled separately by matplotlib and must NOT be counted here.
    # Version-dependent matplotlib behavior can silently misalign the table if
    # the count is wrong. — H7#6.
    assert len(col_widths) == len(col_labels), (
        f"col_widths length ({len(col_widths)}) != col_labels length ({len(col_labels)}). "
        "Update col_widths when adding or removing columns."
    )

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        colWidths=col_widths,
    )

    table.auto_set_font_size(False)
    table.set_fontsize(LABEL_SIZE)
    table.scale(1.0, 1.5)

    # Style header row
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor(HEADER_BG)
        cell.set_text_props(color="white", fontweight="bold")

    # Style row labels
    for i in range(len(row_labels)):
        cell = table[i + 1, -1]
        cell.set_facecolor(ROW_LABEL_BG)
        cell.set_text_props(fontweight="bold")

    finalize(fig, output_path)


def generate_standard_plots(
    all_bg: dict[str, np.ndarray],
    all_metrics: dict[str, dict],
    output_dir: Path,
    *,
    subtitle: str = "",
) -> list[Path]:
    """Generate the standard set of post-simulation images.

    Always produces:
        1. agp_comparison.png       — side-by-side AGP
        2. agp_<archetype>.png      — individual AGP per archetype
        3. metrics_comparison.png   — bar chart
        4. metrics_table.png        — full table

    Args:
        all_bg: Mapping archetype_name -> BG array.
        all_metrics: Mapping archetype_name -> metrics dict (with bg_min/bg_max).
        output_dir: Directory to save all images.
        subtitle: Extra info line for chart titles.

    Returns:
        List of paths to generated images.
    """
    from simada.analysis.agp import generate_agp, generate_comparison_agp

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    # 1. AGP comparison
    bg_series_dict = {k: pd.Series(v) for k, v in all_bg.items()}
    agp_cmp_path = output_dir / "agp_comparison.png"
    generate_comparison_agp(
        bg_series_dict,
        output_path=agp_cmp_path,
        title=f"AGP Comparison by Archetype\n{subtitle}" if subtitle else "AGP Comparison by Archetype",
    )
    generated.append(agp_cmp_path)

    # 2. Individual AGPs (consistent colors per archetype)
    for arch_name, bg in all_bg.items():
        agp_path = output_dir / f"agp_{arch_name}.png"
        generate_agp(
            pd.Series(bg),
            output_path=agp_path,
            title=f"AGP \u2014 {arch_name.capitalize()}" + (f"\n{subtitle}" if subtitle else ""),
            color=COLORS.get(arch_name),
        )
        generated.append(agp_path)

    # 3. Metrics comparison bar chart
    summary_rows = []
    for arch_name, m in all_metrics.items():
        row = {"patient": f"patient_{arch_name}_000"}
        row.update(m)
        summary_rows.append(row)
    metrics_cmp_path = output_dir / "metrics_comparison.png"
    plot_metrics_comparison(pd.DataFrame(summary_rows), metrics_cmp_path)
    generated.append(metrics_cmp_path)

    # 4. Metrics table
    table_path = output_dir / "metrics_table.png"
    render_metrics_table(all_metrics, table_path, subtitle=subtitle)
    generated.append(table_path)

    return generated



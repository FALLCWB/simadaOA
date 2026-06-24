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

"""Ambulatory Glucose Profile (AGP) generator.

The AGP is the clinical standard for summarizing CGM data:
- Median glucose trace (bold line)
- 25th-75th percentile band (dark shading)
- 10th-90th percentile band (light shading)
- Target range overlay (70-180 mg/dL)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from simada.analysis.style import (
    COLORS,
    FIGSIZE,
    LINE_WIDTH_PRIMARY,
    LINE_WIDTH_SECONDARY,
    SUBTITLE_SIZE,
    TITLE_SIZE,
    add_bg_reference_labels,
    add_target_range,
    add_hypo_line,
    finalize,
    simada_figure,
)


def _reshape_to_daily(
    bg: np.ndarray,
    samples_per_day: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reshape BG array into (days x time_of_day) matrix and hours axis."""
    if len(bg) == 0:
        warnings.warn(
            "AGP computed from empty BG array; percentile bands are meaningless.",
            UserWarning,
            stacklevel=3,
        )
        return np.full((1, samples_per_day), np.nan), 0
    n_complete_days = len(bg) // samples_per_day
    if n_complete_days < 1:
        # Fewer than one full day of samples -- percentile bands will be
        # degenerate (a single value at each time-of-day) and clinically
        # uninformative. Clinical AGP convention recommends >=14 days.
        warnings.warn(
            f"AGP computed from <1 day of data ({len(bg)} samples, "
            f"need {samples_per_day} for one day); percentile bands will "
            "be degenerate. Clinical AGP recommends >=14 days.",
            UserWarning,
            stacklevel=3,
        )
        n_complete_days = 1
        bg_matrix = np.full((1, samples_per_day), np.nan)
        bg_matrix[0, : len(bg)] = bg[:samples_per_day]
    else:
        usable = n_complete_days * samples_per_day
        dropped = len(bg) - usable
        if dropped > 0:
            # Warn so callers are aware that up to (samples_per_day - 1) samples
            # (up to ~24 h of data at 3-min intervals) are silently discarded. H7#4.
            warnings.warn(
                f"Dropped {dropped} samples ({dropped * 3} min) from the end of "
                "the BG array to complete a whole number of days for the AGP. "
                "Ensure the simulation output is a multiple of samples_per_day "
                "to avoid data loss.",
                UserWarning,
                stacklevel=3,
            )
        bg_matrix = bg[:usable].reshape(n_complete_days, samples_per_day)
    return bg_matrix, n_complete_days


def _draw_agp_on_axis(
    ax,
    bg: np.ndarray,
    samples_per_day: int,
    color: str,
    *,
    show_p10_p90: bool = True,
) -> None:
    """Draw AGP bands and median on a given axis."""
    bg_matrix, _ = _reshape_to_daily(bg, samples_per_day)
    sample_min = 24 * 60 / samples_per_day
    hours = np.arange(samples_per_day) * sample_min / 60.0

    p25 = np.nanpercentile(bg_matrix, 25, axis=0)
    p50 = np.nanpercentile(bg_matrix, 50, axis=0)
    p75 = np.nanpercentile(bg_matrix, 75, axis=0)

    if show_p10_p90:
        p10 = np.nanpercentile(bg_matrix, 10, axis=0)
        p90 = np.nanpercentile(bg_matrix, 90, axis=0)
        ax.fill_between(hours, p10, p90, alpha=0.10, color=color, label="10th\u201390th", zorder=2)

    ax.fill_between(hours, p25, p75, alpha=0.25, color=color, label="25th\u201375th", zorder=3)
    ax.plot(hours, p50, color=color, linewidth=LINE_WIDTH_PRIMARY, label="Median", zorder=4)

    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel("Hour of Day")


def generate_agp(
    bg_series: pd.Series,
    sample_interval_minutes: int = 3,
    output_path: Path | None = None,
    title: str = "Ambulatory Glucose Profile",
    color: str | None = None,
) -> None:
    """Generate a single-archetype AGP chart.

    Args:
        bg_series: BG values.
        sample_interval_minutes: Time between readings (default 3 for simglucose).
        output_path: If provided, save the figure as PNG.
        title: Chart title.
        color: Primary color. Defaults to adherent blue.
    """
    if color is None:
        color = COLORS.adherent

    samples_per_day = 24 * 60 // sample_interval_minutes
    fig, ax = simada_figure(figsize=FIGSIZE.agp_single)

    add_target_range(ax)
    add_hypo_line(ax)
    # _draw_agp_on_axis must run before add_bg_reference_labels so that
    # set_xlim(0, 24) has already executed. Labels read ax.get_xlim()[1] to
    # position at the right margin; reading before set_xlim lands them at
    # x≈0.99 (default xlim upper bound = 1.0). — H7#7.
    _draw_agp_on_axis(ax, bg_series.values, samples_per_day, color)
    add_bg_reference_labels(ax)

    ax.set_ylim(30, 400)
    ax.set_ylabel("Blood Glucose (mg/dL)")
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.legend(loc="upper right")

    finalize(fig, str(output_path) if output_path else None)


def generate_comparison_agp(
    bg_dict: dict[str, pd.Series],
    sample_interval_minutes: int = 3,
    output_path: Path | None = None,
    title: str = "AGP Comparison by Archetype",
) -> None:
    """Generate stacked AGP charts for multiple archetypes.

    Each archetype gets its own panel (vertically stacked for readability).
    """
    if not bg_dict:
        return
    n = len(bg_dict)
    samples_per_day = 24 * 60 // sample_interval_minutes

    fig, axes = simada_figure(nrows=n, ncols=1, figsize=FIGSIZE.agp_comparison, sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (label, bg_series) in zip(axes, bg_dict.items()):
        color = COLORS.get(label)

        add_target_range(ax)
        add_hypo_line(ax)
        # _draw_agp_on_axis sets xlim(0, 24); add_bg_reference_labels must
        # run after so it reads the correct xlim upper bound. H7#7.
        _draw_agp_on_axis(ax, bg_series.values, samples_per_day, color)
        add_bg_reference_labels(ax)

        # Archetype label as panel title
        ax.set_title(label.capitalize(), fontsize=SUBTITLE_SIZE, loc="left")
        ax.set_ylim(30, 400)
        ax.set_ylabel("BG (mg/dL)")

    # Only bottom axis gets x-label and tick labels
    for ax in axes[:-1]:
        ax.set_xlabel("")
        ax.tick_params(axis='x', labelbottom=False)

    fig.suptitle(title, fontsize=TITLE_SIZE, y=1.01)
    finalize(fig, str(output_path) if output_path else None)

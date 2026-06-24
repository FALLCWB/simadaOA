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

"""Publication-quality design system for simada figures.

All figures follow journal standards (Diabetes Care, CMPB):
- 300 DPI minimum
- Single-column: 3.5 in (89 mm)
- Double-column: 7.0 in (178 mm)
- Full-page: 7.0 x 9.0 in
- Fonts: sans-serif (Arial/Helvetica family) per Nature/ADA guidelines
- Color-blind safe palette with semantic archetype colors
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

# ---------------------------------------------------------------------------
# Watermark
# ---------------------------------------------------------------------------
WATERMARK = "Lemos, F.A.L. et al., 2026 \u2014 simada"

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArchetypeColors:
    """Semantic colors for each archetype — Okabe-Ito colorblind-safe palette."""
    adherent: str = "#0072B2"          # Okabe-Ito blue
    moderate: str = "#E69F00"          # Okabe-Ito orange
    nonadherent: str = "#D55E00"       # Okabe-Ito vermillion

    adherent_light: str = "#56B4E9"    # Sky blue
    moderate_light: str = "#F0E442"    # Yellow
    nonadherent_light: str = "#FFAB91" # Lighter vermillion

    def get(self, name: str) -> str:
        if not hasattr(self, name):
            raise AttributeError(f"Unknown archetype: {name}")
        return getattr(self, name)

    def get_light(self, name: str) -> str:
        return getattr(self, f"{name}_light", "#E0E0E0")


COLORS = ArchetypeColors()

# Archetype markers and linestyles for multi-channel accessibility
ARCHETYPE_MARKERS = {"adherent": "o", "moderate": "s", "nonadherent": "D"}
ARCHETYPE_LINESTYLES = {"adherent": "-", "moderate": "--", "nonadherent": "-."}


@dataclass(frozen=True)
class OkabeIto:
    """Okabe-Ito / Wong colorblind-safe palette (Nature recommended)."""
    blue: str = "#0072B2"
    orange: str = "#E69F00"
    vermillion: str = "#D55E00"
    reddish_purple: str = "#CC79A7"
    bluish_green: str = "#009E73"
    sky_blue: str = "#56B4E9"
    black: str = "#000000"


COLORBLIND = OkabeIto()

STEADY_STATE_COLOR = "#66BB6A" # Green 400

# Clinical reference colors
TARGET_RANGE_COLOR = "#9E9E9E"     # Neutral gray — colorblind universal
HYPO_COLOR = "#F44336"             # Red 500
SEVERE_HYPO_COLOR = "#B71C1C"     # Red 900
HYPER_COLOR = "#FF9800"            # Orange 500
REFERENCE_LINE_COLOR = "#9E9E9E"  # Grey 500
GRID_COLOR = "#E0E0E0"            # Grey 300

# TIR stacked bar colors (AGP clinical standard, bottom to top)
TIR_COLORS = {
    "tbr_l2": "#B71C1C",    # Very Low (<54) — Dark Red / Maroon
    "tbr_l1": "#EF5350",    # Low (54-69) — Red
    "tir": "#66BB6A",       # In Range (70-180) — Green
    "tar_l1": "#FFA726",    # High (181-250) — Yellow/Orange
    "tar_l2": "#FF7043",    # Very High (>250) — Orange/Red
}

# TIR hatching patterns for additional accessibility
TIR_HATCHES = {"tbr_l2": "xx", "tbr_l1": "//", "tir": "", "tar_l1": "\\\\", "tar_l2": ".."}

# Metrics table cell colors (clinical target adherence)
CELL_GREEN = "#C8E6C9"    # Within target
CELL_YELLOW = "#FFF9C4"   # Borderline
CELL_RED = "#FFCDD2"      # Outside target
CELL_WHITE = "#FFFFFF"
CELL_GRAY = "#F5F5F5"     # Non-applicable rows
HEADER_BG = "#37474F"     # Blue Grey 800
ROW_LABEL_BG = "#E8EAF6"  # Indigo 50

# ---------------------------------------------------------------------------
# Figure sizes (inches) — journal standards
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FigSize:
    """Standard figure dimensions for journal submission."""
    single_col: tuple[float, float] = (3.5, 2.8)
    single_col_tall: tuple[float, float] = (3.5, 4.5)
    double_col: tuple[float, float] = (7.0, 4.0)
    double_col_tall: tuple[float, float] = (7.0, 6.0)
    full_page: tuple[float, float] = (7.0, 9.0)
    agp_single: tuple[float, float] = (7.0, 3.5)
    agp_comparison: tuple[float, float] = (7.0, 8.0)
    metrics_table: tuple[float, float] = (7.0, 5.5)

FIGSIZE = FigSize()

# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------

FONT_FAMILY = "sans-serif"
FONT_SANS = ["Arial", "Helvetica", "DejaVu Sans"]
TITLE_SIZE = 11
SUBTITLE_SIZE = 9
LABEL_SIZE = 9
TICK_SIZE = 8
LEGEND_SIZE = 8
ANNOTATION_SIZE = 7.5
WATERMARK_SIZE = 6

# ---------------------------------------------------------------------------
# Line widths and marker sizes
# ---------------------------------------------------------------------------

LINE_WIDTH_PRIMARY = 1.8
LINE_WIDTH_SECONDARY = 1.0
LINE_WIDTH_REFERENCE = 0.7
LINE_WIDTH_GRID = 0.4
MARKER_SIZE_PRIMARY = 6
MARKER_SIZE_SECONDARY = 4

# ---------------------------------------------------------------------------
# Matplotlib RC params
# ---------------------------------------------------------------------------

SIMADA_RC: dict[str, Any] = {
    # Font
    "font.family": FONT_FAMILY,
    "font.sans-serif": FONT_SANS,
    "font.size": TICK_SIZE,
    # Axes
    "axes.titlesize": TITLE_SIZE,
    "axes.titleweight": "bold",
    "axes.labelsize": LABEL_SIZE,
    "axes.labelweight": "medium",
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    # Grid
    "grid.color": GRID_COLOR,
    "grid.linewidth": LINE_WIDTH_GRID,
    "grid.alpha": 0.7,
    # Ticks
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    # Legend
    "legend.fontsize": LEGEND_SIZE,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#CCCCCC",
    "legend.fancybox": False,
    # Figure
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "savefig.facecolor": "white",
    # Lines
    "lines.linewidth": LINE_WIDTH_PRIMARY,
    "lines.markersize": MARKER_SIZE_PRIMARY,
}


def apply_style() -> None:
    """Apply simada publication style globally."""
    plt.rcParams.update(SIMADA_RC)


def simada_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> tuple[Figure, Any]:
    """Create a figure with simada style applied.

    Returns (fig, axes) with publication-quality defaults.
    """
    apply_style()
    if figsize is None:
        if nrows == 1 and ncols == 1:
            figsize = FIGSIZE.double_col
        elif ncols >= 3:
            figsize = FIGSIZE.full_page
        else:
            figsize = FIGSIZE.double_col_tall
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, layout="constrained", **kwargs)
    return fig, axes


def add_watermark(fig: Figure) -> None:
    """Add attribution watermark to bottom-right."""
    fig.text(
        0.98, 0.01, WATERMARK,
        fontsize=WATERMARK_SIZE,
        color="#B0B0B0",
        alpha=0.6,
        ha="right",
        va="bottom",
        fontstyle="italic",
        fontfamily=FONT_FAMILY,
    )


def add_target_range(ax: plt.Axes, ymin: float = 70, ymax: float = 180) -> None:
    """Add the clinical target range (70-180 mg/dL) shading and lines."""
    ax.axhspan(ymin, ymax, alpha=0.08, color=TARGET_RANGE_COLOR, zorder=0)
    ax.axhline(ymin, color=TARGET_RANGE_COLOR, linewidth=LINE_WIDTH_REFERENCE,
               linestyle="--", alpha=0.6, zorder=1)
    ax.axhline(ymax, color=TARGET_RANGE_COLOR, linewidth=LINE_WIDTH_REFERENCE,
               linestyle="--", alpha=0.6, zorder=1)


def add_hypo_line(ax: plt.Axes, level: float = 54) -> None:
    """Add severe hypoglycemia reference line."""
    ax.axhline(level, color=SEVERE_HYPO_COLOR, linewidth=LINE_WIDTH_REFERENCE,
               linestyle=":", alpha=0.5, zorder=1)


def format_bg_axis(ax: plt.Axes, ylim: tuple[float, float] = (30, 400)) -> None:
    """Standard BG axis formatting."""
    ax.set_ylim(ylim)
    ax.set_ylabel("Blood Glucose (mg/dL)")
    add_target_range(ax)
    add_hypo_line(ax)
    add_bg_reference_labels(ax)


def add_bg_reference_labels(ax: plt.Axes) -> None:
    """Add numeric labels (54, 70, 180) on reference lines at right margin."""
    for val, color in [
        (180, TARGET_RANGE_COLOR),
        (70, TARGET_RANGE_COLOR),
        (54, SEVERE_HYPO_COLOR),
    ]:
        ax.text(
            ax.get_xlim()[1] * 0.99, val, str(val),
            fontsize=ANNOTATION_SIZE,
            color=color,
            ha="right",
            va="center",
            fontweight="bold",
            alpha=0.7,
        )


def finalize(fig: Figure, output_path: str | None = None) -> None:
    """Finalize figure: watermark, save."""
    add_watermark(fig)
    if output_path is not None:
        fig.savefig(output_path, facecolor="white")
        plt.close(fig)

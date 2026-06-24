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

"""Export utilities for external analysis.

Converts simulation results into CSV summary tables for statistical
analysis and archetype comparison.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from simada.analysis.metrics import compute_cohort_summary
from simada.analysis.utils import extract_archetype


def export_metrics_csv(results_dir: Path, output_path: Path | None = None) -> pd.DataFrame:
    """Export cohort metrics as a CSV summary table.

    Args:
        results_dir: Path to a simulation run directory.
        output_path: Where to save the CSV. If None, only returns DataFrame.

    Returns:
        DataFrame with per-patient metrics.
    """
    summary = compute_cohort_summary(results_dir)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path, index=False)

    return summary


def export_archetype_comparison(
    results_dir: Path, output_path: Path | None = None
) -> pd.DataFrame:
    """Export aggregated metrics grouped by archetype.

    Returns mean and std for each metric per archetype.
    """
    summary = compute_cohort_summary(results_dir)

    # Extract archetype from patient name
    summary["archetype"] = summary["patient"].apply(extract_archetype)

    metric_cols = [
        "tir", "tbr_l1", "tbr_l2", "tar_l1", "tar_l2",
        "gmi", "cv", "mean_bg", "std_bg", "lbgi", "hbgi", "mage",
    ]
    # pandas .std() default is ddof=1, which now matches compute_metrics()
    # (fixed to ddof=1 in H7#1). Both per-patient std_bg and the cohort-level
    # _std aggregation are therefore consistent. H7#5.
    agg = summary.groupby("archetype")[metric_cols].agg(["mean", "std"]).round(2)
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    agg = agg.reset_index()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        agg.to_csv(output_path, index=False)

    return agg



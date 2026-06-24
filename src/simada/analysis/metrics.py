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

"""Clinical glycemic metrics.

All metrics follow the International Consensus on Time in Range
(Battelino et al., Diabetes Care 2019) and standard clinical practice.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


class GlycemicMetricsResult(NamedTuple):
    """Computed glycemic metrics for a single patient or period."""

    tir: float          # Time in Range 70-180 (%)
    tbr_l1: float       # Time Below Range 54-69 (%)
    tbr_l2: float       # Time Below Range <54 (%)
    tar_l1: float       # Time Above Range 181-250 (%)
    tar_l2: float       # Time Above Range >250 (%)
    gmi: float          # Glucose Management Indicator (estimated A1C)
    cv: float           # Coefficient of Variation (%)
    mean_bg: float      # Mean glucose (mg/dL)
    std_bg: float       # Std dev glucose (mg/dL)
    lbgi: float         # Low Blood Glucose Index
    hbgi: float         # High Blood Glucose Index
    mage: float         # Mean Amplitude of Glycemic Excursions
    readings: int       # Number of BG readings
    severe_hypo_episodes: int  # Consecutive BG < 54 stretches lasting >15min
    max_bg: float       # Peak BG observed (mg/dL)
    min_bg: float       # Nadir BG observed (mg/dL)


def compute_metrics(bg_values: np.ndarray) -> GlycemicMetricsResult:
    """Compute all glycemic metrics from a blood glucose time series.

    Args:
        bg_values: Array of BG values in mg/dL.

    Returns:
        GlycemicMetricsResult with all clinical metrics.
    """
    bg = bg_values[~np.isnan(bg_values)]
    n = len(bg)
    if n == 0:
        return GlycemicMetricsResult(
            tir=0, tbr_l1=0, tbr_l2=0, tar_l1=0, tar_l2=0,
            gmi=0, cv=0, mean_bg=0, std_bg=0, lbgi=0, hbgi=0, mage=0, readings=0,
            severe_hypo_episodes=0, max_bg=0, min_bg=0,
        )

    mean_bg = float(np.mean(bg))
    # Use ddof=1 (sample std) not ddof=0 (population std) — H7#1.
    # Matches export.py aggregation (pandas .std() default is ddof=1) and
    # prevents misclassification of borderline patients near the ADA CV=36% cutoff.
    std_bg = float(np.std(bg, ddof=1))

    # Time in Range percentages
    tir = float(np.sum((bg >= 70) & (bg <= 180)) / n * 100)
    tbr_l1 = float(np.sum((bg >= 54) & (bg < 70)) / n * 100)
    tbr_l2 = float(np.sum(bg < 54) / n * 100)
    tar_l1 = float(np.sum((bg > 180) & (bg <= 250)) / n * 100)
    tar_l2 = float(np.sum(bg > 250) / n * 100)

    # GMI (Glucose Management Indicator) — estimated A1C from CGM
    # Formula: GMI = 3.31 + 0.02392 * mean_glucose
    # Clamp to clinically plausible A1C range [3.0, 14.0]: the linear formula
    # extrapolates absurd values (e.g. GMI ~17.7 for mean BG=600) outside the
    # range where it was empirically validated (~5-12). Upper bound 14% is the
    # maximum plausible A1C clinically observed in uncontrolled T1D.
    gmi = float(np.clip(3.31 + 0.02392 * mean_bg, 3.0, 14.0))

    # CV (Coefficient of Variation)
    cv = (std_bg / mean_bg * 100) if mean_bg > 0 else 0.0

    # LBGI and HBGI (Kovatchev et al.)
    lbgi, hbgi = _compute_bgi(bg)

    # MAGE (Mean Amplitude of Glycemic Excursions)
    mage = _compute_mage(bg, std_bg)

    # Severe hypo episodes: consecutive stretches of BG < 54 lasting >15 min.
    # ADA defines >15 min (strict), not >=15 min. At 3-min CGM intervals,
    # >15 min requires >= 6 consecutive readings (6 * 3 = 18 min). — H7#8.
    severe_hypo_episodes = _count_severe_hypo_episodes(bg, threshold=54.0, min_readings=6)

    max_bg = float(np.max(bg))
    min_bg = float(np.min(bg))

    return GlycemicMetricsResult(
        tir=round(tir, 1),
        tbr_l1=round(tbr_l1, 1),
        tbr_l2=round(tbr_l2, 1),
        tar_l1=round(tar_l1, 1),
        tar_l2=round(tar_l2, 1),
        gmi=round(gmi, 2),
        cv=round(cv, 1),
        mean_bg=round(mean_bg, 1),
        std_bg=round(std_bg, 1),
        lbgi=round(lbgi, 2),
        hbgi=round(hbgi, 2),
        mage=round(mage, 1),
        readings=n,
        severe_hypo_episodes=severe_hypo_episodes,
        max_bg=round(max_bg, 1),
        min_bg=round(min_bg, 1),
    )


def compute_metrics_from_dataframe(df: pd.DataFrame, bg_col: str = "BG") -> GlycemicMetricsResult:
    """Compute metrics from a pandas DataFrame."""
    if bg_col not in df.columns:
        msg = f"Column '{bg_col}' not found in DataFrame"
        raise ValueError(msg)
    return compute_metrics(df[bg_col].values)


def compute_metrics_from_parquet(path: Path, bg_col: str = "BG") -> GlycemicMetricsResult:
    """Compute metrics from a Parquet file."""
    table = pq.read_table(path, columns=[bg_col])
    bg = table.column(bg_col).to_numpy()
    return compute_metrics(bg)


def compute_cohort_summary(
    results_dir: Path,
) -> pd.DataFrame:
    """Compute metrics for all patients in a results directory.

    Args:
        results_dir: Path to the run output directory (contains timeseries/).

    Returns:
        DataFrame with one row per patient and columns for each metric.
    """
    ts_dir = results_dir / "timeseries"
    if not ts_dir.exists():
        msg = f"timeseries directory not found in {results_dir}"
        raise FileNotFoundError(msg)

    rows = []
    for pf in sorted(ts_dir.glob("*.parquet")):
        metrics = compute_metrics_from_parquet(pf)
        name = pf.stem
        rows.append({"patient": name, **metrics._asdict()})

    if not rows:
        # Try CSV fallback
        for cf in sorted(ts_dir.glob("*.csv")):
            df = pd.read_csv(cf, index_col=0)
            metrics = compute_metrics_from_dataframe(df)
            name = cf.stem
            rows.append({"patient": name, **metrics._asdict()})

    return pd.DataFrame(rows)


def _compute_bgi(bg: np.ndarray) -> tuple[float, float]:
    """Compute Low and High Blood Glucose Index (Kovatchev et al.).

    The BG risk function transforms glucose values into a symmetric
    scale where hypo and hyper risks can be compared directly.
    """
    # Transform BG to symmetric scale
    # f(BG) = 1.509 * (ln(BG)^1.084 - 5.381)
    bg_clamped = np.clip(bg, 20, 600)  # avoid log(0)
    f_bg = 1.509 * (np.log(bg_clamped) ** 1.084 - 5.381)

    rl = np.where(f_bg < 0, 10 * f_bg ** 2, 0)  # low risk
    rh = np.where(f_bg > 0, 10 * f_bg ** 2, 0)  # high risk

    lbgi = float(np.mean(rl))
    hbgi = float(np.mean(rh))
    return lbgi, hbgi


def _compute_mage(bg: np.ndarray, std_bg: float) -> float:
    """Compute Mean Amplitude of Glycemic Excursions (Rodbard 2009).

    Implementation follows Rodbard D. (2009): zero-crossings of the smoothed
    first derivative define excursion boundaries. For each interval between
    consecutive sign changes, the amplitude is the global max-minus-min.
    Only excursions with amplitude > 1 SD are included in the mean.

    This replaces the earlier greedy valley-search (H7#3), which broke on the
    first non-monotonic step and systematically underestimated amplitudes.
    """
    if len(bg) < 3 or std_bg <= 0:
        return 0.0

    # Step 1: smooth the derivative with a 3-point moving average to reduce
    # noise-induced spurious zero crossings.
    diff = np.diff(bg.astype(float))
    if len(diff) < 3:
        return 0.0
    # 3-point moving average of the derivative
    smoothed = np.convolve(diff, np.ones(3) / 3.0, mode="valid")
    if len(smoothed) == 0:
        return 0.0

    # Step 2: find indices where smoothed derivative changes sign.
    # A zero-crossing at index i means the signal passed through a local
    # extremum between positions i and i+1 in the smoothed array.
    signs = np.sign(smoothed)
    # Remove zeros from sign sequence (treat as continuation of previous sign)
    nonzero_mask = signs != 0
    if not np.any(nonzero_mask):
        return 0.0
    signs_nz = signs[nonzero_mask]
    idx_nz = np.where(nonzero_mask)[0]

    # Detect sign changes in the cleaned sign array
    sign_changes = np.where(np.diff(signs_nz) != 0)[0]
    # Crossing positions in the original diff array
    crossing_positions = idx_nz[sign_changes + 1]

    # Offset: smoothed[i] corresponds to mean of diff[i], diff[i+1], diff[i+2],
    # which relates to bg[i+1..i+3]. Map crossings back to bg indices.
    # Conservative: treat crossing at smoothed index i as bg index i+2.
    bg_crossings = np.concatenate([[0], crossing_positions + 2, [len(bg) - 1]])
    bg_crossings = np.unique(bg_crossings.clip(0, len(bg) - 1))

    # Step 3: for each segment between crossings, compute global amplitude.
    excursions = []
    for k in range(len(bg_crossings) - 1):
        seg = bg[bg_crossings[k]: bg_crossings[k + 1] + 1]
        if len(seg) < 2:
            continue
        amplitude = float(np.max(seg)) - float(np.min(seg))
        if amplitude > std_bg:
            excursions.append(amplitude)

    return float(np.mean(excursions)) if excursions else 0.0


def _count_severe_hypo_episodes(
    bg: np.ndarray,
    threshold: float = 54.0,
    min_readings: int = 6,
) -> int:
    """Count episodes of consecutive BG < threshold lasting min_readings or more.

    At the standard 3-minute CGM sampling interval, min_readings=6 corresponds
    to 18 min (>15 min strict per ADA definition). Using 5 would give exactly
    15 min, which does NOT satisfy the >15 min requirement. — H7#8.
    """
    episodes = 0
    run_length = 0
    for value in bg:
        if value < threshold:
            run_length += 1
        else:
            if run_length >= min_readings:
                episodes += 1
            run_length = 0
    # Check final run
    if run_length >= min_readings:
        episodes += 1
    return episodes

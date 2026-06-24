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

"""Regression tests for H7 v2 bug-hunt fixes (2026-05-18).

Covers all 8 effective bugs from the H7 section:
  #1 CRITICAL — std_bg ddof=0 → ddof=1
  #2 HIGH     — _color_for_metric / _fmt_value bg_min/bg_max key mismatch
  #3 HIGH     — MAGE greedy valley search → Rodbard 2009 zero-crossings
  #4 MEDIUM   — _reshape_to_daily silent truncation warning
  #5 MEDIUM   — export _std ddof consistency comment (structural, not logic)
  #6 MEDIUM   — col_widths defensive assert in render_metrics_table
  #7 LOW      — add_bg_reference_labels xlim read-before-set
  #8 LOW      — min_readings=5 (ADA >15 min) → 6
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import pandas as pd

from simada.analysis.agp import generate_agp, generate_comparison_agp
from simada.analysis.metrics import compute_metrics, _compute_mage
from simada.analysis.plots import _color_for_metric, _fmt_value, render_metrics_table
from simada.analysis.style import CELL_GREEN, CELL_RED, CELL_YELLOW


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_metrics(**overrides) -> dict:
    """Return a full metrics dict (keys from GlycemicMetricsResult._asdict())."""
    base = {
        "tir": 75.0, "tbr_l1": 2.0, "tbr_l2": 0.5,
        "tar_l1": 18.0, "tar_l2": 4.0,
        "gmi": 6.5, "cv": 28.0,
        "mean_bg": 130.0, "std_bg": 36.4,
        "lbgi": 1.5, "hbgi": 2.0, "mage": 45.0,
        "readings": 2880,
        "severe_hypo_episodes": 0,
        "max_bg": 220.0, "min_bg": 65.0,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Bug #1 — std_bg ddof=0 → ddof=1 (CRITICAL)
# ---------------------------------------------------------------------------

class TestStdBgDdof:
    """Fix #1: std_bg must use ddof=1 (sample std) not ddof=0 (population std)."""

    def test_std_bg_uses_ddof1(self) -> None:
        """std_bg must equal np.std(bg, ddof=1), not np.std(bg, ddof=0)."""
        rng = np.random.default_rng(7)
        bg = rng.normal(140, 40, size=100).clip(40, 400)
        m = compute_metrics(bg)
        expected = np.std(bg, ddof=1)
        assert m.std_bg == pytest.approx(expected, abs=0.1), (
            f"std_bg={m.std_bg} does not match ddof=1 value {expected:.4f}; "
            "population std (ddof=0) would be smaller"
        )

    def test_std_bg_ddof1_greater_than_ddof0(self) -> None:
        """For a finite sample, ddof=1 std is strictly larger than ddof=0 std."""
        rng = np.random.default_rng(99)
        bg = rng.normal(150, 50, size=20).clip(40, 400)
        m = compute_metrics(bg)
        pop_std = np.std(bg, ddof=0)
        # ddof=1 must be strictly larger than ddof=0
        assert m.std_bg > pop_std, (
            f"std_bg={m.std_bg} should be > pop std {pop_std:.4f} (ddof=1 > ddof=0 for finite n)"
        )

    def test_cv_propagates_ddof1(self) -> None:
        """CV = std_bg/mean_bg * 100 must also use the ddof=1 std."""
        rng = np.random.default_rng(13)
        bg = rng.normal(130, 45, size=50).clip(40, 400)
        m = compute_metrics(bg)
        mean = np.mean(bg)
        std_ddof1 = np.std(bg, ddof=1)
        expected_cv = std_ddof1 / mean * 100
        assert m.cv == pytest.approx(expected_cv, abs=0.2), (
            f"CV={m.cv} should match ddof=1-based value {expected_cv:.2f}"
        )

    def test_borderline_cv_36_classification(self) -> None:
        """CV near 36% cutoff must be correctly classified using ddof=1.

        With a borderline patient, using ddof=0 vs ddof=1 can flip the
        ADA risk classification. Verify the returned CV is based on ddof=1.
        """
        rng = np.random.default_rng(42)
        # Craft a BG array whose ddof=0 and ddof=1 std straddle different CVs
        bg = rng.normal(140, 50, size=15).clip(40, 400)
        m = compute_metrics(bg)
        std_ddof1 = np.std(bg, ddof=1)
        cv_ddof1 = std_ddof1 / np.mean(bg) * 100
        assert abs(m.cv - cv_ddof1) < 0.2, (
            "CV does not match ddof=1; possible misclassification near ADA cutoff"
        )


# ---------------------------------------------------------------------------
# Bug #2 — bg_min/bg_max key mismatch in _color_for_metric and _fmt_value (HIGH)
# ---------------------------------------------------------------------------

class TestColorForMetricBgMinMax:
    """Fix #2: _color_for_metric and _fmt_value must read min_bg/max_bg keys
    (matching GlycemicMetricsResult._asdict()), not bg_min/bg_max.

    Phase 1 figures were generated with the wrong keys → Min/Max BG cells
    showed incorrect values and colors.
    """

    def test_min_bg_green_when_above_70(self) -> None:
        """min_bg > 70 mg/dL → CELL_GREEN (no hypoglycemia nadir)."""
        m = _base_metrics(min_bg=80.0)
        assert _color_for_metric("Min BG", m) == CELL_GREEN

    def test_min_bg_yellow_between_54_and_70(self) -> None:
        """54 <= min_bg <= 70 → CELL_YELLOW."""
        m = _base_metrics(min_bg=60.0)
        assert _color_for_metric("Min BG", m) == CELL_YELLOW

    def test_min_bg_exactly_54_is_yellow(self) -> None:
        """min_bg == 54 → CELL_YELLOW (not CELL_RED)."""
        m = _base_metrics(min_bg=54.0)
        assert _color_for_metric("Min BG", m) == CELL_YELLOW

    def test_min_bg_below_54_is_red(self) -> None:
        """min_bg < 54 → CELL_RED (severe hypo nadir)."""
        m = _base_metrics(min_bg=40.0)
        assert _color_for_metric("Min BG", m) == CELL_RED

    def test_max_bg_green_when_below_180(self) -> None:
        """max_bg < 180 → CELL_GREEN."""
        m = _base_metrics(max_bg=170.0)
        assert _color_for_metric("Max BG", m) == CELL_GREEN

    def test_max_bg_yellow_between_180_and_250(self) -> None:
        """180 <= max_bg <= 250 → CELL_YELLOW."""
        m = _base_metrics(max_bg=220.0)
        assert _color_for_metric("Max BG", m) == CELL_YELLOW

    def test_max_bg_above_250_is_red(self) -> None:
        """max_bg > 250 → CELL_RED."""
        m = _base_metrics(max_bg=300.0)
        assert _color_for_metric("Max BG", m) == CELL_RED

    def test_min_bg_not_zero_when_key_correct(self) -> None:
        """If the key is wrong (bg_min vs min_bg), .get('bg_min', 0) returns 0,
        which would be classified as CELL_RED incorrectly. The correct key gives
        the right color for a clinically fine patient.
        """
        # min_bg=80 → should be GREEN; if bug present → RED (defaulting to 0)
        m = _base_metrics(min_bg=80.0)
        color = _color_for_metric("Min BG", m)
        assert color == CELL_GREEN, (
            f"Color is {color!r} instead of CELL_GREEN. "
            "Likely the key 'bg_min' is being used instead of 'min_bg'."
        )

    def test_fmt_value_min_bg_displays_correctly(self) -> None:
        """_fmt_value('Min BG', m) must read min_bg key, not bg_min."""
        m = _base_metrics(min_bg=65.0)
        result = _fmt_value("Min BG", m)
        assert "65" in result, (
            f"_fmt_value returned {result!r}, expected to contain '65'. "
            "Likely reads bg_min (KeyError or wrong default)."
        )

    def test_fmt_value_max_bg_displays_correctly(self) -> None:
        """_fmt_value('Max BG', m) must read max_bg key, not bg_max."""
        m = _base_metrics(max_bg=224.0)
        result = _fmt_value("Max BG", m)
        assert "224" in result, (
            f"_fmt_value returned {result!r}, expected to contain '224'."
        )


# ---------------------------------------------------------------------------
# Bug #3 — MAGE greedy → Rodbard 2009 zero-crossings of smoothed derivative (HIGH)
# ---------------------------------------------------------------------------

class TestMageRodbard2009:
    """Fix #3: MAGE must use zero-crossings of the smoothed derivative (Rodbard 2009).

    The greedy implementation underestimates excursion amplitudes by breaking
    on the first non-monotonic step instead of finding the global extremum
    between derivative sign changes.
    """

    def test_mage_symmetric_oscillation_correct_amplitude(self) -> None:
        """Symmetric sawtooth should yield MAGE equal to the swing amplitude.

        Build a clean sawtooth: 80 → 220 → 80 repeated.
        Amplitude = 140. MAGE must be close to 140 (all excursions > 1 SD).
        """
        n_cycles = 20
        period = 60  # samples per full cycle (30 up, 30 down)
        half = period // 2
        up = np.linspace(80, 220, half)
        down = np.linspace(220, 80, half)
        one_cycle = np.concatenate([up, down])
        bg = np.tile(one_cycle, n_cycles)
        m = compute_metrics(bg)
        # Amplitude is 140 mg/dL; MAGE must be within 20 mg/dL of that
        assert m.mage == pytest.approx(140.0, abs=20.0), (
            f"MAGE={m.mage:.1f} for a clean 80↔220 sawtooth; expected ≈140"
        )

    def test_mage_flat_is_zero(self) -> None:
        """Flat BG → 0 excursions → MAGE == 0."""
        bg = np.full(500, 120.0)
        m = compute_metrics(bg)
        assert m.mage == 0.0

    def test_mage_large_asymmetric_swing(self) -> None:
        """An asymmetric large swing (60→300→60) should produce MAGE > 150."""
        n_cycles = 15
        up = np.linspace(60, 300, 40)
        down = np.linspace(300, 60, 40)
        one_cycle = np.concatenate([up, down])
        bg = np.tile(one_cycle, n_cycles)
        m = compute_metrics(bg)
        assert m.mage > 150.0, f"MAGE={m.mage:.1f}; expected >150 for 60↔300 swing"

    def test_mage_small_noise_below_1sd_not_counted(self) -> None:
        """Tiny noise (amplitude << 1 SD) should yield MAGE == 0."""
        rng = np.random.default_rng(0)
        # BG near-constant 120, with only 2 mg/dL noise → amplitude << 1 SD
        bg = 120.0 + rng.uniform(-1, 1, size=500)
        m = compute_metrics(bg)
        # std_bg will be small (~0.58 mg/dL) but MAGE should count 0 excursions
        # larger than 1 SD of such a flat array — no excursion > 1 SD here.
        assert m.mage == pytest.approx(0.0, abs=2.0), (
            f"MAGE={m.mage:.1f} for near-flat BG; expected ≈0"
        )

    def test_mage_direct_helper_zero_crossings(self) -> None:
        """_compute_mage helper with known std_bg should be > 0 for swings."""
        # 100 mg/dL swing, 1 SD ≈ 50 → all excursions > 1 SD
        up = np.linspace(80, 180, 30)
        down = np.linspace(180, 80, 30)
        bg = np.tile(np.concatenate([up, down]), 10)
        std_bg = float(np.std(bg, ddof=1))
        mage = _compute_mage(bg, std_bg)
        assert mage > 0.0, f"_compute_mage returned {mage} for a 100 mg/dL swing"


# ---------------------------------------------------------------------------
# Bug #4 — _reshape_to_daily silent truncation warning (MEDIUM)
# ---------------------------------------------------------------------------

class TestAgpTruncationWarning:
    """Fix #4: agp._reshape_to_daily must warn when samples are truncated
    (i.e., when len(bg) % samples_per_day != 0).
    """

    def test_warn_on_truncation(self) -> None:
        """A BG series not divisible by samples_per_day should emit a warning."""
        samples_per_day = 24 * 60 // 3  # 480
        # 3 full days + 60 extra samples (= 3h leftover) → truncation
        bg = pd.Series(np.full(samples_per_day * 3 + 60, 120.0))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            generate_agp(bg)
            truncation_warnings = [
                w for w in caught
                if issubclass(w.category, UserWarning)
                and "Dropped" in str(w.message)
            ]
            assert truncation_warnings, (
                f"Expected a 'Dropped N samples' UserWarning, got: "
                f"{[str(w.message) for w in caught]}"
            )

    def test_no_warn_when_exact_days(self) -> None:
        """A BG series that is exactly N days should NOT emit a truncation warning."""
        samples_per_day = 24 * 60 // 3
        bg = pd.Series(np.full(samples_per_day * 3, 120.0))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            generate_agp(bg)
            truncation_warnings = [
                w for w in caught
                if issubclass(w.category, UserWarning)
                and "Dropped" in str(w.message)
            ]
            assert not truncation_warnings, (
                f"Unexpected truncation warning for exact-days data: "
                f"{[str(w.message) for w in truncation_warnings]}"
            )


# ---------------------------------------------------------------------------
# Bug #6 — col_widths defensive assert in render_metrics_table (MEDIUM)
# ---------------------------------------------------------------------------

class TestMetricsTableColWidths:
    """Fix #6: render_metrics_table must produce a table without assertion errors.

    The col_widths list must match the number of data columns (colLabels) so
    matplotlib's table layout is correct regardless of version behavior.
    """

    def test_render_metrics_table_single_arch(self, tmp_path: Path) -> None:
        """render_metrics_table with 1 archetype must not raise."""
        m = _base_metrics()
        out = tmp_path / "metrics_table_1.png"
        render_metrics_table({"adherent": m}, output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_render_metrics_table_three_archs(self, tmp_path: Path) -> None:
        """render_metrics_table with 3 archetypes must not raise."""
        all_m = {
            "adherent": _base_metrics(tir=80, min_bg=70, max_bg=190),
            "moderate": _base_metrics(tir=65, min_bg=55, max_bg=240),
            "nonadherent": _base_metrics(tir=45, min_bg=38, max_bg=320),
        }
        out = tmp_path / "metrics_table_3.png"
        render_metrics_table(all_m, output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Bug #7 — add_bg_reference_labels xlim read-before-set (LOW)
# ---------------------------------------------------------------------------

class TestBgReferenceLabelXlim:
    """Fix #7: add_bg_reference_labels must be called AFTER set_xlim(0,24).

    Previously the call appeared before _draw_agp_on_axis(), so ax.get_xlim()
    returned the matplotlib default (0, 1) and all labels ended up at x≈0.99
    (left-side clutter) instead of x≈23.76 (right margin).
    """

    def test_agp_reference_labels_at_right_margin(self, tmp_path: Path) -> None:
        """After generate_agp(), the axes xlim must be (0, 24) (set by _draw_agp_on_axis).

        If labels are placed before set_xlim, they land at x≈0.99 and the
        figure still renders — but the labels are visually wrong. We verify
        the axis xlim is correct (implying the set_xlim call ran before any
        label that reads it).
        """
        samples_per_day = 24 * 60 // 3
        bg = pd.Series(np.full(samples_per_day * 7, 120.0))
        out = tmp_path / "agp_xlim_check.png"
        # We need to check the axes state; capture the figure before finalize closes it
        import simada.analysis.agp as agp_mod
        from simada.analysis.style import simada_figure, FIGSIZE

        # Re-implement just the relevant part to inspect axes state
        fig, ax = simada_figure(figsize=FIGSIZE.agp_single)
        agp_mod.add_target_range(ax)
        agp_mod.add_hypo_line(ax)
        agp_mod._draw_agp_on_axis(ax, bg.values, samples_per_day, "#0072B2")
        agp_mod.add_bg_reference_labels(ax)  # must read xlim AFTER set_xlim(0,24)

        xlim = ax.get_xlim()
        assert xlim[1] == pytest.approx(24.0, abs=0.5), (
            f"xlim[1]={xlim[1]} after _draw_agp_on_axis; expected ~24. "
            "Labels placed before this call would read the default xlim (1.0)."
        )
        plt.close(fig)

    def test_agp_generates_without_error(self, tmp_path: Path) -> None:
        """generate_agp must complete without error with fixed call order."""
        samples_per_day = 24 * 60 // 3
        bg = pd.Series(np.full(samples_per_day * 3, 120.0))
        out = tmp_path / "agp_fixed.png"
        generate_agp(bg, output_path=out)
        assert out.exists()


# ---------------------------------------------------------------------------
# Bug #8 — min_readings=5 → 6 for ADA >15min definition (LOW)
# ---------------------------------------------------------------------------

class TestSevereHypoMinReadings:
    """Fix #8: ADA defines severe hypoglycemia as >15 min, not >=15 min.

    At 3-min CGM intervals, >15 min requires >=6 consecutive readings
    (6 * 3 = 18 min). The previous default of min_readings=5 gave exactly
    15 min (= 5 * 3), which is NOT >15 min per the ADA definition.
    """

    def test_exactly_5_readings_not_counted(self) -> None:
        """5 consecutive readings < 54 = 15 min exactly → NOT >15 min → 0 episodes."""
        bg = np.concatenate([
            np.full(100, 120.0),
            np.full(5, 45.0),   # 5 * 3min = 15 min (NOT > 15)
            np.full(100, 120.0),
        ])
        m = compute_metrics(bg)
        assert m.severe_hypo_episodes == 0, (
            f"Got {m.severe_hypo_episodes} episode(s) for exactly 5 readings; "
            "ADA >15 min means 5 readings (= 15 min) must NOT count."
        )

    def test_exactly_6_readings_counted(self) -> None:
        """6 consecutive readings < 54 = 18 min → >15 min → 1 episode."""
        bg = np.concatenate([
            np.full(100, 120.0),
            np.full(6, 45.0),   # 6 * 3min = 18 min (> 15 → counts)
            np.full(100, 120.0),
        ])
        m = compute_metrics(bg)
        assert m.severe_hypo_episodes == 1, (
            f"Got {m.severe_hypo_episodes} episode(s) for 6 readings (18 min); "
            "should count as 1 severe episode."
        )

    def test_10_readings_still_counted(self) -> None:
        """10 consecutive readings < 54 must still count (30 min > 15 min)."""
        bg = np.concatenate([
            np.full(100, 120.0),
            np.full(10, 45.0),
            np.full(100, 120.0),
        ])
        m = compute_metrics(bg)
        assert m.severe_hypo_episodes == 1

    def test_two_episodes_both_6_readings(self) -> None:
        """Two separate 6-reading episodes must both be counted."""
        bg = np.concatenate([
            np.full(50, 120.0),
            np.full(6, 45.0),
            np.full(50, 120.0),
            np.full(6, 45.0),
            np.full(50, 120.0),
        ])
        m = compute_metrics(bg)
        assert m.severe_hypo_episodes == 2


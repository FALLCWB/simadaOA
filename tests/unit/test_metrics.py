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

"""Tests for glycemic metrics computation."""

from __future__ import annotations

import numpy as np
import pytest

from simada.analysis.metrics import compute_metrics


class TestGlycemicMetrics:
    """Tests for compute_metrics."""

    def test_perfect_control(self) -> None:
        """BG always at 120 → 100% TIR, 0% TBR/TAR, low CV."""
        bg = np.full(1000, 120.0)
        m = compute_metrics(bg)
        assert m.tir == 100.0
        assert m.tbr_l1 == 0.0
        assert m.tbr_l2 == 0.0
        assert m.tar_l1 == 0.0
        assert m.tar_l2 == 0.0
        assert m.cv == 0.0
        assert m.mean_bg == 120.0

    def test_all_hypo(self) -> None:
        """BG always at 40 → 0% TIR, 100% TBR L2."""
        bg = np.full(1000, 40.0)
        m = compute_metrics(bg)
        assert m.tir == 0.0
        assert m.tbr_l2 == 100.0
        assert m.tar_l1 == 0.0

    def test_all_hyper(self) -> None:
        """BG always at 300 → 0% TIR, 100% TAR L2."""
        bg = np.full(1000, 300.0)
        m = compute_metrics(bg)
        assert m.tir == 0.0
        assert m.tbr_l1 == 0.0
        assert m.tar_l2 == 100.0

    def test_mixed_ranges(self) -> None:
        """50% in range, 25% low, 25% high."""
        bg = np.array([60.0] * 250 + [120.0] * 500 + [200.0] * 250)
        m = compute_metrics(bg)
        assert m.tir == pytest.approx(50.0, abs=0.1)
        assert m.readings == 1000

    def test_gmi_formula(self) -> None:
        """GMI = 3.31 + 0.02392 * mean_glucose."""
        bg = np.full(100, 150.0)
        m = compute_metrics(bg)
        expected_gmi = 3.31 + 0.02392 * 150.0
        assert m.gmi == pytest.approx(expected_gmi, abs=0.01)

    def test_gmi_clamped_upper_bound(self) -> None:
        """BG far above the validation range must not produce an absurd GMI.

        Regression for BUG #2: a mean BG of 600 mg/dL would naively give
        GMI ~17.66, which is clinically nonsensical (A1C max plausible is
        ~14%). The clamp guarantees the value stays within the validated
        clinical range.
        """
        bg = np.full(100, 600.0)
        m = compute_metrics(bg)
        assert m.gmi <= 14.0, f"GMI must be clamped to <=14, got {m.gmi}"
        # Should hit the upper bound exactly for BG=600.
        assert m.gmi == pytest.approx(14.0, abs=0.01)

    def test_gmi_clamped_lower_bound(self) -> None:
        """BG far below the validation range must not produce GMI below 3.0.

        At very low mean BG (e.g. constant 40 mg/dL), the formula would
        produce GMI ~4.27 (already above 3.0). But the clamp guards
        against degenerate inputs and future formula edits.
        """
        # mean_bg=40 => 3.31 + 0.02392*40 = 4.27, above the lower clamp.
        bg = np.full(100, 40.0)
        m = compute_metrics(bg)
        assert m.gmi >= 3.0
        # Confirm the in-range path is unaffected.
        assert m.gmi == pytest.approx(4.27, abs=0.01)

    def test_cv_calculation(self) -> None:
        """CV = std/mean * 100, using ddof=1 (sample std). H7#1."""
        bg = np.array([100.0, 200.0] * 500)
        m = compute_metrics(bg)
        # Must use ddof=1 to match compute_metrics() after H7#1 fix.
        expected_cv = np.std(bg, ddof=1) / np.mean(bg) * 100
        assert m.cv == pytest.approx(expected_cv, abs=0.1)

    def test_lbgi_higher_for_hypo(self) -> None:
        """LBGI should be higher when more time in hypoglycemia."""
        bg_normal = np.full(1000, 120.0)
        bg_hypo = np.array([50.0] * 500 + [120.0] * 500)
        m_normal = compute_metrics(bg_normal)
        m_hypo = compute_metrics(bg_hypo)
        assert m_hypo.lbgi > m_normal.lbgi

    def test_hbgi_higher_for_hyper(self) -> None:
        """HBGI should be higher when more time in hyperglycemia."""
        bg_normal = np.full(1000, 120.0)
        bg_hyper = np.array([300.0] * 500 + [120.0] * 500)
        m_normal = compute_metrics(bg_normal)
        m_hyper = compute_metrics(bg_hyper)
        assert m_hyper.hbgi > m_normal.hbgi

    def test_mage_zero_for_flat(self) -> None:
        """MAGE should be exactly 0 for constant BG.

        Using abs=0.01 instead of abs=1.0: a flat BG signal has zero amplitude
        excursions, so any non-zero MAGE is a bug. The loose 1.0 tolerance
        masked rounding/off-by-one errors and prevented detection of small
        but real amplitude bugs in the MAGE computation.
        """
        bg = np.full(1000, 120.0)
        m = compute_metrics(bg)
        assert m.mage == pytest.approx(0.0, abs=0.01)

    def test_mage_positive_for_swings(self) -> None:
        """MAGE should be positive when there are significant swings."""
        # Create oscillating BG: 80 ↔ 220 every 50 readings
        bg = np.tile(
            np.concatenate([np.full(50, 80.0), np.full(50, 220.0)]),
            10,
        )
        m = compute_metrics(bg)
        assert m.mage > 50.0

    def test_empty_array(self) -> None:
        """Empty BG array should return zeros."""
        m = compute_metrics(np.array([]))
        assert m.readings == 0
        assert m.tir == 0.0

    def test_nan_values_ignored(self) -> None:
        """NaN values should be filtered out."""
        bg = np.array([120.0, np.nan, 120.0, np.nan, 120.0])
        m = compute_metrics(bg)
        assert m.readings == 3
        assert m.tir == 100.0

    def test_tir_ranges_sum_to_100(self) -> None:
        """TIR + TBR_L1 + TBR_L2 + TAR_L1 + TAR_L2 should sum to 100%."""
        rng = np.random.default_rng(42)
        bg = rng.normal(150, 60, size=5000).clip(30, 500)
        m = compute_metrics(bg)
        total = m.tir + m.tbr_l1 + m.tbr_l2 + m.tar_l1 + m.tar_l2
        assert total == pytest.approx(100.0, abs=0.5)


class TestSevereHypoEpisodes:
    """Tests for severe_hypo_episodes metric."""

    def test_no_severe_hypo(self) -> None:
        """All BG > 54 should yield 0 severe hypo episodes."""
        bg = np.full(1000, 80.0)
        m = compute_metrics(bg)
        assert m.severe_hypo_episodes == 0

    def test_one_severe_episode(self) -> None:
        """10 consecutive readings < 54 should count as 1 episode."""
        bg = np.concatenate([
            np.full(100, 120.0),
            np.full(10, 45.0),   # 10 readings < 54 (30 min)
            np.full(100, 120.0),
        ])
        m = compute_metrics(bg)
        assert m.severe_hypo_episodes == 1

    def test_short_dip_not_counted(self) -> None:
        """3 readings < 54 (< 15 min) should not count as an episode."""
        bg = np.concatenate([
            np.full(100, 120.0),
            np.full(3, 45.0),    # 3 readings < 54 (9 min, below threshold)
            np.full(100, 120.0),
        ])
        m = compute_metrics(bg)
        assert m.severe_hypo_episodes == 0

    def test_two_separate_episodes(self) -> None:
        """Two distinct stretches of 6+ readings < 54 should count as 2 episodes."""
        bg = np.concatenate([
            np.full(50, 120.0),
            np.full(8, 45.0),    # episode 1
            np.full(50, 120.0),
            np.full(6, 45.0),    # episode 2
            np.full(50, 120.0),
        ])
        m = compute_metrics(bg)
        assert m.severe_hypo_episodes == 2

    def test_boundary_exactly_5_readings(self) -> None:
        """Exactly 5 readings < 54 = 15 min exactly must NOT count as an episode.

        ADA defines severe hypoglycemia as >15 min (strict inequality). At 3-min
        CGM intervals, 5 readings = 15 min exactly, which does NOT satisfy >15 min.
        Threshold corrected from min_readings=5 to min_readings=6 (H7#8).
        """
        bg = np.concatenate([
            np.full(100, 120.0),
            np.full(5, 45.0),    # exactly 5 readings = 15 min (NOT >15 min)
            np.full(100, 120.0),
        ])
        m = compute_metrics(bg)
        assert m.severe_hypo_episodes == 0

    def test_boundary_exactly_6_readings(self) -> None:
        """Exactly 6 readings < 54 = 18 min → >15 min → 1 severe episode (H7#8)."""
        bg = np.concatenate([
            np.full(100, 120.0),
            np.full(6, 45.0),    # 6 * 3 min = 18 min (>15 min → counts)
            np.full(100, 120.0),
        ])
        m = compute_metrics(bg)
        assert m.severe_hypo_episodes == 1

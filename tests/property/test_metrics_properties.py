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

"""Property-based tests for glycemic metrics, bolus calculations, and related models.

Uses Hypothesis to verify invariants that must hold for *any* valid input,
not just the specific examples in unit tests.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
from hypothesis import given, settings, strategies as st
from numpy.random import default_rng

from simada.analysis.metrics import compute_metrics
from simada.core.types import InsulinRegimen
from simada.insulin.basal import BasalProfile
from simada.insulin.calculator import BolusCalculator
from simada.meals.estimation import CarbEstimationModel

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

bg_arrays = (
    st.lists(st.floats(min_value=20, max_value=600), min_size=10, max_size=1000)
    .map(np.array)
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMetricsProperties:
    """Property-based tests for compute_metrics."""

    @given(bg=bg_arrays)
    @settings(max_examples=200, deadline=None)
    def test_tir_ranges_sum_to_100(self, bg: np.ndarray) -> None:
        """TIR + TBR_L1 + TBR_L2 + TAR_L1 + TAR_L2 must always sum to 100%.

        This is a fundamental invariant: every BG reading falls into exactly
        one of the five international consensus ranges.
        """
        result = compute_metrics(bg)
        total = result.tir + result.tbr_l1 + result.tbr_l2 + result.tar_l1 + result.tar_l2
        assert abs(total - 100.0) < 0.5, (
            f"Range percentages sum to {total:.2f}, expected ~100.0. "
            f"TIR={result.tir}, TBR_L1={result.tbr_l1}, TBR_L2={result.tbr_l2}, "
            f"TAR_L1={result.tar_l1}, TAR_L2={result.tar_l2}"
        )


class TestBolusProperties:
    """Property-based tests for BolusCalculator."""

    @given(
        carbs=st.floats(min_value=0, max_value=500),
        bg=st.floats(min_value=20, max_value=600),
    )
    @settings(max_examples=200, deadline=None)
    def test_bolus_dose_non_negative(self, carbs: float, bg: float) -> None:
        """The total bolus dose must never be negative, regardless of inputs.

        The calculator clamps to zero to prevent negative insulin delivery.
        """
        calc = BolusCalculator(cr=10.0, cf=50.0, target_bg=120.0)
        now = datetime(2026, 6, 1, 12, 0)
        result = calc.calculate(carbs, bg, now)
        assert result.total_dose >= 0.0, (
            f"total_dose={result.total_dose} is negative for "
            f"carbs={carbs}, bg={bg}"
        )


class TestCarbEstimationProperties:
    """Property-based tests for CarbEstimationModel."""

    @given(true_carbs=st.floats(min_value=0, max_value=500))
    @settings(max_examples=200, deadline=None)
    def test_carb_estimate_non_negative(self, true_carbs: float) -> None:
        """Estimated carbs must never be negative.

        A patient cannot estimate negative carbohydrate intake.
        """
        from simada.core.config import ArchetypeParams, load_archetype_params
        from pathlib import Path

        project_root = Path(__file__).resolve().parent.parent.parent
        params = load_archetype_params(
            project_root / "configs" / "archetypes" / "nonadherent.yaml"
        )
        model = CarbEstimationModel(params)
        rng = default_rng(42)
        estimate = model.estimate(true_carbs, rng)
        assert estimate >= 0.0, (
            f"Estimated carbs={estimate} is negative for true_carbs={true_carbs}"
        )


class TestBasalProfileProperties:
    """Property-based tests for BasalProfile."""

    @given(
        tdi=st.floats(min_value=10, max_value=200),
        dawn=st.booleans(),
    )
    @settings(max_examples=100, deadline=None)
    def test_basal_profile_24_entries(self, tdi: float, dawn: bool) -> None:
        """BasalProfile.hourly_profile() must always return exactly 24 entries
        with all rates strictly positive (a patient always needs some basal).
        """
        profile = BasalProfile(
            tdi=tdi,
            regimen=InsulinRegimen.PUMP,
            dawn_phenomenon=dawn,
        )
        hourly = profile.hourly_profile()
        assert len(hourly) == 24, (
            f"Expected 24 hourly entries, got {len(hourly)}"
        )
        for hour, rate in hourly:
            assert 0 <= hour <= 23, f"Invalid hour {hour}"
            assert rate > 0, f"Rate at hour {hour} is {rate}, expected > 0"


class TestGMIProperties:
    """Property-based tests for GMI monotonicity."""

    @given(
        mean_low=st.floats(min_value=60, max_value=200),
        delta=st.floats(min_value=1.0, max_value=300),
    )
    @settings(max_examples=200, deadline=None)
    def test_gmi_monotonically_increases(self, mean_low: float, delta: float) -> None:
        """For BG arrays with a higher mean, GMI should be higher.

        GMI is a linear function of mean glucose, so this must always hold.
        """
        mean_high = mean_low + delta

        # Clamp to valid BG range
        mean_low = max(20.0, min(mean_low, 600.0))
        mean_high = max(20.0, min(mean_high, 600.0))

        if mean_high <= mean_low:
            return  # skip if clamping collapsed the gap

        # Create constant arrays at those means (avoids rounding noise)
        bg_low = np.full(100, mean_low)
        bg_high = np.full(100, mean_high)

        result_low = compute_metrics(bg_low)
        result_high = compute_metrics(bg_high)

        assert result_high.gmi >= result_low.gmi, (
            f"GMI should increase with mean BG. "
            f"mean_low={mean_low:.1f} -> GMI={result_low.gmi}, "
            f"mean_high={mean_high:.1f} -> GMI={result_high.gmi}"
        )

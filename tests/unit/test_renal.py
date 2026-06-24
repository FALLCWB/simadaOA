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

"""Tests for the physiological BG ceiling (renal excretion correction)."""

from __future__ import annotations

import pytest

from simada.physiology.renal import (
    BG_HARD_CEILING,
    RENAL_THRESHOLD_MG_DL,
    apply_renal_correction,
)


class TestRenalCorrection:
    """Unit tests for apply_renal_correction."""

    def test_renal_correction_below_threshold(self) -> None:
        """BG below 180 mg/dL should pass through unchanged."""
        assert apply_renal_correction(150.0) == 150.0

    def test_renal_correction_above_threshold(self) -> None:
        """BG=300 should be slightly reduced (between threshold and saturation)."""
        result = apply_renal_correction(300.0)
        assert result < 300.0
        assert result > RENAL_THRESHOLD_MG_DL

    def test_renal_correction_ceiling(self) -> None:
        """BG=800 must asymptote close to the ceiling (600 mg/dL).

        BUG #3 fix: the previous hard ``min(corrected, BG_HARD_CEILING)``
        clamp introduced a non-differentiable corner at 600 that broke
        adaptive-step ODE solvers. The replacement is a soft-min via
        log-sum-exp that asymptotes to the ceiling without ever exceeding
        it, while remaining C-infinity smooth. We therefore relax the
        exact-equality check to "within a few mg/dL of the ceiling and
        never above it."
        """
        result = apply_renal_correction(800.0)
        assert result <= BG_HARD_CEILING
        # Soft saturation is within the softmin width (6 mg/dL) of the
        # ceiling for inputs well past it.
        assert result > BG_HARD_CEILING - 10.0

    def test_renal_correction_preserves_low_bg(self) -> None:
        """BG=50 (hypoglycemia) must not be modified."""
        assert apply_renal_correction(50.0) == 50.0

    def test_renal_correction_at_exact_threshold(self) -> None:
        """BG exactly at 180 should not be corrected."""
        assert apply_renal_correction(180.0) == 180.0

    def test_renal_correction_never_below_threshold(self) -> None:
        """Result should never go below the renal threshold."""
        # Even with very long sample time, should not go below threshold
        result = apply_renal_correction(185.0, sample_time_min=30.0)
        assert result >= RENAL_THRESHOLD_MG_DL

    def test_renal_correction_sample_time_scaling(self) -> None:
        """Excretion should scale with sample time."""
        result_3min = apply_renal_correction(300.0, sample_time_min=3.0)
        result_6min = apply_renal_correction(300.0, sample_time_min=6.0)
        # Longer sample time = more excretion = lower result
        assert result_6min < result_3min

    # --- BUG #3 regression tests -------------------------------------------

    def test_smooth_ceiling_is_differentiable(self) -> None:
        """BUG #3 regression: ceiling saturation must be continuous and
        the derivative must be bounded across the ceiling transition.

        The previous hard clamp produced a derivative jump from 1.0 to 0.0
        at exactly BG=600, which broke adaptive-step ODE integrators. The
        new soft-min keeps the derivative continuous everywhere.
        """
        # Sample around the ceiling with small steps.
        h = 0.5
        bgs = [BG_HARD_CEILING - 10.0 + i * h for i in range(50)]
        vals = [apply_renal_correction(bg) for bg in bgs]
        # First differences (numerical derivative).
        diffs = [(vals[i + 1] - vals[i]) / h for i in range(len(vals) - 1)]
        # No huge spikes -- derivative bounded.
        for d in diffs:
            assert -0.1 <= d <= 1.1, f"Derivative spike: {d}"

    def test_smooth_ceiling_linear_far_below(self) -> None:
        """BG=200 should be ~200 (linear identity well below ceiling)."""
        result = apply_renal_correction(200.0)
        # apply_renal_correction subtracts a small excretion above 180,
        # but BG=200 is barely above the threshold so the correction is
        # tiny; the smooth ceiling does not deflect it noticeably.
        assert abs(result - 200.0) < 5.0, f"Got {result}, expected ~200"

    def test_smooth_ceiling_asymptotic_at_high_bg(self) -> None:
        """BG=1000 should asymptote very close to the ceiling."""
        result = apply_renal_correction(1000.0)
        assert result <= BG_HARD_CEILING
        assert result > BG_HARD_CEILING - 1.0  # within 1 mg/dL

    def test_smooth_ceiling_never_exceeds_ceiling(self) -> None:
        """BUG #3 regression: corrected BG must never exceed the ceiling
        for any input >= the ceiling."""
        for bg in (600.0, 650.0, 800.0, 1500.0, 5000.0):
            assert apply_renal_correction(bg) <= BG_HARD_CEILING + 1e-9

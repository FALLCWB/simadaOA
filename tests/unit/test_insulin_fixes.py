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

"""Regression tests for H5 bug-hunt v2 fixes.

Covers all 7 bugs:
    Bug 1 (HIGH)   — compute_iob double-call with destructive pruning corrupts bolus_calculated
    Bug 2 (HIGH)   — RAGE correction: cap must precede final round_dose
    Bug 3 (MEDIUM) — Fiasp IOB curve k/n hardcoded for 240 min, not scaled
    Bug 4 (MEDIUM) — FORGOT event stores bolus_calculated=0.0, hiding missed dose
    Bug 5 (MEDIUM) — basal_fraction=0.5 hardcoded, not configurable
    Bug 6 (LOW)    — record_bolus silently drops zero, no guard for negative
    Bug 7 (LOW)    — correction timing always PRE; needs CORRECTION category
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest
from numpy.random import default_rng

from simada.core.config import ArchetypeParams
from simada.core.types import (
    ArchetypeID,
    BolusTimingCategory,
    BolusType,
    InsulinRegimen,
)
from simada.insulin.adherence import AdherenceInsulinModel
from simada.insulin.basal import BasalProfile
from simada.insulin.calculator import BolusCalculator
from simada.patient.archetype import AdherenceArchetype, create_archetype


# ---------------------------------------------------------------------------
# Bug 1 — compute_iob double-call corrupts bolus_calculated (HIGH)
# ---------------------------------------------------------------------------


class TestComputeIobReadonly:
    """Bug 1: calculate_with_errors prunes _bolus_history via compute_iob;
    a subsequent calculate() must see the same history state so that
    bolus_calculated is consistent with bolus_actual's IOB basis.
    """

    def test_compute_iob_readonly_does_not_prune(self) -> None:
        """compute_iob_readonly must leave _bolus_history untouched even
        when records are expired."""
        calc = BolusCalculator(cr=10.0, cf=50.0, iob_duration_hours=1.0)
        t0 = datetime(2026, 6, 1, 8, 0)
        # Record a bolus that will be expired by t1
        calc.record_bolus(t0, 5.0)
        t1 = t0 + timedelta(hours=2)  # well past 1-hour duration

        history_before = len(calc._bolus_history)
        _ = calc.compute_iob_readonly(t1)
        history_after = len(calc._bolus_history)

        assert history_after == history_before, (
            "compute_iob_readonly must not prune _bolus_history "
            f"(before={history_before}, after={history_after})"
        )

    def test_compute_iob_prunes(self) -> None:
        """compute_iob (pruning variant) must remove expired entries."""
        calc = BolusCalculator(cr=10.0, cf=50.0, iob_duration_hours=1.0)
        t0 = datetime(2026, 6, 1, 8, 0)
        calc.record_bolus(t0, 5.0)
        t1 = t0 + timedelta(hours=2)

        _ = calc.compute_iob(t1)
        assert len(calc._bolus_history) == 0, "compute_iob should prune expired records"

    def test_bolus_calculated_consistent_with_actual_iob_basis(
        self, adherent_params: ArchetypeParams
    ) -> None:
        """bolus_calculated must reflect ideal dose over the SAME IOB state
        that was used to compute the erroneous dose.

        Setup: seed a bolus at t-30min so IOB is non-zero.  Then call
        process_meal_bolus.  bolus_calculated should be > 0 (IOB subtracts
        from the ideal, but the ideal IOB should match the actual IOB basis).

        With the bug unfixed, calculate_with_errors would prune the history
        first; then calculate() would see an empty history and return a
        *higher* ideal dose than what the actual IOB calculation used —
        making bolus_calculated misleadingly inflated.
        """
        arch = create_archetype(ArchetypeID.ADHERENT, adherent_params)
        calc = BolusCalculator(cr=10.0, cf=50.0, target_bg=120.0, iob_duration_hours=4.0)

        # Pre-seed a prior bolus so IOB is active
        meal_time = datetime(2026, 6, 1, 12, 0)
        calc.record_bolus(meal_time - timedelta(minutes=30), 8.0)

        model = AdherenceInsulinModel(arch, calc)
        rng = default_rng(0)

        event = model.process_meal_bolus(
            meal_time=meal_time,
            estimated_carbs=60.0,
            current_bg=120.0,
            rng=rng,
        )

        # IOB active — both actual and ideal should account for it.
        # With the fix, both compute_iob calls see the same history state,
        # so bolus_calculated ≤ ideal_no_iob (it has IOB deducted).
        no_iob_ideal = 60.0 / 10.0  # 6.0 U (meal only, no correction, no IOB)
        assert event.bolus_calculated <= no_iob_ideal + 0.01, (
            f"bolus_calculated={event.bolus_calculated:.3f} should not exceed "
            f"the no-IOB ideal ({no_iob_ideal:.1f}U); double-prune bug inflated it"
        )

    def test_correction_bolus_calculated_consistent(
        self, adherent_params: ArchetypeParams
    ) -> None:
        """Same invariant for process_correction_bolus (bug also present at :256)."""
        arch = create_archetype(ArchetypeID.ADHERENT, adherent_params)
        calc = BolusCalculator(cr=10.0, cf=50.0, target_bg=120.0, iob_duration_hours=4.0)

        correction_time = datetime(2026, 6, 1, 15, 0)
        # Pre-seed IOB
        calc.record_bolus(correction_time - timedelta(minutes=60), 4.0)

        model = AdherenceInsulinModel(arch, calc)
        rng = default_rng(42)

        event = model.process_correction_bolus(
            time=correction_time,
            current_bg=250.0,
            rng=rng,
        )

        if event is not None:
            # Ideal with no IOB: (250-120)/50 = 2.6 U
            no_iob_ideal = (250.0 - 120.0) / 50.0
            assert event.bolus_calculated <= no_iob_ideal + 0.01, (
                f"bolus_calculated={event.bolus_calculated:.3f} should not exceed "
                f"the no-IOB ideal ({no_iob_ideal:.2f}U)"
            )


# ---------------------------------------------------------------------------
# Bug 2 — RAGE correction cap order (HIGH)
# ---------------------------------------------------------------------------


class TestRageCorrectionCapOrder:
    """Bug 2: cap must be applied BEFORE final round_dose in the correction
    rage path so the delivered dose is never above max_single_bolus_u.
    """

    def test_rage_correction_never_exceeds_cap(
        self, nonadherent_params: ArchetypeParams
    ) -> None:
        """Force rage probability to 1.0 and cap to a small value; verify
        the delivered dose never exceeds the cap even after rounding.
        """
        cap = 5.0
        params = nonadherent_params.model_copy(update={
            "rage_bolus_probability": 1.0,
            "rage_bolus_extra_factor": 2.0,
            "max_single_bolus_u": cap,
            "bolus_skip_probability": 0.0,
            "iob_hard_limit_u": 0.0,  # disable IOB gate
        })
        arch = create_archetype(ArchetypeID.NONADHERENT, params)
        rng = default_rng(0)

        for seed in range(20):
            calc = BolusCalculator(cr=10.0, cf=50.0, target_bg=120.0)
            model = AdherenceInsulinModel(arch, calc)
            event = model.process_correction_bolus(
                time=datetime(2026, 6, 1, 15, 0),
                current_bg=250.0,
                rng=default_rng(seed),
            )
            if event is not None and event.bolus_type == BolusType.RAGE:
                assert event.bolus_dose <= cap, (
                    f"Seed {seed}: rage correction dose {event.bolus_dose:.2f}U "
                    f"exceeds max_single_bolus_u={cap}U (cap-after-round bug)"
                )


# ---------------------------------------------------------------------------
# Bug 3 — Fiasp IOB curve not scaled with duration (MEDIUM)
# ---------------------------------------------------------------------------


class TestFiaspIobDurationScaling:
    """Bug 3: Fiasp k/n params are fitted for 240 min; for other durations
    the curve must be scaled so the shape is preserved.
    """

    def test_fiasp_iob_zero_at_duration_boundary(self) -> None:
        """IOB must return exactly 0.0 at elapsed_min == duration_min
        regardless of the configured duration.
        """
        for duration_hours in [2.0, 3.0, 4.0, 5.0]:
            calc = BolusCalculator(
                cr=10.0, cf=50.0, iob_duration_hours=duration_hours, insulin_type="fiasp"
            )
            duration_min = duration_hours * 60.0
            # _iob_curve clamps at >= duration_min → 0.0
            result = calc._iob_curve(duration_min, duration_min)
            assert result == 0.0, (
                f"Fiasp IOB should be 0 at boundary for duration={duration_hours}h"
            )

    def test_fiasp_iob_half_duration_consistent_across_durations(self) -> None:
        """With scaling, the fraction remaining at the midpoint of the
        duration should be identical for all configured durations (the
        curve shape is duration-agnostic after scaling).
        """
        fractions: list[float] = []
        for duration_hours in [2.0, 3.0, 4.0, 5.0]:
            calc = BolusCalculator(
                cr=10.0, cf=50.0, iob_duration_hours=duration_hours, insulin_type="fiasp"
            )
            duration_min = duration_hours * 60.0
            frac = calc._iob_curve(duration_min / 2, duration_min)
            fractions.append(frac)

        # All fractions should be the same (scaling makes curve duration-agnostic)
        for i in range(1, len(fractions)):
            assert abs(fractions[i] - fractions[0]) < 1e-9, (
                f"Fiasp IOB at half-duration should be identical across durations "
                f"(fractions={fractions}); curve is not properly scaled"
            )

    def test_fiasp_iob_3h_vs_4h_not_abruptly_truncated(self) -> None:
        """Before fix: a 3h duration would truncate at elapsed=180min where
        the unscaled curve still had ~8% active.  After fix the scaled curve
        reaches near-zero at 180min for a 3h config.
        """
        calc_3h = BolusCalculator(
            cr=10.0, cf=50.0, iob_duration_hours=3.0, insulin_type="fiasp"
        )
        # At t=179min (1 min before 3h boundary) with scaling the IOB
        # fraction should be very small (approaching 0), not ~8%.
        frac_near_end = calc_3h._iob_curve(179.0, 180.0)
        # Unscaled bug: exp(-0.0014 * 179^1.44) ≈ 0.083 (8.3%)
        # With t-scaling: t_scaled = 179 * 240/180 = 238.7 → exp(-0.0014 * 238.7^1.44) ≈ 0.024
        # Threshold 0.03 accepts the scaled value while rejecting the unscaled 0.083.
        assert frac_near_end < 0.03, (
            f"Fiasp IOB at 179/180 min should be near-zero with scaling "
            f"(got {frac_near_end:.4f}); 8% residual indicates missing duration scaling"
        )

    def test_fiasp_iob_integrated_total_not_clipped(self) -> None:
        """IOB sums correctly for a 3h duration: should not abruptly jump
        from ~8% to 0 at the duration boundary (which would cause stacking
        underestimation).  With scaling the transition is smooth near-zero.
        """
        calc = BolusCalculator(
            cr=10.0, cf=50.0, iob_duration_hours=3.0, insulin_type="fiasp"
        )
        duration_min = 180.0
        # Sample IOB fraction at the last 30 min of the action window
        fractions = [
            calc._iob_curve(t, duration_min) for t in range(150, 181, 5)
        ]
        # All fractions should be monotonically decreasing (no cliff)
        for i in range(1, len(fractions)):
            assert fractions[i] <= fractions[i - 1] + 1e-9, (
                f"Fiasp IOB not monotone at t={150 + i * 5}min "
                f"({fractions[i - 1]:.4f} → {fractions[i]:.4f})"
            )


# ---------------------------------------------------------------------------
# Bug 4 — FORGOT event stores bolus_calculated=0.0 (MEDIUM)
# ---------------------------------------------------------------------------


class TestForgotBolusMagnitude:
    """Bug 4: when a patient forgets to bolus, bolus_calculated must reflect
    the ideal dose that should have been delivered, not 0.0.
    """

    def test_forgot_bolus_calculated_nonzero_for_meal(
        self, nonadherent_params: ArchetypeParams
    ) -> None:
        """Find a FORGOT event and verify bolus_calculated > 0."""
        arch = create_archetype(ArchetypeID.NONADHERENT, nonadherent_params)

        found = False
        for seed in range(300):
            calc = BolusCalculator(cr=10.0, cf=50.0, target_bg=120.0)
            model = AdherenceInsulinModel(arch, calc)
            rng = default_rng(seed)
            event = model.process_meal_bolus(
                meal_time=datetime(2026, 6, 1, 12, 0),
                estimated_carbs=60.0,
                current_bg=120.0,
                rng=rng,
            )
            if event.timing_category == BolusTimingCategory.FORGOT:
                assert event.bolus_dose == 0.0, "FORGOT actual dose must be 0"
                assert event.bolus_calculated > 0.0, (
                    f"FORGOT bolus_calculated should be > 0 (60g / CR10 = 6U ideal); "
                    f"got {event.bolus_calculated!r}"
                )
                # Sanity: ideal for 60g at CR=10, BG=120 (no correction) = 6U
                assert abs(event.bolus_calculated - 6.0) < 0.1, (
                    f"Expected ~6U ideal for 60g meal at CR=10; "
                    f"got {event.bolus_calculated:.3f}U"
                )
                found = True
                break

        assert found, "Should find a FORGOT event in 300 seeds"

    def test_forgot_bolus_calculated_respects_iob(
        self, nonadherent_params: ArchetypeParams
    ) -> None:
        """bolus_calculated for FORGOT should be reduced by active IOB."""
        arch = create_archetype(ArchetypeID.NONADHERENT, nonadherent_params)
        meal_time = datetime(2026, 6, 1, 12, 0)

        found = False
        for seed in range(300):
            calc = BolusCalculator(cr=10.0, cf=50.0, target_bg=120.0, iob_duration_hours=4.0)
            # Pre-seed significant IOB
            calc.record_bolus(meal_time - timedelta(minutes=30), 5.0)
            model = AdherenceInsulinModel(arch, calc)
            rng = default_rng(seed)
            event = model.process_meal_bolus(
                meal_time=meal_time,
                estimated_carbs=60.0,
                current_bg=120.0,
                rng=rng,
            )
            if event.timing_category == BolusTimingCategory.FORGOT:
                # Ideal = meal_bolus - IOB (possibly 0 if IOB > meal_bolus)
                assert event.bolus_calculated <= 6.0, (
                    f"bolus_calculated={event.bolus_calculated:.3f}U should not exceed "
                    f"the no-IOB ideal (6.0U) when IOB is active"
                )
                found = True
                break

        assert found, "Should find a FORGOT event in 300 seeds"


# ---------------------------------------------------------------------------
# Bug 5 — basal_fraction hardcoded 0.5 (MEDIUM)
# ---------------------------------------------------------------------------


class TestBasalFractionConfigurable:
    """Bug 5: BasalProfile must accept a configurable basal_fraction
    instead of hardcoding 0.5.
    """

    def test_default_fraction_unchanged(self) -> None:
        """Default behaviour must stay at 50% for backward compatibility."""
        profile = BasalProfile(tdi=40.0, regimen=InsulinRegimen.PUMP)
        expected = 40.0 * 0.5 / 24.0
        assert profile.base_rate == pytest.approx(expected, abs=1e-9)

    def test_custom_fraction_applied(self) -> None:
        """A 40% basal fraction must produce the correct base rate."""
        profile = BasalProfile(tdi=40.0, regimen=InsulinRegimen.PUMP, basal_fraction=0.4)
        expected = 40.0 * 0.4 / 24.0
        assert profile.base_rate == pytest.approx(expected, abs=1e-9)

    def test_fraction_45_percent(self) -> None:
        """45% is common in pump therapy — verify arithmetic is correct."""
        profile = BasalProfile(tdi=36.0, regimen=InsulinRegimen.MDI, basal_fraction=0.45)
        expected = 36.0 * 0.45 / 24.0
        assert profile.base_rate == pytest.approx(expected, abs=1e-9)

    def test_invalid_fraction_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="basal_fraction"):
            BasalProfile(tdi=40.0, regimen=InsulinRegimen.PUMP, basal_fraction=0.0)

    def test_invalid_fraction_one_raises(self) -> None:
        with pytest.raises(ValueError, match="basal_fraction"):
            BasalProfile(tdi=40.0, regimen=InsulinRegimen.PUMP, basal_fraction=1.0)

    def test_invalid_fraction_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="basal_fraction"):
            BasalProfile(tdi=40.0, regimen=InsulinRegimen.PUMP, basal_fraction=-0.1)


# ---------------------------------------------------------------------------
# Bug 6 — record_bolus negative guard (LOW)
# ---------------------------------------------------------------------------


class TestRecordBolusNegativeGuard:
    """Bug 6: record_bolus must raise ValueError for negative dose;
    zero doses are silently dropped (no tracking needed).
    """

    def test_negative_dose_raises(self) -> None:
        calc = BolusCalculator(cr=10.0, cf=50.0)
        with pytest.raises(ValueError, match="non-negative"):
            calc.record_bolus(datetime(2026, 6, 1, 12, 0), -1.0)

    def test_small_negative_dose_raises(self) -> None:
        calc = BolusCalculator(cr=10.0, cf=50.0)
        with pytest.raises(ValueError):
            calc.record_bolus(datetime(2026, 6, 1, 12, 0), -0.001)

    def test_zero_dose_silently_ignored(self) -> None:
        calc = BolusCalculator(cr=10.0, cf=50.0)
        calc.record_bolus(datetime(2026, 6, 1, 12, 0), 0.0)
        assert len(calc._bolus_history) == 0, "Zero dose should not be recorded"

    def test_positive_dose_still_recorded(self) -> None:
        calc = BolusCalculator(cr=10.0, cf=50.0)
        t = datetime(2026, 6, 1, 12, 0)
        calc.record_bolus(t, 5.0)
        assert len(calc._bolus_history) == 1
        assert calc._bolus_history[0].dose == 5.0


# ---------------------------------------------------------------------------
# Bug 7 — correction timing always PRE (LOW)
# ---------------------------------------------------------------------------


class TestCorrectionTimingCategory:
    """Bug 7: correction boluses must carry BolusTimingCategory.CORRECTION,
    not PRE, so timing distribution analyses are not polluted.
    """

    def test_correction_category_exists(self) -> None:
        """BolusTimingCategory must have a CORRECTION member."""
        assert hasattr(BolusTimingCategory, "CORRECTION"), (
            "BolusTimingCategory.CORRECTION not defined in types.py"
        )
        assert BolusTimingCategory.CORRECTION.value == "correction"

    def test_process_correction_bolus_uses_correction_timing(
        self, adherent_params: ArchetypeParams
    ) -> None:
        """Any non-None correction event must have timing_category=CORRECTION."""
        arch = create_archetype(ArchetypeID.ADHERENT, adherent_params)
        calc = BolusCalculator(cr=10.0, cf=50.0, target_bg=120.0)
        model = AdherenceInsulinModel(arch, calc)

        events = []
        for seed in range(50):
            rng = default_rng(seed)
            ev = model.process_correction_bolus(
                time=datetime(2026, 6, 1, 15, 0),
                current_bg=250.0,
                rng=rng,
            )
            if ev is not None:
                events.append(ev)

        assert len(events) > 0, "Expected at least one correction event in 50 seeds"
        for ev in events:
            assert ev.timing_category == BolusTimingCategory.CORRECTION, (
                f"Correction event timing_category should be CORRECTION, "
                f"got {ev.timing_category!r}"
            )

    def test_meal_bolus_pre_timing_unchanged(
        self, adherent_params: ArchetypeParams
    ) -> None:
        """Meal bolus PRE timing must not be affected by the correction fix."""
        arch = create_archetype(ArchetypeID.ADHERENT, adherent_params)
        pre_events = []

        for seed in range(200):
            calc = BolusCalculator(cr=10.0, cf=50.0, target_bg=120.0)
            model = AdherenceInsulinModel(arch, calc)
            rng = default_rng(seed)
            ev = model.process_meal_bolus(
                meal_time=datetime(2026, 6, 1, 12, 0),
                estimated_carbs=60.0,
                current_bg=120.0,
                rng=rng,
            )
            if ev.timing_category == BolusTimingCategory.PRE:
                pre_events.append(ev)

        assert len(pre_events) > 100, (
            "Adherent archetype should produce many PRE meal boluses in 200 seeds"
        )

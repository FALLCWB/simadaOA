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

"""Tests for the insulin delivery system (calculator, basal, archetype, adherence)."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
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


class TestBolusCalculator:
    """Tests for BolusCalculator."""

    def test_basic_meal_bolus(self) -> None:
        calc = BolusCalculator(cr=10.0, cf=50.0, target_bg=120.0)
        t = datetime(2026, 6, 1, 12, 0)
        result = calc.calculate(estimated_carbs=60.0, current_bg=120.0, current_time=t)
        assert result.meal_bolus == pytest.approx(6.0)
        assert result.correction_bolus == pytest.approx(0.0)
        assert result.total_dose == pytest.approx(6.0)

    def test_correction_bolus(self) -> None:
        calc = BolusCalculator(cr=10.0, cf=50.0, target_bg=120.0)
        t = datetime(2026, 6, 1, 12, 0)
        result = calc.calculate(estimated_carbs=0.0, current_bg=220.0, current_time=t)
        assert result.meal_bolus == pytest.approx(0.0)
        assert result.correction_bolus == pytest.approx(2.0)  # (220-120)/50
        assert result.total_dose == pytest.approx(2.0)

    def test_meal_plus_correction(self) -> None:
        calc = BolusCalculator(cr=10.0, cf=50.0, target_bg=120.0)
        t = datetime(2026, 6, 1, 12, 0)
        result = calc.calculate(estimated_carbs=50.0, current_bg=170.0, current_time=t)
        assert result.meal_bolus == pytest.approx(5.0)
        assert result.correction_bolus == pytest.approx(1.0)  # (170-120)/50
        assert result.total_dose == pytest.approx(6.0)

    def test_iob_subtracted(self) -> None:
        calc = BolusCalculator(cr=10.0, cf=50.0, target_bg=120.0, iob_duration_hours=4.0)
        t0 = datetime(2026, 6, 1, 12, 0)
        calc.record_bolus(t0, 5.0)
        t1 = t0 + timedelta(minutes=30)
        result = calc.calculate(estimated_carbs=50.0, current_bg=120.0, current_time=t1)
        assert result.iob > 0.0
        assert result.total_dose < 5.0  # less than meal bolus alone due to IOB

    def test_iob_decays_to_zero(self) -> None:
        calc = BolusCalculator(cr=10.0, cf=50.0, iob_duration_hours=4.0)
        t0 = datetime(2026, 6, 1, 12, 0)
        calc.record_bolus(t0, 5.0)
        t_after = t0 + timedelta(hours=5)
        iob = calc.compute_iob(t_after)
        assert iob == pytest.approx(0.0, abs=0.01)

    def test_negative_dose_clamped_to_zero(self) -> None:
        calc = BolusCalculator(cr=10.0, cf=50.0, target_bg=120.0, iob_duration_hours=4.0)
        t0 = datetime(2026, 6, 1, 12, 0)
        calc.record_bolus(t0, 20.0)  # large IOB
        t1 = t0 + timedelta(minutes=10)
        result = calc.calculate(estimated_carbs=10.0, current_bg=100.0, current_time=t1)
        assert result.total_dose == 0.0

    def test_calculate_with_errors(self) -> None:
        calc = BolusCalculator(cr=10.0, cf=50.0, target_bg=120.0)
        t = datetime(2026, 6, 1, 12, 0)
        result = calc.calculate_with_errors(
            estimated_carbs=60.0, current_bg=120.0, current_time=t,
            cr_error=0.8, cf_error=1.0, iob_fraction=1.0,
        )
        # With cr_error=0.8, effective_cr = 10*0.8 = 8, dose = 60/8 = 7.5
        assert result.meal_bolus == pytest.approx(7.5)

    def test_cr_error_below_one_causes_overdose(self) -> None:
        """Verify that cr_error < 1.0 results in MORE insulin (overdose)."""
        calc = BolusCalculator(cr=10.0, cf=50.0, target_bg=120.0)
        t = datetime(2026, 6, 1, 12, 0)
        ideal = calc.calculate(estimated_carbs=60.0, current_bg=120.0, current_time=t)
        overdose = calc.calculate_with_errors(
            estimated_carbs=60.0, current_bg=120.0, current_time=t,
            cr_error=0.8, cf_error=1.0, iob_fraction=1.0,
        )
        underdose = calc.calculate_with_errors(
            estimated_carbs=60.0, current_bg=120.0, current_time=t,
            cr_error=1.2, cf_error=1.0, iob_fraction=1.0,
        )
        assert overdose.total_dose > ideal.total_dose, "cr_error < 1 should overdose"
        assert underdose.total_dose < ideal.total_dose, "cr_error > 1 should underdose"

    def test_cf_error_below_one_causes_more_correction(self) -> None:
        """Verify that cf_error < 1.0 results in MORE correction insulin."""
        calc = BolusCalculator(cr=10.0, cf=50.0, target_bg=120.0)
        t = datetime(2026, 6, 1, 12, 0)
        ideal = calc.calculate(estimated_carbs=0.0, current_bg=220.0, current_time=t)
        more = calc.calculate_with_errors(
            estimated_carbs=0.0, current_bg=220.0, current_time=t,
            cr_error=1.0, cf_error=0.75, iob_fraction=1.0,
        )
        assert more.total_dose > ideal.total_dose, "cf_error < 1 should give more correction"


class TestBasalProfile:
    """Tests for BasalProfile."""

    def test_base_rate_calculation(self) -> None:
        profile = BasalProfile(tdi=40.0, regimen=InsulinRegimen.PUMP)
        expected = 40.0 * 0.5 / 24.0  # ~0.833 U/hr
        assert profile.base_rate == pytest.approx(expected, abs=0.01)

    def test_dawn_phenomenon_increases_rate(self) -> None:
        profile = BasalProfile(tdi=40.0, regimen=InsulinRegimen.PUMP, dawn_phenomenon=True)
        dawn = profile.rate_at(datetime(2026, 6, 1, 7, 0))
        midnight = profile.rate_at(datetime(2026, 6, 1, 1, 0))
        assert dawn.rate_u_per_hr > midnight.rate_u_per_hr
        assert dawn.is_dawn_phenomenon is True
        assert midnight.is_dawn_phenomenon is False

    def test_mdi_flat_profile(self) -> None:
        profile = BasalProfile(tdi=40.0, regimen=InsulinRegimen.MDI)
        r1 = profile.rate_at(datetime(2026, 6, 1, 3, 0))
        r2 = profile.rate_at(datetime(2026, 6, 1, 15, 0))
        assert r1.rate_u_per_hr == r2.rate_u_per_hr

    def test_hourly_profile_has_24_entries(self) -> None:
        profile = BasalProfile(tdi=40.0, regimen=InsulinRegimen.PUMP)
        hourly = profile.hourly_profile()
        assert len(hourly) == 24


class TestAdherenceArchetype:
    """Tests for AdherenceArchetype behavior."""

    def test_adherent_rarely_skips(self, adherent_params: ArchetypeParams) -> None:
        arch = create_archetype(ArchetypeID.ADHERENT, adherent_params)
        rng = default_rng(42)
        skips = sum(1 for _ in range(1000) if arch.should_skip_bolus(rng))
        assert skips < 50  # expect ~20

    def test_nonadherent_skips_often(self, nonadherent_params: ArchetypeParams) -> None:
        arch = create_archetype(ArchetypeID.NONADHERENT, nonadherent_params)
        rng = default_rng(42)
        skips = sum(1 for _ in range(1000) if arch.should_skip_bolus(rng))
        assert skips > 50  # expect ~120

    def test_timing_distribution(self, adherent_params: ArchetypeParams) -> None:
        arch = create_archetype(ArchetypeID.ADHERENT, adherent_params)
        rng = default_rng(42)
        n = 2000
        counts = {t: 0 for t in BolusTimingCategory}
        for _ in range(n):
            timing = arch.resolve_bolus_timing(rng)
            counts[timing] += 1
        # Adherent: 96% pre, 3% late, 1% forgot
        assert counts[BolusTimingCategory.PRE] / n > 0.90
        assert counts[BolusTimingCategory.FORGOT] / n < 0.05

    def test_pre_bolus_is_negative_delay(self, adherent_params: ArchetypeParams) -> None:
        arch = create_archetype(ArchetypeID.ADHERENT, adherent_params)
        rng = default_rng(42)
        delays = []
        for _ in range(100):
            d = arch.bolus_timing_delay(BolusTimingCategory.PRE, rng)
            delays.append(d.total_seconds() / 60)
        avg = np.mean(delays)
        assert avg < 0, f"Pre-bolus should have negative delay, got {avg:.1f} min"

    def test_late_bolus_positive_delay(self, adherent_params: ArchetypeParams) -> None:
        arch = create_archetype(ArchetypeID.ADHERENT, adherent_params)
        rng = default_rng(42)
        for _ in range(100):
            d = arch.bolus_timing_delay(BolusTimingCategory.LATE_HALF, rng)
            assert d.total_seconds() > 0

    def test_dose_rounding(self, nonadherent_params: ArchetypeParams) -> None:
        arch = create_archetype(ArchetypeID.NONADHERENT, nonadherent_params)
        # Nonadherent rounds to 1U (pump rounds for them)
        assert arch.round_dose(3.7) == 4.0
        assert arch.round_dose(7.2) == 7.0
        assert arch.round_dose(12.8) == 13.0

    def test_adherent_fine_rounding(self, adherent_params: ArchetypeParams) -> None:
        arch = create_archetype(ArchetypeID.ADHERENT, adherent_params)
        # Adherent rounds to 0.5U
        assert arch.round_dose(3.7) == 3.5
        assert arch.round_dose(3.8) == 4.0


class TestAdherenceInsulinModel:
    """Tests for the full adherence insulin delivery pipeline."""

    def _make_model(self, params: ArchetypeParams) -> AdherenceInsulinModel:
        arch = create_archetype(ArchetypeID.ADHERENT, params)
        calc = BolusCalculator(cr=10.0, cf=50.0, target_bg=120.0)
        return AdherenceInsulinModel(arch, calc)

    def test_meal_bolus_produces_event(
        self, adherent_params: ArchetypeParams
    ) -> None:
        model = self._make_model(adherent_params)
        rng = default_rng(42)
        event = model.process_meal_bolus(
            meal_time=datetime(2026, 6, 1, 12, 0),
            estimated_carbs=60.0,
            current_bg=120.0,
            rng=rng,
        )
        assert event.bolus_dose > 0
        assert event.was_skipped is False

    def test_skipped_bolus_has_zero_dose(
        self, nonadherent_params: ArchetypeParams
    ) -> None:
        arch = create_archetype(ArchetypeID.NONADHERENT, nonadherent_params)

        found_skip = False
        for seed in range(100):
            # Fresh calculator per iteration: an earlier seed may have
            # produced a LATE bolus recorded after 12:00, and reusing the
            # same calculator would make compute_iob see a "future" bolus
            # on the next iteration (the H3 #3 fix now raises instead of
            # silently skipping).
            calc = BolusCalculator(cr=10.0, cf=50.0)
            model = AdherenceInsulinModel(arch, calc)
            rng = default_rng(seed)
            event = model.process_meal_bolus(
                meal_time=datetime(2026, 6, 1, 12, 0),
                estimated_carbs=60.0,
                current_bg=120.0,
                rng=rng,
            )
            if event.was_skipped:
                assert event.bolus_dose == 0.0
                assert event.timing_category == BolusTimingCategory.SKIPPED
                found_skip = True
                break
        assert found_skip, "Nonadherent should skip at least once in 100 tries"

    def test_forgot_bolus_has_zero_dose(
        self, nonadherent_params: ArchetypeParams
    ) -> None:
        arch = create_archetype(ArchetypeID.NONADHERENT, nonadherent_params)

        found_forgot = False
        for seed in range(200):
            # Fresh calculator per iteration to avoid IOB ordering issues
            # (see test_skipped_bolus_has_zero_dose for rationale).
            calc = BolusCalculator(cr=10.0, cf=50.0)
            model = AdherenceInsulinModel(arch, calc)
            rng = default_rng(seed)
            event = model.process_meal_bolus(
                meal_time=datetime(2026, 6, 1, 12, 0),
                estimated_carbs=60.0,
                current_bg=120.0,
                rng=rng,
            )
            if event.timing_category == BolusTimingCategory.FORGOT:
                assert event.bolus_dose == 0.0
                found_forgot = True
                break
        assert found_forgot, "Nonadherent should forget at least once in 200 tries"

    def test_late_half_gives_reduced_dose(
        self, adherent_params: ArchetypeParams
    ) -> None:
        pre_doses = []
        late_doses = []

        for seed in range(500):
            # Fresh model each iteration to avoid IOB accumulation
            model = self._make_model(adherent_params)
            rng = default_rng(seed)
            event = model.process_meal_bolus(
                meal_time=datetime(2026, 6, 1, 12, 0),
                estimated_carbs=60.0,
                current_bg=120.0,
                rng=rng,
            )
            if event.timing_category == BolusTimingCategory.PRE and event.bolus_dose > 0:
                pre_doses.append(event.bolus_dose)
            elif event.timing_category == BolusTimingCategory.LATE_HALF and event.bolus_dose > 0:
                late_doses.append(event.bolus_dose)

        assert len(pre_doses) > 0, "Should have found pre-bolus events in 500 seeds"
        assert len(late_doses) > 0, "Should have found late-half events in 500 seeds"
        assert np.mean(late_doses) < np.mean(pre_doses), (
            "Late-half doses should be smaller than pre-bolus doses"
        )

    def test_correction_bolus(self, adherent_params: ArchetypeParams) -> None:
        model = self._make_model(adherent_params)
        rng = default_rng(42)
        event = model.process_correction_bolus(
            time=datetime(2026, 6, 1, 15, 0),
            current_bg=250.0,
            rng=rng,
        )
        assert event is not None
        assert event.bolus_dose > 0
        assert event.bolus_type in (BolusType.CORRECTION, BolusType.RAGE, BolusType.PHANTOM)

    def test_no_correction_below_threshold(
        self, adherent_params: ArchetypeParams
    ) -> None:
        model = self._make_model(adherent_params)
        rng = default_rng(42)
        event = model.process_correction_bolus(
            time=datetime(2026, 6, 1, 15, 0),
            current_bg=130.0,  # below threshold (adherent default = 170)
            rng=rng,
        )
        assert event is None


class TestRageDoubleMutualExclusion:
    """BUG #4 regression: in the meal bolus flow, the rage and double-dose
    multipliers cannot stack on the same bolus event. The fix makes them
    mutually exclusive (DOUBLE wins when both roll)."""

    def test_meal_bolus_rage_and_double_never_combine(
        self, nonadherent_params: ArchetypeParams
    ) -> None:
        """Force both probabilities to 1.0 and verify the delivered dose
        equals the DOUBLE-only path (2x base dose, capped by
        max_single_bolus_u), NEVER the combined 2.8x value.
        """
        params = nonadherent_params.model_copy(update={
            "bolus_skip_probability": 0.0,
            "bolus_timing_pre_pct": 1.0,
            "bolus_timing_late_half_pct": 0.0,
            "bolus_timing_forgot_pct": 0.0,
            "double_dose_probability": 1.0,
            "rage_bolus_probability": 1.0,
            "rage_bolus_extra_factor": 1.4,
            "cr_error_factor_mean": 1.0,
            "cr_error_factor_std": 0.0,
            "cf_error_factor_mean": 1.0,
            "cf_error_factor_std": 0.0,
            "max_single_bolus_u": 100.0,  # disable hardware cap for clarity
        })
        arch = create_archetype(ArchetypeID.NONADHERENT, params)

        # 60 g carbs at CR=10 -> 6 U base; DOUBLE delivers 12 U, never
        # 16.8 U (which would be 6 * 1.4 * 2).
        for seed in range(10):
            calc = BolusCalculator(cr=10.0, cf=50.0, target_bg=120.0)
            model = AdherenceInsulinModel(arch, calc)
            rng = default_rng(seed)
            event = model.process_meal_bolus(
                meal_time=datetime(2026, 6, 1, 12, 0),
                estimated_carbs=60.0,
                current_bg=120.0,
                rng=rng,
            )
            assert event.bolus_type == BolusType.DOUBLE, (
                f"Seed {seed}: bolus_type should be DOUBLE (prefer over "
                f"RAGE when both roll), got {event.bolus_type}"
            )
            assert event.bolus_dose <= 12.5, (
                f"Seed {seed}: dose {event.bolus_dose:.2f}U exceeds "
                f"DOUBLE-only ceiling (combined 2.8x would give ~16.8U)"
            )

    def test_meal_bolus_rage_applies_when_double_does_not(
        self, nonadherent_params: ArchetypeParams
    ) -> None:
        """If double_dose probability is 0 but rage is 1, the meal flow
        should still apply the rage factor (rage is not silently dropped
        when double does not fire)."""
        params = nonadherent_params.model_copy(update={
            "bolus_skip_probability": 0.0,
            "bolus_timing_pre_pct": 1.0,
            "bolus_timing_late_half_pct": 0.0,
            "bolus_timing_forgot_pct": 0.0,
            "double_dose_probability": 0.0,
            "rage_bolus_probability": 1.0,
            "rage_bolus_extra_factor": 1.5,
            "cr_error_factor_mean": 1.0,
            "cr_error_factor_std": 0.0,
            "cf_error_factor_mean": 1.0,
            "cf_error_factor_std": 0.0,
            "max_single_bolus_u": 100.0,
        })
        arch = create_archetype(ArchetypeID.NONADHERENT, params)
        calc = BolusCalculator(cr=10.0, cf=50.0, target_bg=120.0)
        model = AdherenceInsulinModel(arch, calc)
        rng = default_rng(0)
        event = model.process_meal_bolus(
            meal_time=datetime(2026, 6, 1, 12, 0),
            estimated_carbs=60.0,
            current_bg=120.0,
            rng=rng,
        )
        assert event.bolus_type == BolusType.RAGE
        # Base 6 U * 1.5 = 9 U, nonadherent rounds to 1.0 U granularity.
        assert 8.5 <= event.bolus_dose <= 9.5


class TestIOBChronologyAssertion:
    """Regression tests for H3 bug #3: compute_iob now refuses to
    silently skip out-of-order bolus records and raises ValueError with
    a full history dump for debugging.
    """

    def test_future_bolus_raises_value_error(self) -> None:
        calc = BolusCalculator(cr=10.0, cf=50.0, iob_duration_hours=4.0)
        # Record a bolus at 12:00
        calc.record_bolus(datetime(2026, 6, 1, 12, 0), 5.0)
        # Computing IOB at 11:00 (earlier than the bolus) must fail loudly
        with pytest.raises(ValueError, match="Non-chronological bolus history"):
            calc.compute_iob(datetime(2026, 6, 1, 11, 0))

    def test_value_error_contains_history_dump(self) -> None:
        calc = BolusCalculator(cr=10.0, cf=50.0, iob_duration_hours=4.0)
        calc.record_bolus(datetime(2026, 6, 1, 12, 0), 5.0)
        calc.record_bolus(datetime(2026, 6, 1, 13, 0), 3.0)
        try:
            calc.compute_iob(datetime(2026, 6, 1, 11, 0))
        except ValueError as exc:
            msg = str(exc)
            assert "5.000U" in msg, "History dump missing first bolus dose"
            assert "3.000U" in msg, "History dump missing second bolus dose"
            assert "current_time=" in msg, "Error missing current_time context"
            assert "elapsed_min=" in msg, "Error missing elapsed_min context"
        else:
            pytest.fail("Expected ValueError was not raised")

    def test_chronological_history_still_works(self) -> None:
        """Sanity: properly ordered records still compute IOB correctly."""
        calc = BolusCalculator(cr=10.0, cf=50.0, iob_duration_hours=4.0)
        calc.record_bolus(datetime(2026, 6, 1, 11, 0), 5.0)
        calc.record_bolus(datetime(2026, 6, 1, 11, 30), 3.0)
        iob = calc.compute_iob(datetime(2026, 6, 1, 12, 0))
        assert iob > 0.0


class TestRNGStreamIsolation:
    """Regression tests for H3 bug #5: meal-occurrence and bolus-skip
    decisions must use independent sub-streams from food-sampling and
    dose-error draws, so changes to one don't perturb the other for
    the same master seed.
    """

    def test_bolus_skip_independent_of_dose_errors(
        self, nonadherent_params: ArchetypeParams
    ) -> None:
        """Sub-stream isolation: cr/cf errors should show real
        variability across seeds (they are continuous draws on a
        dedicated sub-stream). Use a nonadherent archetype because
        its CR/CF noise is wide enough to make the variability easy
        to detect with a modest seed count.
        """
        arch = create_archetype(ArchetypeID.NONADHERENT, nonadherent_params)
        cr_errors: list[float] = []
        cf_errors: list[float] = []

        for seed in range(80):
            calc = BolusCalculator(cr=10.0, cf=50.0)
            model = AdherenceInsulinModel(arch, calc)
            rng = default_rng(seed)
            event = model.process_meal_bolus(
                meal_time=datetime(2026, 6, 1, 12, 0),
                estimated_carbs=60.0,
                current_bg=120.0,
                rng=rng,
            )
            if not event.was_skipped:
                cr_errors.append(event.cr_error_factor)
                cf_errors.append(event.cf_error_factor)

        assert len(cr_errors) >= 30, "Need enough non-skipped events to check"
        # Continuous draws should produce many distinct values
        assert len(set(cr_errors)) > 20, (
            f"CR errors lack variability: {len(set(cr_errors))} distinct"
        )
        assert len(set(cf_errors)) > 20, (
            f"CF errors lack variability: {len(set(cf_errors))} distinct"
        )

    def test_meal_bolus_reproducible_with_substreams(
        self, adherent_params: ArchetypeParams
    ) -> None:
        """Sub-stream split must remain deterministic: same seed in,
        same event out.
        """
        arch = create_archetype(ArchetypeID.ADHERENT, adherent_params)

        calc1 = BolusCalculator(cr=10.0, cf=50.0)
        model1 = AdherenceInsulinModel(arch, calc1)
        ev1 = model1.process_meal_bolus(
            meal_time=datetime(2026, 6, 1, 12, 0),
            estimated_carbs=60.0,
            current_bg=120.0,
            rng=default_rng(2026),
        )

        calc2 = BolusCalculator(cr=10.0, cf=50.0)
        model2 = AdherenceInsulinModel(arch, calc2)
        ev2 = model2.process_meal_bolus(
            meal_time=datetime(2026, 6, 1, 12, 0),
            estimated_carbs=60.0,
            current_bg=120.0,
            rng=default_rng(2026),
        )

        assert ev1.bolus_dose == ev2.bolus_dose
        assert ev1.timing_category == ev2.timing_category
        assert ev1.cr_error_factor == ev2.cr_error_factor

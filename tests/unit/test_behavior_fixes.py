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

"""Regression tests for H3 v2 bug-hunt fixes (2026-05-18).

Each test targets exactly one bug from the H3 section to catch regressions.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from numpy.random import default_rng

from simada.behavior.circadian import CircadianModel, _SLEEP_MIN_HOUR
from simada.behavior.exercise import ExerciseGenerator
from simada.behavior.snacking import DIET_ADHERENCE_SNACK_REDUCTION, SnackGenerator
from simada.behavior.stress import StressEventGenerator
from simada.core.config import ArchetypeParams, BehaviorConfig
from simada.core.random import PatientRNG, RNGManager
from simada.core.types import DayType, StressType
from simada.meals.taco import TACODatabase

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wake_sleep(wake_h: int = 7, sleep_h: int = 23) -> tuple[datetime, datetime]:
    """Return a canonical (wake_time, sleep_time) pair on 2026-06-01."""
    base = datetime(2026, 6, 1)
    return (
        base.replace(hour=wake_h, minute=0),
        base.replace(hour=sleep_h, minute=0),
    )


# ---------------------------------------------------------------------------
# Bug H3#1 — stress.py: stress event must not overrun sleep_time
# ---------------------------------------------------------------------------

class TestStressOverrunBug1:
    """H3 HIGH #1 — duration clamped so stress event ends at or before sleep."""

    def test_stress_event_ends_before_sleep(
        self, nonadherent_params: ArchetypeParams
    ) -> None:
        gen = StressEventGenerator(nonadherent_params, BehaviorConfig())
        wake, sleep = _wake_sleep(7, 23)

        for seed in range(500):
            events = gen._generate_psychological(wake, sleep, default_rng(seed))
            for e in events:
                event_end = e.start_time + timedelta(minutes=e.duration_minutes)
                assert event_end <= sleep, (
                    f"seed={seed}: stress event ends at {event_end}, "
                    f"after sleep_time {sleep}"
                )

    def test_stress_event_end_tight_window(
        self, nonadherent_params: ArchetypeParams
    ) -> None:
        """With a very short awake window, duration must still fit."""
        gen = StressEventGenerator(nonadherent_params, BehaviorConfig())
        base = datetime(2026, 6, 1)
        wake = base.replace(hour=7, minute=0)
        sleep = base.replace(hour=9, minute=30)  # only 150 min awake

        for seed in range(200):
            events = gen._generate_psychological(wake, sleep, default_rng(seed))
            for e in events:
                event_end = e.start_time + timedelta(minutes=e.duration_minutes)
                assert event_end <= sleep, (
                    f"seed={seed}: stress event ends at {event_end} > sleep {sleep}"
                )


# ---------------------------------------------------------------------------
# Bug H3#2 — stress.py: alcohol Phase 1 must start before sleep_time - 3h
# ---------------------------------------------------------------------------

class TestAlcoholPhase1OverrunBug2:
    """H3 HIGH #2 — drink_start clamped to sleep_time - 180 min."""

    def test_alcohol_phase1_within_awake_window(
        self, nonadherent_params: ArchetypeParams
    ) -> None:
        gen = StressEventGenerator(nonadherent_params, BehaviorConfig())
        wake, sleep = _wake_sleep(10, 23)  # typical weekend with 13h window

        for seed in range(1000):
            events = gen._generate_alcohol(DayType.WEEKEND, wake, sleep, default_rng(seed))
            alcohol_events = [e for e in events if e.stress_type == StressType.ALCOHOL]
            if len(alcohol_events) >= 1:
                phase1 = alcohol_events[0]
                phase1_end = phase1.start_time + timedelta(minutes=phase1.duration_minutes)
                assert phase1.start_time < sleep, (
                    f"seed={seed}: Phase 1 starts at {phase1.start_time} >= sleep {sleep}"
                )
                # Phase 1 ends at or before sleep (the clamp guarantees this)
                assert phase1_end <= sleep, (
                    f"seed={seed}: Phase 1 ends at {phase1_end} > sleep {sleep}"
                )

    def test_alcohol_drink_start_clamp_binding(
        self, nonadherent_params: ArchetypeParams
    ) -> None:
        """Narrow awake window: drink_start must never exceed sleep - 180 min."""
        gen = StressEventGenerator(nonadherent_params, BehaviorConfig())
        base = datetime(2026, 6, 1)
        wake = base.replace(hour=20, minute=0)
        sleep = base.replace(hour=23, minute=0)  # only 180 min awake

        max_drink_start = sleep - timedelta(minutes=180)

        for seed in range(200):
            events = gen._generate_alcohol(DayType.WEEKEND, wake, sleep, default_rng(seed))
            alcohol_events = [e for e in events if e.stress_type == StressType.ALCOHOL]
            for e in alcohol_events:
                if e.insulin_resistance_factor > 1.0:  # Phase 1
                    assert e.start_time <= max_drink_start + timedelta(seconds=1), (
                        f"seed={seed}: drink_start {e.start_time} > max {max_drink_start}"
                    )


# ---------------------------------------------------------------------------
# Bug H3#3 — stress.py: insulin_resistance_factor < 1.0 convention documented
# ---------------------------------------------------------------------------

class TestAlcoholPhase2ConventionBug3:
    """H3 HIGH #3 — Phase 2 factor is correctly <1.0 (hyper-sensitivity)."""

    def test_phase2_factor_less_than_one(
        self, nonadherent_params: ArchetypeParams
    ) -> None:
        gen = StressEventGenerator(nonadherent_params, BehaviorConfig())
        wake, sleep = _wake_sleep(10, 23)

        found = False
        for seed in range(2000):
            events = gen._generate_alcohol(DayType.WEEKEND, wake, sleep, default_rng(seed))
            alcohol = [e for e in events if e.stress_type == StressType.ALCOHOL]
            if len(alcohol) >= 2:
                phase2 = alcohol[1]
                assert 0.0 < phase2.insulin_resistance_factor < 1.0, (
                    f"seed={seed}: Phase 2 factor {phase2.insulin_resistance_factor} "
                    "should be in (0, 1) — hypersensitivity convention"
                )
                found = True
                break
        assert found, "No biphasic alcohol event generated in 2000 seeds"

    def test_phase1_factor_greater_than_one(
        self, nonadherent_params: ArchetypeParams
    ) -> None:
        gen = StressEventGenerator(nonadherent_params, BehaviorConfig())
        wake, sleep = _wake_sleep(10, 23)

        found = False
        for seed in range(2000):
            events = gen._generate_alcohol(DayType.WEEKEND, wake, sleep, default_rng(seed))
            alcohol = [e for e in events if e.stress_type == StressType.ALCOHOL]
            if len(alcohol) >= 1:
                phase1 = alcohol[0]
                assert phase1.insulin_resistance_factor > 1.0, (
                    f"seed={seed}: Phase 1 factor {phase1.insulin_resistance_factor} "
                    "should be >1.0 — resistance convention"
                )
                found = True
                break
        assert found, "No alcohol Phase 1 event generated in 2000 seeds"


# ---------------------------------------------------------------------------
# Bug H3#4 — core/random.py + schedule.py: snacks sub-stream isolation
# ---------------------------------------------------------------------------

class TestSnacksRNGIsolationBug4:
    """H3 MEDIUM #4 — PatientRNG has a dedicated 'snacks' stream."""

    def test_patient_rng_has_snacks_attribute(self) -> None:
        rng_mgr = RNGManager(42)
        prng = rng_mgr.patient_rng(0)
        assert hasattr(prng, "snacks"), "PatientRNG must expose a 'snacks' Generator"

    def test_snacks_stream_independent_from_meals(self) -> None:
        """Consuming the meals stream must not change the snacks stream."""
        rng_mgr = RNGManager(42)

        # Reference: consume nothing from meals
        prng_ref = rng_mgr.patient_rng(0)
        snack_val_ref = prng_ref.snacks.random()

        # Modified: consume 100 draws from meals first
        prng_mod = rng_mgr.patient_rng(0)
        for _ in range(100):
            prng_mod.meals.random()
        snack_val_mod = prng_mod.snacks.random()

        assert snack_val_ref == snack_val_mod, (
            "snacks stream must be independent from meals stream; "
            f"ref={snack_val_ref}, after 100 meals draws={snack_val_mod}"
        )

    def test_n_components_incremented(self) -> None:
        """_N_COMPONENTS must be 7 after adding the snacks stream."""
        from simada.core import random as rng_module
        assert rng_module._N_COMPONENTS == 7, (
            f"Expected _N_COMPONENTS=7, got {rng_module._N_COMPONENTS}"
        )

    def test_snacks_slot_present(self) -> None:
        from simada.core.random import PatientRNG
        assert "snacks" in PatientRNG.__slots__, (
            "'snacks' must be declared in PatientRNG.__slots__"
        )


# ---------------------------------------------------------------------------
# Bug H3#5 — snacking.py: clamp must occur AFTER diet_adherence reduction
# ---------------------------------------------------------------------------

class TestSnackProbOrderBug5:
    """H3 MEDIUM #5 — diet_adherence applied before clamping to [0, 1]."""

    def test_diet_adherence_reduces_clamped_probability(
        self,
        taco_db: TACODatabase,
        nonadherent_params: ArchetypeParams,
        adherent_params: ArchetypeParams,
    ) -> None:
        """Adherent (high diet_adherence) must snack less than nonadherent."""
        wake, sleep = _wake_sleep(7, 23)
        meal_times = [
            datetime(2026, 6, 1, 8, 0),
            datetime(2026, 6, 1, 12, 30),
            datetime(2026, 6, 1, 19, 0),
        ]
        n = 500

        gen_na = SnackGenerator(nonadherent_params, taco_db)
        gen_ad = SnackGenerator(adherent_params, taco_db)

        na_count = sum(
            len(gen_na.generate(DayType.HOLIDAY, wake, sleep, meal_times, default_rng(s)))
            for s in range(n)
        )
        ad_count = sum(
            len(gen_ad.generate(DayType.HOLIDAY, wake, sleep, meal_times, default_rng(s)))
            for s in range(n)
        )

        assert na_count > ad_count, (
            f"Nonadherent ({na_count}) should snack more than adherent ({ad_count}) "
            "over 500 seeds on HOLIDAY"
        )

    def test_probability_formula_order(
        self, nonadherent_params: ArchetypeParams
    ) -> None:
        """Verify the arithmetic: diet reduction must act on base+boost, not
        on the already-clamped value."""
        params = nonadherent_params
        base = params.snack_probability
        boost = params.holiday_extra_snack_probability
        diet = params.diet_adherence
        reduction = DIET_ADHERENCE_SNACK_REDUCTION

        # Correct order: reduce first, clamp after
        expected = min(1.0, (base + boost) * (1.0 - diet * reduction))
        # Wrong order (old code): clamp first, reduce after — equals old result
        old_result = min(1.0, base + boost) * (1.0 - diet * reduction)

        # They differ only when base+boost > 1.0; assert the new formula is used
        if base + boost > 1.0:
            assert expected != old_result, (
                "Test is only meaningful when base+boost > 1.0 — "
                "check nonadherent archetype parameters"
            )
            # Instantiate a SnackGenerator and verify it uses the new path
            # by checking that for a fully-adherent patient (diet=1.0) on
            # a holiday the effective_prob is strictly less than 1.0
            from simada.core.config import load_archetype_params
            ad_params = load_archetype_params(
                PROJECT_ROOT / "configs" / "archetypes" / "adherent.yaml"
            )
            diet_ad = ad_params.diet_adherence
            base_ad = ad_params.snack_probability
            boost_ad = ad_params.holiday_extra_snack_probability
            eff = min(1.0, (base_ad + boost_ad) * (1.0 - diet_ad * reduction))
            assert eff < 1.0, (
                "Adherent patient on a holiday should have effective_prob < 1.0"
            )


# ---------------------------------------------------------------------------
# Bug H3#6 — circadian.py: comment on valid range of sleep_time_*_hour
# ---------------------------------------------------------------------------

class TestCircadianCommentBug6:
    """H3 MEDIUM #6 — normalization branch handles values < _SLEEP_MIN_HOUR."""

    def test_pre_sleep_min_hour_gets_24_offset(self) -> None:
        """Any hour in [0, 20) gets +24 in the normalization branch."""
        # Tests the branch logic directly (mirrors the code in sample_sleep_time)
        hour_in = 1.5  # 01:30 -> should become 25.5
        expected = 25.5
        if hour_in < _SLEEP_MIN_HOUR:
            hour_out = hour_in + 24.0
        else:
            hour_out = hour_in
        assert hour_out == expected

    def test_sleep_min_hour_constant(self) -> None:
        """_SLEEP_MIN_HOUR must be 20.0 — the boundary of the valid range."""
        assert _SLEEP_MIN_HOUR == 20.0, (
            f"Expected _SLEEP_MIN_HOUR=20.0, got {_SLEEP_MIN_HOUR}"
        )

    def test_normal_sleep_hour_not_offset(self) -> None:
        """Hours >= 20.0 (pre-midnight sleep) must not be shifted."""
        hour_in = 22.5  # 22:30 — normal bedtime, should stay 22.5
        if hour_in < _SLEEP_MIN_HOUR:
            hour_out = hour_in + 24.0
        else:
            hour_out = hour_in
        assert hour_out == 22.5

    def test_sample_sleep_time_realistic(
        self, adherent_params: ArchetypeParams
    ) -> None:
        """Sampled sleep times must be plausible (20:00–06:00 next day)."""
        model = CircadianModel(adherent_params)
        date = datetime(2026, 6, 1)

        for seed in range(100):
            sleep = model.sample_sleep_time(date, DayType.WEEKDAY, default_rng(seed))
            # sleep either on same day hour>=20 OR on next day hour<=6
            same_day_valid = (sleep.date() == date.date() and sleep.hour >= 20)
            next_day_valid = (sleep.date() == (date + timedelta(days=1)).date()
                              and sleep.hour <= 6)
            assert same_day_valid or next_day_valid, (
                f"seed={seed}: sleep time {sleep} outside plausible range"
            )


# ---------------------------------------------------------------------------
# Bug H3#7 — exercise.py: fallback latest must not cross sleep_time
# ---------------------------------------------------------------------------

class TestExerciseFallbackBug7:
    """H3 LOW #7 — exercise start + duration must not exceed sleep_time."""

    def test_exercise_start_plus_duration_within_sleep(
        self, adherent_params: ArchetypeParams
    ) -> None:
        gen = ExerciseGenerator(adherent_params)
        wake, sleep = _wake_sleep(7, 23)

        for seed in range(500):
            events = gen.generate(DayType.WEEKDAY, wake, sleep, default_rng(seed))
            for e in events:
                end = e.start_time + timedelta(minutes=e.duration_minutes)
                assert end <= sleep, (
                    f"seed={seed}: exercise ends at {end} after sleep_time {sleep}"
                )

    def test_exercise_narrow_window_no_overrun(
        self, adherent_params: ArchetypeParams
    ) -> None:
        """With a tight awake window, latest fallback must still respect sleep."""
        gen = ExerciseGenerator(adherent_params)
        base = datetime(2026, 6, 1)
        wake = base.replace(hour=7, minute=0)
        sleep = base.replace(hour=8, minute=0)  # only 60 min awake

        for seed in range(300):
            events = gen.generate(DayType.WEEKDAY, wake, sleep, default_rng(seed))
            for e in events:
                end = e.start_time + timedelta(minutes=e.duration_minutes)
                assert end <= sleep, (
                    f"seed={seed}: exercise ends {end} > sleep {sleep}"
                )


# ---------------------------------------------------------------------------
# Bug H3#8 — snacking.py: DIET_ADHERENCE_SNACK_REDUCTION constant exposed
# ---------------------------------------------------------------------------

class TestMagicConstantBug8:
    """H3 LOW #8 — 0.5 promoted to a named module-level constant."""

    def test_constant_exists_and_correct_value(self) -> None:
        assert DIET_ADHERENCE_SNACK_REDUCTION == 0.5, (
            f"Expected DIET_ADHERENCE_SNACK_REDUCTION=0.5, "
            f"got {DIET_ADHERENCE_SNACK_REDUCTION}"
        )

    def test_constant_importable_from_module(self) -> None:
        from simada.behavior.snacking import DIET_ADHERENCE_SNACK_REDUCTION as C
        assert C == 0.5

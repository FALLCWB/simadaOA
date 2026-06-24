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

"""Regression tests for H8 v2 bug fixes — scenario + core layer.

Covers bugs 1-9 from the H8 section plus the three cross-zone APIs:
  - general_rng cached_property (Bug 1)
  - sensor_seed hierarchical derivation (F1 cross-zone API)
  - snacks stream present on PatientRNG (F3 cross-zone API)
  - BolusTimingCategory.CORRECTION entry (F5 cross-zone API)
  - meal snap < start_time dropped (Bug 3)
  - get_action normalizes t (Bug 4)
  - duration_hours validates last_sleep > first_wake (Bug 5)
  - ScenarioConfig.cohort type is CohortConfig, not Optional (Bug 6)
  - CohortConfig.archetype_distribution uses Field(default_factory) (Bug 7)
  - get_context no early break on overlapping events (Bug 8)
  - DailySchedule mutable by design, documented (Bug 9)
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from simada.core.config import CohortConfig, ScenarioConfig
from simada.core.random import RNGManager, PatientRNG
from simada.core.types import (
    BolusTimingCategory,
    DailySchedule,
    DayType,
    ExerciseEvent,
    ExerciseIntensity,
    MealEvent,
    MealType,
    StressEvent,
    StressType,
)
from simada.scenario.custom_scenario import SimadaScenario


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_START = datetime(2026, 6, 1, 6, 0)  # 06:00 Monday


def _make_meal(
    time: datetime,
    true_carbs: float = 50.0,
    estimated_carbs: float = 50.0,
) -> MealEvent:
    return MealEvent(
        time=time,
        meal_type=MealType.ALMOCO,
        true_carbs_g=true_carbs,
        estimated_carbs_g=estimated_carbs,
        foods=("arroz", "feijao"),
        glycemic_index=60.0,
    )


def _make_schedule(
    meals: list[MealEvent] | None = None,
    exercise_events: list[ExerciseEvent] | None = None,
    stress_events: list[StressEvent] | None = None,
    wake_time: datetime = _START,
    sleep_time: datetime | None = None,
) -> DailySchedule:
    if sleep_time is None:
        sleep_time = wake_time.replace(hour=22)
    return DailySchedule(
        date=wake_time,
        day_type=DayType.WEEKDAY,
        wake_time=wake_time,
        sleep_time=sleep_time,
        meals=meals or [],
        exercise_events=exercise_events or [],
        stress_events=stress_events or [],
    )


# ---------------------------------------------------------------------------
# Cross-zone APIs
# ---------------------------------------------------------------------------


class TestCrossZoneAPIs:
    """Verify APIs required by parallel agents F1, F3, F5."""

    # F1: sensor_seed
    def test_sensor_seed_returns_int(self) -> None:
        """sensor_seed(patient_idx) must return an integer."""
        rng = RNGManager(master_seed=42)
        seed = rng.sensor_seed(0)
        assert isinstance(seed, int), f"sensor_seed must return int, got {type(seed)}"

    def test_sensor_seed_deterministic(self) -> None:
        """Same master_seed + patient_idx always gives the same sensor seed."""
        rng1 = RNGManager(master_seed=42)
        rng2 = RNGManager(master_seed=42)
        assert rng1.sensor_seed(0) == rng2.sensor_seed(0)
        assert rng1.sensor_seed(5) == rng2.sensor_seed(5)

    def test_sensor_seed_no_aliasing(self) -> None:
        """sensor_seed avoids the aliasing bug (seed=100+idx=42 == seed=142+idx=0).

        With hierarchical SeedSequence derivation, different (master_seed, patient_idx)
        pairs that sum to the same value must produce different sensor seeds.
        """
        rng_a = RNGManager(master_seed=100)
        rng_b = RNGManager(master_seed=142)
        # Old bug: 100+42 == 142+0, so both gave the same CGM seed.
        # Fixed: hierarchical derivation makes them distinct.
        seed_a = rng_a.sensor_seed(42)
        seed_b = rng_b.sensor_seed(0)
        assert seed_a != seed_b, (
            "sensor_seed must not alias: (seed=100, idx=42) and (seed=142, idx=0) "
            "produced the same value, indicating the old arithmetic pattern is still in use."
        )

    def test_sensor_seed_distinct_across_patients(self) -> None:
        """Different patient indices produce different sensor seeds."""
        rng = RNGManager(master_seed=42)
        seeds = [rng.sensor_seed(i) for i in range(10)]
        assert len(set(seeds)) == len(seeds), "sensor_seed values must be unique per patient"

    def test_sensor_seed_out_of_range(self) -> None:
        """sensor_seed raises IndexError for out-of-range patient_idx."""
        rng = RNGManager(master_seed=42, max_patients=10)
        with pytest.raises(IndexError):
            rng.sensor_seed(10)
        with pytest.raises(IndexError):
            rng.sensor_seed(-1)

    # F3: snacks sub-stream
    def test_patient_rng_has_snacks_stream(self) -> None:
        """PatientRNG must expose a `snacks` Generator attribute."""
        rng = RNGManager(master_seed=42)
        p = rng.patient_rng(0)
        assert hasattr(p, "snacks"), "PatientRNG must have a `snacks` stream"
        # Must be an independent stream: drawing from snacks doesn't affect meals
        meals_val = p.meals.random()
        snacks_val = p.snacks.random()
        assert meals_val != snacks_val, "snacks and meals streams must be independent"

    def test_snacks_stream_independent_of_meals(self) -> None:
        """Adding a snack draw should NOT change the meals stream sequence."""
        rng1 = RNGManager(master_seed=42)
        rng2 = RNGManager(master_seed=42)

        p1 = rng1.patient_rng(0)
        p2 = rng2.patient_rng(0)

        # Draw from snacks on p1 before drawing from meals
        _ = p1.snacks.random()
        meal_seq_with_snack_draw = [p1.meals.random() for _ in range(10)]

        # On p2, don't draw from snacks first
        meal_seq_without = [p2.meals.random() for _ in range(10)]

        np.testing.assert_array_equal(
            meal_seq_with_snack_draw, meal_seq_without,
            err_msg="snacks stream must be isolated: drawing from snacks must not affect meals",
        )

    # F5: BolusTimingCategory.CORRECTION
    def test_bolus_timing_category_correction_exists(self) -> None:
        """BolusTimingCategory must have a CORRECTION entry."""
        assert hasattr(BolusTimingCategory, "CORRECTION"), (
            "BolusTimingCategory.CORRECTION is required by the insulin module (H5 #7 fix)"
        )
        assert BolusTimingCategory.CORRECTION.value == "correction"

    def test_bolus_timing_category_correction_distinct(self) -> None:
        """CORRECTION must be distinct from all other BolusTimingCategory entries."""
        correction = BolusTimingCategory.CORRECTION
        others = [c for c in BolusTimingCategory if c is not correction]
        assert all(c != correction for c in others), (
            "BolusTimingCategory.CORRECTION must be unique among enum members"
        )


# ---------------------------------------------------------------------------
# Bug 1: general_rng cached_property
# ---------------------------------------------------------------------------


class TestGeneralRNGCached:
    """Bug 1: general_rng must be a cached_property returning the same Generator."""

    def test_general_rng_is_same_object(self) -> None:
        """Accessing general_rng twice must return the exact same Generator instance."""
        rng = RNGManager(master_seed=42)
        g1 = rng.general_rng
        g2 = rng.general_rng
        assert g1 is g2, (
            "general_rng must be a cached_property: both accesses must return "
            "the same Generator object, not a freshly-seeded copy."
        )

    def test_general_rng_evolves_state(self) -> None:
        """The cached Generator evolves state across draws (not reset on each access)."""
        rng = RNGManager(master_seed=42)
        v1 = rng.general_rng.random()
        v2 = rng.general_rng.random()
        assert v1 != v2, (
            "general_rng must return a stateful Generator: consecutive draws must differ. "
            "If they are equal, general_rng is returning a fresh generator each time."
        )

    def test_general_rng_does_not_interfere_with_patients(self) -> None:
        """The general_rng stream must not affect patient streams."""
        rng1 = RNGManager(master_seed=42)
        rng2 = RNGManager(master_seed=42)

        # Exhaust several draws from general_rng on rng1
        for _ in range(100):
            rng1.general_rng.random()

        # Patient streams must be identical regardless of general_rng usage
        p1 = rng1.patient_rng(0)
        p2 = rng2.patient_rng(0)
        seq1 = [p1.meals.random() for _ in range(10)]
        seq2 = [p2.meals.random() for _ in range(10)]
        np.testing.assert_array_equal(seq1, seq2)


# ---------------------------------------------------------------------------
# Bug 3: meal snap < start_time dropped
# ---------------------------------------------------------------------------


class TestMealSnapBeforeStart:
    """Bug 3: meals that snap before start_time must be silently dropped."""

    def test_meal_very_early_is_dropped(self) -> None:
        """A meal timed 90 seconds before start_time must not appear in events."""
        early_time = _START - timedelta(seconds=90)
        meal = _make_meal(early_time, true_carbs=30.0)
        schedule = _make_schedule(meals=[meal])
        scenario = SimadaScenario.from_schedules([schedule])

        # No scenario events should exist for a time before start
        assert len(scenario._meal_lookup) == 0, (
            "Meal snapped before start_time must be dropped from scenario_events"
        )

    def test_meal_exactly_on_start_is_kept(self) -> None:
        """A meal exactly at start_time must be kept."""
        meal = _make_meal(_START, true_carbs=30.0)
        schedule = _make_schedule(meals=[meal])
        scenario = SimadaScenario.from_schedules([schedule])
        assert len(scenario._meal_lookup) == 1

    def test_meal_after_start_is_kept(self) -> None:
        """A normal meal after start_time must survive snapping."""
        meal = _make_meal(_START + timedelta(hours=6), true_carbs=50.0)
        schedule = _make_schedule(meals=[meal])
        scenario = SimadaScenario.from_schedules([schedule])
        assert len(scenario._meal_lookup) == 1


# ---------------------------------------------------------------------------
# Bug 4: get_action normalizes t
# ---------------------------------------------------------------------------


class TestGetActionNormalization:
    """Bug 4: get_action(t) must normalize t to the same grid as inject_cho."""

    def test_get_action_delivers_scheduled_meal_exactly_once(self) -> None:
        """A scheduled meal fires at its EXACT grid time only, NOT on the adjacent
        sub-step minutes. simglucose queries t, t+1, t+2 within each 3-min step;
        floor-snapping the meal lookup returned it on all three and delivered the
        meal THREE times (a 50 g meal -> 150 g). It must fire exactly once.
        """
        meal_time = _START + timedelta(hours=6)  # 12:00, on the 3-min grid
        meal = _make_meal(meal_time, true_carbs=50.0)
        scenario = SimadaScenario.from_schedules([_make_schedule(meals=[meal])])

        assert scenario.get_action(meal_time).meal == pytest.approx(50.0)
        # the next two sub-step minutes must NOT re-deliver the same meal
        for off in (1, 2):
            action = scenario.get_action(meal_time + timedelta(minutes=off))
            assert action.meal == pytest.approx(0.0), (
                f"meal re-delivered at t+{off}min -> triple delivery. Got {action.meal}.")

    def test_inject_cho_and_get_action_on_same_grid(self) -> None:
        """inject_cho at off-grid time + get_action at slightly different off-grid time
        must both snap to the same grid point and the CHO must be delivered."""
        meal = _make_meal(_START + timedelta(hours=6))
        schedule = _make_schedule(meals=[meal])
        scenario = SimadaScenario.from_schedules([schedule])

        # Inject at 09:01 — snaps to 09:00
        inject_time = _START + timedelta(hours=3, minutes=1)
        scenario.inject_cho(inject_time, 20.0)

        # get_action at 09:02 — also snaps to 09:00
        query_time = _START + timedelta(hours=3, minutes=2)
        action = scenario.get_action(query_time)
        assert action.meal == pytest.approx(20.0), (
            "inject_cho at 09:01 and get_action at 09:02 must both snap to "
            f"09:00. CHO was not delivered; got {action.meal}."
        )


# ---------------------------------------------------------------------------
# Bug 5: duration_hours validates schedule ordering
# ---------------------------------------------------------------------------


class TestDurationHoursValidation:
    """Bug 5: duration_hours must raise ValueError when last_sleep <= first_wake."""

    def test_duration_hours_normal(self) -> None:
        """Normal multi-day scenario returns positive duration."""
        day1_wake = _START
        day1_sleep = _START.replace(hour=22)
        day2_wake = _START + timedelta(days=1)
        day2_sleep = day2_wake.replace(hour=22)
        s1 = _make_schedule(
            meals=[_make_meal(day1_wake + timedelta(hours=6))],
            wake_time=day1_wake,
            sleep_time=day1_sleep,
        )
        s2 = _make_schedule(
            meals=[_make_meal(day2_wake + timedelta(hours=6))],
            wake_time=day2_wake,
            sleep_time=day2_sleep,
        )
        scenario = SimadaScenario.from_schedules([s1, s2])
        hours = scenario.duration_hours
        assert hours > 0, f"duration_hours must be positive for a valid scenario, got {hours}"

    def test_duration_hours_raises_when_inverted(self) -> None:
        """Raises ValueError if schedules are ordered so last_sleep <= first_wake."""
        # Construct an inverted scenario directly (from_schedules won't prevent this)
        wake = _START
        sleep_before_wake = _START - timedelta(hours=2)  # sleep before wake
        schedule = _make_schedule(
            meals=[_make_meal(wake + timedelta(hours=6))],
            wake_time=wake,
            sleep_time=sleep_before_wake,
        )
        scenario = SimadaScenario(
            start_time=wake,
            scenario_events=[],
            schedules=[schedule],
        )
        with pytest.raises(ValueError, match="last_sleep"):
            _ = scenario.duration_hours

    def test_duration_hours_empty_schedules(self) -> None:
        """Empty schedule list returns 0.0 (not an error)."""
        scenario = SimadaScenario(
            start_time=_START,
            scenario_events=[],
            schedules=[],
        )
        assert scenario.duration_hours == 0.0


# ---------------------------------------------------------------------------
# Bug 6: ScenarioConfig.cohort type is CohortConfig, not Optional
# ---------------------------------------------------------------------------


class TestScenarioConfigCohortType:
    """Bug 6: ScenarioConfig.cohort must be CohortConfig, not CohortConfig | None."""

    def test_cohort_is_always_materialised(self) -> None:
        """ScenarioConfig() without explicit cohort must produce a CohortConfig instance."""
        config = ScenarioConfig()
        assert config.cohort is not None, "cohort must never be None after construction"
        assert isinstance(config.cohort, CohortConfig), (
            f"cohort must be CohortConfig, got {type(config.cohort)}"
        )

    def test_cohort_type_annotation_is_not_optional(self) -> None:
        """The runtime type of cohort must be CohortConfig (not NoneType ever)."""
        import inspect
        import typing

        hints = typing.get_type_hints(ScenarioConfig)
        cohort_hint = hints.get("cohort")
        # Should NOT be Optional[CohortConfig] or CohortConfig | None
        args = getattr(cohort_hint, "__args__", None)
        if args is not None:
            assert type(None) not in args, (
                "ScenarioConfig.cohort type annotation must not include None. "
                "The validator always materialises it, so the annotation lied."
            )


# ---------------------------------------------------------------------------
# Bug 7: CohortConfig.archetype_distribution uses Field(default_factory)
# ---------------------------------------------------------------------------


class TestCohortConfigMutableDefault:
    """Bug 7: archetype_distribution must use Field(default_factory=...) not a mutable dict."""

    def test_two_instances_do_not_share_distribution(self) -> None:
        """Mutating one CohortConfig's archetype_distribution must not affect another."""
        c1 = CohortConfig()
        c2 = CohortConfig()

        # Mutate c1's distribution
        c1.archetype_distribution["test_key"] = 0.99

        assert "test_key" not in c2.archetype_distribution, (
            "CohortConfig instances must not share the same dict object. "
            "Use Field(default_factory=lambda: {...}) not a class-level mutable default."
        )

    def test_insulin_regimen_distribution_not_shared(self) -> None:
        """insulin_regimen_distribution must also use default_factory."""
        c1 = CohortConfig()
        c2 = CohortConfig()
        c1.insulin_regimen_distribution["cgm"] = 0.5
        assert "cgm" not in c2.insulin_regimen_distribution


# ---------------------------------------------------------------------------
# Bug 8: get_context accumulates overlapping events (no early break)
# ---------------------------------------------------------------------------


class TestGetContextOverlap:
    """Bug 8: get_context must not break on first matching event; last active wins."""

    def test_get_context_last_overlapping_exercise_wins(self) -> None:
        """With two overlapping exercise events, the last one's intensity is returned."""
        e1 = ExerciseEvent(
            start_time=_START + timedelta(hours=2),
            duration_minutes=120,  # ends 14:00
            intensity=ExerciseIntensity.LIGHT,
            insulin_sensitivity_multiplier=1.2,
        )
        e2 = ExerciseEvent(
            start_time=_START + timedelta(hours=3),  # 09:00, inside e1's window
            duration_minutes=60,  # ends 10:00
            intensity=ExerciseIntensity.VIGOROUS,
            insulin_sensitivity_multiplier=1.8,
        )
        schedule = _make_schedule(
            meals=[_make_meal(_START + timedelta(hours=6))],
            exercise_events=[e1, e2],
        )
        scenario = SimadaScenario.from_schedules([schedule])

        # At 09:30, both events are active; last processed (e2) must win
        ctx = scenario.get_context(_START + timedelta(hours=3, minutes=30))
        assert ctx["exercise_active"] is True
        assert ctx["exercise_intensity"] == "vigorous", (
            f"With overlapping events, last active event must win. "
            f"Got '{ctx['exercise_intensity']}', expected 'vigorous'."
        )

    def test_get_context_last_overlapping_stress_wins(self) -> None:
        """With two overlapping stress events, the last one's resistance factor is returned."""
        s1 = StressEvent(
            start_time=_START + timedelta(hours=2),
            duration_minutes=180,
            stress_type=StressType.PSYCHOLOGICAL,
            insulin_resistance_factor=1.2,
        )
        s2 = StressEvent(
            start_time=_START + timedelta(hours=3),
            duration_minutes=60,
            stress_type=StressType.ILLNESS,
            insulin_resistance_factor=1.5,
        )
        schedule = _make_schedule(
            meals=[_make_meal(_START + timedelta(hours=6))],
            stress_events=[s1, s2],
        )
        scenario = SimadaScenario.from_schedules([schedule])

        ctx = scenario.get_context(_START + timedelta(hours=3, minutes=30))
        assert ctx["stress_active"] is True
        # Last stress event (s2) must be active
        assert ctx["stress_type"] == "illness", (
            f"Expected stress_type='illness' (last event), got '{ctx['stress_type']}'."
        )
        assert ctx["insulin_resistance_factor"] == pytest.approx(1.5)

    def test_get_context_non_overlapping_still_works(self) -> None:
        """Non-overlapping events return the active one correctly."""
        e1 = ExerciseEvent(
            start_time=_START + timedelta(hours=2),
            duration_minutes=30,  # ends 08:30
            intensity=ExerciseIntensity.LIGHT,
            insulin_sensitivity_multiplier=1.2,
        )
        e2 = ExerciseEvent(
            start_time=_START + timedelta(hours=4),  # 10:00, after e1
            duration_minutes=30,
            intensity=ExerciseIntensity.VIGOROUS,
            insulin_sensitivity_multiplier=1.8,
        )
        schedule = _make_schedule(
            meals=[_make_meal(_START + timedelta(hours=6))],
            exercise_events=[e1, e2],
        )
        scenario = SimadaScenario.from_schedules([schedule])

        # During e1, only e1 is active
        ctx_e1 = scenario.get_context(_START + timedelta(hours=2, minutes=15))
        assert ctx_e1["exercise_intensity"] == "light"

        # During e2, only e2 is active
        ctx_e2 = scenario.get_context(_START + timedelta(hours=4, minutes=15))
        assert ctx_e2["exercise_intensity"] == "vigorous"

        # Between events, nothing is active
        ctx_gap = scenario.get_context(_START + timedelta(hours=3))
        assert ctx_gap["exercise_active"] is False


# ---------------------------------------------------------------------------
# Bug 9: DailySchedule documented as intentionally mutable
# ---------------------------------------------------------------------------


class TestDailyScheduleMutable:
    """Bug 9: DailySchedule is intentionally mutable — verify + doc."""

    def test_daily_schedule_is_mutable(self) -> None:
        """DailySchedule must allow appending to lists post-construction."""
        schedule = _make_schedule()
        initial_count = len(schedule.meals)
        new_meal = _make_meal(_START + timedelta(hours=6))
        schedule.meals.append(new_meal)
        assert len(schedule.meals) == initial_count + 1, (
            "DailySchedule.meals must be mutable (controller appends insulin_events at runtime)"
        )

    def test_daily_schedule_not_frozen(self) -> None:
        """DailySchedule must NOT be a frozen dataclass."""
        import dataclasses
        assert not dataclasses.fields(DailySchedule)[0].metadata.get("frozen", False), (
            "DailySchedule must remain mutable for runtime event appending."
        )
        # Try direct attribute set — must not raise FrozenInstanceError
        schedule = _make_schedule()
        try:
            schedule.day_type = DayType.WEEKEND
        except Exception as exc:
            pytest.fail(f"DailySchedule must not be frozen; got: {exc}")

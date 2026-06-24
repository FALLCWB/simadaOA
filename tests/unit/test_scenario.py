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

"""Tests for SimadaScenario — the behavioral scenario engine bridging simada to simglucose."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from simada.core.types import (
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
# Tests
# ---------------------------------------------------------------------------


class TestSimadaScenario:
    """Unit tests for SimadaScenario."""

    def test_from_schedules_creates_scenario(self) -> None:
        """Basic construction from a single schedule with one meal."""
        meal = _make_meal(_START + timedelta(hours=6))
        schedule = _make_schedule(meals=[meal])
        scenario = SimadaScenario.from_schedules([schedule])

        assert scenario.start_time == _START
        assert len(scenario.schedules) == 1
        assert len(scenario._all_meals) == 1

    def test_from_schedules_empty_raises(self) -> None:
        """Empty schedule list raises ValueError."""
        with pytest.raises(ValueError, match="At least one DailySchedule"):
            SimadaScenario.from_schedules([])

    def test_meal_time_snapping(self) -> None:
        """Meal at 12:01 snaps to 12:00 or 12:03 (nearest 3-min grid point).

        The from_schedules method rounds meal times to the nearest
        sample_time (3 min) grid point aligned with start_time.
        """
        # start_time = 06:00. Grid points: 06:00, 06:03, 06:06, ...
        # 12:01 is 361 min from 06:00. 361/3 = 120.33 => round(120.33) = 120
        # Snapped = 06:00 + 120*3 = 06:00 + 360 = 12:00
        meal_time = _START + timedelta(hours=6, minutes=1)  # 12:01
        meal = _make_meal(meal_time, true_carbs=50.0)
        schedule = _make_schedule(meals=[meal])
        scenario = SimadaScenario.from_schedules([schedule])

        # Check that the snapped time is either 12:00 or 12:03
        snapped_12_00 = _START + timedelta(hours=6)
        snapped_12_03 = _START + timedelta(hours=6, minutes=3)
        assert snapped_12_00 in scenario._meal_lookup or snapped_12_03 in scenario._meal_lookup, (
            f"Meal at 12:01 should snap to 12:00 or 12:03. "
            f"Lookup keys: {list(scenario._meal_lookup.keys())}"
        )

    def test_inject_cho_delivers_via_get_action(self) -> None:
        """inject CHO at time T, get_action(T) includes it."""
        meal = _make_meal(_START + timedelta(hours=6))
        schedule = _make_schedule(meals=[meal])
        scenario = SimadaScenario.from_schedules([schedule])

        inject_time = _START + timedelta(hours=3)  # 09:00 (on grid)
        scenario.inject_cho(inject_time, 15.0)

        # get_action at the snapped time should include the injected CHO
        action = scenario.get_action(inject_time)
        assert action.meal == pytest.approx(15.0), (
            f"Injected CHO should appear in get_action, got {action.meal}"
        )

    def test_inject_cho_consumed_once(self) -> None:
        """Injected CHO is popped (not re-delivered on next call)."""
        meal = _make_meal(_START + timedelta(hours=6))
        schedule = _make_schedule(meals=[meal])
        scenario = SimadaScenario.from_schedules([schedule])

        inject_time = _START + timedelta(hours=3)
        scenario.inject_cho(inject_time, 15.0)

        # First call consumes it
        action1 = scenario.get_action(inject_time)
        assert action1.meal == pytest.approx(15.0)

        # Second call should return 0
        action2 = scenario.get_action(inject_time)
        assert action2.meal == pytest.approx(0.0), (
            "Injected CHO should be consumed on first get_action call"
        )

    def test_inject_cho_accumulates(self) -> None:
        """Two injections at same time sum."""
        meal = _make_meal(_START + timedelta(hours=6))
        schedule = _make_schedule(meals=[meal])
        scenario = SimadaScenario.from_schedules([schedule])

        inject_time = _START + timedelta(hours=3)
        scenario.inject_cho(inject_time, 10.0)
        scenario.inject_cho(inject_time, 7.6)

        action = scenario.get_action(inject_time)
        assert action.meal == pytest.approx(17.6), (
            f"Two injections should sum: expected 17.6, got {action.meal}"
        )

    def test_get_meal_at_within_tolerance(self) -> None:
        """Meal at T found at T+30s but not T+90s (tolerance is 60 seconds)."""
        meal_time = _START + timedelta(hours=6)
        meal = _make_meal(meal_time)
        schedule = _make_schedule(meals=[meal])
        scenario = SimadaScenario.from_schedules([schedule])

        # Within 30 seconds — should find it
        result_30s = scenario.get_meal_at(meal_time + timedelta(seconds=30))
        assert result_30s is not None, "Should find meal within 30 seconds"
        assert result_30s.time == meal_time

        # Beyond 90 seconds — should NOT find it (tolerance < 60s)
        result_90s = scenario.get_meal_at(meal_time + timedelta(seconds=90))
        assert result_90s is None, "Should not find meal beyond 60-second tolerance"

    def test_get_meal_at_no_meals(self) -> None:
        """Returns None when no meals exist."""
        schedule = _make_schedule(meals=[])
        # Need at least one meal for from_schedules to produce scenario_events,
        # but we can create the scenario directly.
        scenario = SimadaScenario(
            start_time=_START,
            scenario_events=[],
            schedules=[schedule],
        )
        assert scenario.get_meal_at(_START) is None

    def test_get_context_exercise_active(self) -> None:
        """Returns exercise_active=True during exercise window."""
        exercise = ExerciseEvent(
            start_time=_START + timedelta(hours=2),
            duration_minutes=60,
            intensity=ExerciseIntensity.MODERATE,
            insulin_sensitivity_multiplier=1.5,
        )
        schedule = _make_schedule(
            meals=[_make_meal(_START + timedelta(hours=6))],
            exercise_events=[exercise],
        )
        scenario = SimadaScenario.from_schedules([schedule])

        # During exercise (30 min into 60-min session)
        ctx_during = scenario.get_context(_START + timedelta(hours=2, minutes=30))
        assert ctx_during["exercise_active"] is True
        assert ctx_during["exercise_intensity"] == "moderate"

        # Before exercise
        ctx_before = scenario.get_context(_START + timedelta(hours=1))
        assert ctx_before["exercise_active"] is False

        # After exercise
        ctx_after = scenario.get_context(_START + timedelta(hours=3, minutes=30))
        assert ctx_after["exercise_active"] is False

    def test_get_context_stress_active(self) -> None:
        """Returns stress info during stress window."""
        stress = StressEvent(
            start_time=_START + timedelta(hours=4),
            duration_minutes=120,
            stress_type=StressType.PSYCHOLOGICAL,
            insulin_resistance_factor=1.3,
        )
        schedule = _make_schedule(
            meals=[_make_meal(_START + timedelta(hours=6))],
            stress_events=[stress],
        )
        scenario = SimadaScenario.from_schedules([schedule])

        ctx = scenario.get_context(_START + timedelta(hours=5))
        assert ctx["stress_active"] is True
        assert ctx["stress_type"] == "psychological"
        assert ctx["insulin_resistance_factor"] == pytest.approx(1.3)

    def test_get_context_no_events(self) -> None:
        """Returns defaults (exercise_active=False, resistance=1.0) when no events active."""
        schedule = _make_schedule(meals=[_make_meal(_START + timedelta(hours=6))])
        scenario = SimadaScenario.from_schedules([schedule])

        ctx = scenario.get_context(_START + timedelta(hours=1))
        assert ctx["exercise_active"] is False
        assert ctx["exercise_intensity"] is None
        assert ctx["stress_active"] is False
        assert ctx["stress_type"] is None
        assert ctx["insulin_resistance_factor"] == pytest.approx(1.0)

    def test_get_action_dict_lookup(self) -> None:
        """Verify O(1) lookup returns correct CHO for known meal time."""
        meal_time = _START + timedelta(hours=6)  # 12:00 (on grid)
        meal = _make_meal(meal_time, true_carbs=75.0)
        schedule = _make_schedule(meals=[meal])
        scenario = SimadaScenario.from_schedules([schedule])

        # The meal should be in the lookup dict at the snapped time
        action = scenario.get_action(meal_time)
        assert action.meal == pytest.approx(75.0), (
            f"get_action should return correct CHO via O(1) dict lookup, got {action.meal}"
        )

        # At a non-meal time, should return 0
        action_empty = scenario.get_action(_START + timedelta(hours=1))
        assert action_empty.meal == pytest.approx(0.0)

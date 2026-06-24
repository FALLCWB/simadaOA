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

"""SimadaScenario — extends simglucose CustomScenario with behavioral context.

The key integration point between simada's behavioral model and simglucose's
physiological simulation. Converts multi-day DailySchedules into simglucose-
compatible meal events while preserving side-channel context (exercise, stress,
insulin plan) for the controller and future RL agents.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from simglucose.simulation.scenario import Action as ScenarioAction
from simglucose.simulation.scenario import CustomScenario

from simada.core.types import DailySchedule, ExerciseEvent, MealEvent, StressEvent


class SimadaScenario(CustomScenario):
    """Extends simglucose CustomScenario with rich behavioral context.

    simglucose scenarios only support meal events (CHO in grams at specific
    times). All other effects (exercise, stress, insulin adherence) are
    accessed through the side-channel context API by the controller.

    Usage::

        schedules = [day1_schedule, day2_schedule, ...]
        scenario = SimadaScenario.from_schedules(schedules)
        # scenario is compatible with simglucose SimObj
    """

    def __init__(
        self,
        start_time: datetime,
        scenario_events: list[tuple[datetime, float]],
        schedules: list[DailySchedule],
    ) -> None:
        super().__init__(start_time=start_time, scenario=scenario_events)
        self._schedules = schedules
        self._all_meals: list[MealEvent] = []
        self._exercise_events: list[ExerciseEvent] = []
        self._stress_events: list[StressEvent] = []
        # Dynamic CHO injections from hypo corrections (controller adds these at runtime)
        self._injected_cho: dict[datetime, float] = {}

        for schedule in schedules:
            self._all_meals.extend(schedule.meals)
            self._exercise_events.extend(schedule.exercise_events)
            self._stress_events.extend(schedule.stress_events)

        self._all_meals.sort(key=lambda m: m.time)

        # Pre-build O(1) meal lookup dict from scenario events.
        # The parent class stores self.scenario as list[tuple[datetime, float]].
        # simglucose's get_action() rebuilds lists + linear scans every call
        # (43,200 times per 30-day patient). This dict replaces that with a
        # single O(1) lookup per call.
        self._meal_lookup: dict[datetime, float] = {}
        for time, cho in scenario_events:
            if time in self._meal_lookup:
                self._meal_lookup[time] += cho
            else:
                self._meal_lookup[time] = cho

    @classmethod
    def from_schedules(cls, schedules: list[DailySchedule]) -> SimadaScenario:
        """Create a scenario from a list of DailySchedules.

        Flattens all meal events into simglucose-compatible (time, Action)
        tuples using true_carbs_g (the actual CHO the patient ingests).
        """
        if not schedules:
            raise ValueError("At least one DailySchedule is required")

        # Round start_time to the nearest minute (strip seconds/microseconds)
        # so simulation time steps align cleanly.
        raw_start = schedules[0].wake_time
        start_time = raw_start.replace(second=0, microsecond=0)

        # Snap meal times to the nearest minute on the simulation grid.
        # simglucose uses EXACT datetime comparison (t in times2compare),
        # so meal times must fall on minute boundaries aligned with start_time.
        # The simulation steps every sample_time minutes (default 3) from start.
        sample_time = 3  # minutes (simglucose default)

        scenario_events: list[tuple[datetime, float]] = []
        for schedule in schedules:
            for meal in schedule.meals:
                # Snap to nearest grid point: round to nearest sample_time minutes
                delta = (meal.time - start_time).total_seconds()
                # Drop meals before simulation start (negative delta), even if they
                # would snap onto start_time. A meal recorded before the sim window
                # is not part of this scenario.
                if delta < 0:
                    continue
                snapped_minutes = round(delta / 60 / sample_time) * sample_time
                snapped_time = start_time + timedelta(minutes=snapped_minutes)
                scenario_events.append((snapped_time, meal.true_carbs_g))

        scenario_events.sort(key=lambda x: x[0])

        return cls(
            start_time=start_time,
            scenario_events=scenario_events,
            schedules=schedules,
        )

    def inject_cho(self, time: datetime, grams: float) -> None:
        """Inject dynamic CHO at runtime (for hypo correction).

        Called by the controller when a hypo correction is needed.
        The CHO will be delivered at the next matching simulation step.
        """
        # Snap DOWN to the current 3-min grid cell (floor), so injections
        # and get_action queries within the same cell coalesce. Round-half
        # would put 09:01 and 09:02 in different cells (09:00 vs 09:03).
        delta_min = (time - self.start_time).total_seconds() / 60.0
        snapped_minutes = int(delta_min // 3) * 3
        snapped_time = self.start_time + timedelta(minutes=snapped_minutes)

        if snapped_time in self._injected_cho:
            self._injected_cho[snapped_time] += grams
        else:
            self._injected_cho[snapped_time] = grams

    def get_action(self, t):
        """O(1) meal lookup, bypassing simglucose's O(n) parent implementation.

        The parent CustomScenario.get_action() rebuilds zip + list + linear
        scan on every call. With ~43,200 calls per 30-day patient, this was
        25-35% of total runtime. This override uses a pre-built dict for O(1)
        lookup while still supporting runtime CHO injections from hypo
        corrections.

        Scheduled meals are looked up by EXACT time: simglucose calls get_action
        once per minute (t, t+1, t+2) within each 3-min step, so floor-snapping the
        meal lookup would return the SAME meal on all three sub-steps and deliver it
        THREE times. Meals are already snapped to the 3-min grid at build time, so an
        exact lookup matches once (like the parent CustomScenario). Runtime CHO
        injections still use the floor-snap grid (and ``pop``, so they fire once)
        because the controller may inject at an off-grid current time.
        """
        base_cho = self._meal_lookup.get(t, 0.0)

        # Floor-snap only the injection lookup; pop so it is delivered once.
        delta_min = (t - self.start_time).total_seconds() / 60.0
        snapped_minutes = int(delta_min // 3) * 3
        t_norm = self.start_time + timedelta(minutes=snapped_minutes)
        injected = self._injected_cho.pop(t_norm, 0.0)

        total_cho = base_cho + injected
        return ScenarioAction(meal=total_cho)

    @property
    def schedules(self) -> list[DailySchedule]:
        """All DailySchedules in this scenario."""
        return list(self._schedules)

    def get_meal_at(self, time: datetime) -> MealEvent | None:
        """Look up the MealEvent closest to a given time (within 1 minute).

        Uses bisect for O(log n) lookup on the sorted meal list.
        """
        from bisect import bisect_left

        if not self._all_meals:
            return None

        times = [m.time for m in self._all_meals]
        idx = bisect_left(times, time)

        best = None
        best_delta = 61.0
        for candidate_idx in (idx - 1, idx):
            if 0 <= candidate_idx < len(self._all_meals):
                delta = abs((self._all_meals[candidate_idx].time - time).total_seconds())
                if delta < 60 and delta < best_delta:
                    best = self._all_meals[candidate_idx]
                    best_delta = delta

        return best

    def get_context(self, time: datetime) -> dict:
        """Return context signals at a specific time for the controller/RL agent.

        This is the future Safe RL integration point. Returns:
            exercise_active: bool — is exercise happening now
            exercise_intensity: str | None — light/moderate/vigorous
            stress_active: bool — is a stress event active
            stress_type: str | None — psychological/illness/alcohol
            insulin_resistance_factor: float — current modifier (1.0 = normal)
            meal_announced: MealEvent | None — nearest meal event
        """

        exercise_active = False
        exercise_intensity = None
        for e in self._exercise_events:
            end = e.start_time + timedelta(minutes=e.duration_minutes)
            if e.start_time <= time <= end:
                exercise_active = True
                exercise_intensity = e.intensity.value
                # Continue scanning: if multiple events overlap, last one wins.
                # Callers should use DailySchedule.validate_non_overlap() to
                # prevent accidental overlaps (see H8 #8).

        stress_active = False
        stress_type = None
        resistance_factor = 1.0
        for s in self._stress_events:
            end = s.start_time + timedelta(minutes=s.duration_minutes)
            if s.start_time <= time <= end:
                stress_active = True
                stress_type = s.stress_type.value
                resistance_factor = s.insulin_resistance_factor
                # Continue scanning: last active stress event wins.

        meal = self.get_meal_at(time)

        return {
            "exercise_active": exercise_active,
            "exercise_intensity": exercise_intensity,
            "stress_active": stress_active,
            "stress_type": stress_type,
            "insulin_resistance_factor": resistance_factor,
            "meal_event": meal,
        }

    @property
    def duration_hours(self) -> float:
        """Total scenario duration in hours."""
        if not self._schedules:
            return 0.0
        first_wake = self._schedules[0].wake_time
        last_sleep = self._schedules[-1].sleep_time
        if last_sleep <= first_wake:
            raise ValueError(
                f"duration_hours: last_sleep ({last_sleep}) must be after "
                f"first_wake ({first_wake}). Check schedule ordering."
            )
        return (last_sleep - first_wake).total_seconds() / 3600.0

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

"""Daily schedule compositor.

Orchestrates all behavioral subsystems (circadian, meals, exercise, stress,
snacking) into a single DailySchedule for each simulated day.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from simada.behavior.circadian import CircadianModel
from simada.behavior.exercise import ExerciseGenerator
from simada.behavior.snacking import SnackGenerator
from simada.behavior.stress import StressEventGenerator
from simada.core.config import ArchetypeParams, BehaviorConfig, MealConfig
from simada.core.random import PatientRNG
from simada.core.types import DailySchedule, DayType
from simada.meals.generator import MealGenerator
from simada.meals.taco import TACODatabase
from simada.meals.templates import DayMealPlan


class DailyScheduleBuilder:
    """Builds a complete DailySchedule by composing all behavioral subsystems.

    Usage::

        builder = DailyScheduleBuilder(params, taco_db, weekday_plan,
                                        weekend_plan, meal_config, behavior_config)
        schedule = builder.build(date, day_type, patient_rng)
    """

    def __init__(
        self,
        archetype_params: ArchetypeParams,
        taco_db: TACODatabase,
        weekday_plan: DayMealPlan,
        weekend_plan: DayMealPlan,
        meal_config: MealConfig,
        behavior_config: BehaviorConfig,
        holiday_plan: DayMealPlan | None = None,
    ) -> None:
        self._circadian = CircadianModel(archetype_params)
        self._meal_gen = MealGenerator(taco_db, meal_config, archetype_params)
        self._exercise_gen = ExerciseGenerator(archetype_params)
        self._stress_gen = StressEventGenerator(archetype_params, behavior_config)
        self._snack_gen = SnackGenerator(archetype_params, taco_db)
        self._weekday_plan = weekday_plan
        self._weekend_plan = weekend_plan
        self._holiday_plan = holiday_plan

    def build(
        self,
        date: datetime,
        day_type: DayType,
        patient_rng: PatientRNG,
    ) -> DailySchedule:
        """Build a complete daily schedule for a single day.

        Args:
            date: The calendar date (midnight).
            day_type: Weekday, weekend, or holiday.
            patient_rng: Per-patient RNG with named component streams.

        Returns:
            A fully populated DailySchedule.
        """
        # 1. Sample circadian times
        wake_time = self._circadian.sample_wake_time(
            date, day_type, patient_rng.behavior
        )
        sleep_time = self._circadian.sample_sleep_time(
            date, day_type, patient_rng.behavior
        )

        # BUG-FIX (H3 #4): Normalize sleep_time when it wraps past midnight.
        # CircadianModel can return sleep_time on the *next* calendar date
        # (e.g. 25:30 -> 01:30 next day). Downstream consumers
        # (SnackGenerator._find_gaps, MealGenerator timing clamps) assume
        # all daily events live on the same date as ``date``; a post-midnight
        # sleep_time produced gaps that crossed midnight and placed snacks
        # on the next day, which then conflicted with the next day's meal
        # generation. We clamp sleep_time to 23:59 of the current date so
        # the schedule is internally consistent. Sleep continuing into the
        # next day is modelled as the next day's "wake_time", not as part
        # of the current day's awake window.
        day_end = date.replace(hour=23, minute=59, second=0, microsecond=0)
        if sleep_time.date() > date.date():
            sleep_time = day_end

        # Ensure minimum awake time
        min_awake = self._circadian.minimum_awake_hours()
        if (sleep_time - wake_time).total_seconds() / 3600 < min_awake:
            sleep_time = wake_time + timedelta(hours=min_awake)
            # Re-clamp in case min_awake pushed us across midnight again
            if sleep_time.date() > date.date():
                sleep_time = day_end

        # 2. Generate meals — holiday plan if available, else fall back to weekend
        if day_type == DayType.HOLIDAY and self._holiday_plan is not None:
            plan = self._holiday_plan
        elif day_type in (DayType.WEEKEND, DayType.HOLIDAY):
            plan = self._weekend_plan
        else:
            plan = self._weekday_plan
        meals = self._meal_gen.generate_day(
            plan, day_type, wake_time, sleep_time, patient_rng.meals
        )

        # 3. Generate exercise
        exercise_events = self._exercise_gen.generate(
            day_type, wake_time, sleep_time, patient_rng.exercise
        )

        # 4. Generate stress/illness/alcohol
        stress_events = self._stress_gen.generate(
            day_type, wake_time, sleep_time, patient_rng.stress
        )

        # 5. Generate unplanned snacks (needs existing meal times).
        # Uses patient_rng.snacks (separate stream from patient_rng.meals) so
        # that adding or removing a meal does not shift the snack RNG sequence
        # and vice-versa — avoids cross-contamination (H3 bug #4).
        meal_times = [m.time for m in meals]
        snacks = self._snack_gen.generate(
            day_type, wake_time, sleep_time, meal_times, patient_rng.snacks
        )
        all_meals = sorted([*meals, *snacks], key=lambda m: m.time)

        return DailySchedule(
            date=date,
            day_type=day_type,
            wake_time=wake_time,
            sleep_time=sleep_time,
            meals=all_meals,
            exercise_events=exercise_events,
            stress_events=stress_events,
        )

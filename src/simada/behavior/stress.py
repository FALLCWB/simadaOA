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

"""Stress, illness, and alcohol event generator.

Models transient perturbations to insulin sensitivity:

- **Psychological stress**: cortisol-driven insulin resistance.
  Duration 2-6h, resistance factor 1.1-1.3.

- **Illness**: significant insulin resistance from infection/inflammation.
  Duration 1-5 days, resistance factor 1.3-2.0.

- **Alcohol**: biphasic effect.
  Phase 1 (0-3h): hyperglycemia from carbs in drinks + mild resistance.
  Phase 2 (3-12h): hypoglycemia from suppressed hepatic glucose output.

All probabilities come directly from ArchetypeParams (configurable via YAML).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from numpy.random import Generator

from simada.core.config import ArchetypeParams, BehaviorConfig
from simada.core.types import DayType, StressEvent, StressType


class StressEventGenerator:
    """Generates stress, illness, and alcohol events.

    All probabilities are read directly from ArchetypeParams, making them
    independently configurable per archetype without proxy inference.
    """

    def __init__(
        self, params: ArchetypeParams, behavior_config: BehaviorConfig
    ) -> None:
        self._params = params
        self._config = behavior_config

    def generate(
        self,
        day_type: DayType,
        wake_time: datetime,
        sleep_time: datetime,
        rng: Generator,
    ) -> list[StressEvent]:
        """Generate all stress/illness/alcohol events for a day."""
        events: list[StressEvent] = []

        if self._config.stress.enabled:
            events.extend(self._generate_psychological(wake_time, sleep_time, rng))
            events.extend(
                self._generate_illness(
                    wake_time, rng, self._config.stress.illness_probability_per_day
                )
            )

        if self._config.alcohol.enabled:
            events.extend(
                self._generate_alcohol(day_type, wake_time, sleep_time, rng)
            )

        return events

    def _generate_psychological(
        self,
        wake_time: datetime,
        sleep_time: datetime,
        rng: Generator,
    ) -> list[StressEvent]:
        """Generate psychological stress events."""
        if rng.random() >= self._params.stress_probability_per_day:
            return []

        duration = int(rng.integers(120, 361))  # 2-6 hours
        resistance = float(rng.uniform(1.1, 1.3))

        awake_minutes = (sleep_time - wake_time).total_seconds() / 60.0
        # Clamp duration so the stress event does not overrun sleep_time.
        duration = min(duration, int(awake_minutes))
        # Headroom is the number of minutes we can slide the start forward;
        # use max(0, ...) instead of max(1, ...) so that when duration==awake_minutes
        # the only valid offset is 0 (event fills the whole awake window exactly).
        headroom = max(0.0, awake_minutes - duration)
        offset = float(rng.uniform(0, headroom)) if headroom > 0 else 0.0
        start = wake_time + timedelta(minutes=offset)

        return [
            StressEvent(
                start_time=start,
                duration_minutes=duration,
                stress_type=StressType.PSYCHOLOGICAL,
                insulin_resistance_factor=resistance,
            )
        ]

    def _generate_illness(
        self,
        wake_time: datetime,
        rng: Generator,
        probability: float,
    ) -> list[StressEvent]:
        """Generate illness events (rare, multi-day effect modeled per-day).

        Note: multi-day illness continuity is handled by the scenario builder
        in a future phase. Currently each day rolls independently.
        """
        if rng.random() >= probability:
            return []

        duration = 24 * 60  # full day
        resistance = float(rng.uniform(1.3, 2.0))

        return [
            StressEvent(
                start_time=wake_time,
                duration_minutes=duration,
                stress_type=StressType.ILLNESS,
                insulin_resistance_factor=resistance,
            )
        ]

    def _generate_alcohol(
        self,
        day_type: DayType,
        wake_time: datetime,
        sleep_time: datetime,
        rng: Generator,
    ) -> list[StressEvent]:
        """Generate alcohol events with biphasic glucose effect.

        Phase 1 (0-3h): hyperglycemia — carbs in drinks + mild resistance.
        Phase 2 (3-12h): hypoglycemia — hepatic glucose suppression.
        """
        prob = self._params.alcohol_probability_weekend
        if day_type == DayType.WEEKDAY:
            prob *= self._params.alcohol_weekday_factor

        if rng.random() >= prob:
            return []

        # Drinking typically in the evening (70-90% of awake period).
        # Clamp drink_start so that Phase 1 (3h) fits entirely before sleep_time.
        awake_minutes = (sleep_time - wake_time).total_seconds() / 60.0
        evening_start = wake_time + timedelta(minutes=awake_minutes * 0.70)
        offset = float(rng.uniform(0, awake_minutes * 0.20))
        drink_start = evening_start + timedelta(minutes=offset)
        max_drink_start = sleep_time - timedelta(minutes=180)
        drink_start = min(drink_start, max_drink_start)

        events = []

        # Phase 1: initial hyperglycemia (0-3h)
        events.append(
            StressEvent(
                start_time=drink_start,
                duration_minutes=180,
                stress_type=StressType.ALCOHOL,
                insulin_resistance_factor=float(rng.uniform(1.1, 1.3)),
            )
        )

        # Phase 2: delayed hypoglycemia (3-12h, extends into sleep/next day).
        # Convention for insulin_resistance_factor:
        #   >1.0 = insulin resistance (hyperglycaemia risk, e.g. Phase 1 alcohol, stress, illness)
        #   <1.0 = insulin hyper-sensitivity / hepatic glucose suppression
        #          (hypoglycaemia risk, e.g. Phase 2 alcohol effect)
        # Downstream consumers MUST check both directions — a value of 0.8
        # means the patient is MORE sensitive to insulin, not more resistant.
        phase2_start = drink_start + timedelta(hours=3)
        events.append(
            StressEvent(
                start_time=phase2_start,
                duration_minutes=int(rng.integers(300, 541)),  # 5-9h
                stress_type=StressType.ALCOHOL,
                # <1.0 = hypersensitivity (hepatic glucose suppression in Phase 2)
                insulin_resistance_factor=float(rng.uniform(0.7, 0.9)),
            )
        )

        return events

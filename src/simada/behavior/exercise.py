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

"""Exercise event generator.

Models exercise sessions with physiological effects on insulin sensitivity.
Since simglucose does not natively support exercise, effects are modeled
through controller-side basal adjustments and context annotations for
the future Safe RL integration.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import NamedTuple

from numpy.random import Generator

from simada.core.config import ArchetypeParams
from simada.core.types import DayType, ExerciseEvent, ExerciseIntensity


class _IntensityProfile(NamedTuple):
    """Physiological effect profile for an exercise intensity level."""

    duration_min: int
    duration_max: int
    sensitivity_multiplier: float
    post_sensitivity_hours: float


# Public alias for use by the controller
INTENSITY_PROFILES: dict[ExerciseIntensity, _IntensityProfile] = {
    ExerciseIntensity.LIGHT: _IntensityProfile(
        duration_min=20, duration_max=45,
        sensitivity_multiplier=1.2, post_sensitivity_hours=2.0,
    ),
    ExerciseIntensity.MODERATE: _IntensityProfile(
        duration_min=30, duration_max=60,
        sensitivity_multiplier=1.5, post_sensitivity_hours=4.0,
    ),
    ExerciseIntensity.VIGOROUS: _IntensityProfile(
        duration_min=20, duration_max=45,
        sensitivity_multiplier=2.0, post_sensitivity_hours=8.0,
    ),
}

class ExerciseGenerator:
    """Generates exercise events based on archetype and day type.

    The generator first decides IF the patient exercises today (based on
    archetype probability), then samples the intensity, duration, and timing.
    Intensity weights come directly from ArchetypeParams (configurable via YAML).
    """

    def __init__(self, params: ArchetypeParams) -> None:
        self._params = params

    def generate(
        self,
        day_type: DayType,
        wake_time: datetime,
        sleep_time: datetime,
        rng: Generator,
    ) -> list[ExerciseEvent]:
        """Generate exercise events for a single day.

        Returns an empty list if the patient doesn't exercise today.
        Currently generates at most one session per day.
        """
        prob = (
            self._params.exercise_probability_weekend
            if day_type in (DayType.WEEKEND, DayType.HOLIDAY)
            else self._params.exercise_probability_weekday
        )
        if rng.random() >= prob:
            return []

        intensity = self._sample_intensity(rng)
        profile = INTENSITY_PROFILES[intensity]

        duration = int(rng.integers(profile.duration_min, profile.duration_max + 1))

        # Exercise typically happens in the morning or late afternoon
        awake_minutes = (sleep_time - wake_time).total_seconds() / 60.0
        # Choose a time window: 25-75% of the awake period
        earliest = wake_time + timedelta(minutes=awake_minutes * 0.25)
        latest = sleep_time - timedelta(minutes=awake_minutes * 0.25 + duration)
        if latest <= earliest:
            # Fallback: extend window forward, but never let the session
            # start so late that it overruns sleep_time (H3 bug #7).
            latest = min(
                earliest + timedelta(minutes=30),
                sleep_time - timedelta(minutes=duration),
            )
        if latest <= earliest:
            # Awake window too short for this duration; skip exercise today.
            return []

        offset_minutes = rng.uniform(0, (latest - earliest).total_seconds() / 60.0)
        start_time = earliest + timedelta(minutes=float(offset_minutes))

        return [
            ExerciseEvent(
                start_time=start_time,
                duration_minutes=duration,
                intensity=intensity,
                insulin_sensitivity_multiplier=profile.sensitivity_multiplier,
            )
        ]

    def _sample_intensity(self, rng: Generator) -> ExerciseIntensity:
        """Sample exercise intensity using configurable weights from ArchetypeParams."""
        intensities = [
            ExerciseIntensity.LIGHT,
            ExerciseIntensity.MODERATE,
            ExerciseIntensity.VIGOROUS,
        ]
        raw_weights = [
            self._params.exercise_intensity_light_weight,
            self._params.exercise_intensity_moderate_weight,
            self._params.exercise_intensity_vigorous_weight,
        ]
        total = sum(raw_weights)
        probs = [w / total for w in raw_weights]
        idx = int(rng.choice(len(intensities), p=probs))
        return intensities[idx]

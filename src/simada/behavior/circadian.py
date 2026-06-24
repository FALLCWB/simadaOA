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

"""Circadian rhythm model — sleep/wake cycle generation.

Produces wake and sleep times for each simulated day based on archetype
parameters and day type. Uses truncated normal distributions to avoid
physiologically impossible values (e.g. waking at 3am on a weekday).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from numpy.random import Generator
from scipy.stats import truncnorm

from simada.core.config import ArchetypeParams
from simada.core.types import DayType

# Physiological bounds (hours of day)
_WAKE_MIN_HOUR = 4.0   # no one wakes before 04:00
_WAKE_MAX_HOUR = 14.0  # no one wakes after 14:00
_SLEEP_MIN_HOUR = 20.0  # earliest sleep 20:00 (mapped to 20-30 range)
_SLEEP_MAX_HOUR = 30.0  # latest sleep 06:00 next day (24 + 6)


def _sample_truncated_normal(
    mean: float,
    std: float,
    low: float,
    high: float,
    rng: Generator,
) -> float:
    """Sample from a truncated normal distribution."""
    if std <= 0:
        return max(low, min(high, mean))
    a = (low - mean) / std
    b = (high - mean) / std
    return float(truncnorm.rvs(a, b, loc=mean, scale=std, random_state=rng))


class CircadianModel:
    """Models sleep/wake patterns with weekday/weekend variance.

    All parameters come from ArchetypeParams, making them fully
    configurable per archetype via YAML.
    """

    def __init__(self, params: ArchetypeParams) -> None:
        self._params = params

    def sample_wake_time(
        self, date: datetime, day_type: DayType, rng: Generator
    ) -> datetime:
        """Sample a wake time for the given date and day type.

        Returns a datetime on the given date with the sampled hour/minute.
        """
        if day_type in (DayType.WEEKEND, DayType.HOLIDAY):
            mean_hour = self._params.wake_time_weekend_hour
            std_minutes = self._params.wake_time_weekend_std_minutes
        else:
            mean_hour = self._params.wake_time_weekday_hour
            std_minutes = self._params.wake_time_weekday_std_minutes

        std_hours = std_minutes / 60.0
        hour = _sample_truncated_normal(
            mean_hour, std_hours, _WAKE_MIN_HOUR, _WAKE_MAX_HOUR, rng
        )
        return _hour_to_datetime(date, hour)

    def sample_sleep_time(
        self, date: datetime, day_type: DayType, rng: Generator
    ) -> datetime:
        """Sample a sleep time for the given date and day type.

        Sleep times can extend past midnight (e.g. 01:30 = next day).
        The returned datetime is always on the same date or the next date.
        """
        if day_type in (DayType.WEEKEND, DayType.HOLIDAY):
            mean_hour = self._params.sleep_time_weekend_hour
            std_minutes = self._params.sleep_time_weekend_std_minutes
        else:
            mean_hour = self._params.sleep_time_weekday_hour
            std_minutes = self._params.sleep_time_weekday_std_minutes

        # Normalize sleep hours to the 20-30 range so midnight crossing works.
        # E.g. 0.5 (00:30) becomes 24.5, 2.0 (02:00) becomes 26.0.
        #
        # Valid range for sleep_time_weekday_hour / sleep_time_weekend_hour
        # in ArchetypeParams (set via YAML):
        #   - [20.0, 23.99] → same-day sleep (20:00–23:59), stored as-is.
        #   - [0.0, 6.0]    → post-midnight sleep (00:00–06:00), stored as-is
        #                      and normalized here to [24.0, 30.0].
        # Values in (6.0, 20.0) are physiologically implausible for sleep onset
        # and will be silently mapped into the [20, 30] range by the truncnorm
        # clamp — do NOT use them.  Validation against this constraint should
        # live in ArchetypeParams (out of scope for this module; see config.py).
        if mean_hour < _SLEEP_MIN_HOUR:
            mean_hour += 24.0

        std_hours = std_minutes / 60.0
        hour = _sample_truncated_normal(
            mean_hour, std_hours, _SLEEP_MIN_HOUR, _SLEEP_MAX_HOUR, rng
        )
        return _hour_to_datetime(date, hour)

    def minimum_awake_hours(self) -> float:
        """Minimum plausible awake duration to validate wake/sleep pairs."""
        return 8.0


def _hour_to_datetime(date: datetime, hour: float) -> datetime:
    """Convert a fractional hour to a datetime on the given date.

    Hours >= 24 wrap to the next day (e.g. 25.5 = 01:30 next day).
    """
    base = datetime(date.year, date.month, date.day)
    return base + timedelta(hours=hour)

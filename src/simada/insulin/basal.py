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

"""Basal insulin rate profiles.

Models circadian variation in basal insulin needs, accounting for the
dawn phenomenon (increased insulin resistance in early morning hours
due to cortisol and growth hormone surges).

Supports both pump (programmable hourly profile) and MDI (single daily
injection of long-acting insulin).
"""

from __future__ import annotations

from datetime import datetime
from typing import NamedTuple

from simada.core.types import InsulinRegimen

# Standard circadian basal profile — multipliers of the base rate.
# These reflect physiological insulin resistance patterns over 24 hours.
# Base rate = TDI * 0.5 / 24 (50% of Total Daily Insulin as basal).
_CIRCADIAN_PROFILE: list[tuple[int, int, float]] = [
    # (start_hour, end_hour, multiplier)
    (0, 3, 0.80),    # lowest resistance (deep sleep)
    (3, 6, 1.20),    # dawn phenomenon ramp
    (6, 9, 1.30),    # dawn phenomenon peak
    (9, 12, 1.00),   # morning normalization
    (12, 15, 1.00),  # afternoon
    (15, 18, 0.90),  # late afternoon dip
    (18, 21, 1.00),  # evening
    (21, 24, 0.90),  # pre-sleep reduction
]


class BasalRateInfo(NamedTuple):
    """Basal rate at a specific time."""

    rate_u_per_hr: float
    is_dawn_phenomenon: bool


class BasalProfile:
    """Circadian basal insulin rate profile.

    For pump therapy: programmable hourly rates following the circadian
    pattern with dawn phenomenon support.

    For MDI: long-acting insulin provides a roughly flat basal rate
    with some variability depending on the insulin type.
    """

    def __init__(
        self,
        tdi: float,
        regimen: InsulinRegimen,
        dawn_phenomenon: bool = True,
        basal_fraction: float = 0.5,
    ) -> None:
        """Initialize the basal profile.

        Args:
            tdi: Total Daily Insulin in units.
            regimen: MDI or pump.
            dawn_phenomenon: Whether to model the dawn phenomenon.
            basal_fraction: Fraction of TDI delivered as basal insulin
                (default 0.5 = 50/50 split).  MDI regimens with glargine or
                detemir commonly use 0.40-0.50; pump therapy often closer to
                0.45-0.55.  BUG-FIX H5 #5: was previously hardcoded to 0.5
                with no way to override from configuration.
        """
        if not (0.0 < basal_fraction < 1.0):
            raise ValueError(
                f"basal_fraction must be in (0, 1), got {basal_fraction!r}"
            )
        self._base_rate = tdi * basal_fraction / 24.0  # U/hr
        self._regimen = regimen
        self._dawn_phenomenon = dawn_phenomenon

    @property
    def base_rate(self) -> float:
        """Base basal rate in U/hr (before circadian adjustment)."""
        return self._base_rate

    def rate_at(self, time: datetime) -> BasalRateInfo:
        """Get the basal rate at a specific time of day.

        For pumps: uses the circadian profile.
        For MDI: returns a flat rate (long-acting provides ~constant coverage).
        """
        if self._regimen == InsulinRegimen.MDI:
            return BasalRateInfo(
                rate_u_per_hr=self._base_rate,
                is_dawn_phenomenon=False,
            )

        hour = time.hour
        multiplier = 1.0
        is_dawn = False

        if self._dawn_phenomenon:
            for start, end, mult in _CIRCADIAN_PROFILE:
                if start <= hour < end:
                    multiplier = mult
                    is_dawn = 3 <= hour < 9
                    break
        # Without dawn phenomenon, use flat profile
        return BasalRateInfo(
            rate_u_per_hr=self._base_rate * multiplier,
            is_dawn_phenomenon=is_dawn,
        )

    def hourly_profile(self) -> list[tuple[int, float]]:
        """Return the full 24-hour profile as (hour, rate_u_per_hr) pairs.

        Useful for visualization and debugging.
        """
        profile = []
        for start, end, mult in _CIRCADIAN_PROFILE:
            effective = self._base_rate * (mult if self._dawn_phenomenon else 1.0)
            for h in range(start, end):
                profile.append((h, effective))
        return profile

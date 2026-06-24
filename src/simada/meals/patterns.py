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

"""Day-type modifiers for meal patterns.

Applies scaling and behavioral changes to meals based on whether the
day is a weekday, weekend, or holiday. All modifier values come from
ArchetypeParams so they can be tuned per archetype via YAML config.
"""

from __future__ import annotations

from simada.core.config import ArchetypeParams, MealConfig
from simada.core.types import DayType


class DayPatternModifier:
    """Modifies meal carb content and snacking based on day type.

    Carb scaling comes from MealConfig (shared across archetypes).
    Dessert/snack probabilities come from ArchetypeParams (per archetype).
    """

    def __init__(self, config: MealConfig, archetype_params: ArchetypeParams) -> None:
        self._weekend_factor = 1.0 + config.weekend_carb_increase_pct / 100.0
        self._holiday_factor = 1.0 + config.holiday_carb_increase_pct / 100.0
        self._params = archetype_params

    def carb_scale_factor(self, day_type: DayType) -> float:
        """Return the carb multiplier for a given day type."""
        if day_type == DayType.HOLIDAY:
            return self._holiday_factor
        if day_type == DayType.WEEKEND:
            return self._weekend_factor
        return 1.0

    def dessert_probability_boost(self, day_type: DayType) -> float:
        """Additional probability added to dessert/snack slots on special days."""
        if day_type == DayType.HOLIDAY:
            return self._params.holiday_dessert_probability_boost
        if day_type == DayType.WEEKEND:
            return self._params.weekend_dessert_probability_boost
        return 0.0

    # NOTE: extra_snack_probability() was removed — it was dead code that was
    # never called from generate_day. The actual extra-snack logic for
    # unplanned snacking lives in behavior/snacking.py (SnackGenerator).

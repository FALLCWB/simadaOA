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
"""Canonical structure of an LLM-generated week scenario -- SINGLE SOURCE OF TRUTH.

The validator (plausibility.py), the adapter (scenario_adapter.py) and the
generator prompt/filter (generate_multicountry.py) must all agree on the same
shape. Disagreement between them is what let aberrant weeks (>7 days,
``24:00`` times, exercise without a valid start) slip through one component
while being rejected by another. This module holds the shared constants and a
purely-structural validator so the three stay aligned.
"""

from __future__ import annotations

from simada.core.types import MealType

# Exactly one calendar week.
EXPECTED_DAYS = 7

# Valid clock-field ranges (used by every HH:MM parser).
HOUR_RANGE = (0, 23)
MINUTE_RANGE = (0, 59)

# Valid meal_type values (the MealType enum). Anything else is coerced to SNACK
# by the adapter; the generator should emit only these.
VALID_MEAL_TYPES = frozenset(m.value for m in MealType)

# Substrings (lowercase, accent-insensitive) identifying a "main" meal, across
# the PT enum values and common English variants. A main meal mapping to ~0
# carbs signals a food-mapping failure, not a light snack.
MAIN_MEAL_KEYS = ("cafe", "café", "breakfast", "brunch", "almoc", "almoç",
                  "lunch", "jantar", "janta", "dinner")

# Optional per-day event fields and their item field names.
EXERCISE_FIELDS = ("start_time", "duration_min", "intensity")
ALCOHOL_FIELDS = ("time", "units")


def valid_clock(hhmm: object) -> bool:
    """True if ``hhmm`` is a well-formed HH:MM with in-range hour/minute."""
    try:
        parts = str(hhmm).strip().split(":")
    except (AttributeError, TypeError):
        return False
    if len(parts) != 2:
        return False
    try:
        h, m = int(parts[0]), int(parts[1])
    except (ValueError, TypeError):
        return False
    return HOUR_RANGE[0] <= h <= HOUR_RANGE[1] and MINUTE_RANGE[0] <= m <= MINUTE_RANGE[1]


def validate_structure(week: object) -> list[str]:
    """Return a list of structural error strings (empty == structurally valid).

    Purely structural: exact 7 days, each day a dict with parseable wake/sleep,
    meals as a list of dicts with a valid time and a non-empty ``items`` list of
    {food, portion_g}. Does NOT judge plausibility (that is plausibility.py) --
    only that the shape is the canonical one the adapter can consume.
    """
    errors: list[str] = []
    if not isinstance(week, dict):
        return [f"week is not a dict: {type(week).__name__}"]
    days = week.get("days")
    if not isinstance(days, list):
        return ["'days' missing or not a list"]
    if len(days) != EXPECTED_DAYS:
        errors.append(f"expected {EXPECTED_DAYS} days, got {len(days)}")
    for di, day in enumerate(days):
        if not isinstance(day, dict):
            errors.append(f"day[{di}] is not a dict")
            continue
        for field in ("wake_time", "sleep_time"):
            if not valid_clock(day.get(field)):
                errors.append(f"day[{di}].{field} invalid: {day.get(field)!r}")
        meals = day.get("meals")
        if not isinstance(meals, list) or not meals:
            errors.append(f"day[{di}].meals missing/empty/not-a-list")
            continue
        for mi, meal in enumerate(meals):
            if not isinstance(meal, dict):
                errors.append(f"day[{di}].meals[{mi}] is not a dict")
                continue
            if not valid_clock(meal.get("time")):
                errors.append(f"day[{di}].meals[{mi}].time invalid: {meal.get('time')!r}")
            items = meal.get("items")
            if not isinstance(items, list) or not items:
                errors.append(f"day[{di}].meals[{mi}].items missing/empty/not-a-list")
                continue
            for ii, item in enumerate(items):
                if not isinstance(item, dict) or "food" not in item or "portion_g" not in item:
                    errors.append(f"day[{di}].meals[{mi}].items[{ii}] needs food+portion_g")
        # exercise/alcohol are optional; if present must be lists of dicts with
        # a valid start time.
        for ev_field, time_key in (("exercise", "start_time"), ("alcohol", "time")):
            ev = day.get(ev_field)
            if ev is None:
                continue
            if not isinstance(ev, list):
                errors.append(f"day[{di}].{ev_field} is not a list")
                continue
            for ei, entry in enumerate(ev):
                if not isinstance(entry, dict):
                    errors.append(f"day[{di}].{ev_field}[{ei}] is not a dict")
                elif not valid_clock(entry.get(time_key)):
                    errors.append(f"day[{di}].{ev_field}[{ei}].{time_key} invalid: {entry.get(time_key)!r}")
    return errors

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

"""Brazilian meal templates — defines meal structure as food slots.

Each MealTemplate describes a typical Brazilian meal as a collection of
FoodSlots. The generator samples concrete foods from the TACO database
to fill each slot, producing realistic meal compositions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from simada.core.types import MealType


@dataclass(frozen=True)
class FoodSlot:
    """A slot within a meal template that will be filled with a concrete food.

    Attributes:
        category: TACO food category to sample from.
        probability: Chance this slot is included in the meal [0, 1].
        servings_min: Minimum number of typical servings.
        servings_max: Maximum number of typical servings.
        preferred: Optional list of preferred food names (nome_pt) that
            receive higher sampling weight.
    """

    category: str
    probability: float
    servings_min: float
    servings_max: float
    preferred: tuple[str, ...] = ()


@dataclass(frozen=True)
class MealTemplate:
    """A meal defined as a collection of food slots with a target carb range.

    Attributes:
        name: Template identifier (e.g. "almoco_weekday").
        meal_type: Which meal this represents (cafe_da_manha, almoco, etc.).
        slots: Food slots that compose this meal.
        total_carb_range: Target (min, max) total carbs in grams.
        offset_from_wake_minutes: How many minutes after wake time this
            meal is typically eaten. Mutually exclusive with
            offset_before_sleep_minutes.
        offset_before_sleep_minutes: How many minutes before sleep time.
            Used for ceia (late snack).
        probability: Probability that this meal occurs at all [0, 1].
            Main meals (cafe, almoco, jantar) default to 1.0.
    """

    name: str
    meal_type: MealType
    slots: tuple[FoodSlot, ...]
    total_carb_range: tuple[float, float]
    offset_from_wake_minutes: int | None = None
    offset_before_sleep_minutes: int | None = None
    probability: float = 1.0
    day_of_week: str | None = None


@dataclass
class DayMealPlan:
    """Collection of meal templates for a full day.

    Attributes:
        templates: Ordered list of meal templates for this day type.
        replacements: Templates that replace standard meals conditionally
            (e.g. churrasco replaces almoco on Sundays).
    """

    templates: list[MealTemplate] = field(default_factory=list)
    replacements: list[tuple[MealTemplate, list[MealType]]] = field(default_factory=list)


def _parse_slot(raw: dict[str, Any]) -> FoodSlot:
    """Parse a single food slot from YAML dict."""
    servings = raw.get("servings", [1, 1])
    preferred_raw = raw.get("preferred", [])
    return FoodSlot(
        category=raw["category"],
        probability=float(raw.get("probability", 1.0)),
        servings_min=float(servings[0]),
        servings_max=float(servings[1]),
        preferred=tuple(preferred_raw),
    )


def _parse_meal_template(name: str, raw: dict[str, Any]) -> MealTemplate:
    """Parse a meal template from YAML dict."""
    meal_type_str = raw.get("meal_type", name)
    try:
        meal_type = MealType(meal_type_str)
    except ValueError:
        meal_type = MealType.SNACK

    carb_range = raw.get("total_carb_range", [30, 60])
    slots = tuple(_parse_slot(s) for s in raw.get("slots", []))

    return MealTemplate(
        name=name,
        meal_type=meal_type,
        slots=slots,
        total_carb_range=(float(carb_range[0]), float(carb_range[1])),
        offset_from_wake_minutes=raw.get("offset_from_wake_minutes"),
        offset_before_sleep_minutes=raw.get("offset_before_sleep_minutes"),
        probability=float(raw.get("probability", 1.0)),
        day_of_week=raw.get("day_of_week"),
    )


# Mapping from YAML template keys to MealType for standard meals
_STANDARD_MEAL_KEYS: dict[str, MealType] = {
    "cafe_da_manha": MealType.CAFE_DA_MANHA,
    "lanche_manha": MealType.LANCHE_MANHA,
    "almoco": MealType.ALMOCO,
    "lanche_tarde": MealType.LANCHE_TARDE,
    "jantar": MealType.JANTAR,
    "ceia": MealType.CEIA,
    "brunch": MealType.BRUNCH,
    "churrasco": MealType.CHURRASCO,
}


# Mapping from a config ``meals.locale`` value to the meal-template file prefix.
# Multiple spellings are accepted so configs can use natural names.
_LOCALE_PREFIX: dict[str, str] = {
    "brazil": "brazilian",
    "brazilian": "brazilian",
    "br": "brazilian",
    "usa": "us",
    "us": "us",
    "united_states": "us",
    "japan": "japan",
    "jp": "japan",
}


def load_locale_day_plans(
    locale: str, meals_dir: Path
) -> tuple[DayMealPlan, DayMealPlan, DayMealPlan]:
    """Load (weekday, weekend, holiday) meal plans for a given locale.

    The locale selects a file prefix (e.g. ``brazil`` -> ``brazilian_*.yaml``,
    ``usa`` -> ``us_*.yaml``, ``japan`` -> ``japan_*.yaml``). The holiday plan
    falls back to the weekend plan when no locale-specific holiday file exists.

    Country meal *structure* (which national dishes appear at each meal and in
    what carb range) is held in these templates; the per-country food
    composition (carbs, fibre, GI, protein, fat) comes from the matching food
    table. This is how a single rule-based engine produces Brazilian, US, and
    Japanese dietary patterns. The US and Japanese templates are
    author-constructed representative national patterns (see ASSUMPTIONS.md):
    they are NOT validated against national dietary-survey microdata, which is
    a documented limitation.
    """
    prefix = _LOCALE_PREFIX.get(locale.strip().lower())
    if prefix is None:
        msg = (
            f"Unknown meals.locale {locale!r}. Known locales: "
            f"{sorted(set(_LOCALE_PREFIX))}."
        )
        raise ValueError(msg)

    weekday = load_day_meal_plan(meals_dir / f"{prefix}_weekday.yaml")
    weekend = load_day_meal_plan(meals_dir / f"{prefix}_weekend.yaml")
    holiday_path = meals_dir / f"{prefix}_holiday.yaml"
    holiday = load_day_meal_plan(holiday_path) if holiday_path.exists() else weekend
    return weekday, weekend, holiday


def load_day_meal_plan(yaml_path: Path) -> DayMealPlan:
    """Load a day's meal plan from a YAML config file.

    The YAML file should have top-level keys matching meal names
    (cafe_da_manha, almoco, jantar, etc.) with their slot definitions.
    """
    with open(yaml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    plan = DayMealPlan()
    replacements: list[tuple[MealTemplate, list[MealType]]] = []

    for key, data in raw.items():
        if not isinstance(data, dict):
            continue

        # Determine meal_type from key or explicit field
        if key in _STANDARD_MEAL_KEYS:
            data.setdefault("meal_type", key)
        elif "meal_type" not in data:
            data["meal_type"] = "snack"

        template = _parse_meal_template(key, data)

        # Check if this is a replacement (e.g. churrasco replaces almoco)
        replaces_raw = data.get("replaces", [])
        if replaces_raw:
            replaced_types = []
            for r in replaces_raw:
                try:
                    replaced_types.append(MealType(r))
                except ValueError:
                    pass
            replacements.append((template, replaced_types))
        else:
            plan.templates.append(template)

    # Sort templates by offset time (wake-anchored first, then sleep-anchored)
    plan.templates.sort(
        key=lambda t: (
            t.offset_from_wake_minutes if t.offset_from_wake_minutes is not None else 9999,
        )
    )
    plan.replacements = replacements
    return plan

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
"""Adapt an LLM-generated week scenario (JSON) into simada DailySchedules.

The LLM produces the cultural/behavioral layer (Brazilian dishes, portions,
timing, exercise, alcohol). This module maps each dish to a TACO entry,
computes ``true_carbs_g`` from the portion, derives ``estimated_carbs_g`` via
the archetype's carb-estimation error, and assembles ``DailySchedule`` objects
that feed the SAME downstream (SimadaScenario -> simglucose) as the
rule-based generator. The "with-LLM" path differs from the "rule-based"
path only in the source of the meals/behavior.

LLM output is untrusted: every field may be missing, null, mistyped, or out
of range. The adapter therefore NEVER raises on malformed input -- it degrades
gracefully (skips the malformed fragment or falls back to a safe default).
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from simada.behavior.exercise import INTENSITY_PROFILES
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
from simada.meals.estimation import CarbEstimationModel

if TYPE_CHECKING:
    from numpy.random import Generator

    from simada.llm.taco_mapping import TACOMapper
    from simada.patient.archetype import ArchetypeParams

_MEAL_TYPES = {m.value: m for m in MealType}
_INTENSITIES = {i.value: i for i in ExerciseIntensity}

# Physiologically sane bounds for a single food portion (grams).
# Adapter floor is 0 (a negative/garbage portion contributes nothing); it is NOT
# raised to the validator's 1 g hard floor on purpose -- the validator is the
# authority that flags sub-gram portions, while the adapter must never turn a
# negative portion into a positive carb contribution.
_PORTION_MIN_G = 0.0
_PORTION_MAX_G = 1500.0
_MAX_DAY_INDEX = 60          # week horizon; guards timedelta overflow
_MAX_EXERCISE_MIN = 480      # upper clamp for exercise duration (8h)

# Alcohol effect parameters -- BIPHASIC, mirroring behavior/stress.py:
#   Phase 1 (0-3h): mild insulin resistance / hyperglycaemia (factor > 1).
#   Phase 2 (3h-...): hepatic glucose suppression / delayed hypoglycaemia
#                     (factor < 1), the dominant clinical risk.
# Convention (see stress.py): insulin_resistance_factor > 1 = resistance,
# < 1 = hypersensitivity. Magnitude/duration of Phase 2 scale with the number
# of standard drinks (units), clamped to plausible bounds.
_ALCOHOL_PHASE1_MIN = 180
_ALCOHOL_PHASE1_RESISTANCE = 1.2
_ALCOHOL_PHASE2_SENS_BASE = 0.9      # 1 drink
_ALCOHOL_PHASE2_SENS_PER_UNIT = 0.04  # each extra drink deepens the hypo
_ALCOHOL_PHASE2_SENS_FLOOR = 0.7
_ALCOHOL_PHASE2_MIN_BASE = 300       # 5h for 1 drink
_ALCOHOL_PHASE2_MIN_PER_UNIT = 40
_ALCOHOL_PHASE2_MIN_CAP = 540        # 9h
_ALCOHOL_UNITS_MAX = 12


@dataclass
class MappingReport:
    """Diagnostics on how well LLM foods mapped to TACO."""

    total_items: int = 0
    exact: int = 0
    token: int = 0
    fuzzy: int = 0
    unmatched: int = 0
    unmatched_names: list[str] = field(default_factory=list)

    @property
    def mapped_rate(self) -> float:
        if self.total_items == 0:
            return 1.0
        return (self.total_items - self.unmatched) / self.total_items


def _normalize_label(value: object) -> str:
    """Lowercase, strip, and remove accents so enum lookup is forgiving.

    "Almoço" -> "almoco", " Vigorous " -> "vigorous",
    "Café da manhã" -> "cafe_da_manha" (spaces become underscores).
    """
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKD", value)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", "_", text.strip().lower())


def _parse_hhmm(date: datetime, hhmm: object) -> datetime | None:
    """Parse an "HH:MM" (or "HH:MM:SS") string into a datetime on ``date``.

    Returns None on any malformed input ("25:00", "12:99", null, garbage)
    instead of crashing or silently wrapping with modular arithmetic.
    """
    if not isinstance(hhmm, str):
        return None
    parts = hhmm.strip().split(":")
    if len(parts) < 2:
        return None
    try:
        h = int(parts[0])
        m = int(parts[1])
    except (ValueError, TypeError):
        return None
    if not (0 <= h <= 23 and 0 <= m <= 59):
        return None
    return date.replace(hour=h, minute=m, second=0, microsecond=0)


def _time_or_default(date: datetime, hhmm: object, default_hhmm: str) -> datetime:
    """Parse an LLM time field, falling back to a safe default on failure."""
    parsed = _parse_hhmm(date, hhmm)
    if parsed is not None:
        return parsed
    fallback = _parse_hhmm(date, default_hhmm)
    assert fallback is not None  # defaults are always valid literals
    return fallback


def _safe_float(value: object, default: float) -> float:
    """Coerce an LLM numeric field to float, falling back on failure.

    json.loads accepts NaN/Infinity/-Infinity by default; those must NOT leak
    into the simulation, so non-finite values fall back to ``default``.
    """
    if isinstance(value, bool):  # bool is an int subclass; reject explicitly
        return default
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return default
    return f if math.isfinite(f) else default


def _safe_int(value: object, default: int) -> int:
    """Coerce an LLM numeric field to int, falling back on failure.

    Rejects NaN/Infinity (non-finite) before ``int()`` and catches
    OverflowError (e.g. ``int(float('inf'))``).
    """
    if isinstance(value, bool):
        return default
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(f):
        return default
    try:
        return int(f)
    except (TypeError, ValueError, OverflowError):
        return default


# String spellings that mean False. Without this, ``bool("false")`` is True
# and a day the LLM marked ``is_weekend: "false"`` would be simulated as a
# weekend, shifting its whole meal/behavior pattern.
_FALSE_STRINGS = frozenset({"false", "0", "no", "nao", "não", "off", "", "none", "null"})


def _safe_bool(value: object, default: bool = False) -> bool:
    """Coerce an LLM boolean field, treating "false"/"0"/"no" as False.

    JSON booleans pass through; strings are matched case-insensitively against
    known falsey spellings (so the truthiness of a non-empty string never
    silently flips a weekday into a weekend). Unknown types fall back.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in _FALSE_STRINGS
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _as_dict_list(value: object) -> list[dict[str, Any]]:
    """Return only the dict entries of a value the LLM claimed was a list.

    Handles null (-> []), a bare string (-> []), and lists polluted with
    strings or other non-dict entries (filtered out).
    """
    if not isinstance(value, list):
        return []
    return [entry for entry in value if isinstance(entry, dict)]


def _ordered_unique_days(week: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    """Extract (day_index, day) pairs: sorted by index, first occurrence wins.

    A non-int day_index falls back to the day's position in the list.
    """
    raw_days = week.get("days") or []
    if not isinstance(raw_days, list):
        return []
    # Two passes so a day with a bad/unparseable day_index (e.g. "Monday") does
    # NOT claim a position slot and silently evict a later day that carries that
    # index explicitly. Pass 1: keep explicitly-indexed days (first wins on
    # duplicates). Pass 2: assign bad-index days to the next FREE slot only.
    explicit: dict[int, dict[str, Any]] = {}
    fallbacks: list[dict[str, Any]] = []
    for day in raw_days:
        if not isinstance(day, dict):
            continue
        idx = _safe_int(day.get("day_index"), -1)
        if 0 <= idx <= _MAX_DAY_INDEX:
            explicit.setdefault(idx, day)  # first occurrence wins
        else:
            fallbacks.append(day)
    indexed: list[tuple[int, dict[str, Any]]] = list(explicit.items())
    used = set(explicit)
    next_free = 0
    for day in fallbacks:
        while next_free in used:
            next_free += 1
        used.add(next_free)
        indexed.append((next_free, day))
    indexed.sort(key=lambda pair: pair[0])
    return indexed


def _build_meals(
    day: dict[str, Any],
    date: datetime,
    mapper: TACOMapper,
    estimator: CarbEstimationModel,
    rng: Generator,
    report: MappingReport,
) -> list[MealEvent]:
    meals: list[MealEvent] = []
    for meal in _as_dict_list(day.get("meals")):
        true_carbs = 0.0
        gi_weighted = 0.0
        foods: list[str] = []
        items = meal.get("items")
        if not isinstance(items, list):
            items = []  # LLM sometimes flattens items into a single string
        for item in items:
            if not isinstance(item, dict):
                continue  # bare string or other junk entry; skip
            report.total_items += 1
            food_name = item.get("food")
            if not isinstance(food_name, str) or not food_name.strip():
                report.unmatched += 1  # null/non-string food: never str() it
                continue
            m = mapper.match(food_name)
            if m.food is None:
                report.unmatched += 1
                report.unmatched_names.append(food_name)
                continue
            # m.method is "exact", "token", or "fuzzy" here (unmatched is
            # handled above). Increment the matching typed counter explicitly
            # rather than poking report.__dict__ by the method string.
            if m.method == "exact":
                report.exact += 1
            elif m.method == "token":
                report.token += 1
            else:
                report.fuzzy += 1
            portion = _safe_float(item.get("portion_g"), m.food.porcao_tipica_g)
            portion = min(max(portion, _PORTION_MIN_G), _PORTION_MAX_G)
            c = m.food.carbs_per_100g * portion / 100.0
            true_carbs += c
            gi_weighted += m.food.indice_glicemico * c
            foods.append(m.food.nome_pt)

        if true_carbs <= 0:
            continue  # skip meals with no mappable carbs (e.g. all-protein)

        gi = gi_weighted / true_carbs
        meal_type = _MEAL_TYPES.get(
            _normalize_label(meal.get("meal_type")), MealType.SNACK
        )
        meals.append(
            MealEvent(
                time=_time_or_default(date, meal.get("time"), "12:00"),
                meal_type=meal_type,
                true_carbs_g=true_carbs,
                estimated_carbs_g=estimator.estimate(true_carbs, rng),
                foods=tuple(foods),
                glycemic_index=gi,
            )
        )
    meals.sort(key=lambda m: m.time)
    return meals


def _build_exercises(day: dict[str, Any], date: datetime) -> list[ExerciseEvent]:
    exercises: list[ExerciseEvent] = []
    for ex in _as_dict_list(day.get("exercise")):
        inten = _INTENSITIES.get(
            _normalize_label(ex.get("intensity")), ExerciseIntensity.MODERATE
        )
        prof = INTENSITY_PROFILES[inten]
        duration_raw = ex.get("duration_min")
        if duration_raw is None:
            duration_raw = ex.get("duration_minutes")
        duration = min(max(0, _safe_int(duration_raw, 30)), _MAX_EXERCISE_MIN)
        exercises.append(
            ExerciseEvent(
                start_time=_time_or_default(date, ex.get("start_time"), "18:00"),
                duration_minutes=duration,
                intensity=inten,
                insulin_sensitivity_multiplier=prof.sensitivity_multiplier,
            )
        )
    exercises.sort(key=lambda e: e.start_time)
    return exercises


def _build_alcohol_events(day: dict[str, Any], date: datetime) -> list[StressEvent]:
    """Map day["alcohol"] entries ({"time","units"}) to BIPHASIC ALCOHOL events.

    Each drinking episode becomes TWO StressEvents mirroring the validated
    rule-based model (behavior/stress.py): a Phase 1 (0-3h) mild resistance
    (hyperglycaemia) and a Phase 2 (3h+) hepatic suppression (delayed
    hypoglycaemia), whose depth and duration scale with the number of standard
    drinks. A missing/invalid time degrades to 20:00; missing units -> 1.
    """
    events: list[StressEvent] = []
    for alc in _as_dict_list(day.get("alcohol")):
        start = _time_or_default(date, alc.get("time"), "20:00")
        units = min(max(1, _safe_int(alc.get("units"), 1)), _ALCOHOL_UNITS_MAX)
        # Phase 1: mild resistance for 3h
        events.append(
            StressEvent(
                start_time=start,
                duration_minutes=_ALCOHOL_PHASE1_MIN,
                stress_type=StressType.ALCOHOL,
                insulin_resistance_factor=_ALCOHOL_PHASE1_RESISTANCE,
            )
        )
        # Phase 2: delayed hypo, deeper/longer with more drinks
        sens = max(_ALCOHOL_PHASE2_SENS_FLOOR,
                   _ALCOHOL_PHASE2_SENS_BASE - _ALCOHOL_PHASE2_SENS_PER_UNIT * (units - 1))
        dur = min(_ALCOHOL_PHASE2_MIN_CAP,
                  _ALCOHOL_PHASE2_MIN_BASE + _ALCOHOL_PHASE2_MIN_PER_UNIT * (units - 1))
        events.append(
            StressEvent(
                start_time=start + timedelta(minutes=_ALCOHOL_PHASE1_MIN),
                duration_minutes=dur,
                stress_type=StressType.ALCOHOL,
                insulin_resistance_factor=round(sens, 3),
            )
        )
    events.sort(key=lambda e: e.start_time)
    return events


def build_schedules_from_llm(
    week: dict[str, Any],
    base_date: datetime,
    archetype_params: ArchetypeParams,
    mapper: TACOMapper,
    rng: Generator,
) -> tuple[list[DailySchedule], MappingReport]:
    """Convert an LLM week scenario into DailySchedules + a mapping report.

    Never raises on malformed LLM output: malformed days/meals/items are
    skipped and malformed scalar fields fall back to safe defaults.
    """
    estimator = CarbEstimationModel(archetype_params)
    report = MappingReport()
    schedules: list[DailySchedule] = []

    for idx, day in _ordered_unique_days(week if isinstance(week, dict) else {}):
        date = (base_date + timedelta(days=idx)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        day_type = DayType.WEEKEND if _safe_bool(day.get("is_weekend")) else DayType.WEEKDAY
        wake = _time_or_default(date, day.get("wake_time"), "07:00")
        sleep = _time_or_default(date, day.get("sleep_time"), "23:00")
        if sleep <= wake:
            # Post-midnight bedtime ("01:00" with a morning wake): the LLM
            # means 01:00 of the NEXT day, not 1 AM before waking up.
            sleep += timedelta(days=1)

        schedules.append(
            DailySchedule(
                date=date,
                day_type=day_type,
                wake_time=wake,
                sleep_time=sleep,
                meals=_build_meals(day, date, mapper, estimator, rng, report),
                exercise_events=_build_exercises(day, date),
                stress_events=_build_alcohol_events(day, date),
            )
        )

    return schedules, report

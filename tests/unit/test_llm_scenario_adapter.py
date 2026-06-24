# simada -- Simulation of AID Adherence
# Copyright (C) 2026 Dr. Filipe Augusto da Luz Lemos, MSc. Ph.D.
# Contact: filipellemos@gmail.com | filipe@falleng.com.br | fadaluzl@syr.edu
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
"""Tests for the LLM week-scenario -> DailySchedule adapter."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from simada.core.types import DayType, ExerciseIntensity, MealType, StressType
from simada.llm.scenario_adapter import _parse_hhmm, build_schedules_from_llm
from simada.llm.taco_mapping import TACOMapper
from simada.meals.taco import TACODatabase

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TACO_CSV = PROJECT_ROOT / "data" / "taco" / "taco_foods.csv"


@pytest.fixture
def taco() -> TACODatabase:
    return TACODatabase(TACO_CSV)


@pytest.fixture
def mapper(taco: TACODatabase) -> TACOMapper:
    return TACOMapper(taco)


WEEK = {
    "scenario_id": "test",
    "metadata": {"archetype": "moderate", "season": "verao", "region": "sul"},
    "days": [
        {
            "day_index": 0, "day_of_week": "segunda", "is_weekend": False,
            "wake_time": "06:30", "sleep_time": "23:00",
            "meals": [
                {"meal_type": "cafe_da_manha", "time": "07:00",
                 "items": [{"food": "Pão de queijo", "portion_g": 100},
                           {"food": "Café com leite e açúcar", "portion_g": 200}]},
                {"meal_type": "almoco", "time": "12:30",
                 "items": [{"food": "Arroz branco cozido", "portion_g": 150},
                           {"food": "Feijao carioca", "portion_g": 100},
                           {"food": "Frango grelhado", "portion_g": 120}]},
            ],
            "exercise": [{"start_time": "18:00", "duration_min": 40, "intensity": "moderate", "activity": "caminhada"}],
        },
        {
            "day_index": 1, "day_of_week": "terca", "is_weekend": False,
            "wake_time": "06:30", "sleep_time": "23:00",
            "meals": [
                {"meal_type": "jantar", "time": "20:00",
                 "items": [{"food": "Croissant inventado", "portion_g": 80}]},
            ],
        },
    ],
}


def test_builds_seven_day_compatible_schedules(moderate_params, mapper, taco):
    rng = np.random.default_rng(42)
    base = datetime(2026, 6, 1)
    schedules, report = build_schedules_from_llm(WEEK, base, moderate_params, mapper, rng)

    assert len(schedules) == 2
    assert schedules[0].day_type == DayType.WEEKDAY
    assert schedules[0].wake_time.hour == 6 and schedules[0].wake_time.minute == 30
    # day 0 has 2 meals (cafe + almoco), both have carbs
    assert len(schedules[0].meals) == 2
    cafe = schedules[0].meals[0]
    assert cafe.meal_type == MealType.CAFE_DA_MANHA


def test_true_carbs_match_taco_computation(moderate_params, mapper, taco):
    rng = np.random.default_rng(0)
    schedules, _ = build_schedules_from_llm(WEEK, datetime(2026, 6, 1), moderate_params, mapper, rng)
    # almoco: arroz 150g + feijao 100g (+ frango 0 carb). Compute expected.
    arroz = taco.by_name("Arroz branco cozido")
    feijao = taco.by_name("Feijao carioca cozido")
    expected = arroz.carbs_per_100g * 150 / 100 + feijao.carbs_per_100g * 100 / 100
    almoco = schedules[0].meals[1]
    assert almoco.true_carbs_g == pytest.approx(expected, rel=1e-6)
    assert almoco.true_carbs_g > 0


def test_estimated_differs_from_true(moderate_params, mapper):
    rng = np.random.default_rng(7)
    schedules, _ = build_schedules_from_llm(WEEK, datetime(2026, 6, 1), moderate_params, mapper, rng)
    almoco = schedules[0].meals[1]
    # estimation applies archetype error -> generally different from true
    assert almoco.estimated_carbs_g != almoco.true_carbs_g


def test_unmatched_food_dropped_and_reported(moderate_params, mapper):
    rng = np.random.default_rng(1)
    schedules, report = build_schedules_from_llm(WEEK, datetime(2026, 6, 1), moderate_params, mapper, rng)
    # day 1 meal has only an invented food -> 0 carbs -> meal dropped
    assert len(schedules[1].meals) == 0
    assert report.unmatched >= 1
    assert any("Croissant" in n for n in report.unmatched_names)
    # overall mapping rate should still be high (the rest mapped)
    assert report.mapped_rate >= 0.7


def test_exercise_mapped(moderate_params, mapper):
    rng = np.random.default_rng(3)
    schedules, _ = build_schedules_from_llm(WEEK, datetime(2026, 6, 1), moderate_params, mapper, rng)
    assert len(schedules[0].exercise_events) == 1
    ex = schedules[0].exercise_events[0]
    assert ex.duration_minutes == 40


# ---------------------------------------------------------------------------
# Robustness against malformed LLM output (H1 bughunt)
# ---------------------------------------------------------------------------

BASE = datetime(2026, 6, 1)


def _build(week, params, mapper, seed=0):
    return build_schedules_from_llm(week, BASE, params, mapper, np.random.default_rng(seed))


def _day(**overrides):
    day = {
        "day_index": 0, "is_weekend": False,
        "wake_time": "07:00", "sleep_time": "23:00",
        "meals": [
            {"meal_type": "almoco", "time": "12:30",
             "items": [{"food": "Arroz branco cozido", "portion_g": 150}]},
        ],
    }
    day.update(overrides)
    return day


def test_items_as_bare_string_does_not_crash(moderate_params, mapper):
    week = {"days": [_day(meals=[{"meal_type": "almoco", "time": "12:00",
                                  "items": "arroz e feijao"}])]}
    schedules, report = _build(week, moderate_params, mapper)
    assert len(schedules) == 1
    assert schedules[0].meals == []  # no parseable items -> meal dropped
    assert report.total_items == 0


def test_items_as_list_of_strings_skips_non_dicts(moderate_params, mapper):
    week = {"days": [_day(meals=[{"meal_type": "almoco", "time": "12:00",
                                  "items": ["arroz", "feijao",
                                            {"food": "Arroz branco cozido", "portion_g": 100}]}])]}
    schedules, report = _build(week, moderate_params, mapper)
    # the two bare strings are ignored; the dict item still maps
    assert len(schedules[0].meals) == 1
    assert report.total_items == 1


def test_null_days_and_null_meals_and_string_meals(moderate_params, mapper):
    schedules, _ = _build({"days": None}, moderate_params, mapper)
    assert schedules == []

    schedules, _ = _build({"days": [_day(meals=None)]}, moderate_params, mapper)
    assert len(schedules) == 1 and schedules[0].meals == []

    schedules, _ = _build({"days": [_day(meals="3 refeicoes")]}, moderate_params, mapper)
    assert len(schedules) == 1 and schedules[0].meals == []


def test_portion_null_string_negative_and_huge(moderate_params, mapper, taco):
    arroz = taco.by_name("Arroz branco cozido")
    week = {"days": [_day(meals=[
        # null portion -> falls back to TACO typical portion
        {"meal_type": "almoco", "time": "12:00",
         "items": [{"food": "Arroz branco cozido", "portion_g": None}]},
        # "150g" string -> unparseable -> typical portion (not a crash)
        {"meal_type": "jantar", "time": "19:00",
         "items": [{"food": "Arroz branco cozido", "portion_g": "150g"}]},
        # negative -> clamped to 0 (must not subtract carbs from the meal)
        {"meal_type": "lanche_tarde", "time": "16:00",
         "items": [{"food": "Arroz branco cozido", "portion_g": 100},
                   {"food": "Feijao carioca", "portion_g": -500}]},
        # absurdly huge -> clamped to 1500 g
        {"meal_type": "ceia", "time": "22:00",
         "items": [{"food": "Arroz branco cozido", "portion_g": 99999}]},
    ])]}
    schedules, _ = _build(week, moderate_params, mapper)
    meals = {m.meal_type: m for m in schedules[0].meals}
    assert len(meals) == 4
    typical_carbs = arroz.carbs_per_100g * arroz.porcao_tipica_g / 100.0
    assert meals[MealType.ALMOCO].true_carbs_g == pytest.approx(typical_carbs)
    assert meals[MealType.JANTAR].true_carbs_g == pytest.approx(typical_carbs)
    # negative feijao contributes exactly 0, arroz 100 g remains
    assert meals[MealType.LANCHE_TARDE].true_carbs_g == pytest.approx(arroz.carbs_per_100g)
    assert meals[MealType.CEIA].true_carbs_g == pytest.approx(arroz.carbs_per_100g * 1500 / 100.0)


def test_parse_hhmm_rejects_invalid_without_wrapping():
    date = datetime(2026, 6, 1)
    assert _parse_hhmm(date, "25:00") is None  # must NOT wrap to 01:00
    assert _parse_hhmm(date, "12:99") is None  # must NOT wrap to 12:39
    assert _parse_hhmm(date, None) is None
    assert _parse_hhmm(date, "meio-dia") is None
    assert _parse_hhmm(date, 1230) is None
    ok = _parse_hhmm(date, "12:00:00")  # HH:MM:SS accepted, seconds dropped
    assert ok is not None and ok.hour == 12 and ok.minute == 0


def test_invalid_times_fall_back_to_safe_defaults(moderate_params, mapper):
    week = {"days": [_day(
        wake_time="25:00", sleep_time=None,
        meals=[{"meal_type": "almoco", "time": "12:99",
                "items": [{"food": "Arroz branco cozido", "portion_g": 150}]}],
    )]}
    schedules, _ = _build(week, moderate_params, mapper)
    day = schedules[0]
    assert day.wake_time.hour == 7  # fallback, not 01:00 from %24
    assert day.sleep_time.hour == 23
    assert day.meals[0].time.hour == 12 and day.meals[0].time.minute == 0


def test_sleep_after_midnight_rolls_to_next_day(moderate_params, mapper):
    week = {"days": [_day(wake_time="09:00", sleep_time="01:00", meals=[])]}
    schedules, _ = _build(week, moderate_params, mapper)
    day = schedules[0]
    assert day.sleep_time == datetime(2026, 6, 2, 1, 0)  # next day, not +14h
    assert day.sleep_time > day.wake_time


def test_meal_type_accented_and_intensity_capitalized(moderate_params, mapper):
    week = {"days": [_day(
        meals=[{"meal_type": "Almoço", "time": "12:30",
                "items": [{"food": "Arroz branco cozido", "portion_g": 150}]}],
        exercise=[{"start_time": "18:00", "duration_minutes": 45,
                   "intensity": " Vigorous ", "activity": "corrida"}],
    )]}
    schedules, _ = _build(week, moderate_params, mapper)
    assert schedules[0].meals[0].meal_type == MealType.ALMOCO  # not SNACK
    ex = schedules[0].exercise_events[0]
    assert ex.intensity == ExerciseIntensity.VIGOROUS  # not MODERATE default
    assert ex.duration_minutes == 45  # duration_minutes key accepted


def test_alcohol_becomes_biphasic_stress_events(moderate_params, mapper):
    week = {"days": [_day(alcohol=[{"time": "21:30", "units": 3}])]}
    schedules, _ = _build(week, moderate_params, mapper)
    events = schedules[0].stress_events
    assert len(events) == 2  # biphasic: phase1 (hyper) + phase2 (delayed hypo)
    phase1, phase2 = events  # sorted by start_time
    assert all(e.stress_type == StressType.ALCOHOL for e in events)
    # Phase 1: start at drink time, 3h, mild resistance (>1)
    assert phase1.start_time == datetime(2026, 6, 1, 21, 30)
    assert phase1.duration_minutes == 180
    assert phase1.insulin_resistance_factor > 1.0
    # Phase 2: starts 3h later, hypersensitivity (<1), scaled by units=3
    assert phase2.start_time == datetime(2026, 6, 2, 0, 30)
    assert phase2.insulin_resistance_factor < 1.0
    assert phase2.duration_minutes >= 300


def test_alcohol_units_deepen_phase2(moderate_params, mapper):
    # more drinks -> deeper phase-2 hypo (lower factor) and longer duration
    def p2(units):
        wk = {"days": [_day(alcohol=[{"time": "20:00", "units": units}])]}
        ev = _build(wk, moderate_params, mapper)[0][0].stress_events
        return ev[1]  # phase 2
    light, heavy = p2(1), p2(6)
    assert heavy.insulin_resistance_factor < light.insulin_resistance_factor
    assert heavy.duration_minutes > light.duration_minutes


def test_alcohol_malformed_entries_degrade(moderate_params, mapper):
    week = {"days": [_day(alcohol=["cerveja", {"time": "25:00", "units": 2}, None])]}
    schedules, _ = _build(week, moderate_params, mapper)
    events = schedules[0].stress_events
    # only the dict entry survives -> 2 biphasic events; bad time -> 20:00 default
    assert len(events) == 2
    assert events[0].start_time.hour == 20


def test_day_index_sorted_deduplicated_and_non_int(moderate_params, mapper):
    week = {"days": [
        _day(day_index="2", wake_time="08:00", meals=[]),
        _day(day_index=0, wake_time="06:00", meals=[]),
        _day(day_index=0, wake_time="05:00", meals=[]),  # duplicate: dropped
        _day(day_index="abc", wake_time="10:00", meals=[]),  # bad index -> free slot
        "not a day",  # junk entry: skipped
    ]}
    schedules, _ = _build(week, moderate_params, mapper)
    assert len(schedules) == 3
    # explicit indices 0 and 2 are kept; the bad-index "abc" day takes the next
    # FREE slot (1) rather than claiming a position and evicting an explicit day.
    assert [s.date for s in schedules] == [
        BASE, BASE + (datetime(2026, 6, 2) - BASE), BASE + (datetime(2026, 6, 3) - BASE)
    ]
    assert schedules[0].wake_time.hour == 6   # index 0, first occurrence wins
    assert schedules[1].wake_time.hour == 10  # "abc" -> free slot 1
    assert schedules[2].wake_time.hour == 8   # "2" coerced to 2


def test_food_none_is_not_stringified(moderate_params, mapper):
    week = {"days": [_day(meals=[
        {"meal_type": "almoco", "time": "12:00",
         "items": [{"food": None, "portion_g": 100},
                   {"food": "Arroz branco cozido", "portion_g": 100}]},
    ])]}
    schedules, report = _build(week, moderate_params, mapper)
    assert "None" not in report.unmatched_names
    assert report.unmatched == 1  # the null-food item counts as unmatched
    assert len(schedules[0].meals) == 1  # the valid item still builds the meal


# --- T4: never-raises vs Infinity/NaN/overflow (json.loads accepts these) ---

import math as _math  # noqa: E402

from simada.llm.scenario_adapter import _MAX_DAY_INDEX, _MAX_EXERCISE_MIN  # noqa: E402


def _one_day_week(day):
    return {"metadata": {"archetype": "moderate"}, "days": [day]}


def test_nan_portion_does_not_leak(moderate_params, mapper):
    rng = np.random.default_rng(0)
    week = _one_day_week({
        "day_index": 0, "wake_time": "06:30", "sleep_time": "23:00",
        "meals": [{"meal_type": "almoco", "time": "12:30",
                   "items": [{"food": "Arroz branco cozido", "portion_g": float("nan")}]}],
    })
    scheds, _ = build_schedules_from_llm(week, datetime(2026, 6, 1), moderate_params, mapper, rng)
    for s in scheds:
        for m in s.meals:
            assert _math.isfinite(m.true_carbs_g)
            assert _math.isfinite(m.estimated_carbs_g)
            assert _math.isfinite(m.glycemic_index)


def test_infinity_does_not_crash(moderate_params, mapper):
    rng = np.random.default_rng(0)
    week = _one_day_week({
        "day_index": float("inf"), "wake_time": "06:30", "sleep_time": "23:00",
        "meals": [{"meal_type": "almoco", "time": "12:30",
                   "items": [{"food": "Arroz branco cozido", "portion_g": float("inf")}]}],
        "exercise": [{"start_time": "18:00", "duration_min": float("inf"), "intensity": "moderate"}],
    })
    # must not raise (OverflowError / etc.)
    scheds, _ = build_schedules_from_llm(week, datetime(2026, 6, 1), moderate_params, mapper, rng)
    assert len(scheds) >= 1


def test_huge_day_index_clamped(moderate_params, mapper):
    rng = np.random.default_rng(0)
    week = _one_day_week({
        "day_index": 1_000_000_000, "wake_time": "06:30", "sleep_time": "23:00",
        "meals": [{"meal_type": "almoco", "time": "12:30",
                   "items": [{"food": "Arroz branco cozido", "portion_g": 150}]}],
    })
    scheds, _ = build_schedules_from_llm(week, datetime(2026, 6, 1), moderate_params, mapper, rng)
    # date must stay within the week horizon (no timedelta overflow)
    assert (scheds[0].date - datetime(2026, 6, 1)).days <= _MAX_DAY_INDEX


def test_exercise_duration_clamped(moderate_params, mapper):
    rng = np.random.default_rng(0)
    week = _one_day_week({
        "day_index": 0, "wake_time": "06:30", "sleep_time": "23:00",
        "meals": [{"meal_type": "almoco", "time": "12:30",
                   "items": [{"food": "Arroz branco cozido", "portion_g": 150}]}],
        "exercise": [{"start_time": "18:00", "duration_min": 100000, "intensity": "moderate"}],
    })
    scheds, _ = build_schedules_from_llm(week, datetime(2026, 6, 1), moderate_params, mapper, rng)
    for s in scheds:
        for ex in s.exercise_events:
            assert ex.duration_minutes <= _MAX_EXERCISE_MIN


# ---------------------------------------------------------------------------
# is_weekend parsing + duplicate-day dedup (E5 coverage gaps)
# ---------------------------------------------------------------------------


def test_is_weekend_false_string_is_weekday(moderate_params, mapper):
    # bool("false") is True; without parsing, the day would become a weekend
    week = {"days": [_day(is_weekend="false")]}
    scheds, _ = _build(week, moderate_params, mapper)
    assert scheds[0].day_type == DayType.WEEKDAY


def test_is_weekend_truthy_spellings(moderate_params, mapper):
    for val in (True, "true", "1", 1):
        week = {"days": [_day(is_weekend=val)]}
        scheds, _ = _build(week, moderate_params, mapper)
        assert scheds[0].day_type == DayType.WEEKEND, val
    for val in (False, "false", "no", 0, "", None):
        week = {"days": [_day(is_weekend=val)]}
        scheds, _ = _build(week, moderate_params, mapper)
        assert scheds[0].day_type == DayType.WEEKDAY, val


def test_duplicate_day_index_keeps_first(moderate_params, mapper):
    # two days share day_index 0: the adapter keeps the first occurrence
    first = _day(day_index=0, meals=[{"meal_type": "almoco", "time": "12:30",
                "items": [{"food": "Arroz branco cozido", "portion_g": 150}]}])
    second = _day(day_index=0, meals=[{"meal_type": "jantar", "time": "19:30",
                 "items": [{"food": "Banana prata", "portion_g": 100}]}])
    scheds, _ = _build({"days": [first, second]}, moderate_params, mapper)
    assert len(scheds) == 1
    assert scheds[0].meals[0].meal_type == MealType.ALMOCO

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
"""Tests for the plausibility / aberration validator."""

from simada.llm.plausibility import _to_min, validate_week


def _day(idx=0, wake="06:30", sleep="23:00", meals=None, exercise=None):
    default_meals = [
        {"meal_type": "cafe_da_manha", "time": "07:00", "items": [{"food": "x", "portion_g": 80}]},
        {"meal_type": "almoco", "time": "12:30", "items": [{"food": "x", "portion_g": 100}]},
        {"meal_type": "lanche_tarde", "time": "16:00", "items": [{"food": "x", "portion_g": 50}]},
        {"meal_type": "jantar", "time": "20:00", "items": [{"food": "x", "portion_g": 90}]},
    ]
    return {"day_index": idx, "wake_time": wake, "sleep_time": sleep,
            "meals": meals if meals is not None else default_meals,
            "exercise": exercise or []}


def _week7(day):
    """Embed one (possibly aberrant) day in an otherwise-clean full week so that
    the week-level ``not_seven_days`` rule does not add noise to per-day tests."""
    day = {**day, "day_index": 0}
    days = [day] + [_day(idx=i) for i in range(1, 7)]
    return {"days": days}


def _ex(start="18:00", dur=40):
    return [{"start_time": start, "duration_min": dur, "intensity": "moderate"}]


def _carbs(meal):  # trivial carb function: 50g per item
    return 50.0 * len(meal.get("items", []))


def test_normal_week_no_violations():
    week = {"days": [_day(idx=i) for i in range(7)]}
    rep = validate_week(week, carbs_per_meal=_carbs)
    assert not rep.is_aberration
    assert len(rep.violations) == 0


def test_eating_during_sleep_is_aberration():
    # meal at 03:00, awake window 06:30-23:00 -> eating during sleep
    week = _week7(_day(meals=[{"meal_type": "ceia", "time": "03:00",
                               "items": [{"food": "x", "portion_g": 100}]}]))
    rep = validate_week(week, carbs_per_meal=_carbs)
    assert rep.is_aberration
    assert any(v.kind == "meal_during_sleep" for v in rep.hard)


def test_sleep_12h_eat_18h_overflow():
    # sleeps 12h (awake 12h: 08:00-20:00) but meals span 06:00-23:00 (18h)
    week = _week7(_day(wake="08:00", sleep="20:00", meals=[
        {"meal_type": "cafe_da_manha", "time": "06:00", "items": [{"food": "x", "portion_g": 50}]},
        {"meal_type": "ceia", "time": "23:00", "items": [{"food": "x", "portion_g": 50}]},
    ]))
    rep = validate_week(week, carbs_per_meal=_carbs)
    # both meals fall outside the awake window -> aberration
    assert rep.is_aberration
    assert sum(1 for v in rep.hard if v.kind == "meal_during_sleep") == 2


def test_exercise_but_no_food_is_aberration():
    week = _week7(_day(meals=[], exercise=_ex()))
    rep = validate_week(week, carbs_per_meal=_carbs)
    assert rep.is_aberration
    # no_meals fires; exercise_no_food also fires
    assert any(v.kind == "no_meals" for v in rep.hard)


def test_exercise_with_negligible_carbs_is_aberration():
    # one meal but with ~0 carbs and exercise present
    def zero_carbs(meal):
        return 0.0
    week = _week7(_day(exercise=_ex()))
    rep = validate_week(week, carbs_per_meal=zero_carbs)
    assert any(v.kind == "exercise_no_food" for v in rep.hard)


def test_sleep_too_long_is_aberration():
    # sleeps 16h (awake 8h) -> sleep_hours_out hard
    week = _week7(_day(wake="10:00", sleep="18:00"))
    rep = validate_week(week, carbs_per_meal=_carbs)
    assert any(v.kind == "sleep_hours_out" for v in rep.hard)


def test_absurd_portion_is_aberration():
    week = _week7(_day(meals=[{"meal_type": "almoco", "time": "12:00",
                               "items": [{"food": "x", "portion_g": 5000}]}]))
    rep = validate_week(week, carbs_per_meal=_carbs)
    assert any(v.kind == "portion_out" for v in rep.hard)


def test_unusual_wake_is_soft_not_aberration():
    # wake at 03:00 is unusual (soft) but plausible enough not to be hard
    early_meals = [
        {"meal_type": "cafe_da_manha", "time": "04:00", "items": [{"food": "x", "portion_g": 80}]},
        {"meal_type": "almoco", "time": "11:00", "items": [{"food": "x", "portion_g": 100}]},
        {"meal_type": "lanche_tarde", "time": "14:00", "items": [{"food": "x", "portion_g": 50}]},
        {"meal_type": "jantar", "time": "18:00", "items": [{"food": "x", "portion_g": 90}]},
    ]
    week = _week7(_day(wake="03:00", sleep="19:00", meals=early_meals))
    rep = validate_week(week, carbs_per_meal=_carbs)
    assert not rep.is_aberration
    assert any(v.kind == "wake_time_unusual" and v.severity == "soft" for v in rep.violations)


# --- regression: cases that previously ESCAPED validation -----------------


def test_unparseable_meal_time_is_aberration():
    # a meal whose time is garbage used to slip past every check
    week = _week7(_day(meals=[
        {"meal_type": "almoco", "time": "noon", "items": [{"food": "x", "portion_g": 100}]},
        {"meal_type": "jantar", "time": "20:00", "items": [{"food": "x", "portion_g": 90}]},
    ]))
    rep = validate_week(week, carbs_per_meal=_carbs)
    assert rep.is_aberration
    assert any(v.kind == "bad_meal_time" for v in rep.hard)


def test_empty_week_is_aberration():
    rep = validate_week({"days": []}, carbs_per_meal=_carbs)
    assert rep.is_aberration
    assert any(v.kind == "not_seven_days" for v in rep.hard)


def test_wrong_number_of_days_is_aberration():
    week = {"days": [_day(idx=i) for i in range(5)]}  # only 5 days
    rep = validate_week(week, carbs_per_meal=_carbs)
    assert any(v.kind == "not_seven_days" for v in rep.hard)


def test_to_min_rejects_out_of_range_clock_fields():
    assert _to_min("25:00") is None
    assert _to_min("12:99") is None
    assert _to_min("-1:30") is None
    assert _to_min("23:59") == 23 * 60 + 59
    assert _to_min("00:00") == 0


def test_out_of_range_wake_time_is_caught():
    # 25:00 is now unparseable -> hard bad_time_format (used to read as 1500min)
    week = _week7(_day(wake="25:00"))
    rep = validate_week(week, carbs_per_meal=_carbs)
    assert rep.is_aberration
    assert any(v.kind == "bad_time_format" for v in rep.hard)


def test_exercise_during_sleep_is_aberration():
    # exercise at 03:00 while asleep (awake 06:30-23:00)
    week = _week7(_day(exercise=_ex("03:00")))
    rep = validate_week(week, carbs_per_meal=_carbs)
    assert rep.is_aberration
    assert any(v.kind == "exercise_during_sleep" for v in rep.hard)


def test_absurd_exercise_duration_is_flagged():
    week = _week7(_day(exercise=_ex(dur=600)))
    rep = validate_week(week, carbs_per_meal=_carbs)
    assert any(v.kind == "exercise_duration_out" and v.severity == "soft" for v in rep.violations)


def test_carbs_function_that_raises_does_not_fabricate_exercise_no_food():
    # carb fn blowing up must NOT be read as 0 carbs (false exercise_no_food)
    def bad_carbs(meal):
        raise ValueError("boom")
    week = _week7(_day(exercise=_ex()))
    rep = validate_week(week, carbs_per_meal=bad_carbs)
    assert not any(v.kind == "exercise_no_food" for v in rep.violations)
    assert any(v.kind == "carbs_uncomputable" for v in rep.soft)


def test_malformed_types_do_not_crash():
    # string day_index and a meals field that is not a list must not raise
    week = {"days": [
        {"day_index": "abc", "wake_time": "06:30", "sleep_time": "23:00", "meals": "not-a-list"},
        {"day_index": 1, "wake_time": "06:30", "sleep_time": "23:00", "meals": ["not-a-dict"]},
    ] + [_day(idx=i) for i in range(2, 7)]}
    rep = validate_week(week, carbs_per_meal=_carbs)  # must not raise
    assert rep.is_aberration


def test_duplicate_meal_times_flagged():
    week = _week7(_day(meals=[
        {"meal_type": "almoco", "time": "12:00", "items": [{"food": "x", "portion_g": 100}]},
        {"meal_type": "lanche", "time": "12:00", "items": [{"food": "x", "portion_g": 50}]},
        {"meal_type": "jantar", "time": "20:00", "items": [{"food": "x", "portion_g": 90}]},
    ]))
    rep = validate_week(week, carbs_per_meal=_carbs)
    assert any(v.kind == "duplicate_meal_time" for v in rep.soft)


def test_meals_out_of_order_flagged():
    week = _week7(_day(meals=[
        {"meal_type": "jantar", "time": "20:00", "items": [{"food": "x", "portion_g": 90}]},
        {"meal_type": "almoco", "time": "12:00", "items": [{"food": "x", "portion_g": 100}]},
    ]))
    rep = validate_week(week, carbs_per_meal=_carbs)
    assert any(v.kind == "meals_out_of_order" for v in rep.soft)


def test_sleep_after_midnight_is_valid():
    # wake 07:00, sleep 01:00 next day (awake 18h), late supper 23:30 -> valid
    week = _week7(_day(wake="07:00", sleep="01:00", meals=[
        {"meal_type": "cafe_da_manha", "time": "08:00", "items": [{"food": "x", "portion_g": 60}]},
        {"meal_type": "almoco", "time": "13:00", "items": [{"food": "x", "portion_g": 100}]},
        {"meal_type": "jantar", "time": "20:00", "items": [{"food": "x", "portion_g": 90}]},
        {"meal_type": "ceia", "time": "23:30", "items": [{"food": "x", "portion_g": 40}]},
    ]))
    rep = validate_week(week, carbs_per_meal=_carbs)
    assert not rep.is_aberration


def test_main_meal_with_zero_mappable_carbs_is_aberration():
    # lunch has items but they map to 0 carbs (unmatched dish names) -> the
    # adapter would silently drop it; validator must flag it (T2 / H5 bug2).
    def carbs_collapsed(meal):
        # almoco maps to nothing; everything else normal
        return 0.0 if meal.get("meal_type") == "almoco" else 60.0
    week = _week7(_day(meals=[
        {"meal_type": "cafe_da_manha", "time": "07:00", "items": [{"food": "x", "portion_g": 60}]},
        {"meal_type": "almoco", "time": "12:30", "items": [{"food": "Prato inexistente", "portion_g": 200}]},
        {"meal_type": "jantar", "time": "20:00", "items": [{"food": "x", "portion_g": 90}]},
    ]))
    rep = validate_week(week, carbs_per_meal=carbs_collapsed)
    assert rep.is_aberration
    assert any(v.kind == "meal_carbs_collapsed" for v in rep.hard)


def test_light_snack_zero_carbs_is_not_collapsed():
    # a snack (not a main meal) mapping to ~0 carbs is fine, not an aberration
    def carbs(meal):
        return 0.0 if meal.get("meal_type") == "snack" else 60.0
    week = _week7(_day(meals=[
        {"meal_type": "cafe_da_manha", "time": "07:00", "items": [{"food": "x", "portion_g": 60}]},
        {"meal_type": "snack", "time": "10:00", "items": [{"food": "Cafe preto", "portion_g": 50}]},
        {"meal_type": "almoco", "time": "12:30", "items": [{"food": "x", "portion_g": 100}]},
        {"meal_type": "jantar", "time": "20:00", "items": [{"food": "x", "portion_g": 90}]},
    ]))
    rep = validate_week(week, carbs_per_meal=carbs)
    assert not any(v.kind == "meal_carbs_collapsed" for v in rep.violations)


# ---------------------------------------------------------------------------
# exercise window + week-level orphan violations (E5 coverage gaps)
# ---------------------------------------------------------------------------


def test_exercise_starting_during_sleep_is_aberration():
    # exercise at 03:00, awake window 06:30-23:00 -> starts mid-sleep
    week = _week7(_day(exercise=_ex(start="03:00", dur=40)))
    rep = validate_week(week, carbs_per_meal=_carbs)
    assert rep.is_aberration
    assert any(v.kind == "exercise_during_sleep" for v in rep.hard)


def test_exercise_ending_after_bedtime_is_aberration():
    # starts 22:30 (awake) but 120 min runs to 00:30 -> ends mid-sleep
    week = _week7(_day(wake="06:30", sleep="23:00", exercise=_ex(start="22:30", dur=120)))
    rep = validate_week(week, carbs_per_meal=_carbs)
    assert rep.is_aberration
    assert any(v.kind == "exercise_ends_during_sleep" for v in rep.hard)
    # the start itself is fine, so the start-window rule must NOT fire
    assert not any(v.kind == "exercise_during_sleep" for v in rep.hard)


def test_exercise_fully_within_window_is_clean():
    week = {"days": [_day(idx=i, exercise=_ex(start="18:00", dur=60)) for i in range(7)]}
    rep = validate_week(week, carbs_per_meal=_carbs)
    assert not rep.is_aberration


def test_six_day_week_flags_not_seven_days_as_orphan():
    # a short week is a week-level (orphan, idx -1) hard violation
    week = {"days": [_day(idx=i) for i in range(6)]}
    rep = validate_week(week, carbs_per_meal=_carbs)
    assert rep.is_aberration
    orphans = [v for v in rep.hard if v.kind == "not_seven_days"]
    assert orphans and orphans[0].day_index == -1

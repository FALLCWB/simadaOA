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
"""Tests for the canonical LLM scenario structure (single source of truth)."""

from simada.llm.scenario_schema import (
    EXPECTED_DAYS,
    valid_clock,
    validate_structure,
)


def _good_day(idx=0):
    return {"day_index": idx, "wake_time": "06:30", "sleep_time": "23:00",
            "meals": [{"meal_type": "almoco", "time": "12:30",
                       "items": [{"food": "x", "portion_g": 100}]}]}


def _good_week():
    return {"days": [_good_day(i) for i in range(EXPECTED_DAYS)]}


def test_valid_clock():
    assert valid_clock("06:30")
    assert valid_clock("00:00") and valid_clock("23:59")
    assert not valid_clock("24:00")
    assert not valid_clock("12:99")
    assert not valid_clock("-1:30")
    assert not valid_clock("noon")
    assert not valid_clock(None)


def test_good_week_has_no_structural_errors():
    assert validate_structure(_good_week()) == []


def test_more_than_seven_days_flagged():
    week = _good_week()
    week["days"].append(_good_day(7))  # 8 days
    errs = validate_structure(week)
    assert any("expected 7 days" in e for e in errs)


def test_clock_2400_flagged():
    week = _good_week()
    week["days"][0]["sleep_time"] = "24:00"
    errs = validate_structure(week)
    assert any("sleep_time invalid" in e for e in errs)


def test_exercise_without_start_time_flagged():
    week = _good_week()
    week["days"][0]["exercise"] = [{"duration_min": 40, "intensity": "moderate"}]
    errs = validate_structure(week)
    assert any("exercise" in e and "start_time" in e for e in errs)


def test_meal_without_items_flagged():
    week = _good_week()
    week["days"][0]["meals"] = [{"meal_type": "almoco", "time": "12:00"}]
    errs = validate_structure(week)
    assert any("items missing" in e for e in errs)

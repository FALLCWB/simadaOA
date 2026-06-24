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

"""Property: a validator-approved week is always adapter-safe.

The plausibility gate (``validate_week``) flags a main meal whose mappable
carbs collapse to near zero (``meal_carbs_collapsed``, a hard aberration). The
adapter (``build_schedules_from_llm``) drops any meal whose ``true_carbs_g`` is
<= 0. Together these must be consistent: if the validator approves a week, then
EVERY main meal in it must survive adaptation with carbs > 0 -- otherwise the
comparison silently loses meals the validator vouched for.

This test generates well-formed weeks from the real TACO vocabulary, keeps only
the ones the official validator approves, and asserts that no approved main meal
is dropped by the adapter.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from hypothesis import HealthCheck, assume, given, settings, strategies as st

from simada.core.config import load_archetype_params
from simada.llm.plausibility import _is_main_meal, validate_week
from simada.llm.scenario_adapter import build_schedules_from_llm
from simada.llm.taco_mapping import TACOMapper
from simada.meals.taco import TACODatabase

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Real TACO entries with non-trivial carbs that map exactly (no qualifier drop).
_CARBY_FOODS = [
    "Arroz branco cozido",
    "Feijao carioca cozido",
    "Pao frances",
    "Macarrao cozido",
    "Banana prata",
    "Banana nanica",
]

# (meal_type label, clock time) for the three main meals always present.
_MAIN_MEALS = [
    ("cafe da manha", "08:00"),
    ("almoco", "12:30"),
    ("jantar", "19:30"),
]

_TACO = TACODatabase(PROJECT_ROOT / "data" / "taco" / "taco_foods.csv")
_MAPPER = TACOMapper(_TACO)
_ARCH_PARAMS = load_archetype_params(
    PROJECT_ROOT / "configs" / "archetypes" / "moderate.yaml")


def _carbs_of_meal(meal: dict) -> float:
    total = 0.0
    for it in meal.get("items", []):
        m = _MAPPER.match(str(it.get("food", "")))
        if m.food is not None:
            total += m.food.carbs_per_100g * float(
                it.get("portion_g", m.food.porcao_tipica_g)) / 100.0
    return total


_item = st.fixed_dictionaries({
    "food": st.sampled_from(_CARBY_FOODS),
    "portion_g": st.integers(min_value=60, max_value=250),
})


@st.composite
def _well_formed_week(draw):
    days = []
    for d in range(7):
        meals = []
        for label, time in _MAIN_MEALS:
            items = draw(st.lists(_item, min_size=1, max_size=3))
            meals.append({"meal_type": label, "time": time, "items": items})
        days.append({
            "day_index": d,
            "wake_time": "07:00",
            "sleep_time": "23:00",
            "meals": meals,
        })
    return {"metadata": {"archetype": "moderate"}, "days": days}


@given(week=_well_formed_week())
@settings(max_examples=60, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_validated_week_is_adapter_safe(week):
    """If validate_week approves, the adapter keeps every main meal (carbs > 0)."""
    rep = validate_week(week, carbs_per_meal=_carbs_of_meal)
    # Only the implication matters: condition on validator approval.
    assume(not rep.is_aberration)

    n_main_in = sum(
        1
        for day in week["days"]
        for meal in day["meals"]
        if _is_main_meal(meal.get("meal_type"))
    )

    rng = np.random.default_rng(0)
    schedules, _ = build_schedules_from_llm(
        week, datetime(2026, 6, 1), _ARCH_PARAMS, _MAPPER, rng)

    n_carby_out = sum(
        1 for s in schedules for m in s.meals if m.true_carbs_g > 0
    )

    # Every approved main meal must survive adaptation with real carbs.
    assert n_carby_out >= n_main_in, (
        f"validator approved {n_main_in} main meals but only {n_carby_out} "
        f"meals survived the adapter with carbs > 0")

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

"""Wiring test: the controller drives the optional counterregulation model.

Verifies the integration seam (controller -> CounterregulationModel ->
scenario.inject_cho) deterministically, without a full simglucose run. The
``counterregulation_glucose_g`` property isolates this source from the
voluntary hypo-treatment carbs (which also inject CHO at low BG).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from numpy.random import default_rng

from simada.behavior.counterregulation import (
    CounterregulationConfig,
    CounterregulationModel,
)
from simada.behavior.schedule import DailySchedule
from simada.controller.adherent_bb import AdherentBBController
from simada.core.config import load_archetype_params
from simada.core.types import DayType, InsulinRegimen, MealEvent, MealType
from simada.insulin.adherence import AdherenceInsulinModel
from simada.insulin.basal import BasalProfile
from simada.insulin.calculator import BolusCalculator
from simada.patient.archetype import ArchetypeID, create_archetype
from simada.scenario.custom_scenario import SimadaScenario

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_START = datetime(2026, 6, 2, 6, 0)  # Monday 06:00


def _params():
    return load_archetype_params(
        PROJECT_ROOT / "configs" / "archetypes" / "moderate.yaml")


def _controller(counterreg=None):
    params = _params()
    meal = MealEvent(time=_START + timedelta(hours=6),
                     meal_type=MealType.ALMOCO, true_carbs_g=50.0,
                     estimated_carbs_g=50.0, foods=("arroz",), glycemic_index=60.0)
    schedule = DailySchedule(date=_START, day_type=DayType.WEEKDAY,
                             wake_time=_START, sleep_time=_START.replace(hour=22),
                             meals=[meal], exercise_events=[], stress_events=[])
    scenario = SimadaScenario.from_schedules([schedule])
    archetype = create_archetype(ArchetypeID.MODERATE, params)
    insulin_model = AdherenceInsulinModel(
        archetype, BolusCalculator(cr=10.0, cf=50.0, target_bg=120.0))
    basal = BasalProfile(tdi=40.0, regimen=InsulinRegimen.PUMP)
    return AdherentBBController(
        insulin_model=insulin_model, basal_profile=basal, scenario=scenario,
        archetype_params=params, rng=default_rng(42),
        start_time=scenario.start_time, counterregulation=counterreg)


def _drive(ctrl, cgm, steps):
    for _ in range(steps):
        ctrl.policy(SimpleNamespace(CGM=cgm, CHO=0.0), reward=0, done=False,
                    sample_time=3)


def test_off_by_default_releases_nothing():
    ctrl = _controller(counterreg=None)
    _drive(ctrl, cgm=50.0, steps=10)  # sustained hypo
    assert ctrl.counterregulation_glucose_g == 0.0


def test_enabled_releases_glucose_during_hypo():
    cr = CounterregulationModel(CounterregulationConfig(enabled=True))
    ctrl = _controller(counterreg=cr)
    _drive(ctrl, cgm=50.0, steps=10)
    assert ctrl.counterregulation_glucose_g > 0.0


def test_enabled_releases_nothing_in_euglycemia():
    cr = CounterregulationModel(CounterregulationConfig(enabled=True))
    ctrl = _controller(counterreg=cr)
    _drive(ctrl, cgm=120.0, steps=10)
    assert ctrl.counterregulation_glucose_g == 0.0


def test_reset_clears_accounting_and_restores_glycogen():
    # simglucose calls controller.reset() at the start of every simulate();
    # the counterregulation total and the depleted store must not leak across runs.
    cr = CounterregulationModel(CounterregulationConfig(enabled=True))
    ctrl = _controller(counterreg=cr)
    _drive(ctrl, cgm=45.0, steps=20)  # sustained deep hypo depletes some glycogen
    assert ctrl.counterregulation_glucose_g > 0.0
    assert cr.glycogen_g < cr._cfg.glycogen_store_g
    ctrl.reset()
    assert ctrl.counterregulation_glucose_g == 0.0
    assert cr.glycogen_g == cr._cfg.glycogen_store_g

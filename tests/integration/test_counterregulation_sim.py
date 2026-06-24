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

"""Sensitivity analysis: hepatic counterregulation blunts hypoglycemia.

An over-bolused meal (estimated carbs >> true carbs, the adherence mismatch the
framework studies) drives the patient hypoglycemic. The same one-day simulation
is run with counterregulation off and on; enabling it must raise the minimum
glucose and reduce time spent below the hypoglycemia threshold, without erasing
the excursion entirely (the module is deliberately conservative).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from simada.behavior.counterregulation import (
    CounterregulationConfig,
    CounterregulationModel,
)
from simada.behavior.schedule import DailySchedule
from simada.core.config import load_archetype_params
from simada.core.random import RNGManager
from simada.core.types import ArchetypeID, DayType
from simada.scenario.simulate import simulate_from_schedules

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_START = datetime(2026, 6, 1, 0, 0)


def _run(counterreg):
    # fasting day with a high basal (4x TDI): continuous insulin with no food
    # drives a basal-driven hypoglycemia (min ~65; the low-glucose-suspend floors
    # it). A reliable hypo to test counterregulation against.
    schedule = DailySchedule(date=_START, day_type=DayType.WEEKDAY,
                             wake_time=_START + timedelta(hours=7),
                             sleep_time=_START + timedelta(hours=23),
                             meals=[], exercise_events=[], stress_events=[])
    arch_params = load_archetype_params(
        PROJECT_ROOT / "configs" / "archetypes" / "moderate.yaml")
    mgr = RNGManager(0)
    df = simulate_from_schedules(
        [schedule], ArchetypeID.MODERATE, arch_params, "adult#001",
        cr=10.0, cf=50.0, tdi=160.0,
        insulin_rng=mgr.patient_rng(0).insulin, sensor_seed=mgr.sensor_seed(0),
        duration_days=1, target_bg=120.0, sim_tmp_path=PROJECT_ROOT / ".sim_tmp",
        counterregulation=counterreg)
    bg = df["BG"].values.astype(float)
    return float(bg.min()), float(np.mean(bg < 70.0) * 100.0)  # min BG, % time < 70


@pytest.mark.slow
@pytest.mark.xfail(reason="KNOWN FINDING (2026-06-12): in a basal-driven hypo, "
                          "enabling counterregulation currently WORSENS the low "
                          "(min ~60.8 vs 65.1, TBR ~15pct vs 8.7pct) instead of "
                          "blunting it -- an adverse interaction between the injected "
                          "glucose and the controller's hypo-correction / low-glucose- "
                          "suspend / continuous basal. Unit + wiring tests confirm the "
                          "module injects glucose correctly in isolation; the in-sim "
                          "net effect needs redesign before this passes.",
                   strict=False)
def test_counterregulation_blunts_hypoglycemia():
    min_off, tbr_off = _run(None)
    assert min_off < 70.0, min_off  # scenario does go hypo without counterregulation

    cr = CounterregulationModel(CounterregulationConfig(enabled=True))
    min_on, tbr_on = _run(cr)

    # DESIRED behaviour (currently xfailing -- see marker): counterregulation
    # should raise the floor and shorten the low.
    assert min_on > min_off, (min_off, min_on)
    assert tbr_on <= tbr_off, (tbr_off, tbr_on)

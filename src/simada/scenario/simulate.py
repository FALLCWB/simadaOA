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

"""Shared schedules -> simglucose -> results wiring.

Both the multi-seed batch runner and the LLM-vs-rule-based comparison build
the exact same simulation pipeline from a list of ``DailySchedule`` objects
(scenario -> archetype -> bolus calculator -> adherence model -> basal profile
-> controller -> simglucose env). Keeping two copies risks them silently
diverging, which would invalidate any comparison of their outputs. This module
is the single source of that wiring; callers supply the per-run inputs
(CR/CF/TDI, the insulin RNG and CGM sensor seed) and get back the raw
simglucose results DataFrame to derive metrics from.

Heavy imports (simglucose, archetype/insulin builders) stay inside the function
so importing this module -- e.g. during test collection -- costs nothing.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd
    from numpy.random import Generator

    from simada.behavior.schedule import DailySchedule
    from simada.core.config import ArchetypeParams
    from simada.core.types import ArchetypeID


def simulate_from_schedules(
    schedules: list[DailySchedule],
    arch_id: ArchetypeID,
    arch_params: ArchetypeParams,
    patient_name: str,
    *,
    cr: float,
    cf: float,
    tdi: float,
    insulin_rng: Generator,
    sensor_seed: int,
    duration_days: int,
    target_bg: float,
    sim_tmp_path: Path,
    counterregulation: Any = None,
) -> pd.DataFrame:
    """Run one simulation from ready-built schedules and return the results df.

    The returned DataFrame is simglucose's raw output (BG/CGM/insulin/CHO
    columns); callers compute clinical metrics from it. ``insulin_rng`` and
    ``sensor_seed`` are passed in (rather than derived here) so the caller's
    seed hierarchy stays the single source of reproducibility.

    ``counterregulation`` is an optional ``CounterregulationModel`` (off by
    default); pass one to enable hepatic glycogenolysis during hypoglycemia
    (used for the counterregulation sensitivity analysis).
    """
    from simglucose.actuator.pump import InsulinPump
    from simglucose.patient.t1dpatient import T1DPatient
    from simglucose.sensor.cgm import CGMSensor
    from simglucose.simulation.env import T1DSimEnv
    from simglucose.simulation.sim_engine import SimObj

    from simada.controller.adherent_bb import AdherentBBController
    from simada.core.types import InsulinRegimen
    from simada.insulin.adherence import AdherenceInsulinModel
    from simada.insulin.basal import BasalProfile
    from simada.insulin.calculator import BolusCalculator
    from simada.patient.archetype import create_archetype
    from simada.scenario.custom_scenario import SimadaScenario

    scenario = SimadaScenario.from_schedules(schedules)
    archetype = create_archetype(arch_id, arch_params)
    calculator = BolusCalculator(
        cr=cr, cf=cf, target_bg=target_bg, iob_duration_hours=4.0,
    )
    insulin_model = AdherenceInsulinModel(archetype, calculator)
    basal_profile = BasalProfile(
        tdi=tdi, regimen=InsulinRegimen.PUMP, dawn_phenomenon=True,
    )
    controller = AdherentBBController(
        insulin_model=insulin_model,
        basal_profile=basal_profile,
        scenario=scenario,
        archetype_params=arch_params,
        rng=insulin_rng,
        start_time=scenario.start_time,
        counterregulation=counterregulation,
    )

    patient = T1DPatient.withName(patient_name)
    sensor = CGMSensor.withName("Dexcom", seed=sensor_seed)
    pump = InsulinPump.withName("Insulet")

    env_obj = T1DSimEnv(patient, sensor, pump, scenario)
    # simulate() + results() instead of sim(): avoid simglucose's CSV write and
    # stdout print side effects on every run.
    sim_obj = SimObj(
        env_obj,
        controller,
        timedelta(days=duration_days),
        animate=False,
        path=str(sim_tmp_path),
    )
    sim_obj.simulate()
    return sim_obj.results()

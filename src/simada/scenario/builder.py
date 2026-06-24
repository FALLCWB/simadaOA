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

"""Scenario builder — orchestrates all components into simglucose SimObj.

This is the top-level integration point that wires:
    Config YAML → CohortGenerator → DailyScheduleBuilder → SimadaScenario
    + AdherentBBController → simglucose SimObj

Each SimulationUnit contains everything needed to run a single patient
simulation through simglucose's batch_sim.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, NamedTuple

from simglucose.actuator.pump import InsulinPump
from simglucose.sensor.cgm import CGMSensor
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.sim_engine import SimObj

from simada.behavior.schedule import DailyScheduleBuilder
from simada.controller.adherent_bb import AdherentBBController
from simada.core.config import SimulationConfig
from simada.core.random import RNGManager
from simada.core.types import ArchetypeID, DayType
from simada.insulin.adherence import AdherenceInsulinModel
from simada.insulin.basal import BasalProfile
from simada.insulin.calculator import BolusCalculator
from simada.meals.taco import TACODatabase
from simada.meals.templates import load_locale_day_plans
from simada.patient.archetype import create_archetype
from simada.patient.cohort import CohortGenerator, PatientProfile
from simada.scenario.custom_scenario import SimadaScenario
from simglucose.patient.t1dpatient import T1DPatient


class SimulationUnit(NamedTuple):
    """Everything needed to run one patient's simulation."""

    sim_obj: SimObj
    patient_profile: PatientProfile
    scenario: SimadaScenario


class ScenarioBuilder:
    """Builds simulation units from configuration.

    Usage::

        builder = ScenarioBuilder(config, project_root)
        units = builder.build()
        for unit in units:
            unit.sim_obj.simulate()  # run through simglucose
    """

    def __init__(self, config: SimulationConfig, project_root: Path) -> None:
        self._config = config
        self._project_root = project_root
        self._rng = RNGManager(config.seed)
        self._taco_db = TACODatabase(project_root / config.meals.taco_path)

        # Locale-aware meal templates (brazil / usa / japan). Country meal
        # structure comes from these templates; country food composition comes
        # from config.meals.taco_path. Both must match the locale.
        self._weekday_plan, self._weekend_plan, self._holiday_plan = (
            load_locale_day_plans(
                config.meals.locale, project_root / "configs" / "meals"
            )
        )

        # Build the set of holiday dates from config for O(1) lookup
        self._holiday_dates = self._parse_holiday_dates(config.scenario.holidays)

    @staticmethod
    def _parse_holiday_dates(holidays: list[dict[str, Any]]) -> set[str]:
        """Extract a set of date strings (YYYY-MM-DD) from the holidays config.

        Each holiday entry may contain either a single ``date`` key or a
        ``dates`` list.  Both string values and dict entries with a ``date``
        key are supported.
        """
        date_set: set[str] = set()
        for entry in holidays:
            if isinstance(entry, str):
                date_set.add(entry)
                continue
            # Single date field
            if "date" in entry:
                date_set.add(str(entry["date"]))
            # List of dates field
            if "dates" in entry:
                for d in entry["dates"]:
                    date_set.add(str(d))
        return date_set

    def build(self) -> list[SimulationUnit]:
        """Build simulation units for all patients in the cohort."""
        cohort_gen = CohortGenerator(
            config=self._config.scenario.cohort,
            archetype_configs_dir=self._project_root / "configs" / "archetypes",
            rng=self._rng.general_rng,
        )
        profiles = cohort_gen.generate()

        units: list[SimulationUnit] = []
        for profile in profiles:
            unit = self._build_single(profile)
            units.append(unit)

        return units

    def _build_single(self, profile: PatientProfile) -> SimulationUnit:
        """Build a single patient's simulation unit."""
        patient_rng = self._rng.patient_rng(profile.patient_index)

        # Build daily schedules for the simulation duration
        schedule_builder = DailyScheduleBuilder(
            archetype_params=profile.archetype_params,
            taco_db=self._taco_db,
            weekday_plan=self._weekday_plan,
            weekend_plan=self._weekend_plan,
            holiday_plan=self._holiday_plan,
            meal_config=self._config.meals,
            behavior_config=self._config.behavior,
        )

        start_date = datetime.strptime(self._config.scenario.start_date, "%Y-%m-%d")
        schedules = []
        for day_offset in range(self._config.scenario.duration_days):
            date = start_date + timedelta(days=day_offset)
            date_str = date.strftime("%Y-%m-%d")

            # Holiday takes precedence over weekend
            if date_str in self._holiday_dates:
                day_type = DayType.HOLIDAY
            elif date.weekday() >= 5:
                day_type = DayType.WEEKEND
            else:
                day_type = DayType.WEEKDAY
            schedule = schedule_builder.build(date, day_type, patient_rng)
            schedules.append(schedule)

        # Create simada scenario
        scenario = SimadaScenario.from_schedules(schedules)

        # Create insulin delivery components
        archetype = create_archetype(profile.archetype_id, profile.archetype_params)
        calculator = BolusCalculator(
            cr=profile.cr,
            cf=profile.cf,
            target_bg=self._config.insulin.target_bg,
            iob_duration_hours=self._config.insulin.iob_duration_hours,
        )
        insulin_model = AdherenceInsulinModel(archetype, calculator)
        basal_profile = BasalProfile(
            tdi=profile.tdi,
            regimen=profile.insulin_regimen,
            dawn_phenomenon=self._config.insulin.dawn_phenomenon,
        )

        # Create controller
        controller = AdherentBBController(
            insulin_model=insulin_model,
            basal_profile=basal_profile,
            scenario=scenario,
            archetype_params=profile.archetype_params,
            rng=patient_rng.insulin,
            start_time=scenario.start_time,
        )

        # Create simglucose components. The cohort runs use simglucose's stock
        # dopri5 integrator, which completes the ninety-day horizon for every
        # physiology except the one stiff pediatric case in a few regimes; the
        # stiffness-aware LSODA alternative (patient/resilient.py) was evaluated
        # and agreed to within sensor-negligible tolerance, so dopri5 is kept.
        patient = T1DPatient.withName(profile.simglucose_name)
        sensor = CGMSensor.withName(
            self._config.scenario.cohort.sensor,
            seed=self._rng.sensor_seed(profile.patient_index),
        )
        pump = InsulinPump.withName(self._config.scenario.cohort.pump)

        env = T1DSimEnv(patient, sensor, pump, scenario)
        sim_time = timedelta(days=self._config.scenario.duration_days)
        output_path = self._project_root / self._config.output.directory
        sim_obj = SimObj(env, controller, sim_time, animate=False, path=str(output_path))

        return SimulationUnit(
            sim_obj=sim_obj,
            patient_profile=profile,
            scenario=scenario,
        )

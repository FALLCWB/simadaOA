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

"""Parallel simulation driven by LLM-generated dietary scenarios.

This is the LLM arm of the LLM-vs-rule-based comparison. It mirrors
``pipeline.parallel`` EXACTLY -- same cohort, same per-patient RNG derivation,
same controller/sensor/pump/SimObj assembly, same Parquet output -- so the ONLY
difference from the rule-based arm is the source of the daily meal schedule:

    rule-based arm: DailyScheduleBuilder samples TACO foods into meal templates.
    LLM arm (here): build_schedules_from_llm maps a phi4-generated week scenario
                    onto DailySchedules, tiled across the simulation horizon.

Keeping everything else identical is what makes the comparison fair.
"""

from __future__ import annotations

import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Reuse the rule-based arm's result container so downstream aggregation is shared.
from simada.pipeline.parallel import PatientResult


@dataclass
class LLMPatientJob:
    """Serializable job: one patient simulated from one LLM week scenario."""

    config_dict: dict[str, Any]
    project_root: str
    patient_index: int
    simglucose_name: str
    archetype: str
    cr: float
    cf: float
    tdi: float
    insulin_regimen: str
    scenario_paths: list[str]  # all phi4 week scenarios for this archetype
    output_dir: str
    master_seed: int
    carb_scale: float = 1.0  # optional carb-matching multiplier on meal carbs


def _run_llm_patient_job(job: LLMPatientJob) -> PatientResult:
    """Worker: build and run one patient from an LLM week scenario, tiled."""
    try:
        import json
        from datetime import datetime, timedelta

        import pyarrow as pa
        import pyarrow.parquet as pq

        from simada.core.config import SimulationConfig, load_archetype_params
        from simada.core.random import RNGManager
        from simada.core.types import ArchetypeID
        from simada.controller.adherent_bb import AdherentBBController
        from simada.insulin.adherence import AdherenceInsulinModel
        from simada.insulin.basal import BasalProfile
        from simada.insulin.calculator import BolusCalculator
        from simada.llm.scenario_adapter import build_schedules_from_llm
        from simada.llm.taco_mapping import TACOMapper
        from simada.meals.taco import TACODatabase
        from simada.patient.archetype import create_archetype
        from simada.scenario.custom_scenario import SimadaScenario

        from simglucose.actuator.pump import InsulinPump
        from simglucose.patient.t1dpatient import T1DPatient
        from simglucose.sensor.cgm import CGMSensor
        from simglucose.simulation.env import T1DSimEnv
        from simglucose.simulation.sim_engine import SimObj

        project_root = Path(job.project_root)
        config = SimulationConfig.model_validate(job.config_dict)

        # Identical RNG derivation to the rule-based arm.
        rng_mgr = RNGManager(job.master_seed)
        patient_rng = rng_mgr.patient_rng(job.patient_index)

        arch_id = ArchetypeID(job.archetype)
        arch_params = load_archetype_params(
            project_root / "configs" / "archetypes" / f"{arch_id.value}.yaml"
        )

        taco_db = TACODatabase(project_root / config.meals.taco_path)
        mapper = TACOMapper(taco_db)
        weeks = []
        for p in job.scenario_paths:
            with open(p, encoding="utf-8") as f:
                weeks.append(json.load(f))
        if not weeks:
            raise ValueError("no LLM scenarios supplied for patient")

        start_date = datetime.strptime(config.scenario.start_date, "%Y-%m-%d")
        n_days = config.scenario.duration_days

        # Cover the horizon by ROTATING through all the archetype's week
        # scenarios (not repeating a single week), starting at a per-patient
        # offset so patients differ. This gives day-to-day dietary variety
        # comparable to the rule-based arm's fresh stochastic days, removing the
        # single-week-tiling confound (which artificially lowered CV). Calendar
        # dates stay consecutive regardless of each week's length.
        schedules = []
        block = 0
        while len(schedules) < n_days:
            wk = weeks[(job.patient_index + block) % len(weeks)]
            base = start_date + timedelta(days=len(schedules))
            sch, _ = build_schedules_from_llm(
                wk, base, arch_params, mapper, patient_rng.meals
            )
            if not sch:
                break
            schedules.extend(sch)
            block += 1
        schedules = schedules[:n_days]
        if not schedules:
            raise ValueError("LLM scenarios produced no schedules")

        # Optional carb-matching: scale every meal's true + estimated carbs by a
        # constant so the LLM arm's per-day carbohydrate load matches the rule
        # arm's, isolating meal PATTERN from carb LOAD in the comparison. The
        # estimation-error ratio is preserved (both scaled), and the controller
        # doses on the scaled estimate, so insulin scales with intake.
        if abs(job.carb_scale - 1.0) > 1e-9:
            from dataclasses import replace as _replace
            schedules = [
                _replace(sch, meals=[
                    _replace(
                        m,
                        true_carbs_g=m.true_carbs_g * job.carb_scale,
                        estimated_carbs_g=m.estimated_carbs_g * job.carb_scale,
                    )
                    for m in sch.meals
                ])
                for sch in schedules
            ]

        scenario = SimadaScenario.from_schedules(schedules)

        # --- everything below is identical to the rule-based worker ---
        archetype = create_archetype(arch_id, arch_params)
        calculator = BolusCalculator(
            cr=job.cr,
            cf=job.cf,
            target_bg=config.insulin.target_bg,
            iob_duration_hours=config.insulin.iob_duration_hours,
        )
        insulin_model = AdherenceInsulinModel(archetype, calculator)
        basal_profile = BasalProfile(
            tdi=job.tdi,
            regimen=job.insulin_regimen,
            dawn_phenomenon=config.insulin.dawn_phenomenon,
        )
        controller = AdherentBBController(
            insulin_model=insulin_model,
            basal_profile=basal_profile,
            scenario=scenario,
            archetype_params=arch_params,
            rng=patient_rng.insulin,
            start_time=scenario.start_time,
        )

        # Stock dopri5 integrator (see scenario/builder.py for the rationale).
        patient = T1DPatient.withName(job.simglucose_name)
        sensor = CGMSensor.withName(
            config.scenario.cohort.sensor,
            seed=rng_mgr.sensor_seed(job.patient_index),
        )
        pump = InsulinPump.withName(config.scenario.cohort.pump)
        env = T1DSimEnv(patient, sensor, pump, scenario)
        sim_time = timedelta(days=n_days)
        sim_obj = SimObj(env, controller, sim_time, animate=False, path=job.output_dir)
        sim_obj.simulate()
        results = sim_obj.results()

        output_path = Path(job.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        safe_name = job.simglucose_name.replace("#", "")
        filename = f"{safe_name}_{job.archetype}_{job.patient_index:03d}.parquet"
        parquet_path = output_path / filename
        pq.write_table(
            pa.Table.from_pandas(results), parquet_path, compression="snappy"
        )

        return PatientResult(
            simglucose_name=job.simglucose_name,
            archetype=job.archetype,
            patient_index=job.patient_index,
            parquet_path=str(parquet_path),
            n_rows=len(results),
            bg_min=float(results["BG"].min()) if "BG" in results.columns else None,
            bg_max=float(results["BG"].max()) if "BG" in results.columns else None,
            error=None,
        )
    except Exception as e:
        return PatientResult(
            simglucose_name=job.simglucose_name,
            archetype=job.archetype,
            patient_index=job.patient_index,
            parquet_path=None,
            n_rows=0,
            bg_min=None,
            bg_max=None,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )


def archetype_scenarios(scenario_files: list[Path], archetype: str) -> list[Path]:
    """All week scenarios for ``archetype`` (the worker rotates through them)."""
    matching = sorted(p for p in scenario_files if f"_{archetype}" in p.name)
    if not matching:
        msg = f"no LLM scenarios for archetype {archetype!r} in library"
        raise FileNotFoundError(msg)
    return matching


def run_llm_parallel_from_config(
    config_dict: dict[str, Any],
    project_root: Path,
    scenarios_dir: Path,
    n_workers: int = 4,
    carb_scale_by_arch: dict[str, float] | None = None,
) -> list[PatientResult]:
    """Run the LLM arm for one country.

    Each patient in the cohort (generated identically to the rule-based arm) is
    assigned a phi4 week scenario matching its archetype from ``scenarios_dir``
    and simulated. Returns the same PatientResult list shape as the rule arm.
    """
    from datetime import datetime as _datetime

    from simada.core.config import SimulationConfig
    from simada.core.random import RNGManager
    from simada.patient.cohort import CohortGenerator

    config = SimulationConfig.model_validate(config_dict)
    rng = RNGManager(config.seed)
    cohort_gen = CohortGenerator(
        config=config.scenario.cohort,
        archetype_configs_dir=project_root / "configs" / "archetypes",
        rng=rng.general_rng,
    )
    profiles = cohort_gen.generate()

    scenario_files = sorted(scenarios_dir.glob("*.json"))
    if not scenario_files:
        msg = f"no LLM scenario JSONs found in {scenarios_dir}"
        raise FileNotFoundError(msg)

    timestamp = _datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = str(
        project_root / config.output.directory / f"run_{config.seed}_{timestamp}"
    )

    # Each patient receives ALL its archetype's week scenarios; the worker
    # rotates through them across the horizon (per-patient offset) so the LLM
    # arm has day-to-day dietary variety like the rule arm, instead of tiling a
    # single week (which artificially lowered CV).
    jobs: list[LLMPatientJob] = []
    for profile in profiles:
        arch = profile.archetype_id.value
        paths = [str(p) for p in archetype_scenarios(scenario_files, arch)]
        jobs.append(
            LLMPatientJob(
                config_dict=config_dict,
                project_root=str(project_root),
                patient_index=profile.patient_index,
                simglucose_name=profile.simglucose_name,
                archetype=arch,
                cr=profile.cr,
                cf=profile.cf,
                tdi=profile.tdi,
                insulin_regimen=profile.insulin_regimen.value,
                scenario_paths=paths,
                output_dir=output_dir,
                master_seed=config.seed,
                carb_scale=(carb_scale_by_arch or {}).get(arch, 1.0),
            )
        )

    results: list[PatientResult] = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_run_llm_patient_job, job): job for job in jobs}
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda r: r.patient_index)
    return results

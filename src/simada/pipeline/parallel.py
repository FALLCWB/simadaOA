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

"""Parallel simulation execution using ProcessPoolExecutor.

Each patient simulation is independent. Since simglucose SimObj is not
picklable, each worker process builds and runs its patient from scratch
using serializable configuration data.

Key design decisions:
    - Each worker receives an explicit PatientSpec (archetype, simglucose
      name, CR, CF, TDI, patient_index) so it builds exactly the right
      patient instead of rebuilding the whole cohort.
    - Seed derivation uses the same RNGManager + patient_index as the
      serial path, guaranteeing identical results regardless of
      parallelism.
    - Workers write Parquet directly to disk and return only lightweight
      metadata, avoiding large DataFrame transfers through IPC.
    - Uses concurrent.futures.ProcessPoolExecutor for cleaner API and
      better exception propagation than multiprocessing.Pool.
"""

from __future__ import annotations

import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PatientJob:
    """Serializable job description for a single patient simulation.

    Contains everything a worker needs to build and run exactly one
    patient without regenerating the cohort.
    """

    config_dict: dict[str, Any]
    project_root: str
    patient_index: int
    simglucose_name: str
    archetype: str
    cr: float
    cf: float
    tdi: float
    insulin_regimen: str
    output_dir: str
    master_seed: int


@dataclass
class PatientResult:
    """Lightweight result returned by a worker process."""

    simglucose_name: str
    archetype: str
    patient_index: int
    parquet_path: str | None
    n_rows: int
    bg_min: float | None
    bg_max: float | None
    error: str | None


def _run_patient_job(job: PatientJob) -> PatientResult:
    """Worker function: build and run one patient from explicit specification.

    Uses the same RNGManager + patient_index as the serial path so that
    results are deterministic and identical regardless of parallelism.
    Writes Parquet directly to disk to avoid sending large DataFrames
    through IPC.

    Returns a lightweight PatientResult with metadata only.
    """
    try:
        # Import inside worker to avoid pickling issues
        from datetime import datetime, timedelta

        import pyarrow as pa
        import pyarrow.parquet as pq

        from simada.core.config import SimulationConfig, load_archetype_params
        from simada.core.random import RNGManager
        from simada.core.types import ArchetypeID, DayType
        from simada.behavior.schedule import DailyScheduleBuilder
        from simada.controller.adherent_bb import AdherentBBController
        from simada.insulin.adherence import AdherenceInsulinModel
        from simada.insulin.basal import BasalProfile
        from simada.insulin.calculator import BolusCalculator
        from simada.meals.taco import TACODatabase
        from simada.meals.templates import load_locale_day_plans
        from simada.patient.archetype import create_archetype
        from simada.scenario.custom_scenario import SimadaScenario

        from simglucose.actuator.pump import InsulinPump
        from simglucose.patient.t1dpatient import T1DPatient
        from simglucose.sensor.cgm import CGMSensor
        from simglucose.simulation.env import T1DSimEnv
        from simglucose.simulation.sim_engine import SimObj

        project_root = Path(job.project_root)
        config = SimulationConfig.model_validate(job.config_dict)

        # Use the MASTER seed (not a derived one) so that RNGManager
        # produces the exact same per-patient sub-streams as the serial path.
        rng_mgr = RNGManager(job.master_seed)
        patient_rng = rng_mgr.patient_rng(job.patient_index)

        arch_id = ArchetypeID(job.archetype)
        arch_params = load_archetype_params(
            project_root / "configs" / "archetypes" / f"{arch_id.value}.yaml"
        )

        # Build daily schedules. Locale-aware templates (brazil/usa/japan):
        # meal structure from the locale templates, food composition from
        # config.meals.taco_path. Both must match config.meals.locale.
        taco_db = TACODatabase(project_root / config.meals.taco_path)
        weekday_plan, weekend_plan, holiday_plan = load_locale_day_plans(
            config.meals.locale, project_root / "configs" / "meals"
        )

        schedule_builder = DailyScheduleBuilder(
            archetype_params=arch_params,
            taco_db=taco_db,
            weekday_plan=weekday_plan,
            weekend_plan=weekend_plan,
            holiday_plan=holiday_plan,
            meal_config=config.meals,
            behavior_config=config.behavior,
        )

        # Parse holiday dates for day-type detection
        holiday_dates: set[str] = set()
        for entry in config.scenario.holidays:
            if isinstance(entry, str):
                holiday_dates.add(entry)
                continue
            if "date" in entry:
                holiday_dates.add(str(entry["date"]))
            if "dates" in entry:
                for d in entry["dates"]:
                    holiday_dates.add(str(d))

        start_date = datetime.strptime(config.scenario.start_date, "%Y-%m-%d")
        schedules = []
        for day_offset in range(config.scenario.duration_days):
            date = start_date + timedelta(days=day_offset)
            date_str = date.strftime("%Y-%m-%d")
            if date_str in holiday_dates:
                day_type = DayType.HOLIDAY
            elif date.weekday() >= 5:
                day_type = DayType.WEEKEND
            else:
                day_type = DayType.WEEKDAY
            schedule = schedule_builder.build(date, day_type, patient_rng)
            schedules.append(schedule)

        scenario = SimadaScenario.from_schedules(schedules)

        # Build insulin delivery components using job's explicit CR/CF/TDI
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

        # Build simglucose components. Stock dopri5 integrator (see
        # scenario/builder.py for the LSODA comparison and the rationale).
        patient = T1DPatient.withName(job.simglucose_name)
        # BUG 2: sensor seed was computed as master_seed + patient_index which
        # aliases (seed=100 + idx=42 == seed=142 + idx=0) and diverges from the
        # serial path (which used spec.seed directly). Both paths now use
        # RNGManager.sensor_seed(patient_index) for a consistent, collision-free
        # derivation from the hierarchical SeedSequence.
        sensor = CGMSensor.withName(
            config.scenario.cohort.sensor,
            seed=rng_mgr.sensor_seed(job.patient_index),
        )
        pump = InsulinPump.withName(config.scenario.cohort.pump)

        env = T1DSimEnv(patient, sensor, pump, scenario)
        sim_time = timedelta(days=config.scenario.duration_days)
        sim_obj = SimObj(
            env, controller, sim_time, animate=False, path=job.output_dir
        )

        # Run simulation (bypass sim() to skip redundant CSV + print)
        sim_obj.simulate()
        results = sim_obj.results()

        # Write Parquet directly
        output_path = Path(job.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        safe_name = job.simglucose_name.replace("#", "")
        filename = f"{safe_name}_{job.archetype}_{job.patient_index:03d}.parquet"
        parquet_path = output_path / filename

        table = pa.Table.from_pandas(results)
        pq.write_table(table, parquet_path, compression="snappy")

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
        # Capture the FULL traceback, not just str(e). Some failures surface a
        # masked message (e.g. simglucose's ODE-solver failure can propagate as
        # RuntimeError("No active exception to reraise") from a bare ``raise``),
        # which hides the real cause. The traceback preserves the origin frame
        # (e.g. the dopri5 step-size failure) so partial-cohort failures are
        # diagnosable from the returned metadata alone.
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


def run_parallel_from_config(
    config_dict: dict[str, Any],
    project_root: Path,
    n_workers: int = 4,
) -> list[PatientResult]:
    """Run a cohort simulation in parallel using ProcessPoolExecutor.

    Each worker builds and runs one patient independently using the same
    RNGManager + patient_index as the serial path.

    Args:
        config_dict: Serializable config dict (from model_dump).
        project_root: Project root path.
        n_workers: Number of parallel worker processes.

    Returns:
        List of PatientResult with per-patient metadata and Parquet paths.
    """
    from datetime import datetime as _datetime

    from simada.core.config import SimulationConfig
    from simada.core.random import RNGManager
    from simada.patient.cohort import CohortGenerator

    config = SimulationConfig.model_validate(config_dict)
    rng = RNGManager(config.seed)

    # Generate the cohort profile list to know what patients to create
    cohort_gen = CohortGenerator(
        config=config.scenario.cohort,
        archetype_configs_dir=project_root / "configs" / "archetypes",
        rng=rng.general_rng,
    )
    profiles = cohort_gen.generate()

    # BUG 8: output_dir was flat (no timestamp subdirectory). Consecutive runs
    # with the same config would silently overwrite each other's Parquet files.
    # We now append a seed+timestamp subdir so each invocation gets a unique
    # output location, matching the convention used by SimulationRunner.run().
    timestamp = _datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = str(
        project_root / config.output.directory / f"run_{config.seed}_{timestamp}"
    )

    # Create jobs with explicit patient specs (one per patient)
    jobs = []
    for profile in profiles:
        jobs.append(
            PatientJob(
                config_dict=config_dict,
                project_root=str(project_root),
                patient_index=profile.patient_index,
                simglucose_name=profile.simglucose_name,
                archetype=profile.archetype_id.value,
                cr=profile.cr,
                cf=profile.cf,
                tdi=profile.tdi,
                insulin_regimen=profile.insulin_regimen.value,
                output_dir=output_dir,
                master_seed=config.seed,
            )
        )

    # Run in parallel with ProcessPoolExecutor
    results: list[PatientResult] = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_run_patient_job, job): job for job in jobs}
        for future in as_completed(futures):
            results.append(future.result())

    # Sort by patient_index to maintain deterministic order
    results.sort(key=lambda r: r.patient_index)
    return results

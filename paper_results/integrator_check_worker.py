"""Run ONE (physiology, solver) controller-in-the-loop simulation and save the
true plasma-glucose (BG) trace, for the dopri5-vs-LSODA integrator comparison
for the integrator comparison. Isolated in a subprocess so a stiff dopri5 run can be
timed out by the driver.
Usage: python integrator_check_worker.py <name> <stock|lsoda> <days> <out.npy>"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


def repo_root(start):
    for q in [Path(start).resolve(), *Path(start).resolve().parents]:
        if (q / "pyproject.toml").exists():
            return q
    raise RuntimeError("repo root (pyproject.toml) not found")


sys.path.insert(0, str(repo_root(__file__) / "src"))

from simada.patient.resilient import ResilientT1DPatient
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.sim_engine import SimObj
from simglucose.simulation.scenario import CustomScenario
from simglucose.controller.basal_bolus_ctrller import BBController

name, solver, days, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
cls = ResilientT1DPatient if solver == "lsoda" else T1DPatient
scen = CustomScenario(start_time=datetime(2026, 6, 1, 0, 0),
                      scenario=[(7, 50), (13, 70), (19, 60)])
env = T1DSimEnv(cls.withName(name), CGMSensor.withName("Dexcom", seed=1),
                InsulinPump.withName("Insulet"), scen)
so = SimObj(env, BBController(), timedelta(days=days), animate=False, path="/tmp")
so.simulate()
np.save(out, so.results()["BG"].to_numpy())

"""dopri5-vs-LSODA integrator comparison.

Runs every standard simglucose physiology under both solvers (controller in the
loop, 30 days) and compares the reported clinical metrics (Time in Range, CV,
mean glucose). Confirms that the solver choice does not change the reported
metrics: the cohort uses the stock dopri5 solver, and LSODA is the evaluated
stiffness-aware alternative.

NOTE: requires the LSODA-reset fix in src/simada/patient/resilient.py (without
it, ResilientT1DPatient silently reverts to dopri5 on reset and both arms are
identical). Production (scenario/builder.py, pipeline/*) uses stock dopri5.

Usage: uv run --project .. python integrator_check.py [days] [timeout_s] [workers]
"""
import csv
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
WORKER = HERE / "integrator_check_worker.py"
DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
TIMEOUT = int(sys.argv[2]) if len(sys.argv) > 2 else 700
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 12
NAMES = [f"{g}#{i:03d}" for g in ("adult", "adolescent", "child") for i in range(1, 11)]
OUT = HERE / "integrator_check_out"
OUT.mkdir(exist_ok=True)


def run(args):
    name, solver = args
    out = OUT / f"{name.replace('#', '_')}_{solver}.npy"
    try:
        r = subprocess.run([sys.executable, str(WORKER), name, solver, str(DAYS), str(out)],
                           capture_output=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return name, solver, None, "timeout"
    if r.returncode != 0 or not out.exists():
        return name, solver, None, "failed"
    return name, solver, np.load(out), "ok"


def metrics(bg):
    bg = bg[~np.isnan(bg)]
    return (100.0 * np.mean((bg >= 70) & (bg <= 180)), 100.0 * bg.std() / bg.mean(), float(bg.mean()))


with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    res = list(ex.map(run, [(n, s) for n in NAMES for s in ("lsoda", "stock")]))
bg = {(n, s): a for n, s, a, st in res if st == "ok"}
status = {(n, s): st for n, s, _, st in res}

rows = []
for name in NAMES:
    if status[(name, "stock")] != "ok" or (name, "lsoda") not in bg:
        print(f"  {name:15s} dopri5 {status[(name, 'stock')]}"); continue
    tl, ts = metrics(bg[(name, "lsoda")]), metrics(bg[(name, "stock")])
    rows.append((name, tl[0] - ts[0], tl[1] - ts[1], tl[2] - ts[2]))
    print(f"  {name:15s} dTIR={rows[-1][1]:+.2f} dCV={rows[-1][2]:+.2f} dMean={rows[-1][3]:+.1f}")

for lbl, idx, unit in (("dTIR", 1, "pp"), ("dCV", 2, "pp"), ("dMean", 3, "mg/dL")):
    a = np.array([r[idx] for r in rows])
    ci = stats.t.interval(0.95, len(a) - 1, loc=a.mean(), scale=a.std(ddof=1) / np.sqrt(len(a)))
    print(f"{lbl}: mean={a.mean():+.2f}{unit} mean|.|={np.abs(a).mean():.2f} max|.|={np.abs(a).max():.2f} "
          f"95%CI=({ci[0]:+.2f},{ci[1]:+.2f})")
with open(HERE / "integrator_metric_diffs.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["physiology", "dTIR_pp", "dCV_pp", "dMean_mgdl"]); w.writerows(rows)

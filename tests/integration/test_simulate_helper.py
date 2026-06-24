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

"""Parity tests for the extracted ``simulate_from_schedules`` helper.

After the DRY extraction (the batch runner and the LLM-vs-rule-based
comparison both delegate to one wiring), two things must hold: the helper is
deterministic for fixed inputs, and the comparison's ``_simulate`` is a faithful
delegation -- replicating its seed derivation by hand and calling the helper
must reproduce its metrics exactly.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from simada.analysis.metrics import compute_metrics
from simada.core.config import load_archetype_params
from simada.core.random import RNGManager
from simada.core.types import ArchetypeID
from simada.llm.plausibility import validate_week
from simada.llm.scenario_adapter import build_schedules_from_llm
from simada.llm.taco_mapping import TACOMapper
from simada.meals.taco import TACODatabase
from simada.scenario.simulate import simulate_from_schedules

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WEEKS_DIR = PROJECT_ROOT / "avaliacao" / "llm_weeks"
COMPARISON_SCRIPT = PROJECT_ROOT / "scripts" / "llm_vs_rulebased_comparison.py"
PATIENT = "adult#001"


def _load_comparison_module():
    spec = importlib.util.spec_from_file_location("llm_vs_rulebased_comparison",
                                                  COMPARISON_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _first_approved_schedules(mapper):
    def carbs_fn(meal):
        total = 0.0
        for it in (meal.get("items") or []):
            if isinstance(it, dict):
                m = mapper.match(str(it.get("food", "")))
                if m.food is not None:
                    total += m.food.carbs_per_100g * float(
                        it.get("portion_g", m.food.porcao_tipica_g)) / 100.0
        return total

    for wf in sorted(WEEKS_DIR.glob("week_*.json")):
        try:
            week = json.loads(wf.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if validate_week(week, carbs_per_meal=carbs_fn).is_aberration:
            continue
        arch = (week.get("metadata", {}).get("_archetype")
                or week.get("metadata", {}).get("archetype", "moderate"))
        arch_params = load_archetype_params(
            PROJECT_ROOT / "configs" / "archetypes" / f"{arch}.yaml")
        schedules, _ = build_schedules_from_llm(
            week, datetime(2026, 6, 1), arch_params, mapper, np.random.default_rng(0))
        return schedules, ArchetypeID(arch), arch_params
    return None, None, None


@pytest.mark.slow
@pytest.mark.skipif(not WEEKS_DIR.exists(), reason="no real LLM weeks available")
def test_helper_is_deterministic():
    """Same schedules + same seeds -> identical glucose trajectory."""
    mapper = TACOMapper(TACODatabase(PROJECT_ROOT / "data" / "taco" / "taco_foods.csv"))
    schedules, arch_id, arch_params = _first_approved_schedules(mapper)
    assert schedules is not None

    def _run():
        mgr = RNGManager(7)
        return simulate_from_schedules(
            schedules, arch_id, arch_params, PATIENT,
            cr=10.0, cf=50.0, tdi=40.0,
            insulin_rng=mgr.patient_rng(0).insulin,
            sensor_seed=mgr.sensor_seed(0),
            duration_days=1, target_bg=120.0,
            sim_tmp_path=PROJECT_ROOT / ".sim_tmp",
        )["BG"].values.astype(float)

    assert np.array_equal(_run(), _run())


@pytest.mark.slow
@pytest.mark.skipif(not WEEKS_DIR.exists(), reason="no real LLM weeks available")
def test_comparison_simulate_delegates_faithfully():
    """``_simulate`` must equal a by-hand helper call with the same seed wiring."""
    comparison = _load_comparison_module()
    mapper = TACOMapper(TACODatabase(PROJECT_ROOT / "data" / "taco" / "taco_foods.csv"))
    schedules, arch_id, arch_params = _first_approved_schedules(mapper)
    assert schedules is not None

    seed, dur = 3, 1
    via_simulate = comparison._simulate(
        schedules, arch_id, arch_params, PATIENT, seed=seed, duration_days=dur)

    mgr = RNGManager(seed)
    cr, cf, tdi = comparison._quest_params(PATIENT)
    df = simulate_from_schedules(
        schedules, arch_id, arch_params, PATIENT,
        cr=cr, cf=cf, tdi=tdi,
        insulin_rng=mgr.patient_rng(0).insulin,
        sensor_seed=mgr.sensor_seed(0),
        duration_days=dur, target_bg=comparison._TARGET_BG,
        sim_tmp_path=PROJECT_ROOT / ".sim_tmp",
    )
    via_helper = compute_metrics(df["BG"].values.astype(float))._asdict()

    assert via_simulate == via_helper

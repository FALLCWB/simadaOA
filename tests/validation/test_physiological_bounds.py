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

"""Physiological plausibility tests.

These tests run actual simulations and verify that the results
stay within clinically expected bounds. They are slower than unit
tests but catch systemic issues like the BG=1267 problem.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from simglucose.simulation.sim_engine import sim

from simada.analysis.metrics import compute_metrics
from simada.core.config import SimulationConfig
from simada.scenario.builder import ScenarioBuilder

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _make_config(
    duration_days: int = 7,
    archetype: str = "adherent",
    output_dir: str = "results",
) -> SimulationConfig:
    """Create a simulation config for validation tests."""
    arch_dist = {"adherent": 0.0, "moderate": 0.0, "nonadherent": 0.0}
    arch_dist[archetype] = 1.0

    return SimulationConfig.model_validate({
        "seed": 42,
        "scenario": {
            "name": "validation",
            "duration_days": duration_days,
            "start_date": "2026-06-01",
            "cohort": {
                "size": 1,
                "archetype_distribution": arch_dist,
                "insulin_regimen_distribution": {"pump": 1.0, "mdi": 0.0},
            },
        },
        "output": {"directory": output_dir},
    })


def _run_simulation(tmp_path: Path, archetype: str = "adherent", duration_days: int = 7):
    """Run a single-patient simulation and return the results DataFrame."""
    config = _make_config(
        duration_days=duration_days,
        archetype=archetype,
        output_dir=str(tmp_path),
    )
    builder = ScenarioBuilder(config, PROJECT_ROOT)
    units = builder.build()
    assert len(units) == 1
    results = sim(units[0].sim_obj)
    return results


class TestPhysiologicalBounds:
    """Validation tests ensuring simulation outputs remain physiologically plausible."""

    @pytest.mark.slow
    def test_nonadherent_bg_below_600(self, tmp_path: Path) -> None:
        """Nonadherent patient BG must stay below 600 mg/dL over 7 days.

        The renal correction and BG ceiling should prevent unrealistic
        hyperglycemia even with missed boluses and poor adherence.
        """
        results = _run_simulation(tmp_path, archetype="nonadherent")
        bg_max = results["BG"].max()
        assert bg_max < 600, f"Max BG {bg_max:.0f} mg/dL exceeds 600 ceiling"

    @pytest.mark.slow
    def test_adherent_tir_above_75(self, tmp_path: Path) -> None:
        """Adherent patient should achieve TIR > 75% over 7 days.

        Well-controlled T1D patients on pump therapy with proper bolusing
        should easily achieve >70% TIR (consensus target). We use 75% as
        a reasonable lower bound for a simulated adherent patient.
        """
        results = _run_simulation(tmp_path, archetype="adherent")
        bg_values = results["BG"].values
        metrics = compute_metrics(bg_values)
        assert metrics.tir > 75.0, (
            f"Adherent TIR {metrics.tir:.1f}% below 75% threshold "
            f"(mean BG={metrics.mean_bg:.0f}, CV={metrics.cv:.1f}%)"
        )

    @pytest.mark.slow
    def test_nonadherent_tar_above_20(self, tmp_path: Path) -> None:
        """Nonadherent patient should spend >20% time above range.

        This validates that the adherence model creates meaningful
        differentiation from the adherent archetype. Nonadherent patients
        miss boluses and have poor timing, leading to sustained hyperglycemia.
        """
        results = _run_simulation(tmp_path, archetype="nonadherent")
        bg_values = results["BG"].values
        metrics = compute_metrics(bg_values)
        tar_total = metrics.tar_l1 + metrics.tar_l2
        assert tar_total > 20.0, (
            f"Nonadherent TAR {tar_total:.1f}% not above 20% — "
            f"adherence model may not differentiate enough"
        )

    @pytest.mark.slow
    def test_no_bg_below_15(self, tmp_path: Path) -> None:
        """No patient should ever reach BG < 15 mg/dL.

        Values below 15 mg/dL are incompatible with consciousness and
        represent model failure. The glucagon rescue system should prevent
        BG from dropping this low.
        """
        results = _run_simulation(tmp_path, archetype="nonadherent")
        bg_min = results["BG"].min()
        assert bg_min >= 15.0, (
            f"Min BG {bg_min:.1f} mg/dL below 15 — "
            f"glucagon rescue or hypo correction may have failed"
        )

    @pytest.mark.slow
    def test_severe_hypo_episodes_bounded(self, tmp_path: Path) -> None:
        """Nonadherent patient should have at most 5 severe hypo episodes in 7 days.

        Severe hypoglycemia (BG < 54 for >15 min) is a serious adverse event.
        Even nonadherent patients have physiological counter-regulatory responses
        and access to carbohydrate correction. More than 5 episodes in a week
        suggests the model is too aggressive or corrections are failing.
        """
        results = _run_simulation(tmp_path, archetype="nonadherent")
        bg_values = results["BG"].values
        metrics = compute_metrics(bg_values)
        assert metrics.severe_hypo_episodes <= 5, (
            f"Severe hypo episodes ({metrics.severe_hypo_episodes}) > 5 in 7 days — "
            f"hypo correction model may be insufficient"
        )

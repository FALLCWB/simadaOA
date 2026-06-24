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

"""Integration tests — run actual simglucose simulations end-to-end."""

from __future__ import annotations

from pathlib import Path

import pytest
from simglucose.simulation.sim_engine import sim

from simada.core.config import SimulationConfig
from simada.scenario.builder import ScenarioBuilder

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _make_config(
    duration_days: int = 1,
    cohort_size: int = 1,
    archetype: str = "adherent",
    output_dir: str = "results",
) -> SimulationConfig:
    """Create a minimal simulation config for testing."""
    arch_dist = {"adherent": 0.0, "moderate": 0.0, "nonadherent": 0.0}
    arch_dist[archetype] = 1.0

    return SimulationConfig.model_validate({
        "seed": 42,
        "scenario": {
            "name": "test",
            "duration_days": duration_days,
            "start_date": "2026-06-01",
            "cohort": {
                "size": cohort_size,
                "archetype_distribution": arch_dist,
                "insulin_regimen_distribution": {"pump": 1.0, "mdi": 0.0},
            },
        },
        "output": {"directory": output_dir},
    })


class TestSingleDaySimulation:
    """Integration tests running actual simglucose simulations."""

    @pytest.mark.slow
    def test_single_patient_one_day(self, tmp_path: Path) -> None:
        """Run 1 adherent patient for 1 day."""
        config = _make_config(output_dir=str(tmp_path))
        builder = ScenarioBuilder(config, PROJECT_ROOT)
        units = builder.build()

        assert len(units) == 1
        results = sim(units[0].sim_obj)

        assert hasattr(results, "columns")
        assert "BG" in results.columns
        assert "CGM" in results.columns
        assert len(results) > 0

        bg_min = results["BG"].min()
        bg_max = results["BG"].max()
        assert bg_min > 20, f"BG {bg_min:.0f} too low"
        assert bg_max < 600, f"BG {bg_max:.0f} too high"

    @pytest.mark.slow
    def test_single_patient_three_days(self, tmp_path: Path) -> None:
        """Run 1 adherent patient for 3 days."""
        config = _make_config(duration_days=3, output_dir=str(tmp_path))
        builder = ScenarioBuilder(config, PROJECT_ROOT)
        units = builder.build()
        results = sim(units[0].sim_obj)

        assert len(results) > 1000

        bg_min = results["BG"].min()
        bg_max = results["BG"].max()
        assert bg_min > 20
        assert bg_max < 600

    @pytest.mark.slow
    def test_nonadherent_completes_without_crash(self, tmp_path: Path) -> None:
        """Nonadherent patient should complete simulation without crashing."""
        config = _make_config(archetype="nonadherent", output_dir=str(tmp_path))
        builder = ScenarioBuilder(config, PROJECT_ROOT)
        units = builder.build()
        results = sim(units[0].sim_obj)

        assert len(results) > 0
        assert results["BG"].min() > 10  # nonadherent may go lower
        # Renal correction in simglucose caps BG at ~600 mg/dL. Using 900 as
        # the upper bound would mask a broken renal correction (cap at 600
        # but test allows up to 900 ⇒ regression undetected).
        assert results["BG"].max() < 600, (
            f"BG max {results['BG'].max():.0f} exceeds renal-correction cap (~600 mg/dL)"
        )

    @pytest.mark.slow
    def test_moderate_completes_without_crash(self, tmp_path: Path) -> None:
        """Moderate patient should complete simulation without crashing."""
        config = _make_config(archetype="moderate", output_dir=str(tmp_path))
        builder = ScenarioBuilder(config, PROJECT_ROOT)
        units = builder.build()
        results = sim(units[0].sim_obj)

        assert len(results) > 0
        assert results["BG"].min() > 10

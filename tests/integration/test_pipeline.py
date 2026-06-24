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

"""Integration tests for the full pipeline: config → run → Parquet output."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from simada.core.config import SimulationConfig
from simada.pipeline.runner import SimulationRunner

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TestPipeline:
    """Tests for the full simulation pipeline with Parquet output."""

    def _make_config(self, output_dir: str) -> SimulationConfig:
        return SimulationConfig.model_validate({
            "seed": 42,
            "scenario": {
                "name": "pipeline_test",
                "duration_days": 1,
                "start_date": "2026-06-01",
                "cohort": {
                    "size": 2,
                    "archetype_distribution": {
                        "adherent": 0.5,
                        "moderate": 0.0,
                        "nonadherent": 0.5,
                    },
                    "insulin_regimen_distribution": {"pump": 1.0, "mdi": 0.0},
                },
            },
            "output": {
                "directory": output_dir,
                "formats": ["parquet", "csv"],
            },
        })

    @pytest.mark.slow
    def test_pipeline_produces_parquet(self, tmp_path: Path) -> None:
        """Run 2 patients through the full pipeline, verify Parquet output."""
        config = self._make_config(str(tmp_path))
        runner = SimulationRunner(config, PROJECT_ROOT)
        output_dir = runner.run()

        # Check output structure
        ts_dir = output_dir / "timeseries"
        assert ts_dir.exists()

        parquet_files = list(ts_dir.glob("*.parquet"))
        assert len(parquet_files) == 2, f"Expected 2 Parquet files, got {len(parquet_files)}"

        csv_files = list(ts_dir.glob("*.csv"))
        assert len(csv_files) == 2

        # Check Parquet content
        for pf in parquet_files:
            table = pq.read_table(pf)
            df = table.to_pandas()
            assert len(df) > 0
            assert "BG" in df.columns

        # Check manifest
        manifest_path = output_dir / "metadata" / "manifest.json"
        assert manifest_path.exists()
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert manifest["total_patients"] == 2
        assert len(manifest["patients"]) == 2

        # Check config snapshot
        config_snap = output_dir / "metadata" / "config_snapshot.yaml"
        assert config_snap.exists()

    @pytest.mark.slow
    def test_pipeline_csv_readable(self, tmp_path: Path) -> None:
        """Verify CSV output is readable by pandas."""
        config = self._make_config(str(tmp_path))
        runner = SimulationRunner(config, PROJECT_ROOT)
        output_dir = runner.run()

        csv_files = list((output_dir / "timeseries").glob("*.csv"))
        for cf in csv_files:
            df = pd.read_csv(cf, index_col=0)
            assert len(df) > 0
            assert "BG" in df.columns
            assert df["BG"].min() > 10

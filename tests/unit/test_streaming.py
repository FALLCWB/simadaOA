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

"""Unit tests for StreamingSink (streaming Parquet/CSV output)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from simada.core.config import ArchetypeParams, load_archetype_params
from simada.core.types import ArchetypeID, InsulinRegimen
from simada.patient.cohort import PatientProfile
from simada.pipeline.streaming import StreamingSink

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _make_profile(index: int = 0) -> PatientProfile:
    """Create a minimal PatientProfile for testing."""
    params = load_archetype_params(
        PROJECT_ROOT / "configs" / "archetypes" / "adherent.yaml"
    )
    return PatientProfile(
        simglucose_name="adult#001",
        archetype_id=ArchetypeID.ADHERENT,
        archetype_params=params,
        insulin_regimen=InsulinRegimen.PUMP,
        cr=10.0,
        cf=50.0,
        tdi=40.0,
        patient_index=index,
    )


def _make_results_df(n_rows: int = 100) -> pd.DataFrame:
    """Create a minimal simulation-like DataFrame."""
    import numpy as np

    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "BG": rng.uniform(60, 300, n_rows),
        "CGM": rng.uniform(60, 300, n_rows),
        "CHO": rng.uniform(0, 80, n_rows),
        "insulin": rng.uniform(0, 10, n_rows),
    })


class TestStreamingSink:
    """Tests for StreamingSink."""

    def test_write_patient_results(self, tmp_path: Path) -> None:
        """Writing a DataFrame should create a Parquet file in timeseries/."""
        sink = StreamingSink(tmp_path, formats=["parquet"])
        profile = _make_profile()
        df = _make_results_df()

        sink.write_patient_results(df, profile)

        parquet_files = list((tmp_path / "timeseries").glob("*.parquet"))
        assert len(parquet_files) == 1, (
            f"Expected 1 Parquet file, found {len(parquet_files)}"
        )
        assert "adult001" in parquet_files[0].name
        assert "adherent" in parquet_files[0].name

    def test_finalize_creates_manifest(self, tmp_path: Path) -> None:
        """finalize() should create metadata/manifest.json with patient entries."""
        sink = StreamingSink(tmp_path, formats=["parquet"])
        profile = _make_profile()
        df = _make_results_df(50)

        sink.write_patient_results(df, profile)
        result_path = sink.finalize()

        manifest_path = result_path / "metadata" / "manifest.json"
        assert manifest_path.exists(), "manifest.json was not created"

        with open(manifest_path) as f:
            manifest = json.load(f)

        assert "patients" in manifest
        assert manifest["total_patients"] == 1
        assert manifest["patients"][0]["patient"] == "adult#001"
        assert manifest["patients"][0]["rows"] == 50

    def test_config_snapshot_saved(self, tmp_path: Path) -> None:
        """save_config_snapshot() should create metadata/config_snapshot.yaml."""
        sink = StreamingSink(tmp_path, formats=["parquet"])
        config = {"seed": 42, "scenario": {"duration_days": 7}}

        sink.save_config_snapshot(config)

        snapshot_path = tmp_path / "metadata" / "config_snapshot.yaml"
        assert snapshot_path.exists(), "config_snapshot.yaml was not created"

        import yaml

        with open(snapshot_path) as f:
            loaded = yaml.safe_load(f)

        assert loaded["seed"] == 42
        assert loaded["scenario"]["duration_days"] == 7

    def test_empty_dataframe(self, tmp_path: Path) -> None:
        """Writing an empty DataFrame should not crash.

        The sink should handle it gracefully -- the Parquet file will be
        written (albeit with zero rows) and the manifest should reflect
        n_rows=0.  We use a DataFrame with no BG column so the manifest
        records bg_min/bg_max as None (avoids NaN JSON serialization).
        """
        sink = StreamingSink(tmp_path, formats=["parquet"])
        profile = _make_profile()
        df = pd.DataFrame({"CHO": pd.Series(dtype=float)})

        # Should not raise
        sink.write_patient_results(df, profile)

        parquet_files = list((tmp_path / "timeseries").glob("*.parquet"))
        assert len(parquet_files) == 1

        result_path = sink.finalize()
        with open(result_path / "metadata" / "manifest.json") as f:
            manifest = json.load(f)
        assert manifest["patients"][0]["rows"] == 0
        assert manifest["patients"][0]["bg_min"] is None
        assert manifest["patients"][0]["bg_max"] is None

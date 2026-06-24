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

"""Unit tests for the parallel simulation module.

Tests serialization and data integrity of PatientJob and PatientResult
without actually running simglucose (which requires the full simulation
environment).
"""

from __future__ import annotations

import pickle

from simada.pipeline.parallel import PatientJob, PatientResult


class TestPatientJob:
    """Tests for PatientJob serializability."""

    def test_patient_job_serializable(self) -> None:
        """PatientJob must be picklable since ProcessPoolExecutor sends it
        to worker processes via pickle.
        """
        job = PatientJob(
            config_dict={"seed": 42, "scenario": {"duration_days": 7}},
            project_root="/tmp/simada",
            patient_index=5,
            simglucose_name="adult#003",
            archetype="adherent",
            cr=10.0,
            cf=50.0,
            tdi=40.0,
            insulin_regimen="pump",
            output_dir="/tmp/output",
            master_seed=42,
        )

        # Round-trip through pickle
        data = pickle.dumps(job)
        restored = pickle.loads(data)

        assert restored.patient_index == job.patient_index
        assert restored.simglucose_name == job.simglucose_name
        assert restored.archetype == job.archetype
        assert restored.cr == job.cr
        assert restored.cf == job.cf
        assert restored.tdi == job.tdi
        assert restored.insulin_regimen == job.insulin_regimen
        assert restored.master_seed == job.master_seed
        assert restored.config_dict == job.config_dict

    def test_patient_job_round_trip_preserves_config(self) -> None:
        """Config dict must survive serialization without loss."""
        nested_config = {
            "seed": 99,
            "scenario": {
                "name": "test",
                "duration_days": 14,
                "cohort": {"size": 10},
            },
            "meals": {"locale": "brazil"},
        }
        job = PatientJob(
            config_dict=nested_config,
            project_root="/tmp/simada",
            patient_index=0,
            simglucose_name="adolescent#001",
            archetype="nonadherent",
            cr=12.0,
            cf=40.0,
            tdi=35.0,
            insulin_regimen="mdi",
            output_dir="/tmp/out",
            master_seed=123,
        )

        restored = pickle.loads(pickle.dumps(job))
        assert restored.config_dict == nested_config
        assert restored.config_dict["scenario"]["cohort"]["size"] == 10


class TestPatientResult:
    """Tests for PatientResult structure and fields."""

    def test_patient_result_fields(self) -> None:
        """PatientResult must have all expected fields for downstream processing."""
        result = PatientResult(
            simglucose_name="adult#001",
            archetype="adherent",
            patient_index=0,
            parquet_path="/tmp/results/adult001_adherent_000.parquet",
            n_rows=2016,
            bg_min=54.3,
            bg_max=312.7,
            error=None,
        )

        assert result.simglucose_name == "adult#001"
        assert result.archetype == "adherent"
        assert result.patient_index == 0
        assert result.parquet_path is not None
        assert result.n_rows == 2016
        assert result.bg_min == 54.3
        assert result.bg_max == 312.7
        assert result.error is None

    def test_patient_result_with_error(self) -> None:
        """PatientResult should cleanly represent a failed simulation."""
        result = PatientResult(
            simglucose_name="child#001",
            archetype="moderate",
            patient_index=3,
            parquet_path=None,
            n_rows=0,
            bg_min=None,
            bg_max=None,
            error="SimObj crashed: some internal error",
        )

        assert result.parquet_path is None
        assert result.n_rows == 0
        assert result.bg_min is None
        assert result.bg_max is None
        assert "crashed" in result.error

    def test_patient_result_serializable(self) -> None:
        """PatientResult should also survive pickle for IPC."""
        result = PatientResult(
            simglucose_name="adult#010",
            archetype="moderate",
            patient_index=9,
            parquet_path="/tmp/out/adult010_moderate_009.parquet",
            n_rows=1440,
            bg_min=62.1,
            bg_max=285.4,
            error=None,
        )

        restored = pickle.loads(pickle.dumps(result))
        assert restored.simglucose_name == result.simglucose_name
        assert restored.n_rows == result.n_rows
        assert restored.bg_min == result.bg_min

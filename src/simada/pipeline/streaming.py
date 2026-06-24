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

"""Streaming result sink for memory-efficient output.

Writes simulation results incrementally to Parquet and/or CSV files,
partitioned by patient. Keeps memory usage constant regardless of
cohort size by writing each patient's results immediately after
simulation completes.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from simada.patient.cohort import PatientProfile


def _safe_float_for_json(val: float | None) -> float | None:
    """Convert a float to None if it is NaN or Inf, else return it.

    Used when writing numeric values to JSON manifests. ``json.dump``
    without ``allow_nan=False`` silently serialises NaN as the literal
    ``NaN`` which is rejected by strict JSON parsers.  This guard
    converts any non-finite value to JSON-safe ``null``.
    """
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


class StreamingSink:
    """Writes simulation results to disk incrementally.

    Output structure::

        output_dir/
            timeseries/
                adult_001_adherent.parquet
                adult_002_nonadherent.parquet
                ...
            metadata/
                config_snapshot.yaml
                manifest.json
    """

    def __init__(self, output_dir: Path, formats: list[str]) -> None:
        self._output_dir = output_dir
        self._formats = formats
        self._ts_dir = output_dir / "timeseries"
        self._ts_dir.mkdir(parents=True, exist_ok=True)
        self._manifest: list[dict[str, Any]] = []

    def write_patient_results(
        self,
        results: pd.DataFrame,
        profile: PatientProfile,
    ) -> None:
        """Write one patient's simulation results to disk.

        Args:
            results: DataFrame from simglucose with BG, CGM, CHO, insulin, etc.
            profile: The patient's profile for naming and metadata.
        """
        safe_name = profile.simglucose_name.replace("#", "")
        filename_base = f"{safe_name}_{profile.archetype_id.value}_{profile.patient_index:03d}"

        if "parquet" in self._formats:
            path = self._ts_dir / f"{filename_base}.parquet"
            table = pa.Table.from_pandas(results)
            pq.write_table(table, path, compression="snappy")

        if "csv" in self._formats:
            path = self._ts_dir / f"{filename_base}.csv"
            results.to_csv(path, index=True)

        # BUG 7: BG.min()/max() on an all-NaN column returns NaN which
        # json.dump serialises as the literal `NaN` — rejected by strict JSON
        # parsers. Guard with _safe_float_for_json which converts NaN/Inf to None.
        bg_min: float | None = None
        bg_max: float | None = None
        if "BG" in results.columns and len(results) > 0:
            bg_min = _safe_float_for_json(results["BG"].min())
            bg_max = _safe_float_for_json(results["BG"].max())

        self._manifest.append({
            "patient": profile.simglucose_name,
            "archetype": profile.archetype_id.value,
            "regimen": profile.insulin_regimen.value,
            "cr": profile.cr,
            "cf": profile.cf,
            "tdi": profile.tdi,
            "rows": len(results),
            "bg_min": bg_min,
            "bg_max": bg_max,
        })

    def save_config_snapshot(self, config_dict: dict[str, Any]) -> None:
        """Save the configuration used for this run."""
        meta_dir = self._output_dir / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        with open(meta_dir / "config_snapshot.yaml", "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def finalize(self) -> Path:
        """Write the manifest and return the output directory path."""
        meta_dir = self._output_dir / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        with open(meta_dir / "manifest.json", "w") as f:
            json.dump(
                {
                    "created": datetime.now().isoformat(),
                    "patients": self._manifest,
                    "total_patients": len(self._manifest),
                },
                f,
                indent=2,
            )
        return self._output_dir

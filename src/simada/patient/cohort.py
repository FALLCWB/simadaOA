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

"""Patient cohort generation.

Generates a cohort of virtual patients by combining simglucose's 30
physiological models with simada's adherence archetypes. Patients are
assigned archetypes according to a configurable distribution.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from importlib.resources import files
from numpy.random import Generator

from simada.core.config import ArchetypeParams, CohortConfig, load_archetype_params
from simada.core.types import ArchetypeID, InsulinRegimen


class PatientProfile(NamedTuple):
    """A virtual patient combining physiological model and adherence archetype."""

    simglucose_name: str
    archetype_id: ArchetypeID
    archetype_params: ArchetypeParams
    insulin_regimen: InsulinRegimen
    cr: float  # carb ratio (g/U) from Quest.csv
    cf: float  # correction factor (mg/dL per U) from Quest.csv
    tdi: float  # total daily insulin (U) from Quest.csv
    patient_index: int  # index for RNG derivation


def _load_quest_data() -> pd.DataFrame:
    """Load simglucose Quest.csv with patient clinical parameters."""
    quest_path = str(files("simglucose").joinpath("params", "Quest.csv"))
    return pd.read_csv(quest_path)


def _get_simglucose_patient_names() -> list[str]:
    """Get all available simglucose patient names."""
    params_path = str(files("simglucose").joinpath("params", "vpatient_params.csv"))
    df = pd.read_csv(params_path)
    return list(df["Name"])


def _interleave_exact_archetypes(counts: dict[str, int], n: int) -> list[str]:
    """Produce an interleaved archetype assignment honouring exact counts.

    Archetypes are emitted round-robin (in sorted order for determinism) so
    that, when mapped onto the ordered simglucose pool, each archetype receives
    a balanced spread of physiologies instead of a contiguous block. For
    ``{"adherent": 10, "moderate": 10, "nonadherent": 10}`` and ``n=30`` this
    yields ``[adherent, moderate, nonadherent, adherent, ...]`` — exactly 10 of
    each, spread evenly across adolescent/adult/child groups.
    """
    arch_ids = sorted(counts.keys())
    remaining = {a: int(counts[a]) for a in arch_ids}
    out: list[str] = []
    while len(out) < n:
        progressed = False
        for a in arch_ids:
            if remaining[a] > 0:
                out.append(a)
                remaining[a] -= 1
                progressed = True
                if len(out) == n:
                    break
        if not progressed:  # counts summed to less than n (guarded upstream)
            break
    return out


class CohortGenerator:
    """Generates patient cohorts with configurable archetype distributions.

    Usage::

        gen = CohortGenerator(config, archetype_configs_dir, rng)
        cohort = gen.generate()
    """

    def __init__(
        self,
        config: CohortConfig,
        archetype_configs_dir: Path,
        rng: Generator,
    ) -> None:
        self._config = config
        self._rng = rng
        self._quest = _load_quest_data()
        self._all_patients = _get_simglucose_patient_names()

        # Load archetype params from YAML
        self._archetype_params: dict[ArchetypeID, ArchetypeParams] = {}
        for arch_id in ArchetypeID:
            yaml_path = archetype_configs_dir / f"{arch_id.value}.yaml"
            self._archetype_params[arch_id] = load_archetype_params(yaml_path)

    def generate(self) -> list[PatientProfile]:
        """Generate the configured number of patient profiles.

        Assigns archetypes and insulin regimens according to the configured
        distributions. Cycles through simglucose patients if cohort size
        exceeds 30.
        """
        n = self._config.size

        # Determine which simglucose patients to use
        if self._config.patient_pool == "all":
            pool = self._all_patients
        else:
            pool = list(self._config.patient_pool)

        # Assign archetypes.
        if self._config.archetype_counts is not None:
            # EXACT counts: deterministic, balanced design (e.g. 10/10/10).
            # Interleave (round-robin) the requested archetypes across the pool
            # so each archetype receives a spread of physiologies rather than a
            # contiguous block (which would confound archetype with age group,
            # since the simglucose pool is ordered adolescent->adult->child).
            arch_assignments = _interleave_exact_archetypes(
                self._config.archetype_counts, n
            )
        else:
            # Probabilistic assignment.
            # BUG #4 fix (H2): sort arch_ids so the assignment is independent of
            # dict insertion order.  Python 3.7+ preserves insertion order but
            # configs with the same probabilities in a different key order would
            # otherwise produce different archetype assignments from the same
            # seed.
            arch_ids = sorted(self._config.archetype_distribution.keys())
            arch_probs = [self._config.archetype_distribution[a] for a in arch_ids]
            arch_assignments = self._rng.choice(arch_ids, size=n, p=arch_probs)

        # Sample insulin regimens
        reg_ids = list(self._config.insulin_regimen_distribution.keys())
        reg_probs = [self._config.insulin_regimen_distribution[r] for r in reg_ids]
        reg_assignments = self._rng.choice(reg_ids, size=n, p=reg_probs)

        # When the cohort size exceeds the pool, patients are cycled.
        #
        # CRITICAL FIX: the previous implementation appended a "_v{n}" suffix to
        # the cycled name (e.g. "adult#009_v2"). That string is NOT a valid
        # simglucose model name, so ``T1DPatient.withName`` raised and crashed
        # every cohort larger than the 30-patient pool; it also missed the
        # Quest.csv row, silently falling back to CR=10/CF=50/TDI=40 and losing
        # the real clinical parameters. We now keep the VALID base simglucose
        # name so the physiological model resolves and the real CR/CF/TDI are
        # used. Profile uniqueness is already guaranteed by ``patient_index``
        # (0..n-1), which is part of every output filename and drives the
        # per-patient RNG sub-stream, so cycled patients never collide on disk
        # or in their stochastic realisation despite sharing a model name.
        _pool_size = len(pool)
        _cycled = False

        profiles: list[PatientProfile] = []
        for i in range(n):
            sg_name = pool[i % _pool_size]
            if i >= _pool_size and not _cycled:
                _cycled = True
                warnings.warn(
                    f"CohortGenerator: cohort size ({n}) exceeds pool size "
                    f"({_pool_size}); simglucose physiological models are reused "
                    "(cycled). Each reused patient keeps its real clinical "
                    "parameters and a distinct patient_index / RNG stream, so "
                    "stochastic realisations differ. Group results by "
                    "patient_index, not simglucose_name.",
                    stacklevel=2,
                )
            arch_str = str(arch_assignments[i])
            arch_id = ArchetypeID(arch_str)
            reg_str = str(reg_assignments[i])
            regimen = InsulinRegimen(reg_str)

            # Get clinical params from Quest.csv
            quest_row = self._quest[self._quest["Name"] == sg_name]
            if quest_row.empty:
                # BUG #10 fix (H2): warn when a non-synthetic patient is missing
                # from Quest.csv — this masks a data integrity issue.  Synthetic
                # patients (name starts with "synthetic_") are expected to not
                # appear in Quest.csv and are silently assigned defaults.
                if not sg_name.startswith("synthetic_"):
                    warnings.warn(
                        f"CohortGenerator: patient {sg_name!r} not found in "
                        "Quest.csv; using fallback CR=10, CF=50, TDI=40. "
                        "Verify the patient name or supply a custom Quest.csv.",
                        stacklevel=2,
                    )
                cr, cf, tdi = 10.0, 50.0, 40.0
            else:
                row = quest_row.iloc[0]
                cr = float(row["CR"])
                cf = float(row["CF"])
                tdi = float(row["TDI"])

            profiles.append(
                PatientProfile(
                    simglucose_name=sg_name,
                    archetype_id=arch_id,
                    archetype_params=self._archetype_params[arch_id],
                    insulin_regimen=regimen,
                    cr=cr,
                    cf=cf,
                    tdi=tdi,
                    patient_index=i,
                )
            )

        return profiles

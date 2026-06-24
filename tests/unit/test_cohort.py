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

"""Tests for patient cohort generation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from numpy.random import default_rng
from pytest import approx

from simada.core.config import CohortConfig
from simada.core.types import ArchetypeID
from simada.patient.cohort import CohortGenerator, PatientProfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARCHETYPES_DIR = PROJECT_ROOT / "configs" / "archetypes"


def _make_cohort_config(
    size: int = 30,
    archetype_distribution: dict[str, float] | None = None,
) -> CohortConfig:
    """Helper to create a CohortConfig with sensible defaults."""
    if archetype_distribution is None:
        archetype_distribution = {
            "adherent": 0.30,
            "moderate": 0.50,
            "nonadherent": 0.20,
        }
    return CohortConfig(
        size=size,
        archetype_distribution=archetype_distribution,
        insulin_regimen_distribution={"pump": 0.60, "mdi": 0.40},
    )


class TestCohortGenerator:
    """Tests for CohortGenerator."""

    def test_generate_correct_size(self) -> None:
        """A cohort configured for 30 patients produces exactly 30 PatientProfiles."""
        config = _make_cohort_config(size=30)
        gen = CohortGenerator(config, ARCHETYPES_DIR, default_rng(42))
        cohort = gen.generate()

        assert len(cohort) == 30
        assert all(isinstance(p, PatientProfile) for p in cohort)

    def test_archetype_distribution(self) -> None:
        """With 100 patients and known distribution, counts are approximately correct.

        Uses generous tolerance to avoid flaky tests while still catching
        gross errors in the sampling logic.
        """
        dist = {"adherent": 0.50, "moderate": 0.30, "nonadherent": 0.20}
        config = _make_cohort_config(size=100, archetype_distribution=dist)
        gen = CohortGenerator(config, ARCHETYPES_DIR, default_rng(123))
        cohort = gen.generate()

        counts = Counter(p.archetype_id for p in cohort)
        # Generous tolerance: within +/- 15 of expected count for n=100
        assert counts[ArchetypeID.ADHERENT] == approx(50, abs=15)
        assert counts[ArchetypeID.MODERATE] == approx(30, abs=15)
        assert counts[ArchetypeID.NONADHERENT] == approx(20, abs=15)

    def test_archetype_counts_exact_and_balanced(self) -> None:
        """archetype_counts assigns EXACTLY the requested numbers, interleaved
        across the pool so each archetype gets a balanced physiology spread.
        """
        config = CohortConfig(
            size=30,
            archetype_counts={"adherent": 10, "moderate": 10, "nonadherent": 10},
            insulin_regimen_distribution={"pump": 1.0, "mdi": 0.0},
        )
        gen = CohortGenerator(config, ARCHETYPES_DIR, default_rng(42))
        cohort = gen.generate()

        counts = Counter(p.archetype_id for p in cohort)
        assert counts[ArchetypeID.ADHERENT] == 10
        assert counts[ArchetypeID.MODERATE] == 10
        assert counts[ArchetypeID.NONADHERENT] == 10
        # Interleaved (round-robin in sorted order: adherent, moderate, nonadh.)
        assert cohort[0].archetype_id == ArchetypeID.ADHERENT
        assert cohort[1].archetype_id == ArchetypeID.MODERATE
        assert cohort[2].archetype_id == ArchetypeID.NONADHERENT
        assert cohort[3].archetype_id == ArchetypeID.ADHERENT
        # Each archetype spans more than one age group (not a contiguous block).
        adherent_models = {
            p.simglucose_name.split("#")[0]
            for p in cohort
            if p.archetype_id == ArchetypeID.ADHERENT
        }
        assert len(adherent_models) >= 2

    def test_archetype_counts_must_sum_to_size(self) -> None:
        """archetype_counts that do not sum to size are rejected at config time."""
        with pytest.raises(ValueError, match="must sum to cohort size"):
            CohortConfig(
                size=30,
                archetype_counts={"adherent": 10, "moderate": 10, "nonadherent": 5},
                insulin_regimen_distribution={"pump": 1.0, "mdi": 0.0},
            )

    def test_quest_csv_lookup(self) -> None:
        """A known simglucose patient gets CR/CF/TDI from Quest.csv."""
        # Use a small pool containing only adult#001 to guarantee it's assigned
        config = CohortConfig(
            size=1,
            patient_pool=["adult#001"],
            archetype_distribution={"adherent": 1.0, "moderate": 0.0, "nonadherent": 0.0},
            insulin_regimen_distribution={"pump": 1.0, "mdi": 0.0},
        )
        gen = CohortGenerator(config, ARCHETYPES_DIR, default_rng(42))
        cohort = gen.generate()

        p = cohort[0]
        assert p.simglucose_name == "adult#001"
        # adult#001 has known values from Quest.csv — just check they're positive numbers
        assert p.cr > 0
        assert p.cf > 0
        assert p.tdi > 0

    def test_quest_csv_fallback(self) -> None:
        """Unknown patient name gets default cr=10, cf=50, tdi=40.

        Bug #10 fix (H2): a UserWarning is now emitted when the patient is not
        found in Quest.csv and the name does not match "synthetic_*".
        """
        import warnings

        config = CohortConfig(
            size=1,
            patient_pool=["nonexistent_patient"],
            archetype_distribution={"adherent": 1.0, "moderate": 0.0, "nonadherent": 0.0},
            insulin_regimen_distribution={"pump": 1.0, "mdi": 0.0},
        )
        gen = CohortGenerator(config, ARCHETYPES_DIR, default_rng(42))
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            cohort = gen.generate()

        p = cohort[0]
        assert p.cr == approx(10.0)
        assert p.cf == approx(50.0)
        assert p.tdi == approx(40.0)
        # The warning must be emitted for non-synthetic unknown patients.
        assert any(
            issubclass(w.category, UserWarning) and "Quest.csv" in str(w.message)
            for w in captured
        ), "Expected a UserWarning about missing Quest.csv entry"

    def test_patient_pool_cycling(self) -> None:
        """Cohort size > 30 cycles through simglucose patients.

        CRITICAL fix: cycled patients keep the VALID base simglucose name (so
        ``T1DPatient.withName`` resolves and the real Quest.csv CR/CF/TDI are
        used) instead of an unresolvable ``_v<n>`` suffix that crashed every
        cohort larger than the 30-patient pool. Uniqueness is carried by
        ``patient_index`` (used in output filenames + per-patient RNG), NOT by
        the model name.
        """
        import warnings

        config = _make_cohort_config(size=35)
        gen = CohortGenerator(config, ARCHETYPES_DIR, default_rng(42))
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            cohort = gen.generate()

        assert len(cohort) == 35
        # Cycled entry 30 reuses pool[0]'s VALID model name (no "_v" suffix).
        assert cohort[30].simglucose_name == cohort[0].simglucose_name
        assert "_v" not in cohort[30].simglucose_name
        assert cohort[31].simglucose_name == cohort[1].simglucose_name
        # The real clinical parameters must be preserved on cycling (the bug
        # silently dropped them to CR=10/CF=50/TDI=40 fallback defaults).
        assert cohort[30].cr == cohort[0].cr
        assert cohort[30].cf == cohort[0].cf
        assert cohort[30].tdi == cohort[0].tdi
        # patient_index — not the name — guarantees per-profile uniqueness.
        all_indices = [p.patient_index for p in cohort]
        assert len(all_indices) == len(set(all_indices)), "patient_index must be unique"
        assert all_indices == list(range(35))

    def test_deterministic_with_seed(self) -> None:
        """Same seed produces the exact same cohort twice."""
        config = _make_cohort_config(size=10)

        gen1 = CohortGenerator(config, ARCHETYPES_DIR, default_rng(999))
        cohort1 = gen1.generate()

        gen2 = CohortGenerator(config, ARCHETYPES_DIR, default_rng(999))
        cohort2 = gen2.generate()

        for p1, p2 in zip(cohort1, cohort2, strict=True):
            assert p1.simglucose_name == p2.simglucose_name
            assert p1.archetype_id == p2.archetype_id
            assert p1.insulin_regimen == p2.insulin_regimen
            assert p1.cr == approx(p2.cr)
            assert p1.cf == approx(p2.cf)
            assert p1.tdi == approx(p2.tdi)


# ---------------------------------------------------------------------------
# Regression tests for BUG #5: CR/CF error factor must be clamped on BOTH
# sides, not just from below. This sits next to the cohort tests because
# the archetype lives in the same patient/ package.
# ---------------------------------------------------------------------------


class TestArchetypeErrorClamping:
    """Tests that sample_cr_error / sample_cf_error clamp to [0.1, 3.0]."""

    def _make_archetype(self, cr_mean: float, cf_mean: float, std: float):
        from simada.core.config import load_archetype_params
        from simada.patient.archetype import create_archetype

        # Start from a real YAML so all the unrelated fields are valid,
        # then override the four CR/CF error fields with extreme values.
        base = load_archetype_params(ARCHETYPES_DIR / "nonadherent.yaml")
        params = base.model_copy(
            update={
                "cr_error_factor_mean": cr_mean,
                "cr_error_factor_std": std,
                "cf_error_factor_mean": cf_mean,
                "cf_error_factor_std": std,
            }
        )
        return create_archetype(ArchetypeID.NONADHERENT, params)

    def test_cr_error_clamped_above(self) -> None:
        """A pathologically high mean must be clamped to 3.0, not unbounded."""
        archetype = self._make_archetype(cr_mean=10.0, cf_mean=1.0, std=0.0)
        rng = default_rng(0)
        for _ in range(50):
            assert archetype.sample_cr_error(rng) <= 3.0

    def test_cf_error_clamped_above(self) -> None:
        """Same upper bound applies to CF error."""
        archetype = self._make_archetype(cr_mean=1.0, cf_mean=10.0, std=0.0)
        rng = default_rng(0)
        for _ in range(50):
            assert archetype.sample_cf_error(rng) <= 3.0

    def test_cr_cf_error_still_clamped_below(self) -> None:
        """The lower clamp at 0.1 must continue to hold (regression)."""
        archetype = self._make_archetype(cr_mean=-5.0, cf_mean=-5.0, std=0.0)
        rng = default_rng(0)
        for _ in range(50):
            assert archetype.sample_cr_error(rng) >= 0.1
            assert archetype.sample_cf_error(rng) >= 0.1

    def test_cr_cf_error_within_band_unchanged(self) -> None:
        """Values inside [0.1, 3.0] must pass through unmodified (no clamp)."""
        archetype = self._make_archetype(cr_mean=1.2, cf_mean=0.9, std=0.0)
        rng = default_rng(0)
        # With std=0 the sample equals the mean exactly.
        assert archetype.sample_cr_error(rng) == approx(1.2)
        assert archetype.sample_cf_error(rng) == approx(0.9)

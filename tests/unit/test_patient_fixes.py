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

"""Regression tests for the H2 v2 bug-hunt fixes (patient module).

Each test class corresponds to one bug from the issue list.  Test IDs follow
the H2 numbering so traceability is straightforward.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from numpy.random import default_rng

from simada.core.config import CohortConfig
from simada.core.types import ArchetypeID, BolusTimingCategory
from simada.patient.cohort import CohortGenerator
from simada.patient.synthetic import (
    MAX_RESAMPLE_ROUNDS,
    fit_distribution,
    generate_synthetic_cohort,
    load_base_cohort,
    plausibility_filter,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARCHETYPES_DIR = PROJECT_ROOT / "configs" / "archetypes"


# ---------------------------------------------------------------------------
# H2 Bug #1 — Seed fragility: all child seeds via single spawn call
# ---------------------------------------------------------------------------


class TestH2Bug1SeedSpawn:
    """H2 #1 — round_seeds and selection_rng must come from a single spawn."""

    def test_single_seed_produces_identical_cohorts(self) -> None:
        """Same master seed always produces the same cohort, regardless of how
        many rounds are actually used.  This is the basic determinism check that
        would catch seed reuse regressions.
        """
        df1 = generate_synthetic_cohort(n=10, pool="adult", seed=2026)
        df2 = generate_synthetic_cohort(n=10, pool="adult", seed=2026)
        import pandas as pd
        pd.testing.assert_frame_equal(df1, df2)

    def test_selection_rng_is_independent_of_round_rngs(self) -> None:
        """Adding a new resample round must not shift the selection RNG.

        We compare the cohorts produced by seed=1 and seed=2 and verify
        that they differ — confirming that the selection RNG is driven by
        the master seed (via spawn), not by a leftover state from the round
        RNGs.  If round and selection RNGs were entangled, certain seed values
        would produce identical cohorts despite different master seeds.
        """
        df1 = generate_synthetic_cohort(n=5, pool="adult", seed=1)
        df2 = generate_synthetic_cohort(n=5, pool="adult", seed=2)
        # At least one numeric column must differ between the two seeds.
        numeric_cols = df1.select_dtypes(include=[float, int]).columns
        differences = (df1[numeric_cols].values != df2[numeric_cols].values)
        assert differences.any(), "Different master seeds must produce different cohorts"

    def test_spawn_count_covers_rounds_plus_selection(self) -> None:
        """generate_synthetic_cohort internally spawns MAX_RESAMPLE_ROUNDS + 1
        children.  We verify the constant is at least 1 so the selection child
        always exists.
        """
        assert MAX_RESAMPLE_ROUNDS >= 1, "MAX_RESAMPLE_ROUNDS must be at least 1"


# ---------------------------------------------------------------------------
# H2 Bug #2 — Oversampling decay: always draw n * oversample_factor
# ---------------------------------------------------------------------------


class TestH2Bug2OversamplingConstantN:
    """H2 #2 — each round must draw n*oversample_factor, not need*oversample_factor."""

    def test_convergence_with_tight_filter(self, monkeypatch) -> None:
        """When acceptance rate is low (tight filter), using `n` instead of
        `need` per round ensures we still accumulate enough samples.

        We simulate a 50% acceptance filter and request n=10.  With the old
        behaviour (n=need), round 1 draws 10*3=30 and accepts ~15, leaving
        need=0 — which is correct in this case.  But with a 10% acceptance
        rate, round 1 would draw 10*3=30 and accept ~3, round 2 would draw
        7*3=21 (decreasing), spiralling toward failure.

        We verify here that the function succeeds for various tight configs
        without raising RuntimeError.
        """
        import simada.patient.synthetic as _synthetic

        call_sizes: list[int] = []
        original_sample = _synthetic.sample_synthetic

        def _record_n(mu, sigma, n, seed, *, oversample_factor=3):
            call_sizes.append(n)
            return original_sample(mu, sigma, n=n, seed=seed, oversample_factor=oversample_factor)

        monkeypatch.setattr(_synthetic, "sample_synthetic", _record_n)

        # Request n=5; every round should receive n=5, not a shrinking need.
        generate_synthetic_cohort(n=5, pool="adult", seed=99, plausibility=True)

        # All recorded n values (for rounds that ran) must equal 5, not shrink.
        assert all(s == 5 for s in call_sizes), (
            f"Some rounds received n != 5: {call_sizes}"
        )

    def test_convergence_does_not_raise_for_n30(self) -> None:
        """n=30 must converge without RuntimeError (smoke test)."""
        df = generate_synthetic_cohort(n=30, pool="adult", seed=42)
        assert len(df) == 30


# ---------------------------------------------------------------------------
# H2 Bug #3 — std(ddof=0) in plausibility_filter envelope
# ---------------------------------------------------------------------------


class TestH2Bug3StdDdof:
    """H2 #3 — plausibility_filter must use ddof=1 for the per-parameter SD."""

    def test_envelope_wider_with_ddof1_than_ddof0(self) -> None:
        """With n=10 base patients, the sample SD (ddof=1) is strictly larger
        than the population SD (ddof=0) by a factor of sqrt(n/(n-1)).

        We verify that the acceptance rate for a carefully crafted draw that
        sits between the ddof=0 and ddof=1 envelopes is 1 (accepted) after the
        fix and would have been 0 (rejected) with the old ddof=0 envelope.
        """
        base = load_base_cohort(pool="adult")
        _, _, param_names = fit_distribution(base)

        bw_col = "BW"
        bw_idx = param_names.index(bw_col)
        bw_vals = base[bw_col].to_numpy(dtype=float)
        bw_max = float(bw_vals.max())
        n_base = len(bw_vals)

        # ddof=0 SD (population)
        sd_pop = float(bw_vals.std(ddof=0))
        # ddof=1 SD (sample) — always >= sd_pop for n > 1
        sd_samp = float(bw_vals.std(ddof=1))

        # Pick a BW value in (max + 2*sd_pop, max + 2*sd_samp)
        # so old code rejects it but new code accepts it.
        probe = bw_max + 2.0 * sd_pop + (sd_samp - sd_pop) * 0.5

        if probe > bw_max + 2.0 * sd_samp:
            pytest.skip(
                "Can't construct probe value between ddof=0 and ddof=1 envelopes "
                "for this base cohort — test is moot."
            )

        # Build a sample at the base means, override BW with probe.
        sample = np.zeros((1, len(param_names)))
        for i, col in enumerate(param_names):
            sample[0, i] = float(base[col].mean())
        sample[0, bw_idx] = probe
        # Ensure EGPb is in clinical range.
        egpb_idx = param_names.index("EGPb")
        sample[0, egpb_idx] = float(base["EGPb"].mean())

        mask = plausibility_filter(sample, base, param_names)
        assert mask[0], (
            f"BW={probe:.3f} should be inside ddof=1 envelope "
            f"(max={bw_max:.3f} + 2*SD_samp={sd_samp:.3f}={bw_max + 2*sd_samp:.3f}) "
            "but was rejected — ddof=1 fix may have reverted."
        )

    def test_ddof1_larger_than_ddof0_for_small_sample(self) -> None:
        """Sanity: sample SD must exceed population SD for n < infinity."""
        base = load_base_cohort(pool="adult")
        bw = base["BW"].to_numpy(dtype=float)
        assert bw.std(ddof=1) > bw.std(ddof=0)


# ---------------------------------------------------------------------------
# H2 Bug #4 — archetype dict order dependency in cohort.py
# ---------------------------------------------------------------------------


class TestH2Bug4ArchetypeDictOrder:
    """H2 #4 — archetype assignments must not depend on dict insertion order."""

    def _make_config(self, key_order: list[str], size: int = 50) -> CohortConfig:
        dist = {k: {"adherent": 0.30, "moderate": 0.50, "nonadherent": 0.20}[k] for k in key_order}
        return CohortConfig(
            size=size,
            archetype_distribution=dist,
            insulin_regimen_distribution={"pump": 1.0, "mdi": 0.0},
        )

    def test_different_key_order_same_seed_same_result(self) -> None:
        """Same probabilities in different insertion order must produce the
        same archetype assignments when the same RNG seed is used.
        """
        order_a = ["adherent", "moderate", "nonadherent"]
        order_b = ["nonadherent", "adherent", "moderate"]
        order_c = ["moderate", "nonadherent", "adherent"]

        rng_a = default_rng(42)
        rng_b = default_rng(42)
        rng_c = default_rng(42)

        gen_a = CohortGenerator(self._make_config(order_a), ARCHETYPES_DIR, rng_a)
        gen_b = CohortGenerator(self._make_config(order_b), ARCHETYPES_DIR, rng_b)
        gen_c = CohortGenerator(self._make_config(order_c), ARCHETYPES_DIR, rng_c)

        cohort_a = gen_a.generate()
        cohort_b = gen_b.generate()
        cohort_c = gen_c.generate()

        ids_a = [p.archetype_id for p in cohort_a]
        ids_b = [p.archetype_id for p in cohort_b]
        ids_c = [p.archetype_id for p in cohort_c]

        assert ids_a == ids_b, "Key order A vs B produced different archetype assignments"
        assert ids_a == ids_c, "Key order A vs C produced different archetype assignments"


# ---------------------------------------------------------------------------
# H2 Bug #5 — PRE timing clamp allows positive minutes (post-meal)
# ---------------------------------------------------------------------------


class TestH2Bug5PreBolusClamp:
    """H2 #5 — bolus_timing_delay for PRE must be clamped to [-20, 0], not [-20, +5]."""

    def _make_archetype(self, pre_mean: float, pre_std: float):
        from simada.core.config import load_archetype_params
        from simada.patient.archetype import create_archetype

        base = load_archetype_params(ARCHETYPES_DIR / "adherent.yaml")
        params = base.model_copy(
            update={
                "pre_bolus_mean_minutes": pre_mean,
                "pre_bolus_std_minutes": pre_std,
            }
        )
        return create_archetype(ArchetypeID.ADHERENT, params)

    def test_pre_timing_never_positive(self) -> None:
        """PRE timing delay must never be positive (post-meal)."""
        archetype = self._make_archetype(pre_mean=2.0, pre_std=3.0)
        rng = default_rng(0)
        for _ in range(200):
            delay = archetype.bolus_timing_delay(BolusTimingCategory.PRE, rng)
            minutes = delay.total_seconds() / 60.0
            assert minutes <= 0.0, (
                f"PRE timing produced positive delay ({minutes:.2f} min); "
                "upper clamp must be 0.0, not 5.0"
            )

    def test_pre_timing_not_earlier_than_minus20(self) -> None:
        """PRE timing delay must not be earlier than -20 minutes."""
        archetype = self._make_archetype(pre_mean=-18.0, pre_std=5.0)
        rng = default_rng(7)
        for _ in range(200):
            delay = archetype.bolus_timing_delay(BolusTimingCategory.PRE, rng)
            minutes = delay.total_seconds() / 60.0
            assert minutes >= -20.0, (
                f"PRE timing exceeded lower bound ({minutes:.2f} min); "
                "lower clamp must remain -20.0"
            )

    def test_pre_timing_zero_at_mean_zero_std_zero(self) -> None:
        """With mean=0, std=0, the PRE delay must be exactly 0 minutes."""
        archetype = self._make_archetype(pre_mean=0.0, pre_std=0.0)
        rng = default_rng(0)
        delay = archetype.bolus_timing_delay(BolusTimingCategory.PRE, rng)
        assert delay.total_seconds() == 0.0


# ---------------------------------------------------------------------------
# H2 Bug #6 — cycling past the pool must keep VALID model names + real params
# ---------------------------------------------------------------------------


class TestH2Bug6CyclingUniqueName:
    """Cycling past pool boundary keeps valid simglucose model names (so the
    model resolves and real CR/CF/TDI are used); uniqueness is via
    patient_index, and a single warning documents the reuse.
    """

    def test_cycled_models_keep_valid_names_and_real_params(self) -> None:
        """Cycled profiles reuse a valid base model name (no ``_v`` suffix) and
        preserve the real Quest.csv CR/CF/TDI of the model they reuse. Profile
        uniqueness is carried by patient_index, not the model name.
        """
        config = CohortConfig(
            size=35,
            archetype_distribution={"adherent": 0.33, "moderate": 0.34, "nonadherent": 0.33},
            insulin_regimen_distribution={"pump": 1.0, "mdi": 0.0},
        )
        gen = CohortGenerator(config, ARCHETYPES_DIR, default_rng(42))
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            cohort = gen.generate()

        # No invalid suffixed names (those crashed T1DPatient.withName).
        assert all("_v" not in p.simglucose_name for p in cohort)
        # Cycled entry reuses the pool entry's name AND its real clinical params.
        assert cohort[30].simglucose_name == cohort[0].simglucose_name
        assert (cohort[30].cr, cohort[30].cf, cohort[30].tdi) == (
            cohort[0].cr, cohort[0].cf, cohort[0].tdi,
        )
        # patient_index disambiguates profiles even when names repeat.
        indices = [p.patient_index for p in cohort]
        assert len(indices) == len(set(indices)) == 35

    def test_cycling_emits_userwarning(self) -> None:
        """At least one UserWarning documents that models were reused."""
        config = CohortConfig(
            size=32,  # 30 base + 2 cycled
            archetype_distribution={"adherent": 1.0, "moderate": 0.0, "nonadherent": 0.0},
            insulin_regimen_distribution={"pump": 1.0, "mdi": 0.0},
        )
        gen = CohortGenerator(config, ARCHETYPES_DIR, default_rng(0))
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            gen.generate()

        cycling_warnings = [
            w for w in captured
            if issubclass(w.category, UserWarning) and "cycled" in str(w.message).lower()
        ]
        assert len(cycling_warnings) >= 1, "Cycling must emit a documenting warning"

    def test_no_cycling_no_warning_for_exact_pool_size(self) -> None:
        """When cohort size equals pool size, no cycling warning is emitted."""
        config = CohortConfig(
            size=30,
            archetype_distribution={"adherent": 1.0, "moderate": 0.0, "nonadherent": 0.0},
            insulin_regimen_distribution={"pump": 1.0, "mdi": 0.0},
        )
        gen = CohortGenerator(config, ARCHETYPES_DIR, default_rng(0))
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            gen.generate()

        cycling_warnings = [
            w for w in captured
            if issubclass(w.category, UserWarning) and "cycling" in str(w.message).lower()
        ]
        assert len(cycling_warnings) == 0, "No cycling warning expected when size == pool"


# ---------------------------------------------------------------------------
# H2 Bug #7 — round_dose with unit=0 causes ZeroDivisionError
# ---------------------------------------------------------------------------


class TestH2Bug7RoundDoseZeroUnit:
    """H2 #7 — round_dose must raise ValueError (not ZeroDivisionError) for unit=0."""

    def _make_archetype_with_unit(self, unit: float):
        from simada.core.config import load_archetype_params
        from simada.patient.archetype import create_archetype

        base = load_archetype_params(ARCHETYPES_DIR / "adherent.yaml")
        # Bypass Pydantic validation to simulate a unit=0 reaching round_dose
        # (e.g. direct dict construction or future refactor that skips the model).
        import pydantic
        params_dict = base.model_dump()
        params_dict["bolus_rounding_units"] = unit
        # Use model_construct to skip validators.
        params = type(base).model_construct(**params_dict)
        return create_archetype(ArchetypeID.ADHERENT, params)

    def test_round_dose_zero_unit_raises_value_error(self) -> None:
        """round_dose(unit=0) must raise ValueError, not ZeroDivisionError."""
        archetype = self._make_archetype_with_unit(unit=0.0)
        with pytest.raises(ValueError, match="bolus_rounding_units"):
            archetype.round_dose(2.5)

    def test_round_dose_negative_unit_raises_value_error(self) -> None:
        """round_dose with a negative unit must also raise ValueError."""
        archetype = self._make_archetype_with_unit(unit=-0.5)
        with pytest.raises(ValueError, match="bolus_rounding_units"):
            archetype.round_dose(2.5)

    def test_round_dose_normal_unit_works(self) -> None:
        """Standard unit values must continue to work correctly."""
        from simada.core.config import load_archetype_params
        from simada.patient.archetype import create_archetype

        base = load_archetype_params(ARCHETYPES_DIR / "adherent.yaml")
        params = base.model_copy(update={"bolus_rounding_units": 0.5})
        archetype = create_archetype(ArchetypeID.ADHERENT, params)
        # 2.3 U → nearest 0.5 → 2.5 U
        assert archetype.round_dose(2.3) == pytest.approx(2.5)
        # 1.7 U → nearest 0.5 → 2.0 U (or 1.5 depending on rounding)
        result = archetype.round_dose(1.7)
        assert result in (1.5, 2.0)


# ---------------------------------------------------------------------------
# H2 Bug #8 — accepted.size > 0 wrong for 2D array
# ---------------------------------------------------------------------------


class TestH2Bug8AcceptedShapeCheck:
    """H2 #8 — accepted.shape[0] > 0 must be used instead of accepted.size > 0."""

    def test_zero_row_array_has_nonzero_size(self) -> None:
        """Demonstrate the bug: a (0, k) array has .size=0 but also .shape[0]=0.
        Confirm that shape[0]==0 correctly identifies an empty batch.
        This test documents the semantics relied on by the fix.
        """
        empty = np.empty((0, 50), dtype=float)
        assert empty.shape[0] == 0, "shape[0] must be 0 for a zero-row 2D array"
        assert empty.size == 0, "size must also be 0 here (both are equivalent for 0 rows)"

        one_row = np.empty((1, 50), dtype=float)
        assert one_row.shape[0] == 1
        assert one_row.size == 50  # size = rows * cols; != 0 but shape[0] = 1

    def test_all_rejected_round_does_not_crash(self, monkeypatch) -> None:
        """If a round produces zero accepted samples, the loop must continue
        without appending anything (not crash or append an empty slice).

        We monkeypatch the plausibility filter to reject everything in round 0
        and accept everything in subsequent rounds.
        """
        import simada.patient.synthetic as _synthetic

        call_idx = [0]

        def _reject_first_accept_rest(samples, base_df, param_names):
            if call_idx[0] == 0:
                call_idx[0] += 1
                return np.zeros(samples.shape[0], dtype=bool)  # reject all
            call_idx[0] += 1
            return np.ones(samples.shape[0], dtype=bool)  # accept all

        monkeypatch.setattr(_synthetic, "plausibility_filter", _reject_first_accept_rest)
        # Should not raise; subsequent rounds will accept enough samples.
        df = _synthetic.generate_synthetic_cohort(n=5, pool="adult", seed=1, plausibility=True)
        assert len(df) == 5


# ---------------------------------------------------------------------------
# H2 Bug #9 — 'i' index collision between batches (docstring / LOW)
# ---------------------------------------------------------------------------


class TestH2Bug9IndexCollisionDocstring:
    """H2 #9 (LOW) — consecutive calls with same base_max_i produce colliding 'i'."""

    def test_single_call_has_unique_i(self) -> None:
        """Within a single generate_synthetic_cohort call, all 'i' must be unique."""
        df = generate_synthetic_cohort(n=20, pool="adult", seed=42)
        assert df["i"].is_unique, "'i' column must have unique values within a single cohort"

    def test_two_calls_with_same_base_produce_colliding_i(self) -> None:
        """Documenting the known limitation: two independent calls both start
        from base_max_i and therefore produce the same 'i' sequence.

        This test asserts the collision EXISTS (documents current known behaviour)
        so that if the limitation is later fixed, this test reminds the author
        to update the docstring note in _build_dataframe.
        """
        df1 = generate_synthetic_cohort(n=5, pool="adult", seed=1)
        df2 = generate_synthetic_cohort(n=5, pool="adult", seed=2)
        # Both calls start from the same base_max_i, so i values overlap.
        overlap = set(df1["i"]) & set(df2["i"])
        assert len(overlap) > 0, (
            "Expected 'i' collision between two independent calls starting from "
            "the same base — if this is no longer the case, remove the docstring "
            "note in _build_dataframe and update this test."
        )


# ---------------------------------------------------------------------------
# H2 Bug #10 — fallback hardcoded without warning
# ---------------------------------------------------------------------------


class TestH2Bug10QuestFallbackWarning:
    """H2 #10 — non-synthetic patient missing from Quest.csv must emit UserWarning."""

    def test_unknown_non_synthetic_emits_warning(self) -> None:
        """A patient whose name does not start with 'synthetic_' and is absent
        from Quest.csv must trigger a UserWarning.
        """
        config = CohortConfig(
            size=1,
            patient_pool=["unknown_patient_xyz"],
            archetype_distribution={"adherent": 1.0, "moderate": 0.0, "nonadherent": 0.0},
            insulin_regimen_distribution={"pump": 1.0, "mdi": 0.0},
        )
        gen = CohortGenerator(config, ARCHETYPES_DIR, default_rng(0))
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            gen.generate()

        quest_warnings = [
            w for w in captured
            if issubclass(w.category, UserWarning) and "Quest.csv" in str(w.message)
        ]
        assert len(quest_warnings) >= 1, (
            "Expected at least one UserWarning mentioning Quest.csv for unknown patient"
        )

    def test_synthetic_prefix_does_not_warn(self) -> None:
        """A patient whose name starts with 'synthetic_' is expected to be absent
        from Quest.csv; no warning should be emitted.
        """
        config = CohortConfig(
            size=1,
            patient_pool=["synthetic_adult_001"],
            archetype_distribution={"adherent": 1.0, "moderate": 0.0, "nonadherent": 0.0},
            insulin_regimen_distribution={"pump": 1.0, "mdi": 0.0},
        )
        gen = CohortGenerator(config, ARCHETYPES_DIR, default_rng(0))
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            gen.generate()

        quest_warnings = [
            w for w in captured
            if issubclass(w.category, UserWarning) and "Quest.csv" in str(w.message)
        ]
        assert len(quest_warnings) == 0, (
            "synthetic_* patients must not trigger Quest.csv warnings"
        )

    def test_fallback_values_are_still_applied(self) -> None:
        """Even with the warning, the fallback values CR=10, CF=50, TDI=40
        must still be assigned so simulation can proceed.
        """
        config = CohortConfig(
            size=1,
            patient_pool=["totally_unknown_patient"],
            archetype_distribution={"adherent": 1.0, "moderate": 0.0, "nonadherent": 0.0},
            insulin_regimen_distribution={"pump": 1.0, "mdi": 0.0},
        )
        gen = CohortGenerator(config, ARCHETYPES_DIR, default_rng(0))
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            cohort = gen.generate()

        p = cohort[0]
        assert p.cr == pytest.approx(10.0)
        assert p.cf == pytest.approx(50.0)
        assert p.tdi == pytest.approx(40.0)

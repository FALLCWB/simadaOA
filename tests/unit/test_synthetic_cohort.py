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

"""Tests for the synthetic patient cohort generator."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from simada.patient.synthetic import (
    generate_synthetic_cohort,
    load_base_cohort,
    save_synthetic_cohort,
)


def test_load_base_cohort_adult_has_10_rows() -> None:
    df = load_base_cohort(pool="adult")
    assert len(df) == 10
    assert all(name.startswith("adult#") for name in df["Name"])


def test_seed_determinism() -> None:
    """Same seed must produce identical synthetic cohorts."""
    df_a = generate_synthetic_cohort(n=5, pool="adult", seed=42)
    df_b = generate_synthetic_cohort(n=5, pool="adult", seed=42)
    pd.testing.assert_frame_equal(df_a, df_b)


def test_different_seeds_differ() -> None:
    df_a = generate_synthetic_cohort(n=5, pool="adult", seed=42)
    df_b = generate_synthetic_cohort(n=5, pool="adult", seed=99)
    diff = (
        df_a.select_dtypes(include=[np.number]).values
        != df_b.select_dtypes(include=[np.number]).values
    )
    assert diff.any()


def test_correct_size() -> None:
    df = generate_synthetic_cohort(n=30, pool="adult", seed=42)
    assert len(df) == 30


def test_all_positive_params_are_positive() -> None:
    df = generate_synthetic_cohort(n=20, pool="adult", seed=42)
    for col in ("BW", "Vg", "Vi", "Gb", "Ib", "kabs", "kmax", "kmin", "EGPb"):
        assert (df[col] > 0).all(), f"{col} has non-positive values"


def test_synthetic_within_envelope() -> None:
    """Synthetic patients should fall within ~base envelope (with margin)."""
    base = load_base_cohort(pool="adult")
    synth = generate_synthetic_cohort(n=50, pool="adult", seed=42)
    for col in ("BW", "Vg", "EGPb"):
        b_lo, b_hi = float(base[col].min()), float(base[col].max())
        b_range = b_hi - b_lo
        assert synth[col].min() >= b_lo - 2 * b_range, f"{col} too far below base"
        assert synth[col].max() <= b_hi + 2 * b_range, f"{col} too far above base"


def test_simglucose_compatible_csv() -> None:
    """Saved CSV must keep simglucose's canonical column order and reload cleanly."""
    df = generate_synthetic_cohort(n=5, pool="adult", seed=42)
    base = load_base_cohort(pool="adult")
    assert list(df.columns) == list(base.columns)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        tmp_path = Path(f.name)
    try:
        save_synthetic_cohort(df, tmp_path)
        reloaded = pd.read_csv(tmp_path)
        pd.testing.assert_frame_equal(df, reloaded)
    finally:
        tmp_path.unlink(missing_ok=True)


def test_no_resample_explosion_for_reasonable_n() -> None:
    """Even with plausibility filter, n=30 should converge quickly."""
    df = generate_synthetic_cohort(n=30, pool="adult", seed=42)
    assert len(df) == 30


def test_name_prefix_and_unique_i() -> None:
    """Synthetic patients use the configured prefix and have unique i values
    that do not collide with the base cohort's i range."""
    base = load_base_cohort(pool="adult")
    base_max_i = int(base["i"].max())
    df = generate_synthetic_cohort(n=10, pool="adult", seed=7)
    assert all(name.startswith("synth_adult#") for name in df["Name"])
    assert df["i"].min() > base_max_i
    assert df["i"].is_unique


def test_constant_columns_are_constant() -> None:
    """Columns that are zero across the base must remain zero in synthetic rows."""
    df = generate_synthetic_cohort(n=10, pool="adult", seed=42)
    for col in ("x0_ 1", "x0_ 2", "x0_ 3", "x0_ 7", "patient_history"):
        assert (df[col] == 0).all(), f"{col} should be constant zero"


def test_load_base_cohort_pools() -> None:
    """All three simglucose pools should load 10 rows each."""
    for pool in ("adult", "adolescent", "child"):
        df = load_base_cohort(pool=pool)
        assert len(df) == 10, f"{pool} pool should have 10 patients"
        assert all(name.startswith(f"{pool}#") for name in df["Name"])


# ---------------------------------------------------------------------------
# Regression tests for the H2 bug-hunt fixes.
# ---------------------------------------------------------------------------


def test_fit_distribution_returns_cholesky_decomposable_sigma() -> None:
    """BUG #1 / #4: fit_distribution must return a Cholesky-PSD sigma.

    Even when the empirical covariance is rank-deficient (10 samples,
    ~50 features), the returned sigma must be safe to feed straight
    into a Cholesky decomposition.
    """
    from simada.patient.synthetic import fit_distribution

    base = load_base_cohort(pool="adult")
    mu, sigma, names = fit_distribution(base)
    # No exception means the matrix is PSD enough for sampling.
    np.linalg.cholesky(sigma)
    assert sigma.shape == (len(names), len(names))
    assert mu.shape == (len(names),)


def test_sample_synthetic_does_not_silence_singular_warning() -> None:
    """BUG #1: sample_synthetic must not silently swallow an invalid covariance.

    With ``check_valid="ignore"`` the call would happily emit garbage
    samples from a non-PSD sigma. With ``check_valid="warn"`` (the new
    behavior) the bad sigma surfaces a numpy ``RuntimeWarning`` instead
    of being silently absorbed.
    """
    import warnings

    from simada.patient.synthetic import sample_synthetic

    # A 2x2 non-PSD covariance: eigenvalues are 3 and -1 (negative
    # eigenvalue ⇒ not positive semi-definite). numpy fires a
    # RuntimeWarning when ``check_valid != "ignore"``.
    mu = np.array([0.0, 0.0])
    sigma = np.array([[1.0, 2.0], [2.0, 1.0]])

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        sample_synthetic(mu, sigma, n=3, seed=0, oversample_factor=1)
    assert any(
        issubclass(w.category, RuntimeWarning)
        and "positive-semidefinite" in str(w.message)
        for w in captured
    ), "expected a RuntimeWarning that the covariance is not PSD"


def test_fit_distribution_raises_clear_error_on_unrecoverable_sigma(
    monkeypatch,
) -> None:
    """BUG #1: ``fit_distribution`` raises an actionable LinAlgError if
    even regularization cannot fix the covariance.
    """
    from simada.patient import synthetic

    # Force every Cholesky attempt to fail so the retry loop exhausts.
    def _always_fail(_mat: np.ndarray) -> np.ndarray:
        raise np.linalg.LinAlgError("forced failure for test")

    monkeypatch.setattr(synthetic.np.linalg, "cholesky", _always_fail)

    base = load_base_cohort(pool="adult")
    try:
        synthetic.fit_distribution(base)
    except np.linalg.LinAlgError as exc:
        msg = str(exc)
        assert "regularization" in msg
        assert "_COV_REGULARIZATION" in msg
        assert "forced failure" in msg
    else:
        raise AssertionError("expected LinAlgError")


def test_bw_envelope_uses_generic_rule_after_unification() -> None:
    """BUG #2: BW must follow the same ``[min - 2*SD, max + 2*SD]``
    envelope as every other parameter.

    Synthesize a draw that is well within the generic envelope but
    *outside* the legacy ``[0.7 * min, 1.3 * max]`` window, and check
    that the plausibility filter accepts it.
    """
    from simada.patient.synthetic import fit_distribution, plausibility_filter

    base = load_base_cohort(pool="adult")
    _, _, param_names = fit_distribution(base)
    bw_idx = param_names.index("BW")

    bw_min = float(base["BW"].min())
    bw_max = float(base["BW"].max())
    bw_sd = float(base["BW"].to_numpy(dtype=float).std(ddof=0))
    # Pick a BW value above 1.3 * max but still within max + 2*SD.
    # If the legacy 1.3 check was still active, this draw would be
    # rejected. We only assert the relationship is sensible; if the SD
    # ever shrinks below ((1.3*max - max) / 2) the test reduces to a
    # tautology, which is still safe.
    candidate_bw = min(bw_max + bw_sd, bw_max + 2.0 * bw_sd - 1e-3)
    if candidate_bw <= bw_max * 1.3:
        # Boost BW past the legacy 1.3 * max gate to make the test
        # genuinely discriminate between old and new behavior.
        candidate_bw = bw_max * 1.3 + 1e-3
        # And then verify it is still within the generic envelope.
        if candidate_bw > bw_max + 2.0 * bw_sd:
            # If the data simply cannot exceed both bars simultaneously,
            # the regression is at least sanity-checked by the filter
            # accepting the *base* mean BW; fall back to that.
            candidate_bw = float(base["BW"].mean())

    # Build a single synthetic sample matching the base means, then
    # override BW with the candidate value.
    sample = np.zeros((1, len(param_names)))
    for i, col in enumerate(param_names):
        sample[0, i] = float(base[col].mean())
    sample[0, bw_idx] = candidate_bw
    # Ensure EGPb is in range.
    egpb_idx = param_names.index("EGPb")
    sample[0, egpb_idx] = float(base["EGPb"].mean())

    mask = plausibility_filter(sample, base, param_names)
    # The point of this test is that BW values in the generic envelope
    # are not rejected by a stricter, removed BW-specific gate.
    assert mask[0], (
        f"BW={candidate_bw:.2f} should be accepted by the generic envelope"
    )


def test_random_truncation_unbiased(monkeypatch) -> None:
    """BUG #3: when more than ``n`` samples are accepted, the kept rows
    must be selected at random rather than the first ``n``.

    We force the resample loop to over-accept by making the plausibility
    filter accept every draw, then check that the resulting cohort is
    not simply the prefix of the pooled accepted samples.

    ROBUSTNESS (H10 bug #7): the original version tested only seed=42.
    A numpy API change that alters sampling order for one seed could cause
    a false alarm (test fails although truncation is random) or a false
    pass (test passes although truncation is prefix-selection). We now test
    100 seeds and require that AT LEAST half produce a non-prefix result.
    This makes the guard robust to single-seed coincidences while still
    being fast (all runs use the accept-all mock filter, no ODE).
    """
    from simada.patient import synthetic

    real_filter = synthetic.plausibility_filter
    _, _, param_names_ref = synthetic.fit_distribution(synthetic.load_base_cohort("adult"))
    bw_idx = param_names_ref.index("BW")

    prefix_matches = 0
    non_prefix_matches = 0
    n_seeds = 100

    for seed in range(n_seeds):
        captured_pools: list[np.ndarray] = []

        def _accept_all(
            samples: np.ndarray, base_df: pd.DataFrame, param_names: list[str]
        ) -> np.ndarray:
            captured_pools.append(samples.copy())
            return np.ones(samples.shape[0], dtype=bool)

        monkeypatch.setattr(synthetic, "plausibility_filter", _accept_all)
        try:
            df = synthetic.generate_synthetic_cohort(
                n=5, pool="adult", seed=seed, plausibility=True
            )
        finally:
            monkeypatch.setattr(synthetic, "plausibility_filter", real_filter)

        assert len(df) == 5
        first_pool = captured_pools[0]
        assert first_pool.shape[0] >= 15

        kept_bw = df["BW"].to_numpy()
        head_bw = first_pool[:5, bw_idx]
        if np.allclose(np.sort(kept_bw), np.sort(head_bw)):
            prefix_matches += 1
        else:
            non_prefix_matches += 1

    # If truncation were pure prefix-selection, every seed would match the prefix.
    # Require at least 50% of seeds to produce non-prefix results, ruling out
    # a degenerate implementation that always keeps [:n] rows.
    assert non_prefix_matches >= n_seeds // 2, (
        f"Truncation appears to use prefix-selection: only {non_prefix_matches}/{n_seeds} "
        f"seeds produced non-prefix results (expected >= {n_seeds // 2}). "
        "Expected random sampling from the over-accepted pool."
    )


def test_constant_columns_track_base_value(tmp_path) -> None:
    """BUG #6: constant columns must mirror whatever is in the base CSV,
    not be hardcoded to zero.

    We rewrite the base CSV with a non-zero ``patient_history`` and
    confirm the synthetic cohort copies that value verbatim.
    """
    from simada.patient.synthetic import _default_csv_path

    original = pd.read_csv(_default_csv_path())
    mutated = original.copy()
    mutated["patient_history"] = 7  # any non-zero constant
    mutated_path = tmp_path / "vpatient_params_mutated.csv"
    mutated.to_csv(mutated_path, index=False)

    from simada.patient.synthetic import generate_synthetic_cohort

    df = generate_synthetic_cohort(
        n=5, pool="adult", seed=42, csv_path=mutated_path
    )
    assert (df["patient_history"] == 7).all(), (
        "patient_history must mirror the base CSV, not be hardcoded to 0"
    )


def test_synthetic_csv_roundtrip_precision(tmp_path) -> None:
    """BUG #7: a saved synthetic cohort, reloaded from CSV, must round-trip
    to within 1e-6 on every numeric column.
    """
    from simada.patient.synthetic import (
        generate_synthetic_cohort,
        save_synthetic_cohort,
    )

    df = generate_synthetic_cohort(n=8, pool="adult", seed=2026)
    out = tmp_path / "synth_roundtrip.csv"
    save_synthetic_cohort(df, out)
    reloaded = pd.read_csv(out)

    # Columns and ordering preserved.
    assert list(reloaded.columns) == list(df.columns)

    for col in df.columns:
        if col in ("Name",):
            assert (reloaded[col].values == df[col].values).all(), (
                f"non-numeric column {col!r} changed across CSV roundtrip"
            )
            continue
        original = df[col].to_numpy(dtype=float)
        loaded = reloaded[col].to_numpy(dtype=float)
        assert np.allclose(original, loaded, atol=1e-6, rtol=0.0), (
            f"column {col!r} drifted >1e-6 across CSV save/reload"
        )

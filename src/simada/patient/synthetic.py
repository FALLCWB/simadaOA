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

"""Synthetic T1D patient cohort generator (parametric perturbation).

simglucose ships with only 10 adults, 10 adolescents and 10 children. For
Phase 1 v2 and Phase 4 we need cohorts of 30-100 patients while preserving
the physiologically valid covariance structure of the base population.

Strategy: fit a multivariate Gaussian to the numeric parameters of one
of the simglucose pools (adult / adolescent / child) and sample from it,
then reject implausible draws (negative-only positive parameters,
out-of-range BW or EGPb, parameters far outside the base envelope). The
covariance matrix preserves inter-parameter correlations (e.g. CR with
BW), which independent per-parameter perturbation would destroy.

Constant columns of the base CSV (`x0_ 1`, `x0_ 2`, `x0_ 3`, `x0_ 7`,
`patient_history` are all zero across every pool) are copied verbatim
rather than sampled, because zero-variance columns are degenerate in a
multivariate Gaussian fit.

The output DataFrame has the exact column layout simglucose expects in
`vpatient_params.csv`, so a synthetic row can be fed directly into
`simglucose.patient.t1dpatient.T1DPatient(params=row)`.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd

# Identity columns: never fitted, regenerated for synthetic patients.
_IDENTITY_COLUMNS: tuple[str, ...] = ("Name", "i")

# Columns that are never sampled by the Gaussian fit. ``patient_history``
# is always zero in the bundled simglucose CSV and is metadata, not a
# physiological parameter, so we always treat it as a constant copy.
_FORCED_CONSTANT_COLUMNS: tuple[str, ...] = ("patient_history",)

# How many resample rounds we allow before giving up on the plausibility
# filter. Each round draws `n * oversample_factor` samples.
MAX_RESAMPLE_ROUNDS: int = 5

# Variance threshold below which a column is treated as constant.
_VARIANCE_EPSILON: float = 1e-12

# Tikhonov-style regularization added to the covariance diagonal if the
# fitted matrix is not positive-definite (numerical stabilization).
_COV_REGULARIZATION: float = 1e-9

# Maximum number of attempts to regularize the covariance matrix into a
# Cholesky-decomposable PSD form before giving up. Each retry multiplies
# the regularization by ``_COV_REGULARIZATION_GROWTH``.
_COV_REGULARIZATION_MAX_RETRIES: int = 3
_COV_REGULARIZATION_GROWTH: float = 100.0

# Parameters that must be strictly positive (physiologically meaningful).
_POSITIVE_PARAMS: tuple[str, ...] = (
    "BW",
    "EGPb",
    "Gb",
    "Ib",
    "kabs",
    "kmax",
    "kmin",
    "Vg",
    "Vi",
    "Ipb",
    "Vmx",
    "Km0",
    "k2",
    "k1",
    "p2u",
    "m1",
    "m5",
    "CL",
    "m2",
    "m4",
    "m30",
    "Ilb",
    "ki",
    "kp2",
    "kp3",
    "Gpb",
    "Gtb",
    "Rdb",
    "PCRb",
    "kd",
    "ksc",
    "ka1",
    "ka2",
    "u2ss",
    "isc1ss",
    "isc2ss",
    "kp1",
)

# Clinical plausibility envelope for EGPb (mg/kg/min). Glucose production
# below 0.5 or above 5.0 is outside the range reported in the literature
# (Cobelli et al., UVA/Padova validation set).
_EGPB_MIN: float = 0.5
_EGPB_MAX: float = 5.0


def _default_csv_path() -> Path:
    """Return the path to simglucose's bundled `vpatient_params.csv`."""
    return Path(str(files("simglucose").joinpath("params", "vpatient_params.csv")))


def load_base_cohort(
    pool: str = "adult",
    csv_path: Path | None = None,
) -> pd.DataFrame:
    """Load the simglucose virtual-patient cohort for the given pool.

    Parameters
    ----------
    pool:
        One of ``"adult"``, ``"adolescent"`` or ``"child"``. simglucose
        ships 10 patients per pool.
    csv_path:
        Optional override for the location of ``vpatient_params.csv``.
        If ``None``, the file shipped with the installed simglucose
        package is used.

    Returns
    -------
    pandas.DataFrame
        The 10 rows of the requested pool, with ``Name`` and 61 numeric
        columns in simglucose's canonical order.
    """
    if pool not in {"adult", "adolescent", "child"}:
        raise ValueError(f"pool must be 'adult', 'adolescent' or 'child', got {pool!r}")
    path = csv_path if csv_path is not None else _default_csv_path()
    df = pd.read_csv(path)
    sub = df[df["Name"].str.startswith(f"{pool}#")].reset_index(drop=True)
    if sub.empty:
        raise ValueError(f"No patients found for pool {pool!r} in {path}")
    return sub


def _split_columns(base_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split numeric columns into (variable, constant) lists.

    A constant column has variance below ``_VARIANCE_EPSILON`` across
    the base cohort. Constants cannot be fit by a multivariate Gaussian
    (degenerate covariance) and are copied verbatim instead.
    """
    variable: list[str] = []
    constant: list[str] = []
    for col in base_df.columns:
        if col in _IDENTITY_COLUMNS:
            continue
        if col in _FORCED_CONSTANT_COLUMNS:
            constant.append(col)
            continue
        values = base_df[col].to_numpy(dtype=float)
        if float(values.var()) < _VARIANCE_EPSILON:
            constant.append(col)
        else:
            variable.append(col)
    return variable, constant


def fit_distribution(
    base_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Estimate the mean vector and covariance matrix of the variable columns.

    Constant columns (zero variance across the base cohort) are excluded
    from the multivariate Gaussian fit. ``param_names`` contains only
    the variable columns, in the same order as ``mu`` and ``sigma``.

    If the empirical covariance matrix is not positive-definite (likely
    because the base cohort has only 10 patients), a small Tikhonov
    regularization (``_COV_REGULARIZATION * I``) is added to the
    diagonal. With 10 samples and ~50 features the empirical covariance
    is rank-deficient by construction, so we additionally project it to
    its nearest PSD form by clipping negative eigenvalues to zero.
    """
    variable, _ = _split_columns(base_df)
    matrix = base_df[variable].to_numpy(dtype=float)
    mu = matrix.mean(axis=0)
    # rowvar=False: each column is a variable, each row an observation.
    sigma_raw = np.cov(matrix, rowvar=False)
    # Symmetrize (numerical noise) and project to PSD.
    sigma_raw = 0.5 * (sigma_raw + sigma_raw.T)
    eigvals, eigvecs = np.linalg.eigh(sigma_raw)
    eigvals = np.clip(eigvals, 0.0, None)
    sigma_psd = (eigvecs * eigvals) @ eigvecs.T

    # BUG #4 fix: validate the PSD projection by attempting Cholesky.
    # If it fails, grow the Tikhonov regularization up to
    # ``_COV_REGULARIZATION_MAX_RETRIES`` times before giving up with a
    # clear, actionable error message (BUG #1).
    reg_eps = _COV_REGULARIZATION
    last_error: np.linalg.LinAlgError | None = None
    for _ in range(_COV_REGULARIZATION_MAX_RETRIES + 1):
        sigma_try = sigma_psd + reg_eps * np.eye(sigma_psd.shape[0])
        try:
            np.linalg.cholesky(sigma_try)
            return mu, sigma_try, variable
        except np.linalg.LinAlgError as exc:
            last_error = exc
            reg_eps *= _COV_REGULARIZATION_GROWTH

    raise np.linalg.LinAlgError(
        "fit_distribution: covariance matrix could not be made "
        "positive-definite even after "
        f"{_COV_REGULARIZATION_MAX_RETRIES} regularization retries "
        f"(final epsilon = {reg_eps / _COV_REGULARIZATION_GROWTH:g}). "
        "Consider increasing the base regularization epsilon "
        f"(currently _COV_REGULARIZATION = {_COV_REGULARIZATION:g}) or "
        "supplying a larger base cohort. "
        f"Underlying Cholesky failure: {last_error}"
    )


def sample_synthetic(
    mu: np.ndarray,
    sigma: np.ndarray,
    n: int,
    seed: int,
    *,
    oversample_factor: int = 3,
) -> np.ndarray:
    """Sample ``n * oversample_factor`` patients from ``N(mu, sigma)``.

    Oversampling lets a downstream :func:`plausibility_filter` reject
    bad draws and still return ``n`` patients without further draws in
    the common case. The caller is responsible for filtering and
    truncating to ``n``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n * oversample_factor, len(mu))``.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if oversample_factor < 1:
        raise ValueError(f"oversample_factor must be >= 1, got {oversample_factor}")
    rng = np.random.default_rng(seed)
    draws = n * oversample_factor
    # BUG #1 fix: do not silently swallow ill-conditioned covariance.
    # ``check_valid="warn"`` surfaces issues that would otherwise hide
    # silently degraded samples. ``fit_distribution`` is responsible for
    # ensuring ``sigma`` is positive-definite (Cholesky-decomposable),
    # so any warning here indicates a regression in the caller.
    samples = rng.multivariate_normal(mu, sigma, size=draws, check_valid="warn")
    return samples


def plausibility_filter(
    samples: np.ndarray,
    base_df: pd.DataFrame,
    param_names: list[str],
) -> np.ndarray:
    """Return a boolean mask selecting plausible synthetic patients.

    A patient is plausible if **all** of the following hold:

    * ``EGPb`` is in ``[0.5, 5.0]`` mg/kg/min (clinical envelope);
    * every parameter listed in :data:`_POSITIVE_PARAMS` is strictly
      positive;
    * every parameter (including ``BW``) is within
      ``[base_min - 2*SD, base_max + 2*SD]`` of the base cohort
      (per-parameter envelope).

    Parameters that are not present in ``param_names`` (because they
    are constant in the base cohort) are not checked here.

    .. note::
       The BW-specific ``[0.7 * base_min, 1.3 * base_max]`` envelope
       that previously coexisted with the generic per-parameter
       envelope was removed (BUG #2). The two checks were conflicting:
       depending on the base cohort's spread, one was strictly stricter
       than the other, making the BW envelope either redundant or
       silently overriding the generic rule. We now use the single
       generic ``[base_min - 2*SD, base_max + 2*SD]`` envelope for
       every parameter to keep the rejection criterion consistent.
    """
    if samples.shape[1] != len(param_names):
        raise ValueError(
            f"samples has {samples.shape[1]} columns but param_names has "
            f"{len(param_names)} entries"
        )

    n_samples = samples.shape[0]
    ok = np.ones(n_samples, dtype=bool)

    name_to_idx = {name: i for i, name in enumerate(param_names)}

    # 1. Strict positivity for biologically positive params.
    for col in _POSITIVE_PARAMS:
        idx = name_to_idx.get(col)
        if idx is None:
            continue
        ok &= samples[:, idx] > 0.0

    # 2. EGPb in clinical envelope.
    egpb_idx = name_to_idx.get("EGPb")
    if egpb_idx is not None:
        ok &= (samples[:, egpb_idx] >= _EGPB_MIN) & (samples[:, egpb_idx] <= _EGPB_MAX)

    # 3. Per-parameter envelope: [base_min - 2*SD, base_max + 2*SD].
    # Applies uniformly to every variable parameter, including BW
    # (see BUG #2 note above).
    for col, idx in name_to_idx.items():
        values = base_df[col].to_numpy(dtype=float)
        # BUG #3 fix (H2): use ddof=1 (sample std) so the envelope is not
        # systematically too tight for small base cohorts (n=10).
        lo = float(values.min()) - 2.0 * float(values.std(ddof=1))
        hi = float(values.max()) + 2.0 * float(values.std(ddof=1))
        ok &= (samples[:, idx] >= lo) & (samples[:, idx] <= hi)

    return ok


def _build_dataframe(
    accepted: np.ndarray,
    base_df: pd.DataFrame,
    param_names: list[str],
    constant_cols: list[str],
    name_prefix: str,
    base_max_i: int,
) -> pd.DataFrame:
    """Assemble a simglucose-compatible DataFrame from accepted samples.

    Constant columns are filled from ``base_df`` (first row, since they
    are constant by definition). ``Name`` is auto-generated as
    ``f"{name_prefix}#{i:03d}"``. ``i`` is generated as
    ``base_max_i + 1, base_max_i + 2, ...`` to avoid colliding with
    simglucose's internal indexing.
    """
    n = accepted.shape[0]
    rows: dict[str, np.ndarray] = {}

    # Variable columns from the accepted Gaussian samples.
    for idx, col in enumerate(param_names):
        rows[col] = accepted[:, idx]

    # Constant columns: tile the base value (all rows of base have the
    # same value by construction, so we read from the first row).
    # BUG #6 fix: do not hardcode ``patient_history`` to zero. Treat it
    # as a true constant copy of whatever the base cohort has, so that
    # if a future simglucose release changes the default value (or if
    # the caller supplies a customized base CSV) the synthetic rows
    # follow suit instead of silently zeroing the column.
    for col in constant_cols:
        rows[col] = np.full(n, float(base_df[col].iloc[0]))

    # Identity columns.
    rows["Name"] = np.array(
        [f"{name_prefix}#{i + 1:03d}" for i in range(n)], dtype=object
    )
    # BUG #9 note (H2, LOW): ``i`` is derived from ``base_max_i`` (the
    # highest ``i`` across the base CSV), so consecutive calls with
    # different batches sharing the same ``base_max_i`` will produce
    # colliding ``i`` values.  Callers that concatenate multiple synthetic
    # batches must recompute ``base_max_i`` from the combined DataFrame or
    # use a single call to ``generate_synthetic_cohort`` for the full size.
    rows["i"] = np.arange(base_max_i + 1, base_max_i + 1 + n, dtype=int)

    df = pd.DataFrame(rows)
    # Reorder to the simglucose canonical column order.
    return df[list(base_df.columns)]


def generate_synthetic_cohort(
    n: int,
    pool: str = "adult",
    seed: int = 42,
    *,
    name_prefix: str | None = None,
    plausibility: bool = True,
    csv_path: Path | None = None,
) -> pd.DataFrame:
    """Generate a synthetic patient cohort of size ``n``.

    Parameters
    ----------
    n:
        Number of synthetic patients to produce.
    pool:
        Base pool to fit the distribution to (``"adult"``,
        ``"adolescent"`` or ``"child"``).
    seed:
        Master seed for full reproducibility.
    name_prefix:
        Prefix for the generated ``Name`` column. Defaults to
        ``f"synth_{pool}"``.
    plausibility:
        If ``True``, run :func:`plausibility_filter` over each batch of
        draws and resample (with deterministically derived seeds) until
        ``n`` plausible patients are accumulated, up to
        :data:`MAX_RESAMPLE_ROUNDS` rounds.
    csv_path:
        Optional override for the base ``vpatient_params.csv``.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the same columns as the base CSV.

    Raises
    ------
    RuntimeError
        If plausibility filtering fails to accumulate ``n`` samples
        within :data:`MAX_RESAMPLE_ROUNDS` rounds.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    base_df = load_base_cohort(pool=pool, csv_path=csv_path)
    base_max_i = int(base_df["i"].max())
    _, constant_cols = _split_columns(base_df)
    mu, sigma, param_names = fit_distribution(base_df)

    if name_prefix is None:
        name_prefix = f"synth_{pool}"

    # Derive child seeds from the master seed so that successive
    # resample rounds are independent but reproducible.
    # BUG #1 fix (H2): derive ALL child seeds via a single spawn call so that
    # adding new consumers never causes accidental seed reuse.  The last child
    # is reserved for the selection RNG; the first MAX_RESAMPLE_ROUNDS children
    # are used for the individual sampling rounds.
    seed_seq = np.random.SeedSequence(seed)
    _all_children = seed_seq.spawn(MAX_RESAMPLE_ROUNDS + 1)
    round_seeds = _all_children[:MAX_RESAMPLE_ROUNDS]
    _selection_child = _all_children[MAX_RESAMPLE_ROUNDS]

    accepted_chunks: list[np.ndarray] = []
    total_accepted = 0

    if plausibility:
        for round_idx in range(MAX_RESAMPLE_ROUNDS):
            need = n - total_accepted
            if need <= 0:
                break
            # BUG #2 fix (H2): always draw n * oversample_factor candidates per
            # round, not just `need * oversample_factor`.  Shrinking the draw to
            # `need` causes geometric decay in accepted samples per round when
            # acceptance rate is low, leading to convergence failure on
            # plausible but tight distributions.
            samples = sample_synthetic(
                mu,
                sigma,
                n=n,
                seed=int(round_seeds[round_idx].generate_state(1, dtype=np.uint32)[0]),
                oversample_factor=3,
            )
            mask = plausibility_filter(samples, base_df, param_names)
            accepted = samples[mask]
            # BUG #8 fix (H2): use shape[0] to test for accepted rows; .size is
            # the total element count (rows * cols) which is always > 0 for a 2D
            # array with any columns, even when there are zero accepted rows.
            if accepted.shape[0] > 0:
                accepted_chunks.append(accepted)
                total_accepted += accepted.shape[0]
        if total_accepted < n:
            raise RuntimeError(
                f"plausibility filter failed to converge: {total_accepted}/{n} "
                f"plausible samples after {MAX_RESAMPLE_ROUNDS} rounds"
            )
        # BUG #3 fix: do not bias selection by always keeping the first
        # ``n`` accepted draws. With oversampling, the early rounds
        # contribute disproportionately; deterministic head-truncation
        # also throws away later rounds entirely, which biases the
        # resulting cohort toward whatever the first round produced.
        # Use a separate, deterministic RNG keyed off the master seed to
        # pick ``n`` samples uniformly at random without replacement
        # from the pooled accepted draws.
        all_accepted_pool = np.concatenate(accepted_chunks, axis=0)
        if all_accepted_pool.shape[0] == n:
            all_accepted = all_accepted_pool
        else:
            # BUG #1 fix (cont.): use the pre-derived selection child seed so
            # the selection RNG is fully independent from the round RNGs.
            selection_rng = np.random.default_rng(_selection_child)
            picks = selection_rng.choice(
                np.arange(all_accepted_pool.shape[0]), size=n, replace=False
            )
            picks.sort()
            all_accepted = all_accepted_pool[picks]
    else:
        samples = sample_synthetic(
            mu, sigma,
            n=n,
            seed=int(round_seeds[0].generate_state(1, dtype=np.uint32)[0]),
            oversample_factor=1,
        )
        all_accepted = samples[:n]

    return _build_dataframe(
        accepted=all_accepted,
        base_df=base_df,
        param_names=param_names,
        constant_cols=constant_cols,
        name_prefix=name_prefix,
        base_max_i=base_max_i,
    )


def save_synthetic_cohort(df: pd.DataFrame, output_path: Path) -> None:
    """Save a synthetic cohort to CSV in simglucose-compatible format.

    The output preserves simglucose's canonical column order so the
    file can be substituted in place of ``vpatient_params.csv`` or
    concatenated with the base cohort.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a synthetic T1D patient cohort by parametric perturbation"
    )
    parser.add_argument("--n", type=int, required=True, help="number of patients")
    parser.add_argument(
        "--pool",
        default="adult",
        choices=("adult", "adolescent", "child"),
        help="base simglucose pool to fit the distribution to",
    )
    parser.add_argument("--seed", type=int, default=42, help="master RNG seed")
    parser.add_argument(
        "--output", type=Path, required=True, help="destination CSV path"
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="disable plausibility filtering (debug only)",
    )
    args = parser.parse_args()

    cohort = generate_synthetic_cohort(
        n=args.n,
        pool=args.pool,
        seed=args.seed,
        plausibility=not args.no_filter,
    )
    save_synthetic_cohort(cohort, args.output)
    print(f"Wrote {len(cohort)} synthetic patients to {args.output}")

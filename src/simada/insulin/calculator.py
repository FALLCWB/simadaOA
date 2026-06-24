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

"""Bolus calculator with IOB tracking.

Standard clinical formula:
    meal_bolus   = estimated_carbs / CR
    correction   = max(0, (current_bg - target_bg)) / CF
    total_bolus  = meal_bolus + correction - IOB_adjustment
    final_dose   = max(0, total_bolus)

IOB (Insulin on Board) uses a configurable action curve to estimate
remaining active insulin from prior boluses.  Two curves are available:

* **walsh** (default) -- piecewise linear approximation of regular rapid
  insulin (Humalog/NovoRapid).  Peak at 25 % of duration, conservative
  decay (90 % remaining at 60 min for a 4 h duration).

* **fiasp** -- exponential decay model tuned for ultra-rapid insulin
  (Fiasp / insulin aspart).  Faster onset, earlier peak (~35 min),
  steeper initial decay matching published pharmacokinetic data:

      T=0 -> 100 %,  T=30 min -> ~83 %,  T=60 min -> ~60 %,
      T=120 min -> ~25 %,  T=180 min -> ~8 %,  T=duration -> 0 %.

  Model:  IOB(t) = exp(-k * t^n)  with k = 0.0014, n = 1.44,
  clamped to 0 at the configured duration.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Literal, NamedTuple

#: Supported insulin action curve types.
InsulinType = Literal["walsh", "fiasp"]


class BolusRecord(NamedTuple):
    """A record of a past bolus for IOB tracking."""

    time: datetime
    dose: float  # Units


class BolusCalculation(NamedTuple):
    """Result of a bolus calculation."""

    meal_bolus: float
    correction_bolus: float
    iob: float
    total_dose: float  # after IOB subtraction, before rounding


class BolusCalculator:
    """Clinical bolus calculator with IOB tracking.

    This computes the *ideal* dose. Adherence modifications (errors,
    rounding, timing) are applied by the AdherenceArchetype separately.
    """

    def __init__(
        self,
        cr: float,
        cf: float,
        target_bg: float = 120.0,
        iob_duration_hours: float = 4.0,
        insulin_type: InsulinType = "walsh",
    ) -> None:
        """Initialize the calculator with patient-specific parameters.

        Args:
            cr: Carb ratio -- grams of carbs per 1 unit of insulin.
            cf: Correction factor -- mg/dL drop per 1 unit of insulin.
            target_bg: Target blood glucose in mg/dL.
            iob_duration_hours: Insulin action duration in hours.
            insulin_type: IOB curve model -- ``"walsh"`` (piecewise linear,
                default) or ``"fiasp"`` (exponential decay for ultra-rapid
                insulin).
        """
        if insulin_type not in ("walsh", "fiasp"):
            raise ValueError(
                f"Unknown insulin_type {insulin_type!r}; "
                "expected 'walsh' or 'fiasp'"
            )
        self._cr = cr
        self._cf = cf
        self._target_bg = target_bg
        self._iob_duration = iob_duration_hours
        self._insulin_type: InsulinType = insulin_type
        self._bolus_history: list[BolusRecord] = []

    def reset(self) -> None:
        """Clear the bolus/IOB history for a new simulation run."""
        self._bolus_history = []

    @property
    def cr(self) -> float:
        return self._cr

    @property
    def cf(self) -> float:
        return self._cf

    def record_bolus(self, time: datetime, dose: float) -> None:
        """Record a delivered bolus for IOB tracking.

        Silently ignores zero doses (nothing to track).  Negative doses are
        a caller bug (no such thing as a negative insulin delivery) and raise
        ``ValueError`` immediately so the error surface is visible rather than
        silently corrupting the IOB integral.
        """
        if dose < 0:
            raise ValueError(
                f"record_bolus: dose must be non-negative, got {dose:.3f}U. "
                "Negative insulin delivery is not physiologically meaningful."
            )
        if dose > 0:
            self._bolus_history.append(BolusRecord(time=time, dose=dose))

    def compute_iob(self, current_time: datetime) -> float:
        """Compute current Insulin on Board from bolus history.

        Uses the configured IOB curve (Walsh or Fiasp) to estimate
        remaining active insulin from prior boluses.  Expired entries are
        pruned from ``_bolus_history`` as a memory optimisation.

        BUG-FIX (H3 #3): Previously a record with ``elapsed_min < 0``
        (bolus recorded with a future timestamp relative to
        ``current_time``) was silently skipped, masking ordering bugs
        in the caller. Now we treat any non-chronological record as a
        hard error and raise ``ValueError`` with a dump of the bolus
        history so the upstream bug can be diagnosed.

        .. warning::
            This method **mutates** ``_bolus_history`` by removing expired
            entries.  Use :meth:`compute_iob_readonly` when you need an IOB
            estimate without side-effects (e.g. when computing the *ideal*
            dose after already calling ``calculate_with_errors``).
        """
        iob = 0.0
        duration_min = self._iob_duration * 60.0
        expired: list[int] = []

        for i, record in enumerate(self._bolus_history):
            elapsed_min = (current_time - record.time).total_seconds() / 60.0
            if elapsed_min < 0:
                history_dump = "\n".join(
                    f"  [{j}] time={r.time.isoformat()} dose={r.dose:.3f}U"
                    for j, r in enumerate(self._bolus_history)
                )
                # NOTE: a LATE timing delay records a bolus at meal_time + delay
                # (a future stamp); if compute_iob were called in the meal_time..
                # delivery_time window this would raise. Not observed in any run
                # (the controller delivers and queries IOB on the same step), so
                # the guard is kept to catch genuine out-of-order caller bugs.
                raise ValueError(
                    "Non-chronological bolus history: bolus at "
                    f"{record.time.isoformat()} is in the future relative "
                    f"to current_time={current_time.isoformat()} "
                    f"(elapsed_min={elapsed_min:.1f}). "
                    "This indicates a caller bug (out-of-order "
                    "record_bolus / compute_iob).\n"
                    f"Bolus history:\n{history_dump}"
                )
            if elapsed_min > duration_min:
                expired.append(i)
                continue
            iob += record.dose * self._iob_curve(elapsed_min, duration_min)

        # Clean up expired entries
        for i in reversed(expired):
            self._bolus_history.pop(i)

        return iob

    def compute_iob_readonly(self, current_time: datetime) -> float:
        """Compute IOB without mutating ``_bolus_history``.

        Identical arithmetic to :meth:`compute_iob` but expired entries are
        **not** pruned.  Use this whenever you need an IOB snapshot for
        informational or comparison purposes after ``calculate_with_errors``
        has already been called (which calls the pruning variant), so the
        ideal-dose computation operates on the same history state.

        The chronological guard is applied identically — out-of-order records
        still raise ``ValueError``.
        """
        iob = 0.0
        duration_min = self._iob_duration * 60.0

        for record in self._bolus_history:
            elapsed_min = (current_time - record.time).total_seconds() / 60.0
            if elapsed_min < 0:
                history_dump = "\n".join(
                    f"  [{j}] time={r.time.isoformat()} dose={r.dose:.3f}U"
                    for j, r in enumerate(self._bolus_history)
                )
                raise ValueError(
                    "Non-chronological bolus history: bolus at "
                    f"{record.time.isoformat()} is in the future relative "
                    f"to current_time={current_time.isoformat()} "
                    f"(elapsed_min={elapsed_min:.1f}). "
                    "This indicates a caller bug (out-of-order "
                    "record_bolus / compute_iob_readonly).\n"
                    f"Bolus history:\n{history_dump}"
                )
            if elapsed_min > duration_min:
                continue  # expired — skip but do NOT remove
            iob += record.dose * self._iob_curve(elapsed_min, duration_min)

        return iob

    def calculate(
        self,
        estimated_carbs: float,
        current_bg: float,
        current_time: datetime,
    ) -> BolusCalculation:
        """Calculate the ideal bolus dose.

        This is the *correct* calculation. Adherence errors (wrong CR/CF,
        ignored IOB, rounding) are applied by the archetype layer.

        Uses :meth:`compute_iob_readonly` so that calling ``calculate``
        after ``calculate_with_errors`` (which already pruned expired records
        via :meth:`compute_iob`) produces an IOB value over the **same**
        history state — preventing the BUG-FIX H5 #1 double-prune corruption
        of ``bolus_calculated``.

        Args:
            estimated_carbs: Carbs the patient/controller estimates (grams).
            current_bg: Current blood glucose (mg/dL).
            current_time: Current simulation time (for IOB computation).

        Returns:
            BolusCalculation with meal, correction, IOB, and total dose.
        """
        meal_bolus = estimated_carbs / self._cr if self._cr > 0 else 0.0

        correction = 0.0
        if current_bg > self._target_bg and self._cf > 0:
            correction = (current_bg - self._target_bg) / self._cf

        iob = self.compute_iob_readonly(current_time)

        total = max(0.0, meal_bolus + correction - iob)

        return BolusCalculation(
            meal_bolus=meal_bolus,
            correction_bolus=correction,
            iob=iob,
            total_dose=total,
        )

    def calculate_with_errors(
        self,
        estimated_carbs: float,
        current_bg: float,
        current_time: datetime,
        cr_error: float,
        cf_error: float,
        iob_fraction: float,
    ) -> BolusCalculation:
        """Calculate bolus with archetype-specific errors applied.

        Args:
            estimated_carbs: What the patient thinks the carbs are.
            current_bg: Current BG.
            current_time: For IOB.
            cr_error: Multiplier on CR (1.0 = correct).
            cf_error: Multiplier on CF (1.0 = correct).
            iob_fraction: How much IOB to subtract (0.0 = ignore, 1.0 = full).

        Returns:
            BolusCalculation reflecting the patient's actual calculation.
        """
        effective_cr = self._cr * cr_error
        meal_bolus = estimated_carbs / effective_cr if effective_cr > 0 else 0.0

        correction = 0.0
        effective_cf = self._cf * cf_error
        if current_bg > self._target_bg and effective_cf > 0:
            correction = (current_bg - self._target_bg) / effective_cf

        iob = self.compute_iob(current_time) * iob_fraction

        total = max(0.0, meal_bolus + correction - iob)

        return BolusCalculation(
            meal_bolus=meal_bolus,
            correction_bolus=correction,
            iob=iob,
            total_dose=total,
        )

    def _iob_curve(self, elapsed_min: float, duration_min: float) -> float:
        """Fraction of insulin remaining at *elapsed_min*.

        Dispatches to the curve selected by ``self._insulin_type``.
        """
        if self._insulin_type == "fiasp":
            return self._fiasp_iob_curve(elapsed_min, duration_min)
        return self._walsh_iob_curve(elapsed_min, duration_min)

    # ------------------------------------------------------------------
    # Walsh curve (regular rapid insulin)
    # ------------------------------------------------------------------

    @staticmethod
    def _walsh_iob_curve(elapsed_min: float, duration_min: float) -> float:
        """Walsh IOB curve -- fraction of insulin remaining at elapsed_min.

        Piecewise linear approximation:
            0 to peak:    slight decrease from 1.0
            peak to end:  linear decay to 0.0

        Peak is at 25 % of the total duration.
        """
        if elapsed_min <= 0:
            return 1.0
        if elapsed_min >= duration_min:
            return 0.0

        peak_time = duration_min * 0.25
        if elapsed_min <= peak_time:
            # Slight ramp down from 1.0 to 0.9 during absorption
            return 1.0 - 0.1 * (elapsed_min / peak_time)
        else:
            # Linear decay from 0.9 to 0.0
            remaining_frac = (duration_min - elapsed_min) / (duration_min - peak_time)
            return 0.9 * remaining_frac

    # ------------------------------------------------------------------
    # Fiasp curve (ultra-rapid insulin)
    # ------------------------------------------------------------------

    @staticmethod
    def _fiasp_iob_curve(elapsed_min: float, duration_min: float) -> float:
        """Fiasp IOB curve -- fraction of insulin remaining at elapsed_min.

        Exponential decay model tuned for ultra-rapid insulin action
        (Fiasp / insulin aspart):

            IOB(t) = exp(-k * t_scaled^n)

        with k = 0.0014, n = 1.44 (fitted to published PK data at 240 min).

        The time axis is **scaled** to the reference 240-minute duration so
        the curve shape is preserved for any configured ``iob_duration_hours``
        (BUG-FIX H5 #3).  Without scaling the parameters are only correct for
        a 4-hour duration; shorter durations (e.g. 3 h / 180 min) would leave
        ~8 % IOB active past the declared duration, causing bolus-stacking
        underestimation and hypoglycaemia risk.

        Approximate profile for the reference 240 min duration::

            T =   0 min -> 100 %
            T =  30 min ->  83 %
            T =  60 min ->  60 %
            T = 120 min ->  25 %
            T = 180 min ->   8 %
            T = 240 min ->   0 % (clamped)
        """
        if elapsed_min <= 0:
            return 1.0
        if elapsed_min >= duration_min:
            return 0.0

        # Exponential decay parameters (fitted to Fiasp PK data at 240 min)
        _K = 0.0014
        _N = 1.44
        _REF_MIN = 240.0  # reference duration the k/n parameters are tuned for

        # Scale elapsed time to the reference 240-min axis so the curve shape
        # is duration-agnostic.  At t=duration_min the scaled value is 240,
        # which gives exp(-0.0014 * 240^1.44) ≈ 0.003 (near-zero before clamp).
        t_scaled = elapsed_min * _REF_MIN / duration_min

        remaining = math.exp(-_K * t_scaled ** _N)

        # Hard floor at zero (shouldn't happen before duration, but guard)
        return max(0.0, remaining)

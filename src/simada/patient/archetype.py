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

"""Patient adherence archetypes.

Defines the AdherenceArchetype interface and provides a factory to
instantiate archetypes from ArchetypeParams (loaded from YAML).

The archetype encapsulates all behavioral decisions around insulin
delivery: when to bolus, how much to bolus, calculation errors,
phantom boluses, rage corrections, and dose rounding.

All numeric parameters come from ArchetypeParams — the archetype class
is a behavioral engine, not a parameter store.
"""

from __future__ import annotations

from datetime import timedelta

from numpy.random import Generator

from simada.core.config import ArchetypeParams
from simada.core.types import ArchetypeID, BolusTimingCategory


class AdherenceArchetype:
    """Behavioral engine for a patient adherence profile.

    This class answers the key behavioral questions about insulin delivery:
    - Should this bolus be skipped?
    - When is the bolus delivered relative to the meal?
    - How much dose error is introduced?
    - Are there phantom/rage/double boluses?
    - How is the dose rounded?

    All behavior is driven by ArchetypeParams, making every aspect
    tunable via YAML without subclassing.
    """

    def __init__(self, archetype_id: ArchetypeID, params: ArchetypeParams) -> None:
        self.id = archetype_id
        self.params = params

    def should_skip_bolus(self, rng: Generator) -> bool:
        """Whether the patient consciously decides to skip this bolus."""
        return float(rng.random()) < self.params.bolus_skip_probability

    def resolve_bolus_timing(self, rng: Generator) -> BolusTimingCategory:
        """Determine when the bolus is delivered relative to the meal.

        Returns one of:
            PRE:        full dose, 5-15 min before meal (ideal for Fiasp)
            LATE_HALF:  half dose, up to 30 min after meal
            FORGOT:     no bolus, patient forgot too long
        """
        roll = float(rng.random())
        if roll < self.params.bolus_timing_pre_pct:
            return BolusTimingCategory.PRE
        if roll < self.params.bolus_timing_pre_pct + self.params.bolus_timing_late_half_pct:
            return BolusTimingCategory.LATE_HALF
        return BolusTimingCategory.FORGOT

    def bolus_timing_delay(self, timing: BolusTimingCategory, rng: Generator) -> timedelta:
        """Compute the actual delay for the resolved timing category.

        Returns negative timedelta for pre-bolus, positive for late bolus.
        """
        if timing == BolusTimingCategory.PRE:
            minutes = rng.normal(
                self.params.pre_bolus_mean_minutes,
                self.params.pre_bolus_std_minutes,
            )
            # BUG #5 fix (H2): clamp upper bound to 0.0 (meal time), not 5.0.
            # A positive value means the bolus is delivered *after* the meal
            # starts, which contradicts the PRE category semantics and would
            # cause the patient to be classified as PRE while behaving as POST.
            minutes = max(-20.0, min(0.0, float(minutes)))
        elif timing == BolusTimingCategory.LATE_HALF:
            minutes = rng.normal(
                self.params.late_bolus_mean_minutes,
                self.params.late_bolus_std_minutes,
            )
            # Clamp late bolus to [1, 30] minutes
            minutes = max(1.0, min(30.0, float(minutes)))
        else:
            minutes = 0.0  # FORGOT/SKIPPED — no delay, no bolus

        return timedelta(minutes=minutes)

    def dose_multiplier_for_timing(self, timing: BolusTimingCategory) -> float:
        """Return the dose fraction based on timing (Fiasp protocol).

        PRE:        1.0  (full dose)
        LATE_HALF:  0.5  (half dose)
        FORGOT:     0.0  (no bolus)
        SKIPPED:    0.0  (no bolus)
        """
        if timing == BolusTimingCategory.PRE:
            return 1.0
        if timing == BolusTimingCategory.LATE_HALF:
            return 0.5
        return 0.0

    def sample_cr_error(self, rng: Generator) -> float:
        """Sample carb ratio error factor.

        Returns a multiplier: 1.0 = correct, <1 = underdose, >1 = overdose.

        Clamped to ``[0.1, 3.0]`` (BUG #5 fix): the upper bound prevents
        absurd overdoses that a real patient would never tolerate
        (>3x intended insulin) while still allowing a wide range of
        realistic misestimation.
        """
        factor = float(
            rng.normal(self.params.cr_error_factor_mean, self.params.cr_error_factor_std)
        )
        return max(0.1, min(3.0, factor))

    def sample_cf_error(self, rng: Generator) -> float:
        """Sample correction factor error.

        Clamped to ``[0.1, 3.0]`` for the same reason as
        :meth:`sample_cr_error` (BUG #5 fix).
        """
        factor = float(
            rng.normal(self.params.cf_error_factor_mean, self.params.cf_error_factor_std)
        )
        return max(0.1, min(3.0, factor))

    def should_phantom_bolus(self, rng: Generator) -> bool:
        """Whether the patient does a correction bolus without a real meal."""
        return float(rng.random()) < self.params.phantom_bolus_probability

    def should_rage_bolus(self, rng: Generator) -> bool:
        """Whether the patient does an aggressive correction out of frustration."""
        return float(rng.random()) < self.params.rage_bolus_probability

    def should_double_dose(self, rng: Generator) -> bool:
        """Whether the patient accidentally doses twice (forgot they bolused)."""
        return float(rng.random()) < self.params.double_dose_probability

    def should_correct(self, current_bg: float, rng: Generator) -> bool:
        """Whether the patient performs a correction bolus at this BG level."""
        if current_bg < self.params.correction_threshold_mg_dl:
            return False
        return float(rng.random()) < self.params.correction_probability

    def round_dose(self, dose: float) -> float:
        """Round a dose to the patient's typical rounding granularity.

        .. note::
           ``bolus_rounding_units`` is validated to be > 0 by the Pydantic
           ``ArchetypeParams`` model (``gt=0.0``), so ZeroDivisionError should
           not occur in normal use.  This guard provides defence-in-depth for
           callers that bypass the model validator (e.g. direct dict construction
           in tests or future refactors).  BUG #7 fix (H2).
        """
        unit = self.params.bolus_rounding_units
        if unit <= 0.0:
            raise ValueError(
                f"bolus_rounding_units must be > 0, got {unit!r}. "
                "Check ArchetypeParams validation."
            )
        return round(dose / unit) * unit

    def effective_iob_fraction(self) -> float:
        """How much of the IOB the patient actually considers."""
        return self.params.iob_consideration


def create_archetype(
    archetype_id: ArchetypeID, params: ArchetypeParams
) -> AdherenceArchetype:
    """Factory to create an archetype from an ID and parameters.

    All archetypes use the same AdherenceArchetype class — behavioral
    differences come entirely from the ArchetypeParams values loaded
    from YAML. No subclassing needed.
    """
    return AdherenceArchetype(archetype_id, params)

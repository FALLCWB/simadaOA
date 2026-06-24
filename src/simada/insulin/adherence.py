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

"""Adherence-modified insulin delivery model.

Bridges the gap between the ideal bolus calculation and what actually
gets delivered. Applies archetype-specific modifications:

1. Timing: pre-bolus / late-half / forgot (Fiasp protocol)
2. Dose calculation errors: wrong CR/CF, IOB ignorance
3. Dose rounding: precision depends on archetype
4. Special events: phantom bolus, rage bolus, double dose
"""

from __future__ import annotations

from datetime import datetime

from numpy.random import Generator

from simada.core.types import BolusTimingCategory, BolusType, InsulinEvent
from simada.insulin.calculator import BolusCalculator
from simada.patient.archetype import AdherenceArchetype


class AdherenceInsulinModel:
    """Applies archetype-specific modifications to insulin delivery.

    This is the key bridge between the ideal BolusCalculator output
    and what actually gets delivered to the patient in the simulation.
    """

    def __init__(
        self,
        archetype: AdherenceArchetype,
        calculator: BolusCalculator,
    ) -> None:
        self._archetype = archetype
        self._calculator = calculator

    def severe_hyper_rescue_dose(
        self, time: datetime, current_bg: float, rng: Generator
    ) -> float:
        """Correction dose for the SEVERE-hyperglycemia self-rescue.

        Mirror of the hypo carb-rescue: every real T1D patient eventually acts
        on a sustained severe high (the pump's repeating high alert forces
        attention), so WHETHER it fires is NOT gated by adherence -- it fires
        for all archetypes. But the DOSE itself carries the same archetype
        errors as a normal correction: the patient may mis-set CF, ignore part
        of their insulin-on-board, and (when frustrated) rage-correct. So the
        rescue is NOT perfectly IOB-aware -- a poorly-adherent patient can
        ignore IOB and over-bolus here, risking a rebound hypo (a realistic
        adverse event, not engineered away). Capped at the pump per-bolus
        hardware limit (max_single_bolus_u). Returns 0.0 if no dose warranted.
        """
        cf_error = self._archetype.sample_cf_error(rng)
        iob_fraction = self._archetype.effective_iob_fraction()
        calc = self._calculator.calculate_with_errors(
            estimated_carbs=0.0,
            current_bg=current_bg,
            current_time=time,
            cr_error=1.0,
            cf_error=cf_error,
            iob_fraction=iob_fraction,
        )
        dose = calc.total_dose
        # Frustration "rage" over-correction (more likely in poor adherence) —
        # a realistic way the self-rescue itself can overshoot toward a hypo.
        if self._archetype.should_rage_bolus(rng):
            dose *= self._archetype.params.rage_bolus_extra_factor
        dose = min(dose, self._archetype.params.max_single_bolus_u)
        return self._archetype.round_dose(dose)

    def process_meal_bolus(
        self,
        meal_time: datetime,
        estimated_carbs: float,
        current_bg: float,
        rng: Generator,
    ) -> InsulinEvent:
        """Process a meal bolus with all adherence modifications.

        Returns an InsulinEvent describing what actually happened —
        including skips, timing, errors, and special events.
        """
        scheduled_time = meal_time

        # BUG-FIX (H3 #5): Spawn independent sub-streams so the
        # skip/timing decisions ("did the patient bolus / when") are
        # statistically independent from the dose-error draws
        # ("if they bolused, what CR/CF/IOB errors did they make").
        # Without this, all draws share state and a perturbation in the
        # skip probability shifts the entire downstream error profile
        # for the same seed, polluting reproducibility studies.
        decision_rng, dose_rng = rng.spawn(2)

        # Step 1: Should this bolus be skipped?
        if self._archetype.should_skip_bolus(decision_rng):
            return self._make_event(
                time=meal_time,
                scheduled_time=scheduled_time,
                basal_rate=0.0,
                bolus_dose=0.0,
                bolus_calculated=0.0,
                bolus_type=BolusType.MEAL,
                timing=BolusTimingCategory.SKIPPED,
                was_skipped=True,
                cr_error=1.0,
                cf_error=1.0,
                iob_considered=0.0,
            )

        # Step 2: Resolve timing
        timing = self._archetype.resolve_bolus_timing(decision_rng)
        if timing == BolusTimingCategory.FORGOT:
            # BUG-FIX H5 #4: Compute the ideal dose even when the patient
            # forgot to bolus so the bolus_calculated field reflects the true
            # magnitude of the missed dose in adherence metrics (previously
            # stored 0.0, hiding the dose loss entirely).
            ideal_forgot = self._calculator.calculate(
                estimated_carbs, current_bg, meal_time
            )
            return self._make_event(
                time=meal_time,
                scheduled_time=scheduled_time,
                basal_rate=0.0,
                bolus_dose=0.0,
                bolus_calculated=ideal_forgot.total_dose,
                bolus_type=BolusType.MEAL,
                timing=BolusTimingCategory.FORGOT,
                was_skipped=False,
                cr_error=1.0,
                cf_error=1.0,
                iob_considered=0.0,
            )

        # Step 3: Compute timing delay
        delay = self._archetype.bolus_timing_delay(timing, decision_rng)
        delivery_time = meal_time + delay
        dose_multiplier = self._archetype.dose_multiplier_for_timing(timing)

        # Step 4: Calculate dose with errors (independent sub-stream)
        cr_error = self._archetype.sample_cr_error(dose_rng)
        cf_error = self._archetype.sample_cf_error(dose_rng)
        iob_fraction = self._archetype.effective_iob_fraction()

        calc = self._calculator.calculate_with_errors(
            estimated_carbs=estimated_carbs,
            current_bg=current_bg,
            current_time=delivery_time,
            cr_error=cr_error,
            cf_error=cf_error,
            iob_fraction=iob_fraction,
        )

        # Step 5: Apply timing dose multiplier (half dose for late)
        dose = calc.total_dose * dose_multiplier

        # Step 6: Round dose
        dose = self._archetype.round_dose(dose)

        # Step 7: Pump hardware limit — cap single bolus delivery
        dose = min(dose, self._archetype.params.max_single_bolus_u)

        # Step 8: Check for double dose (forgot they already bolused) and
        # rage bolus (frustration overdose). BUG #4 fix: these two events
        # are mutually exclusive -- a patient cannot simultaneously "forget
        # that they already dosed" AND consciously "rage-correct". Both
        # rolls are sampled so the RNG sequence does not depend on which
        # branch wins, but only one effect is applied. Clinical choice:
        # prefer the DOUBLE path because forgetting a prior bolus is the
        # documented more-common error mode and stacking rage on top of
        # double yielded a 2.8x worst case (rage 1.4 x double 2.0) that
        # could spike to ~84U with large meals -- well past any pump
        # hardware limit.
        actual_dose = dose
        bolus_type = BolusType.MEAL
        rolled_rage = self._archetype.should_rage_bolus(decision_rng)
        rolled_double = self._archetype.should_double_dose(decision_rng)
        if rolled_double:
            actual_dose = dose * 2.0
            bolus_type = BolusType.DOUBLE
        elif rolled_rage:
            rage_factor = self._archetype.params.rage_bolus_extra_factor
            actual_dose = self._archetype.round_dose(dose * rage_factor)
            bolus_type = BolusType.RAGE
        # Re-apply pump hardware limit after any multiplier so the cap is
        # always honoured, even on the DOUBLE/RAGE branch.
        actual_dose = min(
            actual_dose, self._archetype.params.max_single_bolus_u
        )

        # Ideal dose (what the correct calculation would give)
        ideal = self._calculator.calculate(
            estimated_carbs, current_bg, delivery_time
        )

        # Record the bolus for IOB tracking
        self._calculator.record_bolus(delivery_time, actual_dose)

        return self._make_event(
            time=delivery_time,
            scheduled_time=scheduled_time,
            basal_rate=0.0,
            bolus_dose=actual_dose,
            bolus_calculated=ideal.total_dose,
            bolus_type=bolus_type,
            timing=timing,
            was_skipped=False,
            cr_error=cr_error,
            cf_error=cf_error,
            iob_considered=iob_fraction,
        )

    def process_correction_bolus(
        self,
        time: datetime,
        current_bg: float,
        rng: Generator,
        allow_phantom: bool = True,
    ) -> InsulinEvent | None:
        """Process a potential correction bolus (between meals).

        Args:
            time: Current simulation time.
            current_bg: Current blood glucose reading.
            rng: Random generator for stochastic decisions.
            allow_phantom: BUG #6 fix -- when False, suppress the phantom
                bolus roll regardless of probability. The controller uses
                this gate to enforce a 30 min cooldown between phantom
                events so it cannot stack 3 phantoms in 10 minutes.

        Returns None if no correction is warranted.
        """
        if not self._archetype.should_correct(current_bg, rng):
            return None

        # IOB hard limit — suppress correction when IOB is dangerously high
        if self._archetype.params.iob_hard_limit_u > 0:
            current_iob = self._calculator.compute_iob(time)
            if current_iob >= self._archetype.params.iob_hard_limit_u:
                return None  # pump warning suppresses correction

        cf_error = self._archetype.sample_cf_error(rng)
        iob_fraction = self._archetype.effective_iob_fraction()

        # Correction-only calculation (no carbs)
        calc = self._calculator.calculate_with_errors(
            estimated_carbs=0.0,
            current_bg=current_bg,
            current_time=time,
            cr_error=1.0,
            cf_error=cf_error,
            iob_fraction=iob_fraction,
        )

        dose = self._archetype.round_dose(calc.total_dose)
        if dose <= 0:
            return None

        # Determine correction sub-type (mutually exclusive: rage > phantom > normal)
        # BUG #6 fix: the phantom roll is gated by allow_phantom so the
        # controller can enforce a 30 min cooldown. We still consume the
        # RNG draw on the suppressed branch to keep reproducibility intact
        # across allow_phantom toggles.
        bolus_type = BolusType.CORRECTION
        if self._archetype.should_rage_bolus(rng):
            dose *= self._archetype.params.rage_bolus_extra_factor
            # BUG-FIX H5 #2: cap BEFORE final round_dose so the hardware
            # limit is applied on the pre-rounded value.  Applying cap after
            # round_dose could permit a dose up to (granularity - epsilon)
            # above max_single_bolus_u in borderline cases.
            dose = min(dose, self._archetype.params.max_single_bolus_u)
            dose = self._archetype.round_dose(dose)
            bolus_type = BolusType.RAGE
        else:
            phantom_rolled = self._archetype.should_phantom_bolus(rng)
            if phantom_rolled and allow_phantom:
                bolus_type = BolusType.PHANTOM

        # Pump hardware limit — cap non-rage paths (rage already capped above)
        dose = min(dose, self._archetype.params.max_single_bolus_u)

        ideal = self._calculator.calculate(0.0, current_bg, time)
        self._calculator.record_bolus(time, dose)

        return self._make_event(
            time=time,
            scheduled_time=time,
            basal_rate=0.0,
            bolus_dose=dose,
            bolus_calculated=ideal.total_dose,
            bolus_type=bolus_type,
            # BUG-FIX H5 #7: correction boluses are not meal-relative; use
            # the dedicated CORRECTION category instead of PRE so timing
            # distribution analyses are not polluted by off-label PRE events.
            timing=BolusTimingCategory.CORRECTION,
            was_skipped=False,
            cr_error=1.0,
            cf_error=cf_error,
            iob_considered=iob_fraction,
        )

    @staticmethod
    def _make_event(
        time: datetime,
        scheduled_time: datetime,
        basal_rate: float,
        bolus_dose: float,
        bolus_calculated: float,
        bolus_type: BolusType,
        timing: BolusTimingCategory,
        was_skipped: bool,
        cr_error: float,
        cf_error: float,
        iob_considered: float,
    ) -> InsulinEvent:
        return InsulinEvent(
            time=time,
            scheduled_time=scheduled_time,
            basal_rate=basal_rate,
            bolus_dose=bolus_dose,
            bolus_calculated=bolus_calculated,
            bolus_type=bolus_type,
            timing_category=timing,
            was_skipped=was_skipped,
            cr_error_factor=cr_error,
            cf_error_factor=cf_error,
            iob_considered=iob_considered,
        )

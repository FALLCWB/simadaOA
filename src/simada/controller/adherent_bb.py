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

"""Archetype-aware basal-bolus controller for simglucose.

The controller:
1. Delivers basal insulin following the circadian profile
2. Adjusts basal for exercise (increased sensitivity) and stress/illness
   (increased resistance) by querying scenario context
3. Responds to meal announcements via the AdherenceInsulinModel
4. Corrects hyperglycemia (>correction_threshold) with phantom boluses every 30 min
5. Corrects hypoglycemia (<70) with carb intake (tablets/juice/soda)
6. Triggers emergency glucagon rescue when BG < 30 mg/dL (severe neuroglycopenia)
7. Applies all archetype-specific behavioral modifications
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

from numpy.random import Generator

from simglucose.controller.base import Action as CtrlAction
from simglucose.controller.base import Controller

from simada.behavior.counterregulation import CounterregulationModel
from simada.behavior.exercise import INTENSITY_PROFILES
from simada.core.config import ArchetypeParams
from simada.core.types import (
    BolusType,
    ExerciseIntensity,
    GlucagonEvent,
    PhysiologicalLimitEvent,
)
from simada.insulin.adherence import AdherenceInsulinModel
from simada.insulin.basal import BasalProfile
from simada.physiology.renal import BG_HARD_CEILING, apply_renal_correction
from simada.scenario.custom_scenario import SimadaScenario

# How often (in simulation steps) to check for correction boluses (hyper).
# At 3-min sample time, every 10 steps = every 30 minutes.
_CORRECTION_CHECK_INTERVAL = 10

# Hypo treatment options: (name, carbohydrates in grams)
_HYPO_TREATMENTS = {
    "tablete_glicose": 3.8,   # 1 glucose tablet — adherent uses 2 or 4 depending on severity
    "suco": 20.0,             # orange juice 200ml
    "refrigerante": 26.0,     # cola soda 200ml
}

# Below this, escalate the hypo treatment (double dose). Aligned to ADA Level 2
# clinically-significant hypoglycemia (<54 mg/dL; ADA Standards of Care 2024,
# from the 2017 ADA/EASD International Hypoglycaemia Study Group) and to the
# pump low-glucose-suspend threshold, which is also 54.
_SEVERE_HYPO_MG_DL = 54.0

# BUG #1 fix: emergency glucagon rescue threshold.
# Clinical practice teaches that severe neuroglycopenia (loss of consciousness,
# seizures) typically begins below 30-40 mg/dL -- by BG<20 the patient is
# already in coma, far too late to trigger a rescue. We use 30 mg/dL as a
# conservative threshold so the simulation triggers glucagon while the
# patient is still treatable (Cryer 2009; ADA Standards of Care 2024).
_GLUCAGON_RESCUE_MG_DL = 30.0

# --- Severe hyperglycemia self-rescue (high-side mirror of the carb/glucagon
# rescue). Real T1D patients do not sit in severe hyperglycemia: the pump's
# repeating high alert forces attention and they take correction insulin, then
# re-dose periodically until glucose comes down. Without this, a poorly
# adherent virtual patient drifts into DKA/HHS range (BG>600) where the
# UVA/Padova model is invalid and the ODE solver fails. This is a patient
# REACTION (NOT a closed loop -- that is a separate, future control module).
# Grounded in: CGM high-alert default 250 mg/dL repeating every 60 min; sick-day
# rules correcting + rechecking sustained hyperglycemia (Control-IQ high alert /
# CGM alert literature). Fires for ALL archetypes (adherence-independent), just
# like the low-glucose suspend and the carb rescue. The dose is IOB-aware
# (self-limiting -- will not stack into a hypo).
_SEVERE_HYPER_MG_DL = 350.0          # threshold to start the self-rescue
_SEVERE_HYPER_REDOSE_MIN = 60.0      # Y: re-dose interval while still >= threshold
# X (onset delay before the patient reacts) is per-archetype
# (ArchetypeParams.severe_hyper_onset_minutes: adherent 15 / moderate 30 /
# nonadherent 45) because chronic hyperglycemia blunts symptom perception.

# BUG #6 fix: phantom bolus cooldown (minutes). Real patients cannot reasonably
# stack phantom corrections every 30 minutes; impose a 30 min hard cooldown
# beyond the existing _CORRECTION_CHECK_INTERVAL.
_PHANTOM_BOLUS_COOLDOWN_MIN = 30.0


class AdherentBBController(Controller):
    """Basal-bolus controller modified by adherence archetype.

    Handles both hyper corrections (insulin) and hypo corrections (carbs).
    """

    def __init__(
        self,
        insulin_model: AdherenceInsulinModel,
        basal_profile: BasalProfile,
        scenario: SimadaScenario,
        archetype_params: ArchetypeParams,
        rng: Generator,
        start_time: datetime,
        counterregulation: CounterregulationModel | None = None,
    ) -> None:
        super().__init__(init_state=0)
        self._insulin_model = insulin_model
        self._basal_profile = basal_profile
        self._scenario = scenario
        self._params = archetype_params
        self._rng = rng
        self._start_time = start_time
        self._current_time = start_time
        # Optional hepatic counterregulation (physiology, reactive to BG; off
        # unless a model is supplied). Releases glucose during hypoglycemia.
        self._counterregulation = counterregulation
        self._counterreg_total_g: float = 0.0
        self._pending_bolus: float = 0.0
        self._processed_meals: set[datetime] = set()
        self._step_count: int = 0
        # H6#7 fix: dedicated correction step counter, not advanced during meal
        # steps, so interleaved meal boluses do not shift correction timing.
        self._correction_step_count: int = 0
        self._first_call: bool = True
        # Hypo correction state
        self._last_hypo_correction: datetime | None = None
        self._hypo_correction_count: int = 0
        self._bg_at_last_correction: float | None = None
        self._episode_cho_total: float = 0.0
        # Basal suspend state (generic pump low-glucose suspend, not closed-loop)
        self._basal_suspended: bool = False
        # Glucagon rescue state
        self._glucagon_events: list[GlucagonEvent] = []
        self._last_glucagon_time: datetime | None = None
        # Severe-hyperglycemia self-rescue state (high-side mirror of glucagon)
        self._severe_hyper_since: datetime | None = None
        self._last_severe_hyper_dose: datetime | None = None
        self._severe_hyper_doses: int = 0
        # H6#2 fix: suppress hypo correction for 20 min after glucagon fires.
        # Replaced the old 1-step bool flag (_glucagon_just_fired) with a
        # datetime so the suppression window is time-based (20 min), not
        # step-count-based (which was only ~3 min at 3-min sample time).
        self._glucagon_fired_time: datetime | None = None
        # BUG #6 fix: phantom bolus cooldown -- track last phantom firing so
        # we never deliver another phantom within 30 minutes of the previous.
        self._last_phantom_bolus: datetime | None = None
        # Post-exercise tracking: (end_time, sensitivity_multiplier, post_hours)
        self._last_exercise_end: datetime | None = None
        self._last_exercise_sensitivity: float = 1.0
        self._last_exercise_post_hours: float = 0.0
        # Last known good CGM value (fallback for NaN sensor dropout)
        self._last_known_cgm: float = 120.0
        # BG ceiling tracking (renal correction validation metric)
        self._bg_ceiling_hits: int = 0
        # Physiological limit events (WP4 validation)
        self._limit_events: list[PhysiologicalLimitEvent] = []

    @property
    def glucagon_events(self) -> list[GlucagonEvent]:
        """Emergency glucagon rescue events logged during the simulation."""
        return list(self._glucagon_events)

    @property
    def counterregulation_glucose_g(self) -> float:
        """Total carb-equivalent glucose released by hepatic counterregulation."""
        return self._counterreg_total_g

    @property
    def bg_ceiling_hits(self) -> int:
        """Number of time steps where raw BG exceeded the hard ceiling (600 mg/dL)."""
        return self._bg_ceiling_hits

    @property
    def limit_events(self) -> list[PhysiologicalLimitEvent]:
        """Physiological limit events logged during the simulation."""
        return list(self._limit_events)

    def policy(self, observation, reward, done, **info) -> CtrlAction:
        """Compute insulin delivery for this time step."""
        sample_time = info.get("sample_time", 3)

        if self._first_call:
            self._first_call = False
        else:
            self._current_time += timedelta(minutes=sample_time)
        self._step_count += 1

        cgm = observation.CGM if hasattr(observation, "CGM") else 120.0
        if math.isnan(cgm):
            cgm = self._last_known_cgm
        else:
            self._last_known_cgm = cgm

        # Physiological BG ceiling — renal excretion correction
        if cgm > BG_HARD_CEILING:
            self._bg_ceiling_hits += 1
            self._limit_events.append(PhysiologicalLimitEvent(
                time=self._current_time,
                limit_type="bg_ceiling",
                original_value=cgm,
                clamped_value=BG_HARD_CEILING,
                context=f"Raw CGM {cgm:.0f} exceeded ceiling {BG_HARD_CEILING:.0f}",
            ))
        cgm = apply_renal_correction(cgm, sample_time_min=sample_time)

        # --- COUNTERREGULATION (optional physiology, reactive to BG) ---
        # Hepatic glycogenolysis defends against hypoglycemia by releasing
        # glucose; modeled as a carb-equivalent injection (off unless enabled).
        # Runs before insulin/rescue logic since it is involuntary physiology.
        # LIMITATION: driven by the (renal-corrected) CGM value, not true plasma
        # glucose -- a real glycogenolytic response keys off plasma glucose, so
        # sensor lag/noise desynchronizes the release. The controller has no
        # plasma-BG channel; this CGM-driven approximation is a known fidelity
        # gap and a likely co-cause of the adverse interaction in the xfail test.
        if self._counterregulation is not None:
            cr_g = self._counterregulation.step(cgm, float(sample_time))
            if cr_g > 0.0:
                self._scenario.inject_cho(self._current_time, cr_g)
                self._counterreg_total_g += cr_g

        # Basal delivery
        basal_info = self._basal_profile.rate_at(self._current_time)
        basal_u_per_min = basal_info.rate_u_per_hr / 60.0

        # --- CONTEXT: exercise and stress effects on basal rate ---
        basal_u_per_min = self._apply_context_adjustment(basal_u_per_min)

        # --- GLUCAGON RESCUE (BG < 30 — emergency, max 1 per 24h) ---
        # BUG #1 fix: 20 mg/dL was too late -- the patient is typically
        # comatose by then. 30 mg/dL still represents severe neuroglycopenia
        # but leaves a window where rescue can be administered.
        glucagon_fired_now = False
        if cgm < _GLUCAGON_RESCUE_MG_DL:
            glucagon_fired_now = self._handle_glucagon_rescue(cgm)

        # --- HYPO CORRECTION (all archetypes correct when BG < 70) ---
        # H6#2 fix: suppress hypo correction for 20 min after glucagon fires.
        # Glucagon already injected ~50g CHO; piling more on top risks rebound
        # hyper while the CGM still reads in the danger zone.
        if cgm < self._params.hypo_threshold_mg_dl:
            glucagon_suppressed = False
            # NOTE: suppressing the voluntary carb-treatment while counterregulation
            # is active was TESTED and made the hypo much WORSE (min 49.5 vs 65.1) --
            # the voluntary correction is the dominant defense, so it must NOT be
            # gated on counterregulation. The adverse interaction (xfail) stems from
            # counterreg raising the CGM and thereby reducing the voluntary
            # correction; it is not fixable by suppression. Left documented.
            if glucagon_fired_now:
                glucagon_suppressed = True
            elif self._glucagon_fired_time is not None:
                elapsed_min = (
                    self._current_time - self._glucagon_fired_time
                ).total_seconds() / 60.0
                # Suppress for a full 3-min-step-aligned window of >=20 min.
                # 7 steps × 3 min = 21 min; we suppress while elapsed <= 21 min
                # so the first allowed correction step is at 24 min (step 8).
                if elapsed_min <= 21.0:
                    glucagon_suppressed = True
            if not glucagon_suppressed:
                self._handle_hypo_correction(cgm)

        # --- MEAL BOLUS ---
        meal_cho = observation.CHO if hasattr(observation, "CHO") else 0
        if meal_cho > 0:
            meal_event = self._scenario.get_meal_at(self._current_time)
            if meal_event is not None and meal_event.time not in self._processed_meals:
                self._processed_meals.add(meal_event.time)
                event = self._insulin_model.process_meal_bolus(
                    meal_time=self._current_time,
                    estimated_carbs=meal_event.estimated_carbs_g,
                    current_bg=cgm,
                    rng=self._rng,
                )
                if event.bolus_dose > 0:
                    self._pending_bolus += event.bolus_dose
            # H6#7 fix: meal step does NOT advance _correction_step_count.
            # The counter only ticks on non-meal steps so meal boluses cannot
            # shift the correction interval timing.

        # --- HYPER CORRECTION (phantom bolus every 30min when BG > 250) ---
        # H6#7 fix: use _correction_step_count (not _step_count) so meal steps
        # do not shift correction timing. Only non-meal steps advance the counter.
        else:
            self._correction_step_count += 1
            if self._correction_step_count % _CORRECTION_CHECK_INTERVAL == 0:
                # BUG #6 fix: gate phantom bolus by a 30-minute cooldown so we
                # never stack 3 phantom corrections inside 10 minutes. Regular
                # corrections still fire on every interval; only the random
                # PHANTOM type is suppressed during the cooldown window.
                allow_phantom = True
                if self._last_phantom_bolus is not None:
                    minutes_since = (
                        self._current_time - self._last_phantom_bolus
                    ).total_seconds() / 60.0
                    if minutes_since < _PHANTOM_BOLUS_COOLDOWN_MIN:
                        allow_phantom = False
                correction_event = self._insulin_model.process_correction_bolus(
                    time=self._current_time,
                    current_bg=cgm,
                    rng=self._rng,
                    allow_phantom=allow_phantom,
                )
                if correction_event is not None and correction_event.bolus_dose > 0:
                    self._pending_bolus += correction_event.bolus_dose
                    # Track the phantom firing time for cooldown bookkeeping.
                    if correction_event.bolus_type == BolusType.PHANTOM:
                        self._last_phantom_bolus = self._current_time

        # --- SEVERE HYPERGLYCEMIA SELF-RESCUE (high-side mirror of the carb
        # rescue; runs every step, all archetypes, adherence-independent) ---
        self._handle_severe_hyper_rescue(cgm)

        # If BG is back in range, reset hypo correction state (episode ends)
        if cgm >= self._params.hypo_threshold_mg_dl:
            self._hypo_correction_count = 0
            self._bg_at_last_correction = None
            self._last_hypo_correction = None
            self._episode_cho_total = 0.0

        # --- A1: LOW-GLUCOSE SUSPEND (generic pump, not closed-loop) ---
        # Automatic pump safety feature: suspend basal when BG < 54 (TBR L2),
        # resume when BG >= 70. Applies to ALL archetypes — pump behavior.
        # Note: 1-step (3-min) latency is inherent to the simulation loop —
        # the controller reacts to the PREVIOUS step's BG, same as real pumps.
        if cgm < 54.0:
            basal_u_per_min = 0.0
            self._basal_suspended = True
        elif self._basal_suspended and cgm >= self._params.hypo_threshold_mg_dl:
            self._basal_suspended = False
        if self._basal_suspended:
            basal_u_per_min = 0.0

        # Deliver pending bolus
        bolus_u_per_min = 0.0
        if self._pending_bolus > 0:
            bolus_u_per_min = self._pending_bolus / sample_time
            self._pending_bolus = 0.0

        return CtrlAction(basal=basal_u_per_min, bolus=bolus_u_per_min)

    def _apply_context_adjustment(self, basal_u_per_min: float) -> float:
        """Adjust basal rate based on exercise and stress context.

        Exercise increases insulin sensitivity → reduce basal to avoid hypo.
        The effect persists for hours after exercise ends (post-exercise sensitivity).
        Stress/illness increases insulin resistance → increase basal.
        Alcohol phase 2 decreases resistance (factor < 1.0) → reduce basal.

        The adherent archetype fully adjusts; nonadherent partially ignores.
        """
        ctx = self._scenario.get_context(self._current_time)
        adjustment = 1.0

        # Exercise effect: reduce basal (more sensitive = need less insulin)
        if ctx["exercise_active"]:
            intensity = ExerciseIntensity(ctx["exercise_intensity"])
            profile = INTENSITY_PROFILES[intensity]
            # Reduce basal by inverse of sensitivity multiplier
            # e.g., 2.0x sensitivity → deliver 50% of normal basal
            adjustment /= profile.sensitivity_multiplier
            # Track for post-exercise effect
            self._last_exercise_end = self._current_time
            self._last_exercise_sensitivity = profile.sensitivity_multiplier
            self._last_exercise_post_hours = profile.post_sensitivity_hours

        # Post-exercise residual effect (decaying linearly)
        elif self._last_exercise_end is not None:
            hours_since = (self._current_time - self._last_exercise_end).total_seconds() / 3600
            if hours_since < self._last_exercise_post_hours:
                decay = 1.0 - (hours_since / self._last_exercise_post_hours)
                residual_sensitivity = 1.0 + (self._last_exercise_sensitivity - 1.0) * decay
                adjustment /= residual_sensitivity
            else:
                self._last_exercise_end = None  # expired

        # Stress/illness/alcohol effect on insulin resistance
        resistance_factor = ctx["insulin_resistance_factor"]
        if resistance_factor != 1.0:
            # resistance > 1.0 → need more insulin → multiply basal
            # resistance < 1.0 (alcohol phase 2) → need less → multiply basal
            adjustment *= resistance_factor

        # Archetype modulation: adherent fully adjusts, nonadherent partially.
        # H6#4 fix: use context_attention (dedicated param for exercise/stress
        # awareness, separate from bolus IOB accounting).
        # context_attention defaults to the same value as the IOB param when
        # not set explicitly, preserving backwards compatibility.
        attention = self._params.context_attention  # 1.0 adherent, 0.1 nonadherent
        effective_adjustment = 1.0 + (adjustment - 1.0) * attention

        return basal_u_per_min * effective_adjustment

    def _handle_hypo_correction(self, cgm: float) -> None:
        """Correct hypoglycemia via carbohydrate intake.

        All archetypes correct hypo (survival instinct). The ADA 'Rule of 15'
        (15 g fast carbs, recheck in 15 min; ADA Standards of Care 2024) is the
        clinical reference. NOTE: the simulator uses a SEVERITY-GRADED dose, not a
        flat 15 g -- mild hypo gets a partial dose (~7.6 g), reflecting the common
        real-world under-treatment of mild lows, and only clinically-significant
        hypo (<54 mg/dL, ADA Level 2) gets the full ~15 g. This graded under-
        treatment is a documented modeling choice (see ASSUMPTIONS Limitations),
        not the literal Rule of 15.

        Adherent (severity-graded):
            BG 54-70: 2 glucose tablets (7.6 g), wait 15 min
            BG < 54:  4 glucose tablets (15.2 g), wait 15 min (full Rule of 15)
            No escalation -- repeats same dose if needed.

        Moderate:
            Knows the Rule of 15 but sometimes grabs juice (20 g).
            Waits 15 min. Mild escalation if not improving.

        Nonadherent (impatient, aggressive):
            Prefers soda (26g) or juice (20g).
            Waits 7min. Escalates 1.5x each attempt.

        The correction logic is TREND-BASED, not count-based:
            - First correction in an episode: always correct.
            - Subsequent: wait hypo_recheck_minutes, then check trend.
              If BG is rising (current > BG at last correction): don't correct,
              the previous CHO is working.
              If BG is flat or falling: correct again.
            - hypo_max_corrections is kept as a high safety net (10) but the
              primary gate is the trend check.
        """
        # Safety-net hard limit (should rarely bind -- trend logic is primary)
        if self._hypo_correction_count >= self._params.hypo_max_corrections:
            return

        recheck_minutes = self._params.hypo_recheck_minutes
        if self._last_hypo_correction is not None:
            elapsed = (self._current_time - self._last_hypo_correction).total_seconds() / 60
            if elapsed < recheck_minutes:
                return
            # Trend-based decision: if BG is rising since last correction,
            # the treatment is working -- don't pile on more CHO.
            if self._bg_at_last_correction is not None and cgm > self._bg_at_last_correction:
                return

        # Select treatment based on archetype preference
        prefs = self._params.hypo_treatment_preference
        if prefs:
            treatment_name = prefs[0]
            if len(prefs) > 1:
                idx = int(self._rng.integers(0, len(prefs)))
                treatment_name = prefs[idx]
        else:
            treatment_name = "tablete_glicose"

        base_carbs = _HYPO_TREATMENTS.get(treatment_name, 15.0)

        # For tablets: adjust quantity by hypo severity
        if treatment_name == "tablete_glicose":
            if cgm < _SEVERE_HYPO_MG_DL:
                base_carbs = 3.8 * 4  # 4 tablets = 15.2g (Rule of 15)
            else:
                base_carbs = 3.8 * 2  # 2 tablets = 7.6g (mild hypo)

        # Apply overcorrection factor for repeated attempts
        self._hypo_correction_count += 1
        factor = self._params.hypo_overcorrection_factor
        # Cap escalation at 3 steps (max ~2x base at factor=1.25)
        raw_count = self._hypo_correction_count - 1
        effective_count = min(raw_count, 3)
        if effective_count > 0:
            carbs = base_carbs * (factor ** effective_count)
        else:
            carbs = base_carbs
        # Log escalation cap event if capping occurred
        if raw_count > 3:
            uncapped_carbs = base_carbs * (factor ** raw_count)
            self._limit_events.append(PhysiologicalLimitEvent(
                time=self._current_time,
                limit_type="escalation_cap",
                original_value=uncapped_carbs,
                clamped_value=carbs,
                context=f"Escalation capped at step 3 (attempt {raw_count})",
            ))

        # Stomach capacity limit
        remaining = self._params.hypo_max_episode_cho_g - self._episode_cho_total
        if remaining <= 0:
            self._limit_events.append(PhysiologicalLimitEvent(
                time=self._current_time,
                limit_type="cho_episode_cap",
                original_value=carbs,
                clamped_value=0.0,
                context=f"Episode CHO total {self._episode_cho_total:.0f}g reached cap {self._params.hypo_max_episode_cho_g:.0f}g",
            ))
            return
        if carbs > remaining:
            self._limit_events.append(PhysiologicalLimitEvent(
                time=self._current_time,
                limit_type="cho_episode_cap",
                original_value=carbs,
                clamped_value=remaining,
                context=f"CHO clamped from {carbs:.1f}g to {remaining:.1f}g (episode cap)",
            ))
            carbs = remaining
        self._episode_cho_total += carbs

        self._scenario.inject_cho(self._current_time, carbs)
        self._last_hypo_correction = self._current_time
        self._bg_at_last_correction = cgm

    def _handle_severe_hyper_rescue(self, cgm: float) -> None:
        """Patient self-rescue in SEVERE, sustained hyperglycemia.

        High-side mirror of the carb/glucagon rescue. Once CGM stays at or
        above ``_SEVERE_HYPER_MG_DL`` for ``_SEVERE_HYPER_ONSET_MIN``, the
        patient takes an IOB-aware correction bolus, then re-doses every
        ``_SEVERE_HYPER_REDOSE_MIN`` while still severe, until glucose drops
        below the threshold. Fires for ALL archetypes regardless of adherence
        (the repeating pump high alert forces action) -- this is what stops a
        poorly adherent patient from drifting into DKA/HHS range (BG>600).

        It is a patient REACTION, NOT a closed loop. The dose is IOB-aware and
        capped at the pump per-bolus limit, so it is self-limiting and will not
        stack into a hypo.
        """
        if cgm < _SEVERE_HYPER_MG_DL:
            # Episode over -- reset so a future episode needs its own onset wait.
            self._severe_hyper_since = None
            self._last_severe_hyper_dose = None
            return

        if self._severe_hyper_since is None:
            self._severe_hyper_since = self._current_time
        sustained_min = (
            self._current_time - self._severe_hyper_since
        ).total_seconds() / 60.0
        if sustained_min < self._params.severe_hyper_onset_minutes:
            return  # not severe long enough yet (onset differs by archetype)

        if self._last_severe_hyper_dose is not None:
            since_min = (
                self._current_time - self._last_severe_hyper_dose
            ).total_seconds() / 60.0
            if since_min < _SEVERE_HYPER_REDOSE_MIN:
                return  # already dosed within the re-dose window

        dose = self._insulin_model.severe_hyper_rescue_dose(
            self._current_time, cgm, self._rng
        )
        if dose <= 0:
            return
        self._pending_bolus += dose
        self._last_severe_hyper_dose = self._current_time
        self._severe_hyper_doses += 1
        self._limit_events.append(PhysiologicalLimitEvent(
            time=self._current_time,
            limit_type="severe_hyper_rescue",
            original_value=cgm,
            clamped_value=dose,
            context=(
                f"Sustained CGM>={_SEVERE_HYPER_MG_DL:.0f} for "
                f"{sustained_min:.0f} min; self-rescue correction {dose:.1f} U"
            ),
        ))

    def _handle_glucagon_rescue(self, cgm: float) -> bool:
        """Emergency glucagon rescue when BG < 30 mg/dL.

        This is a serious adverse event. In real life the patient would be
        unconscious or seizing and a bystander would administer glucagon.
        We model it as a 50g CHO injection (physiological equivalent).

        Maximum 1 glucagon per 24 hours. Resets all hypo correction state
        so the trend logic starts fresh after rescue.

        Returns:
            True if glucagon was administered this call, False if it was
            suppressed by the 24h cooldown.
        """
        if self._last_glucagon_time is not None:
            hours_since = (self._current_time - self._last_glucagon_time).total_seconds() / 3600
            if hours_since < 24.0:
                return False

        event = GlucagonEvent(
            time=self._current_time,
            bg_at_rescue=cgm,
            carbs_injected=50.0,
        )
        self._glucagon_events.append(event)
        self._scenario.inject_cho(self._current_time, event.carbs_injected)
        self._last_glucagon_time = self._current_time
        self._glucagon_fired_time = self._current_time  # H6#2: start 20-min suppression window

        # Reset hypo correction state -- fresh start after rescue
        self._hypo_correction_count = 0
        self._bg_at_last_correction = None
        self._last_hypo_correction = None
        self._episode_cho_total = 0.0
        return True

    def reset(self) -> None:
        """Reset controller state for a new simulation run."""
        self._current_time = self._start_time
        self._pending_bolus = 0.0
        self._processed_meals.clear()
        self._step_count = 0
        self._correction_step_count = 0  # H6#7
        self._first_call = True
        self._last_hypo_correction = None
        self._hypo_correction_count = 0
        self._bg_at_last_correction = None
        self._episode_cho_total = 0.0
        self._basal_suspended = False
        self._glucagon_events = []
        self._last_glucagon_time = None
        self._glucagon_fired_time = None  # H6#2
        self._severe_hyper_since = None
        self._last_severe_hyper_dose = None
        self._severe_hyper_doses = 0
        self._last_phantom_bolus = None
        self._last_exercise_end = None
        self._last_exercise_sensitivity = 1.0
        self._last_exercise_post_hours = 0.0
        self._last_known_cgm = 120.0
        self._bg_ceiling_hits = 0
        self._limit_events = []
        # counterregulation accounting + its (depleting) glycogen store must also
        # reset, else state leaks across runs that share this controller/model.
        self._counterreg_total_g = 0.0
        if self._counterregulation is not None:
            self._counterregulation.reset()
        # the bolus calculator carries IOB (bolus history) that must not leak
        # across runs sharing this controller.
        self._insulin_model._calculator.reset()

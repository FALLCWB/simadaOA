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

"""Foundational data types for the simada simulation framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ArchetypeID(str, Enum):
    """Patient adherence archetype identifier."""

    ADHERENT = "adherent"
    MODERATE = "moderate"
    NONADHERENT = "nonadherent"


class DayType(str, Enum):
    """Classification of a calendar day for behavioral modeling."""

    WEEKDAY = "weekday"
    WEEKEND = "weekend"
    HOLIDAY = "holiday"


class InsulinRegimen(str, Enum):
    """Insulin delivery method."""

    MDI = "mdi"
    PUMP = "pump"


class MealType(str, Enum):
    """Structured meal types following Brazilian eating patterns."""

    CAFE_DA_MANHA = "cafe_da_manha"
    LANCHE_MANHA = "lanche_manha"
    ALMOCO = "almoco"
    LANCHE_TARDE = "lanche_tarde"
    JANTAR = "jantar"
    CEIA = "ceia"
    BRUNCH = "brunch"
    CHURRASCO = "churrasco"
    SNACK = "snack"


class ExerciseIntensity(str, Enum):
    """Exercise intensity levels with distinct physiological effects."""

    LIGHT = "light"
    MODERATE = "moderate"
    VIGOROUS = "vigorous"


class StressType(str, Enum):
    """Categories of transient perturbations to insulin sensitivity."""

    PSYCHOLOGICAL = "psychological"
    ILLNESS = "illness"
    ALCOHOL = "alcohol"


# ---------------------------------------------------------------------------
# Event dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MealEvent:
    """A single meal event with both true and estimated carbohydrate content.

    The separation between true_carbs_g and estimated_carbs_g is fundamental:
    true_carbs_g drives the physiological model (simglucose), while
    estimated_carbs_g is what the patient/controller perceives. This mismatch
    is the core challenge that Safe RL must handle.
    """

    time: datetime
    meal_type: MealType
    true_carbs_g: float
    estimated_carbs_g: float
    foods: tuple[str, ...]
    glycemic_index: float


class BolusType(str, Enum):
    """Classification of a bolus event."""

    MEAL = "meal"
    CORRECTION = "correction"
    PHANTOM = "phantom"  # correction without real meal ("ghost" bolus to push BG down)
    RAGE = "rage"  # aggressive extra correction out of frustration
    DOUBLE = "double"  # accidental double-dose (forgot already bolused)


class BolusTimingCategory(str, Enum):
    """When the bolus was delivered relative to the meal.

    Based on Fiasp (ultra-rapid insulin) protocol:
        PRE:        5-15 min before meal → full dose (ideal)
        LATE_HALF:  up to 30 min after meal → half dose
        FORGOT:     >30 min after meal → no bolus, let basal/pump handle
        SKIPPED:    patient consciously decided not to bolus
        CORRECTION: correction bolus (between meals, not meal-relative)
    """

    PRE = "pre"
    LATE_HALF = "late_half"
    FORGOT = "forgot"
    SKIPPED = "skipped"
    CORRECTION = "correction"


@dataclass(frozen=True)
class InsulinEvent:
    """An insulin delivery event (basal segment or bolus).

    Tracks both what the correct dose would have been and what was actually
    delivered, allowing analysis of adherence-related dosing errors.
    """

    time: datetime
    scheduled_time: datetime
    basal_rate: float  # U/hr
    bolus_dose: float  # U (what was actually delivered)
    bolus_calculated: float  # U (what the correct dose would be)
    bolus_type: BolusType
    timing_category: BolusTimingCategory
    was_skipped: bool
    cr_error_factor: float  # multiplier applied to carb ratio (1.0 = correct)
    cf_error_factor: float  # multiplier applied to correction factor (1.0 = correct)
    iob_considered: float  # fraction of IOB that was subtracted (0.0-1.0)


@dataclass(frozen=True)
class ExerciseEvent:
    """An exercise session with its physiological effect parameters."""

    start_time: datetime
    duration_minutes: int
    intensity: ExerciseIntensity
    insulin_sensitivity_multiplier: float


@dataclass(frozen=True)
class StressEvent:
    """A transient perturbation to insulin sensitivity."""

    start_time: datetime
    duration_minutes: int
    stress_type: StressType
    insulin_resistance_factor: float  # >1.0 means more resistant


@dataclass(frozen=True)
class GlucagonEvent:
    """Emergency glucagon rescue -- serious adverse event.

    In real life this corresponds to a glucagon injection or nasal spray
    administered by a bystander when the patient is unable to self-treat.
    We model it as a large CHO injection (physiological equivalent) because
    simglucose does not have a glucagon compartment.
    """

    time: datetime
    bg_at_rescue: float
    carbs_injected: float = 50.0  # glucagon equivalent in grams of CHO


@dataclass(frozen=True)
class PhysiologicalLimitEvent:
    """Logged when a physiological safety limit is hit during simulation."""

    time: datetime
    limit_type: str  # "max_bolus", "bg_ceiling", "cho_episode_cap", "iob_hard_limit", "escalation_cap"
    original_value: float
    clamped_value: float
    context: str = ""


# NOTE: PerturbationEvent will be added here when the perturbation generator
# is implemented (currently scaffolding only — see ScenarioConfig.perturbations).


# ---------------------------------------------------------------------------
# Composite types
# ---------------------------------------------------------------------------


@dataclass
class DailySchedule:
    """Complete schedule of events for a single simulated day.

    This is the primary output of the behavioral model and the input
    to the scenario engine that bridges to simglucose.

    NOTE: Intentionally NOT frozen (mutable). The controller appends
    insulin_events at runtime as the simulation progresses, so the schedule
    must remain mutable after construction. All other dataclasses in this
    module are frozen because they represent immutable point-in-time events.
    If you need to prevent post-construction mutation of schedule times (e.g.,
    wake_time, sleep_time), enforce this at the builder level rather than here.
    """

    date: datetime
    day_type: DayType
    wake_time: datetime
    sleep_time: datetime
    meals: list[MealEvent] = field(default_factory=list)
    insulin_events: list[InsulinEvent] = field(default_factory=list)
    exercise_events: list[ExerciseEvent] = field(default_factory=list)
    stress_events: list[StressEvent] = field(default_factory=list)
    # perturbations: list will be added when the perturbation generator is implemented


# ---------------------------------------------------------------------------
# Clinical target ranges (Battelino et al., 2019 consensus)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GlycemicTargets:
    """International consensus glycemic targets for T1D management."""

    tir_low: float = 70.0  # mg/dL
    tir_high: float = 180.0  # mg/dL
    tbr_level1_low: float = 54.0
    tbr_level1_high: float = 69.0
    tbr_level2_threshold: float = 54.0
    tar_level1_low: float = 181.0
    tar_level1_high: float = 250.0
    tar_level2_threshold: float = 250.0
    target_tir_pct: float = 70.0  # >70%
    target_tbr_l1_pct: float = 4.0  # <4%
    target_tbr_l2_pct: float = 1.0  # <1%
    target_tar_l1_pct: float = 25.0  # <25%
    target_tar_l2_pct: float = 5.0  # <5%
    target_cv_pct: float = 36.0  # <36%


DEFAULT_TARGETS = GlycemicTargets()

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

"""Pydantic configuration models and YAML loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

_TOLERANCE = 1e-6


# ---------------------------------------------------------------------------
# Archetype configuration
# ---------------------------------------------------------------------------


class ArchetypeParams(BaseModel):
    """Behavioral parameters for a single adherence archetype.

    Every numeric value is configurable via YAML so researchers can tune
    individual parameters without touching code.
    """

    # -- Carb estimation --
    carb_estimation_error_mean: float = Field(
        description="Mean relative error in carb estimation (negative = underestimation)"
    )
    carb_estimation_error_std: float = Field(
        description="Std dev of relative carb estimation error"
    )

    # -- Meal timing --
    meal_time_variance_minutes: float = Field(
        description="Std dev of meal timing offset in minutes"
    )

    # -- Bolus behavior --
    bolus_skip_probability: float = Field(
        ge=0.0, le=1.0,
        description="Probability of consciously deciding to skip a meal bolus",
    )

    # Bolus timing model (Fiasp / ultra-rapid):
    #   pre_bolus: 5-15 min BEFORE meal → full dose (ideal)
    #   late_half: remembered up to 30 min AFTER → half dose
    #   forgot:    remembered >30 min AFTER → no bolus, let basal/pump handle
    # These three must sum to 1.0. Applied only when bolus is NOT skipped.
    bolus_timing_pre_pct: float = Field(
        default=0.96, ge=0.0, le=1.0,
        description="Fraction of non-skipped boluses given 5-15 min before meal (full dose)",
    )
    bolus_timing_late_half_pct: float = Field(
        default=0.03, ge=0.0, le=1.0,
        description="Fraction given up to 30 min after meal (half dose, Fiasp protocol)",
    )
    bolus_timing_forgot_pct: float = Field(
        default=0.01, ge=0.0, le=1.0,
        description="Fraction where patient forgot too long (>30 min), no bolus given",
    )
    pre_bolus_mean_minutes: float = Field(
        default=-10.0,
        description="Mean timing for pre-bolus (negative = before meal). Fiasp: -15 to -5.",
    )
    pre_bolus_std_minutes: float = Field(
        default=3.0, ge=0.0,
        description="Std dev of pre-bolus timing",
    )
    late_bolus_mean_minutes: float = Field(
        default=15.0, ge=0.0,
        description="Mean delay for late bolus (within 30 min after meal)",
    )
    late_bolus_std_minutes: float = Field(
        default=8.0, ge=0.0,
        description="Std dev of late bolus timing",
    )

    double_dose_probability: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Probability of accidentally dosing twice (forgot already bolused)",
    )
    phantom_bolus_probability: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Probability of correction stacking without a real meal (inexperience/anxiety)",
    )
    rage_bolus_probability: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Probability of aggressive extra correction when BG stays high (frustration)",
    )
    rage_bolus_extra_factor: float = Field(
        default=1.5, ge=1.0,
        description="Multiplier applied to correction dose during a rage bolus",
    )

    # -- Bolus calculation errors --
    cr_error_factor_mean: float = Field(
        default=1.0,
        description=(
            "Mean multiplier on carb ratio. effective_cr = CR * factor. "
            "1.0 = correct. <1.0 = lower effective CR = more insulin per gram = OVERDOSE. "
            ">1.0 = higher effective CR = less insulin per gram = UNDERDOSE."
        ),
    )
    cr_error_factor_std: float = Field(
        default=0.0, ge=0.0,
        description="Std dev of carb ratio error factor",
    )
    cf_error_factor_mean: float = Field(
        default=1.0,
        description=(
            "Mean multiplier on correction factor. effective_cf = CF * factor. "
            "1.0 = correct. <1.0 = lower effective CF = more correction insulin = OVERDOSE. "
            ">1.0 = higher effective CF = less correction insulin = UNDERDOSE."
        ),
    )
    cf_error_factor_std: float = Field(
        default=0.0, ge=0.0,
        description="Std dev of correction factor error",
    )
    iob_consideration: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="How much IOB is considered: 0.0 = ignores, 1.0 = fully considers",
    )
    context_attention: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description=(
            "How attentively the patient adjusts basal for exercise/stress context. "
            "Defaults to iob_consideration when not set explicitly, preserving "
            "backwards compatibility (H6#4 fix)."
        ),
    )
    correction_threshold_mg_dl: float = Field(
        default=170.0, ge=0.0,
        description=(
            "BG threshold above which a correction bolus is considered. "
            "Archetype defaults (BUG #5 fix): adherent=170, moderate=190, "
            "nonadherent=220. Each archetype YAML overrides this so the "
            "controller threshold is no longer hardcoded uniformly."
        ),
    )
    severe_hyper_onset_minutes: float = Field(
        default=30.0, ge=0.0,
        description=(
            "Minutes that CGM must stay at/above the severe-hyperglycemia "
            "threshold before the patient self-rescues with correction insulin "
            "(high-side mirror of the carb rescue). Differs by archetype because "
            "chronic hyperglycemia blunts symptom perception and the glycemic "
            "thresholds for symptoms reset with recurrent exposure, so a "
            "well-controlled patient reacts sooner than a poorly-controlled one. "
            "Archetype defaults: adherent=15, moderate=30, nonadherent=45 "
            "(within the literature envelope: pump high alert repeats every "
            "60 min; sick-day rules act on sustained hyperglycemia). "
            "Refs: Robinson 2021 (missed boluses); hyperglycemia symptom "
            "blunting / threshold adaptation literature."
        ),
    )
    correction_probability: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Probability of actually doing a correction when BG is above threshold",
    )
    bolus_rounding_units: float = Field(
        default=0.5, gt=0.0,
        description="Dose rounding granularity in units (0.5 = half-unit, 1.0 = full unit)",
    )

    # -- Exercise --
    exercise_probability_weekday: float = Field(default=0.70, ge=0.0, le=1.0)
    exercise_probability_weekend: float = Field(default=0.80, ge=0.0, le=1.0)
    exercise_intensity_light_weight: float = Field(
        default=0.40, ge=0.0,
        description="Relative weight for light exercise selection",
    )
    exercise_intensity_moderate_weight: float = Field(
        default=0.40, ge=0.0,
        description="Relative weight for moderate exercise selection",
    )
    exercise_intensity_vigorous_weight: float = Field(
        default=0.20, ge=0.0,
        description="Relative weight for vigorous exercise selection",
    )

    # -- Snacking --
    snack_probability: float = Field(default=0.10, ge=0.0, le=1.0)

    # -- Stress / illness / alcohol --
    stress_probability_per_day: float = Field(
        default=0.15, ge=0.0, le=1.0,
        description="Daily probability of a psychological stress event",
    )
    alcohol_probability_weekend: float = Field(
        default=0.10, ge=0.0, le=1.0,
        description="Probability of alcohol consumption on a weekend day",
    )
    alcohol_weekday_factor: float = Field(
        default=0.25, ge=0.0, le=1.0,
        description="Multiplier applied to alcohol_probability_weekend for weekdays",
    )

    # -- Monitoring --
    monitor_frequency_per_day: int = Field(default=10, ge=0)

    # -- Circadian / sleep-wake --
    wake_time_weekday_hour: float = Field(default=6.5, description="Mean wake hour on weekdays (e.g. 6.5)")
    wake_time_weekday_std_minutes: float = Field(default=15.0, ge=0.0)
    wake_time_weekend_hour: float = Field(default=7.5, description="Mean wake hour on weekends (e.g. 9.0)")
    wake_time_weekend_std_minutes: float = Field(default=30.0, ge=0.0)
    sleep_time_weekday_hour: float = Field(default=22.5, description="Mean sleep hour on weekdays (e.g. 22.5)")
    sleep_time_weekday_std_minutes: float = Field(default=20.0, ge=0.0)
    sleep_time_weekend_hour: float = Field(default=23.5, description="Mean sleep hour on weekends (e.g. 23.5)")
    sleep_time_weekend_std_minutes: float = Field(default=30.0, ge=0.0)

    # -- Day-type pattern modifiers --
    weekend_dessert_probability_boost: float = Field(
        default=0.15, ge=0.0, le=1.0,
        description="Extra dessert/snack slot probability on weekends",
    )
    holiday_dessert_probability_boost: float = Field(
        default=0.30, ge=0.0, le=1.0,
        description="Extra dessert/snack slot probability on holidays",
    )
    weekend_extra_snack_probability: float = Field(
        default=0.25, ge=0.0, le=1.0,
        description="Probability of additional unplanned snack on weekends",
    )
    holiday_extra_snack_probability: float = Field(
        default=0.50, ge=0.0, le=1.0,
        description="Probability of additional unplanned snack on holidays",
    )
    weekday_extra_snack_probability: float = Field(
        default=0.10, ge=0.0, le=1.0,
        description="Probability of additional unplanned snack on weekdays",
    )

    # -- Diet adherence --
    diet_adherence: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="How closely the patient follows a planned diet (1.0 = strict, 0.0 = no plan)",
    )

    # -- Hypo correction --
    hypo_threshold_mg_dl: float = Field(
        default=70.0, ge=0.0,
        description="BG threshold below which hypoglycemia correction is triggered",
    )
    hypo_recheck_minutes: float = Field(
        default=15.0, gt=0.0,
        description="Minutes between hypo rechecks (adherent=15, moderate=10, nonadherent=5)",
    )
    hypo_treatment_preference: list[str] = Field(
        default=["glucose_tablet"],
        description="Ordered preference: glucose_tablet (3.8g), suco_laranja (20g), refrigerante (26g)",
    )
    hypo_overcorrection_factor: float = Field(
        default=1.0, ge=1.0,
        description="Multiplier for repeated corrections (1.0 = no escalation, >1 = takes more each time)",
    )
    hypo_max_corrections: int = Field(
        default=3, ge=1,
        description="Maximum number of hypo corrections per episode before letting pump/basal handle it",
    )

    # -- Pump safety limits (hardware constraints) --
    max_single_bolus_u: float = Field(
        default=25.0, gt=0.0,
        description="Maximum single bolus delivery in units (pump hardware limit).",
    )
    hypo_max_episode_cho_g: float = Field(
        default=120.0, gt=0.0,
        description="Maximum CHO per hypo correction episode in grams (stomach capacity).",
    )
    iob_hard_limit_u: float = Field(
        default=0.0, ge=0.0,
        description="Suppress correction boluses when IOB exceeds this. 0 = no limit.",
    )

    @model_validator(mode="after")
    def _set_context_attention_default(self) -> ArchetypeParams:
        """H6#4: context_attention defaults to iob_consideration when not set."""
        if self.context_attention is None:
            object.__setattr__(self, "context_attention", self.iob_consideration)
        return self

    @model_validator(mode="after")
    def _validate_bolus_timing_sums_to_one(self) -> ArchetypeParams:
        total = (
            self.bolus_timing_pre_pct
            + self.bolus_timing_late_half_pct
            + self.bolus_timing_forgot_pct
        )
        if abs(total - 1.0) > _TOLERANCE:
            msg = (
                f"bolus_timing percentages must sum to 1.0, "
                f"got {total:.4f} "
                f"(pre={self.bolus_timing_pre_pct}, "
                f"late_half={self.bolus_timing_late_half_pct}, "
                f"forgot={self.bolus_timing_forgot_pct})"
            )
            raise ValueError(msg)
        return self


# ---------------------------------------------------------------------------
# Meal configuration
# ---------------------------------------------------------------------------


class MealConfig(BaseModel):
    """Meal generation settings."""

    locale: str = "brazil"
    taco_path: Path = Path("data/taco/taco_foods.csv")
    weekend_carb_increase_pct: float = 15.0
    holiday_carb_increase_pct: float = 30.0


# ---------------------------------------------------------------------------
# Behavior configuration
# ---------------------------------------------------------------------------


class ExerciseConfig(BaseModel):
    enabled: bool = True


class StressConfig(BaseModel):
    enabled: bool = True
    illness_probability_per_day: float = 0.01


class AlcoholConfig(BaseModel):
    enabled: bool = True


class SnackingConfig(BaseModel):
    enabled: bool = True


class BehaviorConfig(BaseModel):
    exercise: ExerciseConfig = ExerciseConfig()
    stress: StressConfig = StressConfig()
    alcohol: AlcoholConfig = AlcoholConfig()
    snacking: SnackingConfig = SnackingConfig()


# ---------------------------------------------------------------------------
# Insulin configuration
# ---------------------------------------------------------------------------


class InsulinConfig(BaseModel):
    target_bg: float = 120.0
    iob_duration_hours: float = 4.0
    dawn_phenomenon: bool = True


# ---------------------------------------------------------------------------
# Perturbation configuration (placeholder — generator not yet implemented)
# ---------------------------------------------------------------------------

# PerturbationScheduleEntry and PerturbationConfig will be added here when
# the perturbation generator is implemented.  The YAML key ``perturbations``
# is accepted and silently ignored until then.


# ---------------------------------------------------------------------------
# Cohort configuration
# ---------------------------------------------------------------------------


class CohortConfig(BaseModel):
    """Configuration for generating a patient cohort."""

    size: int = 30
    archetype_distribution: dict[str, float] = Field(
        default_factory=lambda: {
            "adherent": 0.30,
            "moderate": 0.50,
            "nonadherent": 0.20,
        }
    )
    # Optional EXACT per-archetype counts. When provided, this overrides the
    # probabilistic ``archetype_distribution`` and assigns exactly the requested
    # number of patients to each archetype, interleaved (round-robin) across the
    # patient pool so each archetype receives a balanced spread of physiologies
    # (adolescent/adult/child) rather than a contiguous block. Used for balanced
    # designs such as "10 adherent / 10 moderate / 10 nonadherent". The counts
    # must sum to ``size``.
    archetype_counts: dict[str, int] | None = None
    patient_pool: str | list[str] = "all"
    insulin_regimen_distribution: dict[str, float] = Field(
        default_factory=lambda: {
            "pump": 0.60,
            "mdi": 0.40,
        }
    )
    sensor: str = "Dexcom"
    pump: str = "Insulet"

    @model_validator(mode="after")
    def _validate_distributions(self) -> CohortConfig:
        for name, dist in [
            ("archetype_distribution", self.archetype_distribution),
            ("insulin_regimen_distribution", self.insulin_regimen_distribution),
        ]:
            total = sum(dist.values())
            if abs(total - 1.0) > _TOLERANCE:
                msg = f"{name} must sum to 1.0, got {total:.4f}"
                raise ValueError(msg)
        if self.archetype_counts is not None:
            if any(c < 0 for c in self.archetype_counts.values()):
                msg = "archetype_counts values must be non-negative"
                raise ValueError(msg)
            total_counts = sum(self.archetype_counts.values())
            if total_counts != self.size:
                msg = (
                    f"archetype_counts must sum to cohort size ({self.size}), "
                    f"got {total_counts}"
                )
                raise ValueError(msg)
        return self


# ---------------------------------------------------------------------------
# Scenario configuration
# ---------------------------------------------------------------------------


class ScenarioConfig(BaseModel):
    """Top-level scenario definition."""

    name: str = "default"
    description: str = ""
    duration_days: int = 7
    start_date: str = "2026-01-05"
    cohort: CohortConfig = Field(default_factory=CohortConfig)
    holidays: list[dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Output configuration
# ---------------------------------------------------------------------------


class OutputConfig(BaseModel):
    directory: Path = Path("results")
    formats: list[str] = ["parquet"]
    streaming: bool = True
    generate_reports: bool = True
    metrics: list[str] = ["tir", "tbr_l1", "tbr_l2", "tar_l1", "tar_l2", "gmi", "cv"]


class ParallelConfig(BaseModel):
    enabled: bool = False
    n_workers: int = 4


# ---------------------------------------------------------------------------
# Root configuration
# ---------------------------------------------------------------------------


class SimulationConfig(BaseModel):
    """Root configuration model — loaded from YAML."""

    model_config = {"extra": "ignore"}  # allows ``perturbations`` key in YAML

    seed: int = 42
    scenario: ScenarioConfig = ScenarioConfig()
    meals: MealConfig = MealConfig()
    behavior: BehaviorConfig = BehaviorConfig()
    insulin: InsulinConfig = InsulinConfig()
    # perturbations: will be re-added when the perturbation generator is implemented
    output: OutputConfig = OutputConfig()
    parallel: ParallelConfig = ParallelConfig()


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def load_config(path: Path) -> SimulationConfig:
    """Load and validate a simulation configuration from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raw = {}
    return SimulationConfig.model_validate(raw)


def load_archetype_params(path: Path) -> ArchetypeParams:
    """Load archetype parameters from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return ArchetypeParams.model_validate(raw)

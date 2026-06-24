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

"""Unplanned snacking generator.

Models between-meal snacking that is NOT part of the meal plan. This is
distinct from planned lanches — these are impulsive snacks driven by
cravings, boredom, or social context. Uses the day-type snack
probability boosts from ArchetypeParams.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from numpy.random import Generator

from simada.core.config import ArchetypeParams
from simada.core.types import DayType, MealEvent, MealType
from simada.meals.estimation import CarbEstimationModel
from simada.meals.taco import TACODatabase

# Unplanned snacks are typically high-GI, small portions
_SNACK_CATEGORIES = ["salgado", "sobremesa", "panificado", "fruta", "bebida"]
_SNACK_CARB_RANGE = (8.0, 40.0)  # grams per snack

# Fraction by which perfect diet adherence (1.0) suppresses snack probability.
# A fully adherent patient has their base+boost probability halved; a fully
# non-adherent patient (diet_adherence=0) retains the full probability.
# Source: clinical assumption (no single physiological paper; chosen to yield
# ~2x more snacking in nonadherent vs adherent archetypes in simulations).
DIET_ADHERENCE_SNACK_REDUCTION = 0.5


class SnackGenerator:
    """Generates unplanned snacking events between meals.

    The probability of snacking depends on:
    1. Base snack_probability from the archetype
    2. Day-type boost from the archetype (weekday/weekend/holiday)
    3. Diet adherence (lower adherence = more snacking)
    """

    def __init__(
        self, params: ArchetypeParams, taco_db: TACODatabase
    ) -> None:
        self._params = params
        self._taco = taco_db
        self._estimator = CarbEstimationModel(params)

    def generate(
        self,
        day_type: DayType,
        wake_time: datetime,
        sleep_time: datetime,
        existing_meal_times: list[datetime],
        rng: Generator,
    ) -> list[MealEvent]:
        """Generate unplanned snack events for a day.

        Snacks are placed in the gaps between existing meals, avoiding
        the 30 minutes before/after each meal.
        """
        # Compute effective snack probability.
        # Apply diet_adherence FIRST so the reduction acts on the full
        # base+boost value; only then clamp to [0, 1].  Previously the clamp
        # happened before the diet reduction, which silently made the boost a
        # no-op for high-probability archetypes (H3 bug #5).
        base_prob = self._params.snack_probability
        day_boost = self._get_day_boost(day_type)
        # Diet adherence reduces snacking
        effective_prob = (base_prob + day_boost) * (
            1.0 - self._params.diet_adherence * DIET_ADHERENCE_SNACK_REDUCTION
        )
        effective_prob = min(1.0, effective_prob)

        snacks: list[MealEvent] = []

        # Maximum number of snacks possible on this day, capped by the
        # archetype's effective snack probability.
        max_snacks = 1
        if effective_prob > 0.40:
            max_snacks = 2
        if effective_prob > 0.60:
            max_snacks = 3

        # BUG-FIX (H3 #1): Previously, the per-attempt loop rolled
        # ``rng.random() < effective_prob`` N independent times. For high
        # ``effective_prob`` (e.g. nonadherent + holiday), this could yield
        # up to ``max_snacks`` snacks every day with probability ~p^N,
        # producing >10 snacks/day across the week. The correct interpretation
        # is: ``effective_prob`` is the probability that the patient snacks
        # AT ALL on this day, then we sample the COUNT in [1, max_snacks]
        # uniformly. This decouples the per-day occurrence decision from the
        # how-many decision and prevents runaway snacking.
        if rng.random() >= effective_prob:
            return snacks

        # Patient snacks today — sample count uniformly in [1, max_snacks].
        n_snacks = 1 if max_snacks == 1 else int(rng.integers(1, max_snacks + 1))

        for _ in range(n_snacks):
            snack = self._make_snack(wake_time, sleep_time, existing_meal_times, rng)
            if snack is not None:
                snacks.append(snack)
                existing_meal_times = sorted(
                    [*existing_meal_times, snack.time]
                )

        return snacks

    def _get_day_boost(self, day_type: DayType) -> float:
        """Get the snack probability boost for this day type."""
        if day_type == DayType.HOLIDAY:
            return self._params.holiday_extra_snack_probability
        if day_type == DayType.WEEKEND:
            return self._params.weekend_extra_snack_probability
        return self._params.weekday_extra_snack_probability

    def _make_snack(
        self,
        wake_time: datetime,
        sleep_time: datetime,
        meal_times: list[datetime],
        rng: Generator,
    ) -> MealEvent | None:
        """Create a single snack event in a gap between meals."""
        gaps = self._find_gaps(wake_time, sleep_time, meal_times)
        if not gaps:
            return None

        # Pick a random gap, weighted by duration
        durations = [(end - start).total_seconds() for start, end in gaps]
        total = sum(durations)
        if total <= 0:
            return None
        probs = [d / total for d in durations]
        gap_idx = int(rng.choice(len(gaps), p=probs))
        gap_start, gap_end = gaps[gap_idx]

        # Place snack randomly within the gap
        gap_minutes = (gap_end - gap_start).total_seconds() / 60.0
        offset = float(rng.uniform(0, gap_minutes))
        snack_time = gap_start + timedelta(minutes=offset)

        # Sample a snack food
        category = _SNACK_CATEGORIES[int(rng.integers(0, len(_SNACK_CATEGORIES)))]
        if not self._taco.has_category(category):
            category = "sobremesa"
            if not self._taco.has_category(category):
                return None

        food = self._taco.random_food(category, rng)
        servings = float(rng.uniform(0.3, 1.0))
        true_carbs = food.carbs_for_serving(servings)
        true_carbs = max(_SNACK_CARB_RANGE[0], min(_SNACK_CARB_RANGE[1], true_carbs))
        estimated_carbs = self._estimator.estimate(true_carbs, rng)

        return MealEvent(
            time=snack_time,
            meal_type=MealType.SNACK,
            true_carbs_g=round(true_carbs, 1),
            estimated_carbs_g=round(estimated_carbs, 1),
            foods=(food.nome_pt,),
            glycemic_index=food.indice_glicemico,
        )

    def _find_gaps(
        self,
        wake_time: datetime,
        sleep_time: datetime,
        meal_times: list[datetime],
    ) -> list[tuple[datetime, datetime]]:
        """Find time gaps between meals (at least 30 min from any meal)."""
        buffer = timedelta(minutes=30)
        sorted_times = sorted(meal_times)

        gaps: list[tuple[datetime, datetime]] = []

        # Gap before first meal
        if sorted_times:
            first_gap_end = sorted_times[0] - buffer
            if first_gap_end > wake_time + buffer:
                gaps.append((wake_time + buffer, first_gap_end))

            # Gaps between meals
            for i in range(len(sorted_times) - 1):
                gap_start = sorted_times[i] + buffer
                gap_end = sorted_times[i + 1] - buffer
                if gap_end > gap_start:
                    gaps.append((gap_start, gap_end))

            # Gap after last meal
            last_gap_start = sorted_times[-1] + buffer
            if last_gap_start < sleep_time - buffer:
                gaps.append((last_gap_start, sleep_time - buffer))
        else:
            gaps.append((wake_time + buffer, sleep_time - buffer))

        return gaps

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

"""Meal generation engine.

Composes TACO foods into Brazilian meal templates, applies archetype
timing/estimation errors, and handles weekday/weekend/holiday patterns.
Produces MealEvent objects ready for the simulation scenario.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from numpy.random import Generator

from simada.core.config import ArchetypeParams, MealConfig
from simada.core.types import DayType, MealEvent, MealType
from simada.meals.estimation import CarbEstimationModel
from simada.meals.patterns import DayPatternModifier
from simada.meals.taco import Food, TACODatabase
from simada.meals.templates import DayMealPlan, FoodSlot, MealTemplate

# Maximum attempts to resample a meal within the target carb range
_MAX_RESAMPLE_ATTEMPTS = 10


class MealGenerator:
    """Generates realistic daily meal schedules.

    Pipeline for each meal:
        1. Check if meal occurs (probability roll)
        2. Calculate meal time anchored to wake/sleep time
        3. Apply archetype timing variance
        4. For each food slot: roll inclusion, sample food, sample serving
        5. Compute true carbs and weighted GI
        6. Scale carbs by day-type modifier
        7. Clamp to template carb range (resample if needed)
        8. Apply carb estimation error
        9. Return MealEvent
    """

    def __init__(
        self,
        taco_db: TACODatabase,
        config: MealConfig,
        archetype_params: ArchetypeParams,
    ) -> None:
        self._taco = taco_db
        self._config = config
        self._pattern = DayPatternModifier(config, archetype_params)
        self._estimator = CarbEstimationModel(archetype_params)
        self._time_variance = archetype_params.meal_time_variance_minutes

    def generate_day(
        self,
        day_plan: DayMealPlan,
        day_type: DayType,
        wake_time: datetime,
        sleep_time: datetime,
        rng: Generator,
    ) -> list[MealEvent]:
        """Generate all meals for a single day.

        Args:
            day_plan: Meal templates for this day type.
            day_type: Weekday, weekend, or holiday.
            wake_time: Patient's wake time for this day.
            sleep_time: Patient's sleep time for this day.
            rng: Random number generator (meals stream).

        Returns:
            Sorted list of MealEvent instances for the day.
        """
        carb_scale = self._pattern.carb_scale_factor(day_type)
        dessert_boost = self._pattern.dessert_probability_boost(day_type)
        meals: list[MealEvent] = []

        # BUG-FIX (H3 #5): Spawn independent sub-streams for meal-occurrence
        # decisions vs. food/serving sampling. Previously every draw came
        # from the same Generator state, so changing template.probability
        # propagated unpredictable correlations into food choice and
        # estimation noise for the same seed. Sub-streams keep the
        # "should this meal occur" coin flips deterministic and decoupled
        # from "given that it occurred, which foods/servings/carbs" draws.
        occurrence_rng, sampling_rng = rng.spawn(2)

        # Process replacement meals first to determine which standard meals
        # to skip (e.g. churrasco replaces almoco on Sunday)
        replaced_types: set[MealType] = set()
        for repl_template, replaces in day_plan.replacements:
            # Filter by day_of_week if the template specifies one.
            # E.g. churrasco (day_of_week="sunday") must not appear on Saturdays.
            if repl_template.day_of_week is not None and (
                wake_time.strftime("%A").lower() != repl_template.day_of_week.lower()
            ):
                continue
            if occurrence_rng.random() < repl_template.probability:
                event = self._generate_meal(
                    repl_template, wake_time, sleep_time, carb_scale,
                    dessert_boost, sampling_rng,
                )
                if event is not None:
                    meals.append(event)
                    replaced_types.update(replaces)

        # Process standard meals, skipping replaced ones
        for template in day_plan.templates:
            if template.meal_type in replaced_types:
                continue
            if occurrence_rng.random() > template.probability:
                continue
            event = self._generate_meal(
                template, wake_time, sleep_time, carb_scale,
                dessert_boost, sampling_rng,
            )
            if event is not None:
                meals.append(event)

        meals.sort(key=lambda m: m.time)
        return meals

    def _generate_meal(
        self,
        template: MealTemplate,
        wake_time: datetime,
        sleep_time: datetime,
        carb_scale: float,
        dessert_boost: float,
        rng: Generator,
    ) -> MealEvent | None:
        """Generate a single meal from a template."""
        # Determine base meal time
        meal_time = self._compute_meal_time(
            template, wake_time, sleep_time, rng
        )
        if meal_time is None:
            return None

        # Ensure meal time is between wake and sleep
        if meal_time < wake_time:
            meal_time = wake_time + timedelta(minutes=15)
        if meal_time > sleep_time:
            meal_time = sleep_time - timedelta(minutes=15)

        # Sample foods and compute carbs (with resampling for carb range)
        target_min = template.total_carb_range[0] * carb_scale
        target_max = template.total_carb_range[1] * carb_scale

        best_foods: list[tuple[Food, float]] = []
        best_raw_carbs = 0.0
        best_gi = 0.0

        for _ in range(_MAX_RESAMPLE_ATTEMPTS):
            foods, raw_carbs, weighted_gi = self._sample_foods(
                template.slots, rng, dessert_boost=dessert_boost,
            )
            scaled_carbs = raw_carbs * carb_scale
            if target_min <= scaled_carbs <= target_max:
                best_foods = foods
                best_raw_carbs = raw_carbs
                best_gi = weighted_gi
                break
            # Keep the best attempt if none fit exactly
            target_mid = (target_min + target_max) / 2
            if not best_foods or abs(scaled_carbs - target_mid) < abs(
                best_raw_carbs * carb_scale - target_mid
            ):
                best_foods = foods
                best_raw_carbs = raw_carbs
                best_gi = weighted_gi

        if not best_foods:
            return None

        # Guard: if no food contributed any carbs (e.g. protein-only churrasco
        # with pure Picanha), return None rather than clamping 0 up to
        # target_min and fabricating phantom carbs that trigger ghost insulin.
        if best_raw_carbs <= 0:
            return None

        # Clamp scaled carbs to range
        true_carbs = max(target_min, min(target_max, best_raw_carbs * carb_scale))

        # Compute weighted average GI using RAW carbs (not scaled) to avoid
        # deflating GI on weekends/holidays
        if best_raw_carbs > 0:
            avg_gi = best_gi / best_raw_carbs
        else:
            avg_gi = 50.0

        # Apply archetype estimation error
        estimated_carbs = self._estimator.estimate(true_carbs, rng)

        food_names = tuple(f.nome_pt for f, _ in best_foods)

        return MealEvent(
            time=meal_time,
            meal_type=template.meal_type,
            true_carbs_g=round(true_carbs, 1),
            estimated_carbs_g=round(estimated_carbs, 1),
            foods=food_names,
            glycemic_index=round(avg_gi, 1),
        )

    def _compute_meal_time(
        self,
        template: MealTemplate,
        wake_time: datetime,
        sleep_time: datetime,
        rng: Generator,
    ) -> datetime | None:
        """Compute the actual meal time with archetype-dependent variance."""
        if template.offset_from_wake_minutes is not None:
            base = wake_time + timedelta(minutes=template.offset_from_wake_minutes)
        elif template.offset_before_sleep_minutes is not None:
            base = sleep_time - timedelta(minutes=template.offset_before_sleep_minutes)
        else:
            return None

        # Add archetype timing variance (truncated to avoid extreme shifts)
        variance_minutes = rng.normal(0, self._time_variance)
        # Truncate at 3 sigma to avoid unrealistic times
        max_shift = 3 * self._time_variance
        variance_minutes = max(-max_shift, min(max_shift, variance_minutes))

        return base + timedelta(minutes=float(variance_minutes))

    def _sample_foods(
        self,
        slots: tuple[FoodSlot, ...],
        rng: Generator,
        dessert_boost: float = 0.0,
    ) -> tuple[list[tuple[Food, float]], float, float]:
        """Sample concrete foods from template slots.

        Args:
            slots: Food slots to sample from.
            rng: Random number generator.
            dessert_boost: Additional probability for dessert (sobremesa) slots.

        Returns:
            Tuple of (food_list, total_carbs, total_weighted_gi).
            food_list contains (Food, servings) pairs.
        """
        foods: list[tuple[Food, float]] = []
        total_carbs = 0.0
        total_weighted_gi = 0.0

        for slot in slots:
            # Boost dessert/sobremesa slot probability on weekends/holidays
            effective_prob = slot.probability
            if dessert_boost > 0.0 and slot.category == "sobremesa":
                effective_prob = min(1.0, slot.probability + dessert_boost)

            # Roll slot inclusion
            if rng.random() > effective_prob:
                continue

            # Sample a food from the category
            if not self._taco.has_category(slot.category):
                continue

            if slot.preferred:
                food = self._taco.random_food_preferred(
                    slot.category, list(slot.preferred), rng
                )
            else:
                food = self._taco.random_food(slot.category, rng)

            # Sample serving size
            servings = float(
                rng.uniform(slot.servings_min, slot.servings_max)
            )

            carbs = food.carbs_for_serving(servings)
            gi_contribution = food.weighted_gi(servings)

            foods.append((food, servings))
            total_carbs += carbs
            total_weighted_gi += gi_contribution

        return foods, total_carbs, total_weighted_gi

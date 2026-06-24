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

"""Tests for the meal generation system."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.random import default_rng

from simada.core.config import ArchetypeParams, MealConfig
from simada.core.types import DayType
from simada.meals.estimation import CarbEstimationModel
from simada.meals.generator import MealGenerator
from simada.meals.patterns import DayPatternModifier
from simada.meals.taco import TACODatabase
from simada.meals.templates import load_day_meal_plan

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TestCarbEstimation:
    """Tests for CarbEstimationModel."""

    def test_adherent_no_bias(self, adherent_params: ArchetypeParams) -> None:
        model = CarbEstimationModel(adherent_params)
        rng = default_rng(42)
        true_carbs = 60.0
        estimates = [model.estimate(true_carbs, rng) for _ in range(5000)]
        mean_error = abs(np.mean(estimates) - true_carbs) / true_carbs
        assert mean_error < 0.02, f"Adherent mean bias {mean_error:.3f} exceeds 2%"

    def test_nonadherent_underestimates(self, nonadherent_params: ArchetypeParams) -> None:
        model = CarbEstimationModel(nonadherent_params)
        rng = default_rng(42)
        true_carbs = 60.0
        estimates = [model.estimate(true_carbs, rng) for _ in range(5000)]
        # Nonadherent has -15% bias
        mean_estimate = np.mean(estimates)
        assert mean_estimate < true_carbs, "Nonadherent should underestimate on average"
        relative_bias = (mean_estimate - true_carbs) / true_carbs
        assert -0.20 < relative_bias < -0.10, f"Bias {relative_bias:.3f} outside expected range"

    def test_estimates_never_negative(self, nonadherent_params: ArchetypeParams) -> None:
        model = CarbEstimationModel(nonadherent_params)
        rng = default_rng(42)
        for true_carbs in [5.0, 10.0, 20.0, 50.0, 100.0]:
            for _ in range(1000):
                assert model.estimate(true_carbs, rng) >= 0.0

    def test_zero_carbs_returns_zero(self, adherent_params: ArchetypeParams) -> None:
        model = CarbEstimationModel(adherent_params)
        rng = default_rng(42)
        assert model.estimate(0.0, rng) == 0.0

    def test_estimate_lower_bound_at_ten_percent(
        self, nonadherent_params: ArchetypeParams
    ) -> None:
        """Regression test for H3 bug #2: extreme negative-tail samples
        must not crash the estimate below 10% of true. Before the fix,
        a nonadherent SD=0.20 archetype could (rarely) draw an error of
        e.g. -true_carbs, collapsing the estimate to 0g and causing the
        downstream bolus calculator to deliver zero meal bolus.
        """
        model = CarbEstimationModel(nonadherent_params)
        true_carbs = 60.0
        rng = default_rng(42)
        for _ in range(5000):
            est = model.estimate(true_carbs, rng)
            assert est >= true_carbs * 0.1 - 1e-9, (
                f"Estimate {est:.2f} below 10% floor "
                f"({true_carbs * 0.1:.2f})"
            )
            assert est <= true_carbs * 2.0 + 1e-9, (
                f"Estimate {est:.2f} above 200% ceiling "
                f"({true_carbs * 2.0:.2f})"
            )

    def test_estimate_clamps_extreme_tails(
        self, nonadherent_params: ArchetypeParams
    ) -> None:
        """Regression test for H3 bug #2: deterministic monkey-patched
        rng returning extreme errors must be clamped to the [10%, 200%]
        band rather than producing absurd values.
        """
        model = CarbEstimationModel(nonadherent_params)
        true_carbs = 80.0

        class _FixedRNG:
            def __init__(self, value: float) -> None:
                self._value = value

            def normal(self, mean: float, std: float) -> float:
                return self._value

        # Massive negative error -> floor at 10%
        est_low = model.estimate(true_carbs, _FixedRNG(-1e6))  # type: ignore[arg-type]
        assert est_low == true_carbs * 0.1

        # Massive positive error -> ceiling at 200%
        est_high = model.estimate(true_carbs, _FixedRNG(1e6))  # type: ignore[arg-type]
        assert est_high == true_carbs * 2.0


class TestDayPatternModifier:
    """Tests for DayPatternModifier."""

    def test_weekday_no_scaling(self, adherent_params: ArchetypeParams) -> None:
        config = MealConfig(weekend_carb_increase_pct=15.0, holiday_carb_increase_pct=30.0)
        mod = DayPatternModifier(config, adherent_params)
        assert mod.carb_scale_factor(DayType.WEEKDAY) == 1.0

    def test_weekend_scales_up(self, adherent_params: ArchetypeParams) -> None:
        config = MealConfig(weekend_carb_increase_pct=15.0, holiday_carb_increase_pct=30.0)
        mod = DayPatternModifier(config, adherent_params)
        assert mod.carb_scale_factor(DayType.WEEKEND) == 1.15

    def test_holiday_scales_up_more(self, adherent_params: ArchetypeParams) -> None:
        config = MealConfig(weekend_carb_increase_pct=15.0, holiday_carb_increase_pct=30.0)
        mod = DayPatternModifier(config, adherent_params)
        assert mod.carb_scale_factor(DayType.HOLIDAY) == 1.30

    def test_snack_probabilities_no_dead_method(
        self, adherent_params: ArchetypeParams, nonadherent_params: ArchetypeParams
    ) -> None:
        # extra_snack_probability() was dead code removed in H4#4 fix.
        # Actual extra-snack logic lives in behavior/snacking.py.
        config = MealConfig()
        mod_c = DayPatternModifier(config, adherent_params)
        assert not hasattr(mod_c, "extra_snack_probability"), (
            "extra_snack_probability() should have been removed (dead code). "
            "Snacking logic lives in behavior/snacking.py."
        )


class TestMealGenerator:
    """Tests for the full MealGenerator pipeline."""

    def _make_generator(
        self, taco_db: TACODatabase, params: ArchetypeParams
    ) -> MealGenerator:
        config = MealConfig(
            taco_path=Path("data/taco/taco_foods.csv"),
            weekend_carb_increase_pct=15.0,
            holiday_carb_increase_pct=30.0,
        )
        return MealGenerator(taco_db, config, params)

    def test_generates_meals_for_weekday(
        self, taco_db: TACODatabase, adherent_params: ArchetypeParams
    ) -> None:
        gen = self._make_generator(taco_db, adherent_params)
        plan = load_day_meal_plan(PROJECT_ROOT / "configs" / "meals" / "brazilian_weekday.yaml")
        rng = default_rng(42)

        wake = datetime(2026, 6, 1, 6, 30)
        sleep = datetime(2026, 6, 1, 22, 30)

        meals = gen.generate_day(plan, DayType.WEEKDAY, wake, sleep, rng)

        assert len(meals) >= 2, "Should generate at least 2 meals (cafe + almoco)"
        assert len(meals) <= 7, "Should not generate more than 7 meals in a day"

        # Meals should be sorted by time
        for i in range(len(meals) - 1):
            assert meals[i].time <= meals[i + 1].time

        # All meals should be between wake and sleep
        for meal in meals:
            assert meal.time >= wake - __import__("datetime").timedelta(minutes=1)
            assert meal.time <= sleep + __import__("datetime").timedelta(minutes=1)

    def test_all_meals_have_positive_carbs(
        self, taco_db: TACODatabase, adherent_params: ArchetypeParams
    ) -> None:
        gen = self._make_generator(taco_db, adherent_params)
        plan = load_day_meal_plan(PROJECT_ROOT / "configs" / "meals" / "brazilian_weekday.yaml")
        rng = default_rng(42)

        wake = datetime(2026, 6, 1, 6, 30)
        sleep = datetime(2026, 6, 1, 22, 30)

        for seed in range(20):
            rng = default_rng(seed)
            meals = gen.generate_day(plan, DayType.WEEKDAY, wake, sleep, rng)
            for meal in meals:
                assert meal.true_carbs_g > 0, f"Meal {meal.meal_type} has 0 carbs"
                assert meal.estimated_carbs_g >= 0

    def test_meals_have_food_names(
        self, taco_db: TACODatabase, adherent_params: ArchetypeParams
    ) -> None:
        gen = self._make_generator(taco_db, adherent_params)
        plan = load_day_meal_plan(PROJECT_ROOT / "configs" / "meals" / "brazilian_weekday.yaml")
        rng = default_rng(42)

        wake = datetime(2026, 6, 1, 6, 30)
        sleep = datetime(2026, 6, 1, 22, 30)
        meals = gen.generate_day(plan, DayType.WEEKDAY, wake, sleep, rng)

        for meal in meals:
            assert len(meal.foods) > 0, f"Meal {meal.meal_type} has no foods"
            for food_name in meal.foods:
                assert isinstance(food_name, str)
                assert len(food_name) > 0

    def test_weekend_has_higher_carbs_on_average(
        self, taco_db: TACODatabase, adherent_params: ArchetypeParams
    ) -> None:
        gen = self._make_generator(taco_db, adherent_params)
        weekday_plan = load_day_meal_plan(
            PROJECT_ROOT / "configs" / "meals" / "brazilian_weekday.yaml"
        )
        weekend_plan = load_day_meal_plan(
            PROJECT_ROOT / "configs" / "meals" / "brazilian_weekend.yaml"
        )

        n_samples = 50
        weekday_totals = []
        weekend_totals = []

        for seed in range(n_samples):
            rng = default_rng(seed)
            wake = datetime(2026, 6, 1, 6, 30)
            sleep = datetime(2026, 6, 1, 22, 30)
            meals = gen.generate_day(weekday_plan, DayType.WEEKDAY, wake, sleep, rng)
            weekday_totals.append(sum(m.true_carbs_g for m in meals))

            rng = default_rng(seed + 10000)
            wake_wk = datetime(2026, 6, 1, 8, 0)
            sleep_wk = datetime(2026, 6, 1, 23, 30)
            meals_wk = gen.generate_day(weekend_plan, DayType.WEEKEND, wake_wk, sleep_wk, rng)
            weekend_totals.append(sum(m.true_carbs_g for m in meals_wk))

        avg_weekday = np.mean(weekday_totals)
        avg_weekend = np.mean(weekend_totals)
        assert avg_weekend > avg_weekday, (
            f"Weekend avg ({avg_weekend:.0f}g) should exceed weekday ({avg_weekday:.0f}g)"
        )

    def test_reproducibility(
        self, taco_db: TACODatabase, adherent_params: ArchetypeParams
    ) -> None:
        gen = self._make_generator(taco_db, adherent_params)
        plan = load_day_meal_plan(PROJECT_ROOT / "configs" / "meals" / "brazilian_weekday.yaml")

        wake = datetime(2026, 6, 1, 6, 30)
        sleep = datetime(2026, 6, 1, 22, 30)

        rng1 = default_rng(42)
        meals1 = gen.generate_day(plan, DayType.WEEKDAY, wake, sleep, rng1)

        rng2 = default_rng(42)
        meals2 = gen.generate_day(plan, DayType.WEEKDAY, wake, sleep, rng2)

        assert len(meals1) == len(meals2)
        for m1, m2 in zip(meals1, meals2):
            assert m1.true_carbs_g == m2.true_carbs_g
            assert m1.estimated_carbs_g == m2.estimated_carbs_g
            assert m1.foods == m2.foods
            assert m1.time == m2.time

    def test_meal_generator_reproducible_after_substream_split(
        self, taco_db: TACODatabase, adherent_params: ArchetypeParams
    ) -> None:
        """Regression test for H3 bug #5: spawning sub-streams inside
        generate_day must preserve deterministic reproducibility for a
        given input seed.
        """
        gen = self._make_generator(taco_db, adherent_params)
        plan = load_day_meal_plan(
            PROJECT_ROOT / "configs" / "meals" / "brazilian_weekday.yaml"
        )
        wake = datetime(2026, 6, 1, 6, 30)
        sleep = datetime(2026, 6, 1, 22, 30)

        meals1 = gen.generate_day(plan, DayType.WEEKDAY, wake, sleep, default_rng(7))
        meals2 = gen.generate_day(plan, DayType.WEEKDAY, wake, sleep, default_rng(7))

        assert len(meals1) == len(meals2)
        for m1, m2 in zip(meals1, meals2):
            assert m1.time == m2.time
            assert m1.true_carbs_g == m2.true_carbs_g
            assert m1.foods == m2.foods

    def test_meal_occurrence_decoupled_from_food_sampling(
        self, taco_db: TACODatabase, adherent_params: ArchetypeParams
    ) -> None:
        """Regression test for H3 bug #5: meal-occurrence rolls and
        food-sampling rolls live on independent sub-streams. As a
        smoke test we just verify that across many seeds we get
        substantial variability in both the *which meals occur* and
        the *food composition*, which would not be the case if both
        decisions were locked to a single shared stream state.
        """
        gen = self._make_generator(taco_db, adherent_params)
        plan = load_day_meal_plan(
            PROJECT_ROOT / "configs" / "meals" / "brazilian_weekday.yaml"
        )
        wake = datetime(2026, 6, 1, 6, 30)
        sleep = datetime(2026, 6, 1, 22, 30)

        meal_counts: set[int] = set()
        carb_totals: set[float] = set()
        for seed in range(40):
            meals = gen.generate_day(
                plan, DayType.WEEKDAY, wake, sleep, default_rng(seed)
            )
            meal_counts.add(len(meals))
            carb_totals.add(round(sum(m.true_carbs_g for m in meals), 1))

        assert len(meal_counts) >= 2, (
            "Expected variability in meal counts across seeds"
        )
        assert len(carb_totals) >= 20, (
            "Expected variability in carb totals across seeds"
        )

    def test_nonadherent_higher_estimation_error(
        self,
        taco_db: TACODatabase,
        adherent_params: ArchetypeParams,
        nonadherent_params: ArchetypeParams,
    ) -> None:
        plan = load_day_meal_plan(PROJECT_ROOT / "configs" / "meals" / "brazilian_weekday.yaml")
        wake = datetime(2026, 6, 1, 6, 30)
        sleep = datetime(2026, 6, 1, 22, 30)

        adherent_errors = []
        nonadherent_errors = []

        for seed in range(30):
            gen_c = self._make_generator(taco_db, adherent_params)
            rng = default_rng(seed)
            meals = gen_c.generate_day(plan, DayType.WEEKDAY, wake, sleep, rng)
            for m in meals:
                if m.true_carbs_g > 0:
                    adherent_errors.append(
                        abs(m.estimated_carbs_g - m.true_carbs_g) / m.true_carbs_g
                    )

            gen_d = self._make_generator(taco_db, nonadherent_params)
            rng = default_rng(seed)
            meals = gen_d.generate_day(plan, DayType.WEEKDAY, wake, sleep, rng)
            for m in meals:
                if m.true_carbs_g > 0:
                    nonadherent_errors.append(
                        abs(m.estimated_carbs_g - m.true_carbs_g) / m.true_carbs_g
                    )

        avg_c = np.mean(adherent_errors)
        avg_d = np.mean(nonadherent_errors)
        assert avg_d > avg_c, (
            f"Nonadherent error ({avg_d:.3f}) should exceed adherent ({avg_c:.3f})"
        )

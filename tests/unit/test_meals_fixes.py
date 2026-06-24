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

"""TDD regression tests for H4 v2 bug-hunt fixes (5 bugs in meals module)."""

from __future__ import annotations

import inspect
from datetime import datetime
from pathlib import Path

import pytest
from numpy.random import default_rng

from simada.core.config import ArchetypeParams, MealConfig
from simada.core.types import DayType, MealType
from simada.meals.generator import MealGenerator
from simada.meals.patterns import DayPatternModifier
from simada.meals.taco import Food, TACODatabase
from simada.meals.templates import MealTemplate, FoodSlot, load_day_meal_plan

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# H4 Bug #1 — Protein-only meal must not fabricate carbs
# ---------------------------------------------------------------------------

class TestProteinOnlyMealNoCarbs:
    """Bug H4#1: generator.py:184-188 — protein-only meal injected phantom carbs.

    When ALL food slots in a template have carbs_per_100g == 0, best_raw_carbs
    stays 0 but the code still calls max(target_min, ...) which clamped 0 up
    to target_min, fabricating carbs for e.g. a pure-churrasco Picanha meal.
    Fix: add guard `if best_raw_carbs <= 0: return None` before the clamp.
    """

    def _make_zero_carb_template(self) -> MealTemplate:
        """A template whose only slot has zero-carb foods (protein only)."""
        return MealTemplate(
            name="proteina_pura",
            meal_type=MealType.ALMOCO,
            slots=(
                FoodSlot(
                    category="proteina",
                    probability=1.0,
                    servings_min=1.0,
                    servings_max=1.0,
                ),
            ),
            total_carb_range=(40.0, 80.0),
            offset_from_wake_minutes=300,
            probability=1.0,
        )

    def test_protein_only_meal_returns_none_when_no_carbs(
        self, taco_db: TACODatabase, adherent_params: ArchetypeParams
    ) -> None:
        """_generate_meal must return None when best_raw_carbs <= 0.

        We monkey-patch _sample_foods to always return zero raw carbs with a
        non-empty foods list, mimicking the bug scenario (e.g. pure Picanha).
        """
        config = MealConfig(taco_path=Path("data/taco/taco_foods.csv"))
        gen = MealGenerator(taco_db, config, adherent_params)

        # Patch _sample_foods to simulate a protein-only result
        def _zero_carb_sample(slots, rng, dessert_boost=0.0):
            # Return a non-empty food list but zero total carbs
            fake_food = list(taco_db.random_food("proteina", default_rng(0)))
            return ([fake_food] if fake_food else [("Picanha", 1.0)], 0.0, 0.0)

        # Use a simpler approach: directly test _generate_meal behavior
        template = self._make_zero_carb_template()

        # We need a mock that returns (foods, 0.0, 0.0) for _sample_foods
        original_sample_foods = gen._sample_foods

        def patched_sample_foods(slots, rng, dessert_boost=0.0):
            foods, _raw_carbs, _gi = original_sample_foods(slots, rng, dessert_boost)
            # Force raw carbs to 0 to simulate protein-only scenario
            return foods, 0.0, 0.0

        gen._sample_foods = patched_sample_foods  # type: ignore[method-assign]

        wake = datetime(2026, 6, 1, 6, 30)
        sleep = datetime(2026, 6, 1, 22, 30)
        rng = default_rng(42)

        result = gen._generate_meal(
            template, wake, sleep, carb_scale=1.0, dessert_boost=0.0, rng=rng
        )

        assert result is None, (
            "A protein-only meal (best_raw_carbs=0) must return None, "
            "not a MealEvent with phantom carbs."
        )

    def test_protein_only_day_no_phantom_carbs(
        self, taco_db: TACODatabase, adherent_params: ArchetypeParams
    ) -> None:
        """Integration: generate_day must not produce zero-carb MealEvents."""
        config = MealConfig(taco_path=Path("data/taco/taco_foods.csv"))
        gen = MealGenerator(taco_db, config, adherent_params)
        template = self._make_zero_carb_template()

        # Patch to force raw_carbs=0
        original_sample_foods = gen._sample_foods

        def patched_sample_foods(slots, rng, dessert_boost=0.0):
            foods, _raw_carbs, _gi = original_sample_foods(slots, rng, dessert_boost)
            return foods, 0.0, 0.0

        gen._sample_foods = patched_sample_foods  # type: ignore[method-assign]

        from simada.meals.templates import DayMealPlan
        plan = DayMealPlan(templates=[template])

        wake = datetime(2026, 6, 1, 6, 30)
        sleep = datetime(2026, 6, 1, 22, 30)
        meals = gen.generate_day(plan, DayType.WEEKDAY, wake, sleep, default_rng(0))

        # With the fix in place, generate_day should produce no meals (all
        # _generate_meal calls return None).
        assert meals == [], (
            "No MealEvents should be emitted when raw carbs are 0 in all slots."
        )


# ---------------------------------------------------------------------------
# H4 Bug #2 — templates.py open() must use encoding="utf-8"
# ---------------------------------------------------------------------------

class TestTemplatesOpenEncoding:
    """Bug H4#2: templates.py:154 — open() without encoding="utf-8".

    We verify via source-code inspection that load_day_meal_plan passes
    encoding="utf-8" to open(). This is a static check — it catches the
    bug without needing a non-UTF-8 locale at test time.
    """

    def test_load_day_meal_plan_uses_utf8_encoding(self) -> None:
        import simada.meals.templates as templates_mod

        source = inspect.getsource(templates_mod.load_day_meal_plan)
        assert 'encoding="utf-8"' in source or "encoding='utf-8'" in source, (
            "load_day_meal_plan must open YAML files with encoding='utf-8' "
            "to handle PT-BR characters in non-UTF-8 locales."
        )


# ---------------------------------------------------------------------------
# H4 Bug #3 — day_of_week field ignored for replacement meals
# ---------------------------------------------------------------------------

class TestDayOfWeekFilterForReplacements:
    """Bug H4#3: templates.py:129 + weekend yaml:74 — day_of_week ignored.

    The churrasco template has day_of_week='sunday' but was being applied on
    Saturdays too, doubling churrasco prevalence. Fix requires:
      1. MealTemplate gains a `day_of_week: str | None` field.
      2. _parse_meal_template reads it from YAML.
      3. generate_day filters replacements by day_of_week before the
         probability roll.
    """

    def test_meal_template_has_day_of_week_field(self) -> None:
        """MealTemplate dataclass must have a day_of_week field."""
        fields = {f.name for f in MealTemplate.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        assert "day_of_week" in fields, (
            "MealTemplate must have a `day_of_week: str | None` field."
        )

    def test_parse_reads_day_of_week_from_yaml(self) -> None:
        """load_day_meal_plan must read day_of_week from the YAML."""
        plan = load_day_meal_plan(
            PROJECT_ROOT / "configs" / "meals" / "brazilian_weekend.yaml"
        )
        churrasco_templates = [
            tmpl for tmpl, _ in plan.replacements if tmpl.name == "churrasco"
        ]
        assert len(churrasco_templates) == 1, "Expected one churrasco replacement template"
        churrasco = churrasco_templates[0]
        assert churrasco.day_of_week is not None, (
            "churrasco template must have day_of_week loaded from YAML"
        )
        assert churrasco.day_of_week.lower() == "sunday", (
            f"Expected day_of_week='sunday', got {churrasco.day_of_week!r}"
        )

    def test_churrasco_never_on_saturday(
        self, taco_db: TACODatabase, adherent_params: ArchetypeParams
    ) -> None:
        """Churrasco (day_of_week=sunday) must not appear on Saturdays."""
        config = MealConfig(taco_path=Path("data/taco/taco_foods.csv"))
        gen = MealGenerator(taco_db, config, adherent_params)
        weekend_plan = load_day_meal_plan(
            PROJECT_ROOT / "configs" / "meals" / "brazilian_weekend.yaml"
        )

        # Saturday = 2026-06-06
        saturday_wake = datetime(2026, 6, 6, 8, 0)   # Saturday
        saturday_sleep = datetime(2026, 6, 6, 23, 0)

        for seed in range(100):
            meals = gen.generate_day(
                weekend_plan, DayType.WEEKEND, saturday_wake, saturday_sleep, default_rng(seed)
            )
            meal_types = [m.meal_type for m in meals]
            assert MealType.CHURRASCO not in meal_types, (
                f"Seed {seed}: churrasco (sunday-only) appeared on a Saturday!"
            )

    def test_churrasco_can_appear_on_sunday(
        self, taco_db: TACODatabase, adherent_params: ArchetypeParams
    ) -> None:
        """Churrasco must still be possible on Sundays (probability=0.40)."""
        config = MealConfig(taco_path=Path("data/taco/taco_foods.csv"))
        gen = MealGenerator(taco_db, config, adherent_params)
        weekend_plan = load_day_meal_plan(
            PROJECT_ROOT / "configs" / "meals" / "brazilian_weekend.yaml"
        )

        # Sunday = 2026-06-07
        sunday_wake = datetime(2026, 6, 7, 8, 0)    # Sunday
        sunday_sleep = datetime(2026, 6, 7, 23, 0)

        churrasco_count = sum(
            1
            for seed in range(200)
            for m in gen.generate_day(
                weekend_plan, DayType.WEEKEND, sunday_wake, sunday_sleep, default_rng(seed)
            )
            if m.meal_type == MealType.CHURRASCO
        )
        assert churrasco_count > 0, (
            "Churrasco should appear at least once across 200 Sunday seeds "
            "(probability=0.40)"
        )


# ---------------------------------------------------------------------------
# H4 Bug #4 — extra_snack_probability is dead code in patterns.py
# ---------------------------------------------------------------------------

class TestExtraSnackProbabilityIsDeadCode:
    """Bug H4#4: patterns.py:62-68 — extra_snack_probability() is dead code.

    The method is never called by generate_day; the actual snacking logic
    lives in behavior/snacking.py. The fix removes the method and adds a
    comment pointing to the real implementation.

    This test checks that the dead method no longer exists on
    DayPatternModifier, and that a comment referencing behavior/snacking
    is present in the source.
    """

    def test_extra_snack_probability_method_removed(self) -> None:
        """DayPatternModifier must NOT have extra_snack_probability method."""
        assert not hasattr(DayPatternModifier, "extra_snack_probability"), (
            "DayPatternModifier.extra_snack_probability() is dead code "
            "and should be removed. Snacking logic lives in behavior/snacking.py."
        )

    def test_patterns_source_references_snacking_module(self) -> None:
        """patterns.py must contain a comment pointing to behavior/snacking.py."""
        import simada.meals.patterns as patterns_mod

        source = inspect.getsource(patterns_mod)
        assert "behavior/snacking" in source or "behavior.snacking" in source, (
            "patterns.py should have a comment directing maintainers to "
            "behavior/snacking.py for the real extra-snack logic."
        )


# ---------------------------------------------------------------------------
# H4 Bug #5 — holidays.yaml never loaded (wrong filename)
# ---------------------------------------------------------------------------

class TestHolidayYamlFilename:
    """Bug H4#5: configs/meals/holidays.yaml never loaded because builder.py
    looks for brazilian_holiday.yaml. Fix: create brazilian_holiday.yaml with
    the same weekday/weekend YAML schema (not the custom holiday-event schema).
    Original holidays.yaml is preserved.
    """

    def test_brazilian_holiday_yaml_exists(self) -> None:
        """configs/meals/brazilian_holiday.yaml must exist."""
        path = PROJECT_ROOT / "configs" / "meals" / "brazilian_holiday.yaml"
        assert path.exists(), (
            "configs/meals/brazilian_holiday.yaml does not exist. "
            "builder.py looks for this file but only holidays.yaml was present."
        )

    def test_original_holidays_yaml_preserved(self) -> None:
        """configs/meals/holidays.yaml must still exist (not renamed)."""
        path = PROJECT_ROOT / "configs" / "meals" / "holidays.yaml"
        assert path.exists(), (
            "configs/meals/holidays.yaml was removed; it must be preserved."
        )

    def test_brazilian_holiday_yaml_loadable(self) -> None:
        """brazilian_holiday.yaml must be loadable by load_day_meal_plan."""
        path = PROJECT_ROOT / "configs" / "meals" / "brazilian_holiday.yaml"
        plan = load_day_meal_plan(path)
        assert len(plan.templates) > 0, (
            "brazilian_holiday.yaml loaded but produced no meal templates."
        )

    def test_holiday_plan_has_more_carbs_than_weekday(
        self, taco_db: TACODatabase, adherent_params: ArchetypeParams
    ) -> None:
        """Holiday meals should have higher carb ranges than weekday."""
        weekday_plan = load_day_meal_plan(
            PROJECT_ROOT / "configs" / "meals" / "brazilian_weekday.yaml"
        )
        holiday_plan = load_day_meal_plan(
            PROJECT_ROOT / "configs" / "meals" / "brazilian_holiday.yaml"
        )

        # Compare the sum of max carb ranges for main meals
        weekday_max = sum(
            t.total_carb_range[1]
            for t in weekday_plan.templates
            if t.meal_type in (MealType.CAFE_DA_MANHA, MealType.ALMOCO, MealType.JANTAR)
        )
        holiday_max = sum(
            t.total_carb_range[1]
            for t in holiday_plan.templates
            if t.meal_type in (MealType.CAFE_DA_MANHA, MealType.ALMOCO, MealType.JANTAR)
        )

        assert holiday_max > weekday_max, (
            f"Holiday meal carb ceiling ({holiday_max}g) should exceed "
            f"weekday ({weekday_max}g) — holidays are more indulgent."
        )

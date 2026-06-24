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

"""Tests for the TACO food database loader."""

from __future__ import annotations

import pytest
from numpy.random import default_rng

from simada.meals.taco import TACODatabase


class TestTACODatabase:
    """Tests for TACODatabase."""

    def test_loads_all_foods(self, taco_db: TACODatabase) -> None:
        assert len(taco_db) >= 100

    def test_has_expected_categories(self, taco_db: TACODatabase) -> None:
        expected = {"cereal", "fruta", "proteina", "leguminosa", "panificado", "bebida"}
        assert expected.issubset(set(taco_db.categories))

    def test_by_category_returns_foods(self, taco_db: TACODatabase) -> None:
        cereals = taco_db.by_category("cereal")
        assert len(cereals) > 0
        for food in cereals:
            assert food.categoria == "cereal"

    def test_loads_protein_and_fat_macros(self, taco_db: TACODatabase) -> None:
        foods = list(taco_db.all_foods)
        # the extended table populates protein/fat for most foods
        assert sum(1 for f in foods if f.proteina_per_100g > 0) > 0.5 * len(foods)
        assert sum(1 for f in foods if f.gordura_per_100g > 0) > 0.3 * len(foods)

    def test_energy_kcal_uses_all_macros(self, taco_db: TACODatabase) -> None:
        chicken = next(f for f in taco_db.all_foods if f.nome_pt == "Frango grelhado")
        assert chicken.proteina_per_100g > 25.0  # lean protein
        avail = max(0.0, chicken.carbs_per_100g - chicken.fibra_per_100g)
        expected = (4 * avail + 2 * chicken.fibra_per_100g
                    + 4 * chicken.proteina_per_100g + 9 * chicken.gordura_per_100g)
        assert chicken.energy_kcal_per_100g == expected
        assert chicken.energy_kcal_per_100g > 100.0

    def test_fiber_counted_at_2_kcal_not_4(self, taco_db: TACODatabase) -> None:
        # high-fiber legume: fiber must not be counted as 4 kcal/g available carb
        bean = next(f for f in taco_db.all_foods if f.nome_pt == "Feijao carioca cozido")
        naive = (4 * bean.carbs_per_100g + 4 * bean.proteina_per_100g
                 + 9 * bean.gordura_per_100g)
        assert bean.energy_kcal_per_100g < naive  # fiber discount applied

    def test_composition_invariants(self, taco_db: TACODatabase) -> None:
        # fibra is a sub-fraction of carbs; macros cannot exceed 100 g/100 g
        for f in taco_db.all_foods:
            assert f.fibra_per_100g <= f.carbs_per_100g + 1e-6, f.nome_pt
            total = f.carbs_per_100g + f.proteina_per_100g + f.gordura_per_100g
            assert total <= 100.0 + 1e-6, (f.nome_pt, total)

    def test_by_category_unknown_raises(self, taco_db: TACODatabase) -> None:
        with pytest.raises(KeyError, match="not found"):
            taco_db.by_category("nonexistent_category")

    def test_by_name_finds_food(self, taco_db: TACODatabase) -> None:
        rice = taco_db.by_name("Arroz branco cozido")
        assert rice.nome_en == "White rice cooked"
        assert rice.carbs_per_100g == pytest.approx(28.1)

    def test_by_name_unknown_raises(self, taco_db: TACODatabase) -> None:
        with pytest.raises(KeyError, match="not found"):
            taco_db.by_name("Comida Inventada")

    def test_carbs_for_serving(self, taco_db: TACODatabase) -> None:
        rice = taco_db.by_name("Arroz branco cozido")
        # 150g serving * 28.1g/100g = 42.15g
        assert rice.carbs_for_serving(1.0) == pytest.approx(42.15, abs=0.1)
        # Double serving
        assert rice.carbs_for_serving(2.0) == pytest.approx(84.3, abs=0.1)

    def test_random_food_returns_from_category(self, taco_db: TACODatabase) -> None:
        rng = default_rng(42)
        for _ in range(20):
            food = taco_db.random_food("fruta", rng)
            assert food.categoria == "fruta"

    def test_random_food_preferred_favors_preferred(self, taco_db: TACODatabase) -> None:
        rng = default_rng(42)
        preferred = ["Banana prata"]
        counts: dict[str, int] = {}
        n = 500
        for _ in range(n):
            food = taco_db.random_food_preferred("fruta", preferred, rng, prefer_weight=10.0)
            counts[food.nome_pt] = counts.get(food.nome_pt, 0) + 1

        # Banana prata should appear much more often than any single other fruit
        assert counts.get("Banana prata", 0) > n * 0.3

    def test_has_category(self, taco_db: TACODatabase) -> None:
        assert taco_db.has_category("cereal") is True
        assert taco_db.has_category("nonexistent") is False

    def test_all_foods_have_valid_data(self, taco_db: TACODatabase) -> None:
        for food in taco_db.all_foods:
            assert food.carbs_per_100g >= 0
            assert food.fibra_per_100g >= 0
            assert food.indice_glicemico >= 0
            assert food.porcao_tipica_g > 0
            assert food.nome_pt
            assert food.nome_en
            assert food.categoria

    def test_protein_foods_have_low_carbs(self, taco_db: TACODatabase) -> None:
        proteins = taco_db.by_category("proteina")
        for food in proteins:
            assert food.carbs_per_100g <= 5.0, (
                f"{food.nome_pt} has {food.carbs_per_100g}g carbs/100g, "
                "expected <= 5 for protein category"
            )

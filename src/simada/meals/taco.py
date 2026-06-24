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

"""TACO (Tabela Brasileira de Composicao de Alimentos) food database loader.

Provides structured access to Brazilian food composition data for realistic
meal generation in T1D simulations.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from numpy.random import Generator


@dataclass(frozen=True)
class Food:
    """A single food item from the TACO table."""

    nome_pt: str
    nome_en: str
    categoria: str
    carbs_per_100g: float
    fibra_per_100g: float
    indice_glicemico: float
    porcao_tipica_g: float
    # Macronutrients beyond carbohydrate (default 0.0 for tables that predate the
    # extension); needed for energy-balance and satiety calculations.
    proteina_per_100g: float = 0.0
    gordura_per_100g: float = 0.0

    @property
    def energy_kcal_per_100g(self) -> float:
        """Atwater energy estimate.

        TACO/USDA/MEXT report carbohydrate "by difference", which INCLUDES
        dietary fiber, and fiber yields ~2 kcal/g (fermentation) rather than 4.
        So available carbohydrate (carbs - fiber) is counted at 4 kcal/g and
        fiber at 2 kcal/g, avoiding a systematic overstatement for high-fiber
        foods (e.g. legumes). Protein 4, fat 9 kcal/g.
        """
        available_carb = max(0.0, self.carbs_per_100g - self.fibra_per_100g)
        return (4.0 * available_carb + 2.0 * self.fibra_per_100g
                + 4.0 * self.proteina_per_100g + 9.0 * self.gordura_per_100g)

    def carbs_for_serving(self, servings: float = 1.0) -> float:
        """Compute total carbs for a given number of typical servings."""
        return self.carbs_per_100g * self.porcao_tipica_g * servings / 100.0

    def weighted_gi(self, servings: float = 1.0) -> float:
        """GI contribution weighted by carb content."""
        carbs = self.carbs_for_serving(servings)
        return self.indice_glicemico * carbs if carbs > 0 else 0.0


class TACODatabase:
    """Brazilian Food Composition Table (TACO) database.

    Loads a curated subset of ~100 foods commonly consumed in Brazil,
    organized by category for meal composition.

    Usage::

        db = TACODatabase(Path("data/taco/taco_foods.csv"))
        cereals = db.by_category("cereal")
        food = db.random_food("fruta", rng)
    """

    def __init__(self, csv_path: Path) -> None:
        self._foods: list[Food] = []
        self._by_category: dict[str, list[Food]] = {}
        self._by_name: dict[str, Food] = {}
        self._load(csv_path)

    def _load(self, csv_path: Path) -> None:
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                food = Food(
                    nome_pt=row["nome_pt"].strip(),
                    nome_en=row["nome_en"].strip(),
                    categoria=row["categoria"].strip(),
                    carbs_per_100g=float(row["carbs_per_100g"]),
                    fibra_per_100g=float(row["fibra_per_100g"]),
                    indice_glicemico=float(row["indice_glicemico"]),
                    porcao_tipica_g=float(row["porcao_tipica_g"]),
                    proteina_per_100g=float(row.get("proteina_per_100g") or 0.0),
                    gordura_per_100g=float(row.get("gordura_per_100g") or 0.0),
                )
                self._foods.append(food)
                self._by_category.setdefault(food.categoria, []).append(food)
                self._by_name[food.nome_pt] = food

    @property
    def categories(self) -> list[str]:
        """All available food categories."""
        return sorted(self._by_category.keys())

    @property
    def all_foods(self) -> list[Food]:
        """All foods in the database."""
        return list(self._foods)

    def __len__(self) -> int:
        return len(self._foods)

    def by_category(self, category: str) -> list[Food]:
        """Return all foods in a given category.

        Raises KeyError if category does not exist.
        """
        if category not in self._by_category:
            msg = (
                f"Category '{category}' not found. "
                f"Available: {', '.join(self.categories)}"
            )
            raise KeyError(msg)
        return list(self._by_category[category])

    def by_name(self, nome_pt: str) -> Food:
        """Look up a food by its Portuguese name.

        Raises KeyError if not found.
        """
        if nome_pt not in self._by_name:
            msg = f"Food '{nome_pt}' not found in database"
            raise KeyError(msg)
        return self._by_name[nome_pt]

    def random_food(self, category: str, rng: Generator) -> Food:
        """Sample a random food from a category."""
        foods = self.by_category(category)
        idx = int(rng.integers(0, len(foods)))
        return foods[idx]

    def random_food_preferred(
        self,
        category: str,
        preferred: list[str],
        rng: Generator,
        prefer_weight: float = 3.0,
    ) -> Food:
        """Sample a food from a category, preferring certain items.

        Items whose nome_pt appears in *preferred* receive *prefer_weight*
        times the selection probability of non-preferred items.
        """
        foods = self.by_category(category)
        weights = []
        for food in foods:
            if food.nome_pt in preferred:
                weights.append(prefer_weight)
            else:
                weights.append(1.0)

        total = sum(weights)
        probs = [w / total for w in weights]
        idx = int(rng.choice(len(foods), p=probs))
        return foods[idx]

    def has_category(self, category: str) -> bool:
        """Check if a category exists in the database."""
        return category in self._by_category

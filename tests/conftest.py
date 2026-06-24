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

"""Shared test fixtures.

Loads archetype parameters from the YAML configs so tests always match
the authoritative config files (single source of truth).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from numpy.random import default_rng

from simada.core.config import ArchetypeParams, load_archetype_params
from simada.core.random import RNGManager
from simada.meals.taco import TACODatabase

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def taco_db() -> TACODatabase:
    """Load the TACO database from the project data directory."""
    return TACODatabase(PROJECT_ROOT / "data" / "taco" / "taco_foods.csv")


@pytest.fixture
def rng_manager() -> RNGManager:
    """Create a seeded RNG manager for reproducible tests."""
    return RNGManager(master_seed=42)


@pytest.fixture
def rng():
    """Simple seeded RNG for tests that don't need the full manager."""
    return default_rng(42)


@pytest.fixture
def adherent_params() -> ArchetypeParams:
    """Adherent archetype parameters — loaded from YAML config."""
    return load_archetype_params(
        PROJECT_ROOT / "configs" / "archetypes" / "adherent.yaml"
    )


@pytest.fixture
def nonadherent_params() -> ArchetypeParams:
    """Nonadherent archetype parameters — loaded from YAML config."""
    return load_archetype_params(
        PROJECT_ROOT / "configs" / "archetypes" / "nonadherent.yaml"
    )


@pytest.fixture
def moderate_params() -> ArchetypeParams:
    """Moderate archetype parameters — loaded from YAML config."""
    return load_archetype_params(
        PROJECT_ROOT / "configs" / "archetypes" / "moderate.yaml"
    )

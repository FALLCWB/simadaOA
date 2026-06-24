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

"""Tests for Pydantic configuration models and YAML loaders."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from simada.core.config import (
    ArchetypeParams,
    CohortConfig,
    SimulationConfig,
    load_archetype_params,
    load_config,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARCHETYPES_DIR = PROJECT_ROOT / "configs" / "archetypes"
SCENARIOS_DIR = PROJECT_ROOT / "configs" / "scenarios"


# ---------------------------------------------------------------------------
# Minimal valid kwargs for ArchetypeParams (all required fields)
# ---------------------------------------------------------------------------

_MINIMAL_ARCHETYPE_KWARGS: dict = {
    "carb_estimation_error_mean": 0.0,
    "carb_estimation_error_std": 0.10,
    "meal_time_variance_minutes": 15.0,
    "bolus_skip_probability": 0.05,
    "bolus_timing_pre_pct": 0.50,
    "bolus_timing_late_half_pct": 0.30,
    "bolus_timing_forgot_pct": 0.20,
    "exercise_probability_weekday": 0.50,
    "exercise_probability_weekend": 0.60,
    "snack_probability": 0.20,
    "monitor_frequency_per_day": 6,
    "wake_time_weekday_hour": 6.5,
    "wake_time_weekday_std_minutes": 15.0,
    "wake_time_weekend_hour": 8.0,
    "wake_time_weekend_std_minutes": 30.0,
    "sleep_time_weekday_hour": 22.5,
    "sleep_time_weekday_std_minutes": 20.0,
    "sleep_time_weekend_hour": 23.5,
    "sleep_time_weekend_std_minutes": 30.0,
}


class TestArchetypeParams:
    """Tests for ArchetypeParams validation."""

    def test_bolus_timing_must_sum_to_one(self) -> None:
        """Bolus timing percentages that don't sum to 1.0 raise ValueError."""
        kwargs = {
            **_MINIMAL_ARCHETYPE_KWARGS,
            "bolus_timing_pre_pct": 0.50,
            "bolus_timing_late_half_pct": 0.30,
            "bolus_timing_forgot_pct": 0.10,  # total = 0.9
        }
        with pytest.raises(ValidationError, match="bolus_timing"):
            ArchetypeParams(**kwargs)

    def test_bolus_timing_valid(self) -> None:
        """Bolus timing percentages that sum to 1.0 are accepted."""
        kwargs = {
            **_MINIMAL_ARCHETYPE_KWARGS,
            "bolus_timing_pre_pct": 0.50,
            "bolus_timing_late_half_pct": 0.30,
            "bolus_timing_forgot_pct": 0.20,  # total = 1.0
        }
        params = ArchetypeParams(**kwargs)
        assert params.bolus_timing_pre_pct == 0.50
        assert params.bolus_timing_late_half_pct == 0.30
        assert params.bolus_timing_forgot_pct == 0.20

    def test_load_archetype_params(self) -> None:
        """Loading adherent.yaml produces a valid ArchetypeParams."""
        params = load_archetype_params(ARCHETYPES_DIR / "adherent.yaml")
        assert isinstance(params, ArchetypeParams)
        assert 0.0 <= params.bolus_skip_probability <= 1.0
        assert params.monitor_frequency_per_day > 0

    @pytest.mark.parametrize("archetype_file", [
        "adherent.yaml", "moderate.yaml", "nonadherent.yaml",
    ])
    def test_archetype_params_have_safety_fields(self, archetype_file: str) -> None:
        """All archetype YAMLs must load safety guard fields correctly."""
        params = load_archetype_params(ARCHETYPES_DIR / archetype_file)
        # Fields must exist and be positive
        assert params.max_single_bolus_u > 0, (
            f"{archetype_file}: max_single_bolus_u must be positive"
        )
        assert params.hypo_max_episode_cho_g > 0, (
            f"{archetype_file}: hypo_max_episode_cho_g must be positive"
        )
        assert params.iob_hard_limit_u >= 0, (
            f"{archetype_file}: iob_hard_limit_u must be non-negative"
        )


class TestCohortConfig:
    """Tests for CohortConfig distribution validation."""

    def test_archetype_distribution_must_sum_to_one(self) -> None:
        """Archetype distribution that doesn't sum to 1.0 raises ValueError."""
        with pytest.raises(ValidationError, match="archetype_distribution"):
            CohortConfig(
                archetype_distribution={
                    "adherent": 0.50,
                    "moderate": 0.30,
                    "nonadherent": 0.10,  # total = 0.9
                },
            )

    def test_archetype_distribution_valid(self) -> None:
        """Archetype distribution that sums to 1.0 is accepted."""
        config = CohortConfig(
            archetype_distribution={
                "adherent": 0.50,
                "moderate": 0.30,
                "nonadherent": 0.20,  # total = 1.0
            },
        )
        assert sum(config.archetype_distribution.values()) == pytest.approx(1.0)


class TestSimulationConfig:
    """Tests for SimulationConfig and YAML loading."""

    def test_load_config(self) -> None:
        """Loading 7day_cohort.yaml produces a valid SimulationConfig."""
        config = load_config(SCENARIOS_DIR / "7day_cohort.yaml")
        assert isinstance(config, SimulationConfig)
        assert config.seed == 42
        assert config.scenario.duration_days == 7
        assert config.scenario.cohort is not None
        assert config.scenario.cohort.size == 30

    def test_extra_fields_ignored(self, tmp_path: Path) -> None:
        """YAML with extra 'perturbations' key doesn't crash (extra='ignore')."""
        yaml_content = """\
seed: 99
scenario:
  name: "test"
  duration_days: 3
  start_date: "2026-01-01"
perturbations:
  enabled: false
  schedule: []
unknown_future_key: "should be ignored"
"""
        yaml_file = tmp_path / "with_extras.yaml"
        yaml_file.write_text(yaml_content)

        config = load_config(yaml_file)
        assert config.seed == 99
        assert config.scenario.duration_days == 3

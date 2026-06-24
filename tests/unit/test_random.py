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

"""Tests for hierarchical seeded RNG manager."""

from __future__ import annotations

import numpy as np
import pytest
from pytest import approx

from simada.core.random import RNGManager


class TestRNGManager:
    """Tests for RNGManager."""

    def test_patient_independence(self) -> None:
        """Changing max_patients does not affect a given patient's random stream.

        Patient 5 should produce the exact same sequence regardless of
        whether the manager was created with max_patients=10 or max_patients=2000.
        """
        rng_small = RNGManager(master_seed=42, max_patients=10)
        rng_large = RNGManager(master_seed=42, max_patients=2000)

        p5_small = rng_small.patient_rng(5)
        p5_large = rng_large.patient_rng(5)

        # Compare 20 draws from the meals stream
        seq_small = [p5_small.meals.random() for _ in range(20)]
        seq_large = [p5_large.meals.random() for _ in range(20)]

        np.testing.assert_array_equal(seq_small, seq_large)

    def test_component_independence(self) -> None:
        """Different component streams for the same patient produce different sequences."""
        rng = RNGManager(master_seed=42)
        p0 = rng.patient_rng(0)

        meals_seq = [p0.meals.random() for _ in range(20)]
        insulin_seq = [p0.insulin.random() for _ in range(20)]

        # The probability that two independent streams produce 20 identical
        # floats is essentially zero
        assert meals_seq != insulin_seq

    def test_reproducibility(self) -> None:
        """Same seed produces identical streams across two separate RNGManagers."""
        rng1 = RNGManager(master_seed=777)
        rng2 = RNGManager(master_seed=777)

        for patient_idx in [0, 5, 99]:
            p1 = rng1.patient_rng(patient_idx)
            p2 = rng2.patient_rng(patient_idx)

            for component in ("meals", "insulin", "exercise", "behavior", "stress", "sensor", "snacks"):
                gen1 = getattr(p1, component)
                gen2 = getattr(p2, component)
                seq1 = [gen1.random() for _ in range(10)]
                seq2 = [gen2.random() for _ in range(10)]
                np.testing.assert_array_equal(
                    seq1, seq2,
                    err_msg=f"Mismatch for patient {patient_idx}, component {component}",
                )

    def test_general_rng_is_cached_property(self) -> None:
        """general_rng is a cached_property: both accesses return the same Generator object.

        The cached Generator is a single evolving stream (not a fresh generator
        on each access). Two accesses must return the exact same object instance
        so callers share one sequence rather than restarting it.
        """
        rng = RNGManager(master_seed=42)
        g1 = rng.general_rng
        g2 = rng.general_rng

        # Must be the exact same object (cached)
        assert g1 is g2, "general_rng must return the same Generator instance on repeated access"

        # Drawing from it evolves the state: consecutive draws differ
        v1 = g1.random()
        v2 = g2.random()  # same object, so this advances the same stream
        assert v1 != v2, "Consecutive draws from the shared generator must differ"

    def test_patient_index_out_of_range(self) -> None:
        """Requesting a patient index >= max_patients raises IndexError."""
        rng = RNGManager(master_seed=42, max_patients=10)

        with pytest.raises(IndexError, match="out of range"):
            rng.patient_rng(10)

        with pytest.raises(IndexError, match="out of range"):
            rng.patient_rng(100)

        with pytest.raises(IndexError, match="out of range"):
            rng.patient_rng(-1)

    def test_master_seed_property(self) -> None:
        """master_seed property returns the entropy used for initialization."""
        rng = RNGManager(master_seed=12345)
        assert rng.master_seed == 12345

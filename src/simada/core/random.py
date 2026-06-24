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

"""Hierarchical seeded RNG for reproducible simulations.

Architecture: master_seed -> per-patient seeds -> per-component seeds.

This design ensures that adding patient #31 to a cohort does not change
the random sequences of patients #1-30, and that each component (meals,
insulin, exercise, etc.) has an independent stream within each patient.
"""

from __future__ import annotations

from functools import cached_property

from numpy.random import Generator, SeedSequence, default_rng

# Number of component streams per patient.  If you add a new component,
# increment this and append the new stream at the end to preserve
# backward-compatibility of existing streams.
_N_COMPONENTS = 7

# Component stream indices
_IDX_MEALS = 0
_IDX_INSULIN = 1
_IDX_EXERCISE = 2
_IDX_BEHAVIOR = 3
_IDX_STRESS = 4
_IDX_SENSOR = 5
_IDX_SNACKS = 6  # separate from meals to avoid RNG cross-contamination (H3 #4)


class PatientRNG:
    """Per-patient RNG with named sub-streams for each simulation component.

    Each component gets its own independent Generator so that, for example,
    changing the meal generation logic does not affect the insulin delivery
    random draws for the same patient and seed.
    """

    __slots__ = ("meals", "insulin", "exercise", "behavior", "stress", "sensor", "snacks")

    def __init__(self, children: list[SeedSequence] | SeedSequence) -> None:
        # Accept pre-spawned children (preferred, deterministic across repeated
        # patient_rng() calls) or a SeedSequence for back-compat (legacy callers
        # that pass a SeedSequence will get fresh-spawn semantics).
        if isinstance(children, SeedSequence):
            children = list(children.spawn(_N_COMPONENTS))
        self.meals: Generator = default_rng(children[_IDX_MEALS])
        self.insulin: Generator = default_rng(children[_IDX_INSULIN])
        self.exercise: Generator = default_rng(children[_IDX_EXERCISE])
        self.behavior: Generator = default_rng(children[_IDX_BEHAVIOR])
        self.stress: Generator = default_rng(children[_IDX_STRESS])
        self.sensor: Generator = default_rng(children[_IDX_SENSOR])
        self.snacks: Generator = default_rng(children[_IDX_SNACKS])


class RNGManager:
    """Master RNG manager that spawns stable per-patient sub-streams.

    Usage::

        rng = RNGManager(seed=42)
        patient_0_rng = rng.patient_rng(0)
        patient_1_rng = rng.patient_rng(1)

        # Each patient has independent component streams:
        meal_carbs = patient_0_rng.meals.normal(50, 10)
        bolus_skip = patient_0_rng.insulin.random() < skip_prob
    """

    def __init__(self, master_seed: int, max_patients: int = 1000) -> None:
        self._master_ss = SeedSequence(master_seed)
        # Spawn max_patients + 1: indices 0..max_patients-1 for patients,
        # index max_patients for the general-purpose stream.
        all_seeds = self._master_ss.spawn(max_patients + 1)
        self._patient_seeds = all_seeds[:max_patients]
        self._general_seed = all_seeds[max_patients]
        self._max_patients = max_patients
        # Pre-spawn each patient's component children ONCE. SeedSequence.spawn()
        # mutates internal n_children_spawned, so repeated calls produce different
        # children. Caching here guarantees patient_rng(i) and sensor_seed(i)
        # return the SAME generators/seed across repeated calls.
        self._patient_children: list[list[SeedSequence]] = [
            list(ss.spawn(_N_COMPONENTS)) for ss in self._patient_seeds
        ]

    @property
    def master_seed(self) -> int:
        """Return the entropy used to initialize this manager."""
        return int(self._master_ss.entropy)  # type: ignore[arg-type]

    def patient_rng(self, patient_index: int) -> PatientRNG:
        """Derive a stable sub-stream for a specific patient.

        The patient_index must be in [0, max_patients). The returned
        PatientRNG is deterministic for a given (master_seed, patient_index)
        pair regardless of how many other patients exist.
        """
        if patient_index < 0 or patient_index >= self._max_patients:
            msg = (
                f"patient_index {patient_index} out of range "
                f"[0, {self._max_patients})"
            )
            raise IndexError(msg)
        return PatientRNG(self._patient_children[patient_index])

    @cached_property
    def general_rng(self) -> Generator:
        """Return a general-purpose RNG for non-patient-specific randomness.

        Uses a pre-allocated seed that does not interfere with patient streams.
        The generator is cached so all callers share a single evolving stream;
        calling this multiple times returns the same Generator instance.
        """
        return default_rng(self._general_seed)

    def sensor_seed(self, patient_idx: int) -> int:
        """Derive a reproducible integer seed for a patient's CGM sensor.

        Uses the patient's pre-allocated SeedSequence to spawn a child for
        the sensor component (same as PatientRNG._IDX_SENSOR), then extracts
        an integer suitable for passing to simglucose's CGMSensor.withName().

        This replaces the aliasing pattern ``config.seed + patient_index``
        (seed=100 + idx=42 == seed=142 + idx=0) with a proper hierarchical
        derivation that is guaranteed distinct for every (master_seed, patient_idx)
        pair within the valid range.
        """
        if patient_idx < 0 or patient_idx >= self._max_patients:
            msg = (
                f"patient_idx {patient_idx} out of range "
                f"[0, {self._max_patients})"
            )
            raise IndexError(msg)
        sensor_child = self._patient_children[patient_idx][_IDX_SENSOR]
        # Extract a 32-bit int from the SeedSequence state for simglucose compatibility
        return int(sensor_child.generate_state(1, dtype="uint32")[0])

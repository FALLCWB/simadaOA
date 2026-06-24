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

"""Carbohydrate estimation error model.

Models the discrepancy between actual carb content and what the patient
estimates. This is a core challenge for Safe RL: the controller only
sees estimated_carbs, not true_carbs.

Error model per archetype:
    Adherent:  N(0, 0.10 * true) — small random error, no bias
    Moderate:     N(-0.05 * true, 0.12 * true) — slight underestimation
    Nonadherent: N(-0.15 * true, 0.20 * true) — systematic undercount

The underestimation bias for non-compliant patients reflects that careless
patients tend to miss counting sauces, dressings, drinks, and sides.
"""

from __future__ import annotations

from numpy.random import Generator

from simada.core.config import ArchetypeParams


class CarbEstimationModel:
    """Estimates carbohydrates with archetype-dependent error.

    The model applies a normally-distributed relative error to the true
    carb value. The error has both a mean component (bias) and a
    standard deviation component (random noise).
    """

    def __init__(self, params: ArchetypeParams) -> None:
        self._mean = params.carb_estimation_error_mean
        self._std = params.carb_estimation_error_std

    def estimate(self, true_carbs_g: float, rng: Generator) -> float:
        """Return the patient's estimated carbs given the true value.

        The estimate is clamped to a physiologically plausible range:

        * Lower bound: ``true_carbs_g * 0.1`` (10 % of true). Even careless
          patients who underestimate sauces and sides do not estimate
          essentially zero carbs for a real meal; ``max(0, ...)`` alone
          allowed the normal tail to drag the estimate to 0 g for
          high-SD archetypes (BUG-FIX H3 #2).
        * Upper bound: ``true_carbs_g * 2.0`` (200 % of true). Symmetric
          guard against the upper tail producing implausible over-counts.

        These bounds preserve the qualitative bias direction while
        preventing pathological tail samples from breaking the
        downstream bolus calculator.
        """
        if true_carbs_g <= 0:
            return 0.0
        error = rng.normal(self._mean * true_carbs_g, self._std * true_carbs_g)
        estimate = true_carbs_g + error
        lower = true_carbs_g * 0.1
        upper = true_carbs_g * 2.0
        return max(lower, min(upper, estimate))

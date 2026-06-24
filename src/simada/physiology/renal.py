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

"""Renal glucose excretion correction.

simglucose's UVA/Padova model has a renal excretion term (ke1=0.0005)
that underestimates real glucosuria by ~10x at extreme BG values
(confirmed by nephrologist analysis: simglucose excretes 61 mg/min at
BG=800 vs real kidney 625 mg/min).

This module provides a post-hoc correction that adds the missing
excretion to the observed BG, producing physiologically plausible
values without modifying simglucose internals.

References:
    - Gerich JE. Diabetologia 2010;53:2135-2142
    - DeFronzo RA et al. Diabetes Care 2012;35(12):2726-2737
    - Expert nephrologist analysis: ke1 produces 10x less excretion
      than GFR-based model above 400 mg/dL
"""

from __future__ import annotations

import math

# Constants
RENAL_THRESHOLD_MG_DL = 180.0  # glucosuria onset
RENAL_SATURATION_MG_DL = 350.0  # full excretion capacity
MAX_EXCRETION_MG_DL_PER_3MIN = 2.5  # ~0.83/min, matching real kidney deficit
BG_HARD_CEILING = 600.0  # absolute safety net (model not validated above this)

# BUG #3 fix: width of the smooth saturation around the ceiling.
# A small width (e.g., 6 mg/dL) keeps the function close to the identity
# well below the ceiling while still being C-infinity smooth across the
# transition. ODE solvers do not see a discontinuity, only a fast but
# differentiable bend.
_CEILING_SOFTMIN_WIDTH = 6.0


def _smooth_ceiling(bg_mg_dl: float, ceiling: float = BG_HARD_CEILING) -> float:
    """Differentiable saturation toward ``ceiling`` from below.

    Implemented as a soft-min via log-sum-exp:

        smooth_min(bg, ceiling) = -w * log(exp(-bg / w) + exp(-ceiling / w))

    Properties (width = 6 mg/dL):
        bg << ceiling  -> result ~= bg            (essentially identity)
        bg == ceiling  -> result ~= ceiling - w*ln(2)  (~595.8 mg/dL)
        bg >> ceiling  -> result ~= ceiling       (asymptotic cap)

    The function is C-infinity, so the ODE solver sees no kink. It also
    preserves monotonicity, which is what we want for a physiological
    saturation. BUG #3 fix: this replaces the previous hard
    ``min(corrected, BG_HARD_CEILING)`` clamp, which produced a
    non-differentiable corner that broke adaptive-step ODE integrators
    when the simglucose state hovered near the ceiling.
    """
    w = _CEILING_SOFTMIN_WIDTH
    # Numerically stable log-sum-exp -- subtract the max of (-bg/w, -ceiling/w)
    a = -bg_mg_dl / w
    b = -ceiling / w
    m = max(a, b)
    return -w * (m + math.log(math.exp(a - m) + math.exp(b - m)))


def apply_renal_correction(bg_mg_dl: float, sample_time_min: float = 3.0) -> float:
    """Apply renal excretion correction to observed BG.

    Below 180: no correction (simglucose is adequate)
    180-350: linear ramp of additional excretion
    350+: full additional excretion
    Above 600: smooth saturation (soft-min) toward the ceiling -- replaces
        the previous hard clamp so the function remains differentiable for
        ODE solvers (BUG #3 fix).

    Returns corrected BG value.
    """
    if sample_time_min <= 0:
        raise ValueError(
            f"sample_time_min must be positive (physics undefined for "
            f"sample_time_min={sample_time_min})"
        )

    if bg_mg_dl <= RENAL_THRESHOLD_MG_DL:
        return bg_mg_dl

    # Additional excretion not captured by simglucose's weak ke1
    fraction = min(
        1.0,
        (bg_mg_dl - RENAL_THRESHOLD_MG_DL)
        / (RENAL_SATURATION_MG_DL - RENAL_THRESHOLD_MG_DL),
    )
    excretion = fraction * MAX_EXCRETION_MG_DL_PER_3MIN * (sample_time_min / 3.0)

    corrected = bg_mg_dl - excretion

    # Smooth saturation toward the ceiling. The function is essentially
    # the identity for corrected << ceiling, asymptotes to the ceiling
    # for corrected >> ceiling, and is C-infinity across the transition.
    corrected = _smooth_ceiling(corrected, BG_HARD_CEILING)

    return max(corrected, RENAL_THRESHOLD_MG_DL)  # never correct below threshold

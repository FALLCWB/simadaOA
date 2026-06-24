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

"""Tests for ResilientT1DPatient.

The integrator restart is a NUMERICAL change only: re-seeding dopri5 from the
exact current state must not alter the physiology, so over a smooth interval the
resilient patient must track the stock simglucose patient closely.
"""

from __future__ import annotations

import numpy as np
from simglucose.patient.t1dpatient import Action, T1DPatient

from simada.patient.resilient import ResilientT1DPatient


def test_is_t1dpatient_subclass_and_constructs():
    p = ResilientT1DPatient.withName("adult#001")
    assert isinstance(p, T1DPatient)
    assert len(p.state) == 13


def test_tracks_stock_patient_over_smooth_interval():
    """Same name + same actions: re-seeding each step must not change the
    physiological trajectory (smooth basal-only interval)."""
    base = T1DPatient.withName("adult#001")
    res = ResilientT1DPatient.withName("adult#001")
    action = Action(CHO=0.0, insulin=0.02)  # constant basal, no meal
    for _ in range(120):  # ~6 h at 3-min steps
        base.step(action)
        res.step(action)
    assert np.allclose(base.state, res.state, rtol=1e-6, atol=1e-6)


def test_resilient_never_blows_up_on_stiff_patient():
    """Robustness claim: a stiff physiology that collapses dopri5 (child#008)
    must integrate to completion with a FINITE state under LSODA -- no
    integrator failure, no NaN/inf blow-up. (No-distortion in the normal regime
    is covered by the smooth-interval test above; aggregate agreement with the
    controller in the loop is shown by avaliacao/validate_lsoda.py: adult#001
    over 30 days matches stock dopri5 to 0.000 mg/dL.)
    """
    import numpy as np

    res = ResilientT1DPatient.withName("child#008")
    meal = Action(CHO=70.0, insulin=0.01)
    basal = Action(CHO=0.0, insulin=0.01)
    for i in range(1440):  # ~3 days of fixed inputs (a deliberately hard case)
        res.step(meal if i % 480 in (40, 240, 380) else basal)
        assert np.all(np.isfinite(res.state)), f"non-finite state at step {i}"
    # Finite and not an inf/explosion (Gp far below any numeric blow-up).
    assert np.max(np.abs(res.state)) < 1e5

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

"""Numerically resilient T1D patient for long-horizon simulation.

simglucose's :class:`T1DPatient` builds a single ``scipy.integrate.ode``
(``dopri5``, an explicit Runge-Kutta method) at ``t0`` and integrates the whole
run incrementally on that one solver instance. The UVA/Padova right-hand side
is **discontinuous at every control step** -- meals enter as carbohydrate
impulses and the insulin input changes each sample. Integrating *across* those
discontinuities with one long-lived adaptive solver lets its internal
step-size / error bookkeeping degrade over hundreds of thousands of steps;
eventually a routine meal triggers "dopri5: step size becomes too small" and
the solver fails (and simglucose then re-``raise``s with no active exception,
killing the patient). Over a 1-year run this struck the majority of patients.

The numerical-analysis remedy for a piecewise / discontinuous RHS is well
established: **stop and restart the integrator at the discontinuity** rather
than carrying internal solver state across it (Hairer & Wanner, *Solving
Ordinary Differential Equations*; Shampine; this is what event-aware solvers
such as SUNDIALS/CVODE, Julia's OrdinaryDiffEq and Wolfram NDSolve do by
default). We therefore re-seed the solver from the CURRENT state at the current
time before every control step. This changes NOTHING physiological -- the state
vector is carried forward exactly; only the solver's internal step-size/error
accumulation is reset, so each control interval is integrated as a fresh,
well-posed sub-problem with continuous inputs.

If a single interval is still not advanced (a genuinely stiff interval, the
other classic cause of step-size collapse), the step is retried by integrating
the interval as several smaller sub-steps; per the standard recommendation, a
truly stiff system would instead call for an implicit/stiff method (Radau, BDF,
LSODA), which is the documented next step if sub-stepping proves insufficient.
"""

from __future__ import annotations

from simglucose.patient.t1dpatient import T1DPatient

# LSODA internal step cap per control interval. Generous: a 3-min interval needs
# very few internal steps even when stiff; the cap just bounds pathological cases.
_LSODA_NSTEPS = 5000


class ResilientT1DPatient(T1DPatient):
    """``T1DPatient`` using an implicit/auto-switching (LSODA) integrator.

    Stock simglucose integrates the whole run on one explicit ``dopri5``
    instance. Over a long horizon (e.g. a year) this fails: "dopri5: step size
    becomes too small" is the classic symptom of a *stiff* right-hand side, and
    one physiology (child#008) is stiff enough in the meal+controller regime to
    collapse dopri5 every run. The documented remedy is an implicit /
    stiffness-aware method (scipy: explicit RK for non-stiff problems;
    ``Radau``/``BDF``/``LSODA`` for stiff ones). We switch the integrator to
    **LSODA** (Petzold), which auto-switches between a non-stiff Adams method
    and a stiff BDF method, so it integrates both the easy and the stiff
    patients efficiently and without failure.

    No manual integrator restart is used. (Restarting at discontinuities is the
    right treatment for an *explicit* solver carrying error across a meal
    impulse, but LSODA is a stable long-horizon implicit solver; re-seeding it
    every meal would discard its adapted internal state and force it to
    re-detect stiffness each interval -- which made the stiff patient ~10x
    slower in testing. LSODA handles the piecewise inputs natively.)

    This is a NUMERICAL change only: both methods approximate the same ODE, and
    on a non-stiff patient LSODA and stock dopri5 agree to integrator tolerance
    -- 0.000 mg/dL over 30 days with the controller in the loop
    (avaliacao/validate_lsoda.py) and 1e-6 over a smooth interval (regression
    test). Where they differ (stiff transients) LSODA is the accurate one;
    dopri5 was failing there. Drop-in: ``withName`` constructs ``cls``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_lsoda()

    def reset(self):
        # T1DPatient.reset() rebuilds the ODE solver as dopri5; re-apply LSODA
        # afterwards, otherwise the integrator swap is silently lost on the
        # reset that SimObj.simulate() performs before stepping.
        super().reset()
        self._use_lsoda()

    def _use_lsoda(self):
        # Swap simglucose's dopri5 for LSODA, preserving the current state/time.
        y, t = self._odesolver.y, self._odesolver.t
        self._odesolver.set_integrator("lsoda", nsteps=_LSODA_NSTEPS)
        self._odesolver.set_initial_value(y, t)

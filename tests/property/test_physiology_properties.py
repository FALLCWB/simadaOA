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

"""Property-based invariants for the new physiology/behavior models."""

from __future__ import annotations

from hypothesis import given, settings, strategies as st
from numpy.random import default_rng

from simada.behavior.counterregulation import (
    CounterregulationConfig,
    CounterregulationModel,
)

_bg = st.floats(min_value=10.0, max_value=400.0)
_nonneg = st.floats(min_value=0.0, max_value=200.0)
_gi = st.floats(min_value=0.0, max_value=120.0)


class TestCounterregulationInvariants:
    @given(bg=_bg, dt=st.floats(min_value=0.0, max_value=30.0))
    @settings(max_examples=200, deadline=None)
    def test_release_nonnegative_and_bounded_by_store(self, bg, dt):
        m = CounterregulationModel(CounterregulationConfig(enabled=True))
        before = m.glycogen_g
        grams = m.step(bg=bg, dt_min=dt)
        assert grams >= 0.0
        assert grams <= before + 1e-9         # cannot release more than was stored
        assert m.glycogen_g >= 0.0            # store never goes negative

    @given(seq=st.lists(st.tuples(_bg, st.floats(min_value=0.0, max_value=10.0)),
                        min_size=1, max_size=50))
    @settings(max_examples=100, deadline=None)
    def test_store_stays_within_capacity(self, seq):
        cfg = CounterregulationConfig(enabled=True)
        m = CounterregulationModel(cfg)
        for bg, dt in seq:
            m.step(bg=bg, dt_min=dt)
            assert 0.0 <= m.glycogen_g <= cfg.glycogen_store_g + 1e-9


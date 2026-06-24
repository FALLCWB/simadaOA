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

"""Unit tests for the optional hepatic counterregulation model."""

from __future__ import annotations

from simada.behavior.counterregulation import (
    CounterregulationConfig,
    CounterregulationModel,
)


def _model(**over):
    cfg = CounterregulationConfig(enabled=True, **over)
    return CounterregulationModel(cfg)


def test_disabled_releases_nothing():
    m = CounterregulationModel(CounterregulationConfig(enabled=False))
    # even deep hypo releases nothing when the module is off (default behaviour)
    assert m.step(bg=30.0, dt_min=3.0) == 0.0


def test_euglycemia_releases_nothing():
    m = _model()
    assert m.step(bg=120.0, dt_min=3.0) == 0.0
    assert m.step(bg=68.0, dt_min=3.0) == 0.0  # exactly at threshold = not below


def test_hypoglycemia_releases_glucose():
    m = _model()
    grams = m.step(bg=50.0, dt_min=1.0)
    assert grams > 0.0


def test_deeper_hypo_releases_more():
    shallow = _model().step(bg=60.0, dt_min=1.0)
    deep = _model().step(bg=42.0, dt_min=1.0)
    assert deep > shallow > 0.0


def test_release_caps_at_floor_depth():
    # below the floor the depth fraction saturates at 1.0 (no more than max rate)
    at_floor = _model().step(bg=40.0, dt_min=1.0)
    below = _model().step(bg=20.0, dt_min=1.0)
    assert below == at_floor
    assert at_floor <= _model()._cfg.max_rate_g_per_min * 1.0 + 1e-9


def test_glycogen_depletes_under_sustained_hypo():
    m = _model(glycogen_store_g=5.0, max_rate_g_per_min=0.5)
    first = m.step(bg=40.0, dt_min=1.0)
    # drive a long sustained severe hypo; output must decline as the store empties
    later = None
    for _ in range(200):
        later = m.step(bg=40.0, dt_min=1.0)
    assert later < first
    assert m.glycogen_g >= 0.0


def test_total_release_never_exceeds_store():
    store = 8.0
    m = _model(glycogen_store_g=store, max_rate_g_per_min=1.0)
    total = sum(m.step(bg=35.0, dt_min=1.0) for _ in range(1000))
    assert total <= store + 1e-9
    assert m.glycogen_g >= 0.0


def test_glycogen_recovers_during_euglycemia():
    m = _model(glycogen_store_g=10.0, max_rate_g_per_min=1.0, recovery_g_per_min=0.5)
    # deplete
    for _ in range(1000):
        m.step(bg=35.0, dt_min=1.0)
    depleted = m.glycogen_g
    # euglycemic recovery refills the store (slowly), then it can release again
    for _ in range(100):
        m.step(bg=120.0, dt_min=1.0)
    assert m.glycogen_g > depleted
    assert m.glycogen_g <= 10.0 + 1e-9  # never above capacity
    assert m.step(bg=40.0, dt_min=1.0) > 0.0


def test_deterministic():
    a = _model(); b = _model()
    seq_a = [a.step(bg=45.0, dt_min=3.0) for _ in range(50)]
    seq_b = [b.step(bg=45.0, dt_min=3.0) for _ in range(50)]
    assert seq_a == seq_b

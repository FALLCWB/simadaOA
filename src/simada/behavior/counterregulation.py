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

"""Optional hepatic counterregulation (glycogenolysis) during hypoglycemia.

The open-source UVA/Padova model integrated by simglucose has only a weak,
linear glucose-autoregulation of endogenous glucose production (EGP rises as
plasma glucose falls) and NO hormone-mediated counterregulation: no glucagon /
epinephrine surge that defends against hypoglycemia by mobilizing hepatic
glycogen. This module adds that defense as an OPTIONAL, reactive physiological
mechanism (off by default), so its effect can be studied as a sensitivity
analysis rather than silently changing the validated base model.

It is physiology, not control: it reacts to live glucose every simulation step
and releases a graded amount of glucose (expressed as a carbohydrate-equivalent,
the same approximation already used by the BG<30 glucagon rescue) into the
scenario via ``SimadaScenario.inject_cho``. Two features keep it from masking
the very hypoglycemia the framework studies:

1. Graded by hypo depth between an activation threshold (~68 mg/dL, where
   glucagon/epinephrine responses begin -- Mitrakou et al. 1991; Cryer 2009)
   and a floor below which glycogenolysis is maximal.
2. Bounded by a finite hepatic glycogen store that DEPLETES under sustained or
   repeated hypoglycemia and resynthesizes slowly during euglycemia. A prolonged
   fast or recurrent lows therefore exhaust the defense -- the physiological
   basis for the danger of prolonged fasting and hypoglycemia-associated
   autonomic failure (HAAF).

The default magnitude is INTENDED as a modest extra hepatic output ON TOP of the
weak glucose-autoregulation already in the ODE -- it is added (not subtracted)
and is not formally decomposed from it, so treat ~0.2 g/min as the model's
gastric INJECTION rate (the effective plasma appearance is lower; see below), not
a rigorously net-of-EGP figure. It is tunable; it should not be increased to the
point of erasing hypoglycemic excursions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CounterregulationConfig:
    """Parameters for hepatic counterregulation. Disabled by default.

    All glucose quantities are carbohydrate-equivalent grams (the gut-absorption
    approximation shared with the glucagon-rescue mechanism).
    """

    enabled: bool = False
    threshold_mg_dl: float = 68.0       # activation onset (Mitrakou 1991; Cryer 2009)
    floor_mg_dl: float = 55.0           # near-maximal glucagon secretion by ~50-60 mg/dL (Cryer 2009)
    # Peak GASTRIC INJECTION rate (carb-equivalent), NOT the plasma appearance
    # rate: the glucose is routed through simglucose's gut absorption (Ra), which
    # spreads and delays it (~15-30 min), so the effective plasma-appearance rate
    # is several-fold lower than this number. Real glycogenolysis appears in plasma
    # within minutes; a direct portal/EGP path is future work (see module docstring).
    max_rate_g_per_min: float = 0.20
    glycogen_store_g: float = 90.0      # hepatic glycogen pool (~70-100 g)
    recovery_g_per_min: float = 0.05    # slow resynthesis during euglycemia

    def __post_init__(self) -> None:
        if self.floor_mg_dl >= self.threshold_mg_dl:
            raise ValueError("floor_mg_dl must be below threshold_mg_dl")
        if self.max_rate_g_per_min <= 0 or self.glycogen_store_g <= 0:
            raise ValueError("max_rate_g_per_min and glycogen_store_g must be positive")
        if self.recovery_g_per_min < 0:
            raise ValueError("recovery_g_per_min must be non-negative")


class CounterregulationModel:
    """Reactive glycogenolysis: glucose (g) released per simulation step.

    Stateful only in the remaining glycogen store; otherwise a pure function of
    the current glucose and elapsed time, so a run is fully reproducible.
    """

    def __init__(self, config: CounterregulationConfig) -> None:
        self._cfg = config
        self._glycogen_g = config.glycogen_store_g

    @property
    def glycogen_g(self) -> float:
        """Remaining hepatic glycogen available for counterregulation (g)."""
        return self._glycogen_g

    def reset(self) -> None:
        """Restore the glycogen store to full (start of a new simulation run).

        simglucose calls ``controller.reset()`` at the start of every
        ``SimObj.simulate()``; without this the depleted store would leak across
        runs that share a model instance.
        """
        self._glycogen_g = self._cfg.glycogen_store_g

    def step(self, bg: float, dt_min: float) -> float:
        """Glucose (carb-equivalent g) released this step for glucose ``bg``.

        Returns 0.0 when disabled or at/above the activation threshold; during
        euglycemia the glycogen store slowly resynthesizes. Below threshold the
        release is graded by hypo depth and scaled by the fraction of glycogen
        still available, and can never exceed the remaining store.
        """
        cfg = self._cfg
        if not cfg.enabled or dt_min <= 0.0:
            return 0.0

        if bg >= cfg.threshold_mg_dl:
            # euglycemic: slowly refill the store (capped at capacity)
            self._glycogen_g = min(
                cfg.glycogen_store_g,
                self._glycogen_g + cfg.recovery_g_per_min * dt_min,
            )
            return 0.0

        span = cfg.threshold_mg_dl - cfg.floor_mg_dl
        depth = 1.0 if span <= 0.0 else (cfg.threshold_mg_dl - bg) / span
        depth = max(0.0, min(1.0, depth))
        # LINEAR depletion then hard cutoff (not capacity-scaled): release at up to
        # max_rate while glycogen remains, and stop exactly when the store empties.
        # Capacity-scaling (rate proportional to remaining store) produced an
        # exponential tail that never truly exhausts -- contradicting the
        # "store exhausted under sustained hypo" rationale. At max rate the
        # 90 g store now depletes in ~7.5 h, matching glycogenolytic kinetics.
        grams = min(cfg.max_rate_g_per_min * depth * dt_min, self._glycogen_g)
        self._glycogen_g -= grams
        return grams

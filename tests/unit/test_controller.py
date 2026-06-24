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

"""Tests for AdherentBBController — the safety-critical insulin delivery controller."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from numpy.random import default_rng

from simada.controller.adherent_bb import AdherentBBController, _CORRECTION_CHECK_INTERVAL
from simada.core.config import ArchetypeParams
from simada.core.types import (
    ArchetypeID,
    DailySchedule,
    DayType,
    ExerciseEvent,
    ExerciseIntensity,
    GlucagonEvent,
    InsulinRegimen,
    MealEvent,
    MealType,
    StressEvent,
    StressType,
)
from simada.insulin.adherence import AdherenceInsulinModel
from simada.insulin.basal import BasalProfile
from simada.insulin.calculator import BolusCalculator
from simada.patient.archetype import create_archetype
from simada.scenario.custom_scenario import SimadaScenario


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_START = datetime(2026, 6, 1, 6, 0)  # 06:00 on a Monday


def _make_obs(cgm: float = 120.0, cho: float = 0.0) -> SimpleNamespace:
    """Create a mock observation matching simglucose's interface."""
    return SimpleNamespace(CGM=cgm, CHO=cho)


def _make_meal(
    time: datetime,
    true_carbs: float = 50.0,
    estimated_carbs: float = 50.0,
) -> MealEvent:
    return MealEvent(
        time=time,
        meal_type=MealType.ALMOCO,
        true_carbs_g=true_carbs,
        estimated_carbs_g=estimated_carbs,
        foods=("arroz", "feijao"),
        glycemic_index=60.0,
    )


def _make_schedule(
    meals: list[MealEvent] | None = None,
    exercise_events: list[ExerciseEvent] | None = None,
    stress_events: list[StressEvent] | None = None,
) -> DailySchedule:
    return DailySchedule(
        date=_START,
        day_type=DayType.WEEKDAY,
        wake_time=_START,
        sleep_time=_START.replace(hour=22),
        meals=meals or [],
        exercise_events=exercise_events or [],
        stress_events=stress_events or [],
    )


def _build_controller(
    params: ArchetypeParams,
    schedules: list[DailySchedule] | None = None,
    seed: int = 42,
) -> AdherentBBController:
    """Build a fully wired controller for testing."""
    if schedules is None:
        # Minimal schedule with one meal so from_schedules is valid
        meal = _make_meal(_START + timedelta(hours=6))
        schedules = [_make_schedule(meals=[meal])]

    scenario = SimadaScenario.from_schedules(schedules)
    archetype = create_archetype(ArchetypeID.ADHERENT, params)
    calculator = BolusCalculator(cr=10.0, cf=50.0, target_bg=120.0)
    insulin_model = AdherenceInsulinModel(archetype, calculator)
    basal_profile = BasalProfile(tdi=40.0, regimen=InsulinRegimen.PUMP)
    rng = default_rng(seed)

    return AdherentBBController(
        insulin_model=insulin_model,
        basal_profile=basal_profile,
        scenario=scenario,
        archetype_params=params,
        rng=rng,
        start_time=scenario.start_time,
    )


def _step(ctrl: AdherentBBController, cgm: float = 120.0, cho: float = 0.0):
    """Advance the controller by one step and return the Action."""
    return ctrl.policy(_make_obs(cgm, cho), reward=0, done=False, sample_time=3)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAdherentBBController:
    """Unit tests for the basal-bolus controller."""

    def test_basal_delivery(self, adherent_params: ArchetypeParams) -> None:
        """Every step should deliver basal > 0 (pump always running)."""
        ctrl = _build_controller(adherent_params)
        for _ in range(5):
            action = _step(ctrl, cgm=120.0)
            assert action.basal > 0, "Basal rate must be positive every step"

    def test_hypo_correction_triggers_at_threshold(
        self, adherent_params: ArchetypeParams
    ) -> None:
        """BG=69 should trigger CHO injection; BG=71 should not.

        The adherent threshold is 70 mg/dL. We detect the correction by
        checking whether inject_cho was called on the scenario (which adds
        to the _injected_cho dict).
        """
        # BG = 69 => below threshold => correction.
        # Both CHO delivery AND the correction counter must be incremented.
        # An OR assertion would allow the mechanism to be half-broken (counter
        # incremented without actual CHO delivery, or vice versa).
        ctrl_low = _build_controller(adherent_params, seed=100)
        _step(ctrl_low, cgm=69.0)
        assert ctrl_low._hypo_correction_count > 0, (
            "BG=69 should increment _hypo_correction_count"
        )
        assert len(ctrl_low._scenario._injected_cho) > 0, (
            "BG=69 should inject CHO into the scenario (delivery mechanism must fire)"
        )

        # BG = 71 => above threshold => no correction
        ctrl_high = _build_controller(adherent_params, seed=100)
        _step(ctrl_high, cgm=71.0)
        assert ctrl_high._hypo_correction_count == 0, (
            "BG=71 should NOT trigger hypo correction"
        )

    def test_hypo_correction_severity(
        self, adherent_params: ArchetypeParams
    ) -> None:
        """Mild hypo (54-70) yields 7.6g (2 tablets); severe (<54, ADA Level 2)
        yields 15.2g (4 tablets)."""
        # Mild hypo: BG=65
        ctrl_mild = _build_controller(adherent_params, seed=42)
        _step(ctrl_mild, cgm=65.0)
        mild_cho = sum(ctrl_mild._scenario._injected_cho.values())
        assert mild_cho == pytest.approx(7.6, abs=0.1), (
            f"Mild hypo should inject 7.6g (2 tablets), got {mild_cho}"
        )

        # Severe hypo: BG=50 (< 54 mg/dL = ADA Level 2)
        ctrl_severe = _build_controller(adherent_params, seed=42)
        _step(ctrl_severe, cgm=50.0)
        severe_cho = sum(ctrl_severe._scenario._injected_cho.values())
        assert severe_cho == pytest.approx(15.2, abs=0.1), (
            f"Severe hypo should inject 15.2g (4 tablets), got {severe_cho}"
        )

    def test_hypo_max_corrections(
        self, adherent_params: ArchetypeParams
    ) -> None:
        """After hypo_max_corrections attempts, no more CHO is injected."""
        ctrl = _build_controller(adherent_params, seed=42)
        max_corrections = adherent_params.hypo_max_corrections  # from adherent.yaml

        # Force enough steps with hypo BG, spacing them far enough apart
        # that recheck_minutes is satisfied (15 min = 5 steps at 3 min each).
        for i in range(max_corrections + 3):
            _step(ctrl, cgm=55.0)
            # Advance time past recheck window by stepping 5 more times
            for _ in range(5):
                _step(ctrl, cgm=55.0)

        assert ctrl._hypo_correction_count == max_corrections, (
            f"Should stop at {max_corrections} corrections, got {ctrl._hypo_correction_count}"
        )

    def test_hypo_recheck_timing(
        self, adherent_params: ArchetypeParams
    ) -> None:
        """Correction should not fire again before hypo_recheck_minutes (15 min)."""
        ctrl = _build_controller(adherent_params, seed=42)

        # First step triggers correction
        _step(ctrl, cgm=65.0)
        assert ctrl._hypo_correction_count == 1

        # Steps within 15 minutes (steps 2-5, i.e. minutes 3-12) should NOT
        # trigger a second correction
        for _ in range(4):
            _step(ctrl, cgm=65.0)
        assert ctrl._hypo_correction_count == 1, (
            "Should not re-correct within 15 minutes"
        )

        # Step at minute 15 (step 6) should trigger second correction
        _step(ctrl, cgm=65.0)
        assert ctrl._hypo_correction_count == 2, (
            "Should re-correct at 15 minutes"
        )

    def test_hypo_reset_when_bg_recovers(
        self, adherent_params: ArchetypeParams
    ) -> None:
        """hypo_correction_count resets when BG >= threshold (70)."""
        ctrl = _build_controller(adherent_params, seed=42)

        # Trigger one correction
        _step(ctrl, cgm=65.0)
        assert ctrl._hypo_correction_count == 1

        # BG recovers to 70 (>= threshold)
        _step(ctrl, cgm=70.0)
        assert ctrl._hypo_correction_count == 0, (
            "Correction count should reset when BG >= threshold"
        )

    def test_meal_bolus_on_cho(
        self, adherent_params: ArchetypeParams
    ) -> None:
        """When observation has CHO > 0 and matching meal in scenario, bolus is added."""
        meal_time = _START + timedelta(hours=6)
        meal = _make_meal(meal_time, true_carbs=60.0, estimated_carbs=60.0)
        schedule = _make_schedule(meals=[meal])
        ctrl = _build_controller(adherent_params, schedules=[schedule], seed=42)

        # Advance to meal time.
        # Step 1: _first_call=True, no time advance => time=_START.
        # Step N (N>1): time = _START + (N-1)*3 min.
        # To reach 12:00 = _START + 360 min => N-1 = 120 => N = 121.
        # So we need 120 pre-steps, then the 121st step is at meal_time.
        for _ in range(120):
            _step(ctrl, cgm=120.0, cho=0.0)

        # At step 121, controller is at meal_time. Deliver CHO.
        action = _step(ctrl, cgm=120.0, cho=60.0)
        assert action.bolus > 0, "Meal bolus should be delivered when CHO is present"

    def test_meal_not_double_processed(
        self, adherent_params: ArchetypeParams
    ) -> None:
        """Same meal time should not produce two boluses."""
        meal_time = _START + timedelta(hours=6)
        meal = _make_meal(meal_time, true_carbs=60.0, estimated_carbs=60.0)
        schedule = _make_schedule(meals=[meal])
        ctrl = _build_controller(adherent_params, schedules=[schedule], seed=42)

        # Advance to meal time (step 121 = _START + 360 min = 12:00)
        for _ in range(120):
            _step(ctrl, cgm=120.0, cho=0.0)

        # First call at meal time
        action1 = _step(ctrl, cgm=120.0, cho=60.0)
        bolus1 = action1.bolus

        # Second call (next step, 3 min later)
        action2 = _step(ctrl, cgm=120.0, cho=60.0)
        bolus2 = action2.bolus

        # The meal should only produce a bolus once (first time)
        assert bolus1 > 0, "First encounter should produce a bolus"
        # The _processed_meals set prevents re-processing
        assert len(ctrl._processed_meals) == 1, "Meal should be processed exactly once"

    def test_hyper_correction_interval(
        self, adherent_params: ArchetypeParams
    ) -> None:
        """Correction fires every _CORRECTION_CHECK_INTERVAL steps when BG > threshold."""
        ctrl = _build_controller(adherent_params, seed=42)

        bolus_steps = []
        for i in range(1, _CORRECTION_CHECK_INTERVAL * 3 + 1):
            action = _step(ctrl, cgm=280.0, cho=0.0)
            if action.bolus > 0:
                bolus_steps.append(ctrl._step_count)

        # Corrections should only fire at multiples of _CORRECTION_CHECK_INTERVAL
        for step in bolus_steps:
            assert step % _CORRECTION_CHECK_INTERVAL == 0, (
                f"Correction at step {step} is not at interval boundary "
                f"(expected multiples of {_CORRECTION_CHECK_INTERVAL})"
            )

    def test_exercise_reduces_basal(
        self, adherent_params: ArchetypeParams
    ) -> None:
        """During exercise, basal should be lower than without exercise."""
        # Controller without exercise
        schedule_no_ex = _make_schedule(meals=[_make_meal(_START + timedelta(hours=6))])
        ctrl_no_ex = _build_controller(adherent_params, schedules=[schedule_no_ex], seed=42)
        action_no_ex = _step(ctrl_no_ex, cgm=120.0)

        # Controller with exercise starting at _START (immediately)
        exercise = ExerciseEvent(
            start_time=_START,
            duration_minutes=60,
            intensity=ExerciseIntensity.MODERATE,
            insulin_sensitivity_multiplier=1.5,
        )
        schedule_ex = _make_schedule(
            meals=[_make_meal(_START + timedelta(hours=6))],
            exercise_events=[exercise],
        )
        ctrl_ex = _build_controller(adherent_params, schedules=[schedule_ex], seed=42)
        action_ex = _step(ctrl_ex, cgm=120.0)

        assert action_ex.basal < action_no_ex.basal, (
            f"Exercise should reduce basal: {action_ex.basal} should be < {action_no_ex.basal}"
        )

    def test_post_exercise_effect_decays(
        self, adherent_params: ArchetypeParams
    ) -> None:
        """After exercise ends, residual sensitivity decays linearly."""
        # Exercise from _START for 30 min (10 steps at 3 min)
        exercise = ExerciseEvent(
            start_time=_START,
            duration_minutes=30,
            intensity=ExerciseIntensity.MODERATE,
            insulin_sensitivity_multiplier=1.5,
        )
        schedule = _make_schedule(
            meals=[_make_meal(_START + timedelta(hours=6))],
            exercise_events=[exercise],
        )
        ctrl = _build_controller(adherent_params, schedules=[schedule], seed=42)

        # Step through exercise (10 steps = 30 min)
        for _ in range(10):
            _step(ctrl, cgm=120.0)

        # Now exercise just ended. Collect basal rates during post-exercise period.
        post_basals = []
        for _ in range(20):
            action = _step(ctrl, cgm=120.0)
            post_basals.append(action.basal)

        # First post-exercise basal should be lower than last (effect decays)
        assert post_basals[0] < post_basals[-1], (
            "Post-exercise sensitivity should decay: early basal should be lower than late basal"
        )

    def test_stress_increases_basal(
        self, adherent_params: ArchetypeParams
    ) -> None:
        """During stress event, basal should be higher than normal."""
        # No stress
        schedule_no_stress = _make_schedule(meals=[_make_meal(_START + timedelta(hours=6))])
        ctrl_no_stress = _build_controller(adherent_params, schedules=[schedule_no_stress], seed=42)
        action_no_stress = _step(ctrl_no_stress, cgm=120.0)

        # With psychological stress starting at _START
        stress = StressEvent(
            start_time=_START,
            duration_minutes=120,
            stress_type=StressType.PSYCHOLOGICAL,
            insulin_resistance_factor=1.3,  # 30% more resistance
        )
        schedule_stress = _make_schedule(
            meals=[_make_meal(_START + timedelta(hours=6))],
            stress_events=[stress],
        )
        ctrl_stress = _build_controller(adherent_params, schedules=[schedule_stress], seed=42)
        action_stress = _step(ctrl_stress, cgm=120.0)

        assert action_stress.basal > action_no_stress.basal, (
            f"Stress should increase basal: {action_stress.basal} should be > {action_no_stress.basal}"
        )

    def test_nonadherent_ignores_context(
        self, nonadherent_params: ArchetypeParams
    ) -> None:
        """Nonadherent with iob_consideration=0.1 should barely adjust basal for exercise."""
        exercise = ExerciseEvent(
            start_time=_START,
            duration_minutes=60,
            intensity=ExerciseIntensity.VIGOROUS,
            insulin_sensitivity_multiplier=2.0,
        )

        # Build controller for nonadherent archetype
        schedule = _make_schedule(
            meals=[_make_meal(_START + timedelta(hours=6))],
            exercise_events=[exercise],
        )
        scenario = SimadaScenario.from_schedules([schedule])
        archetype = create_archetype(ArchetypeID.NONADHERENT, nonadherent_params)
        calculator = BolusCalculator(cr=10.0, cf=50.0, target_bg=120.0)
        insulin_model = AdherenceInsulinModel(archetype, calculator)
        basal_profile = BasalProfile(tdi=40.0, regimen=InsulinRegimen.PUMP)
        rng = default_rng(42)

        ctrl_nonadherent = AdherentBBController(
            insulin_model=insulin_model,
            basal_profile=basal_profile,
            scenario=scenario,
            archetype_params=nonadherent_params,
            rng=rng,
            start_time=scenario.start_time,
        )

        # Also build an adherent controller with same exercise
        from simada.core.config import load_archetype_params
        from pathlib import Path
        project_root = Path(__file__).resolve().parent.parent.parent
        adherent_params = load_archetype_params(
            project_root / "configs" / "archetypes" / "adherent.yaml"
        )
        scenario2 = SimadaScenario.from_schedules([schedule])
        archetype2 = create_archetype(ArchetypeID.ADHERENT, adherent_params)
        calculator2 = BolusCalculator(cr=10.0, cf=50.0, target_bg=120.0)
        insulin_model2 = AdherenceInsulinModel(archetype2, calculator2)
        basal_profile2 = BasalProfile(tdi=40.0, regimen=InsulinRegimen.PUMP)
        rng2 = default_rng(42)

        ctrl_adherent = AdherentBBController(
            insulin_model=insulin_model2,
            basal_profile=basal_profile2,
            scenario=scenario2,
            archetype_params=adherent_params,
            rng=rng2,
            start_time=scenario2.start_time,
        )

        action_nonadherent = _step(ctrl_nonadherent, cgm=120.0)
        action_adherent = _step(ctrl_adherent, cgm=120.0)

        # Nonadherent iob_consideration=0.1 means only 10% of the exercise adjustment
        # is applied. So the nonadherent basal during exercise should be closer to
        # the un-adjusted basal (higher) than the adherent basal.
        assert action_nonadherent.basal > action_adherent.basal, (
            f"Nonadherent should barely adjust basal for exercise: "
            f"nonadherent={action_nonadherent.basal:.6f}, adherent={action_adherent.basal:.6f}"
        )

    # -- A1: Low-glucose suspend tests -------------------------------------------

    def test_basal_suspend_below_54(
        self, adherent_params: ArchetypeParams
    ) -> None:
        """Basal must be zero when CGM < 54 (TBR Level 2 suspend)."""
        ctrl = _build_controller(adherent_params, seed=42)
        action = _step(ctrl, cgm=50.0)
        assert action.basal == 0.0, (
            f"Basal should be suspended at CGM=50, got {action.basal}"
        )

    def test_basal_suspend_stays_until_70(
        self, adherent_params: ArchetypeParams
    ) -> None:
        """Once suspended at <54, basal stays zero until CGM >= 70."""
        ctrl = _build_controller(adherent_params, seed=42)
        # Trigger suspend
        _step(ctrl, cgm=50.0)
        assert ctrl._basal_suspended is True

        # BG rises to 60 — still below 70, basal should stay suspended
        action = _step(ctrl, cgm=60.0)
        assert action.basal == 0.0, "Basal should stay suspended at CGM=60"
        assert ctrl._basal_suspended is True

        # BG rises to 70 — resume
        action = _step(ctrl, cgm=70.0)
        assert action.basal > 0.0, "Basal should resume at CGM=70"
        assert ctrl._basal_suspended is False

    def test_basal_normal_above_54(
        self, adherent_params: ArchetypeParams
    ) -> None:
        """Basal should be positive when CGM is safely above 54."""
        ctrl = _build_controller(adherent_params, seed=42)
        action = _step(ctrl, cgm=120.0)
        assert action.basal > 0.0

    # -- A2: Glucagon rescue tests -----------------------------------------------

    def test_glucagon_fires_below_threshold(
        self, adherent_params: ArchetypeParams
    ) -> None:
        """Glucagon rescue should fire when CGM < 30 (BUG #1 fix)."""
        ctrl = _build_controller(adherent_params, seed=42)
        _step(ctrl, cgm=15.0)
        assert len(ctrl.glucagon_events) == 1
        event = ctrl.glucagon_events[0]
        assert event.bg_at_rescue == 15.0
        assert event.carbs_injected == 50.0

    def test_glucagon_fires_at_25_mg_dl(
        self, adherent_params: ArchetypeParams
    ) -> None:
        """BUG #1 regression: BG=25 must now trigger glucagon.

        Under the old 20 mg/dL threshold, BG=25 was treated as recoverable
        and no rescue was issued; the patient was already deep into
        neuroglycopenia. With the new 30 mg/dL threshold, BG=25 fires.
        """
        ctrl = _build_controller(adherent_params, seed=42)
        _step(ctrl, cgm=25.0)
        assert len(ctrl.glucagon_events) == 1
        assert ctrl.glucagon_events[0].bg_at_rescue == 25.0

    def test_glucagon_max_once_per_24h(
        self, adherent_params: ArchetypeParams
    ) -> None:
        """Only 1 glucagon per 24 hours."""
        ctrl = _build_controller(adherent_params, seed=42)
        _step(ctrl, cgm=10.0)
        assert len(ctrl.glucagon_events) == 1

        # Still < 20 but within 24h — no second glucagon
        for _ in range(10):
            _step(ctrl, cgm=10.0)
        assert len(ctrl.glucagon_events) == 1

    def test_glucagon_resets_hypo_state(
        self, adherent_params: ArchetypeParams
    ) -> None:
        """After glucagon rescue, hypo correction state is reset and the
        immediate hypo re-trigger is suppressed (BUG #7 fix).

        Updated for H6#2: previously the controller cleared the hypo
        counter inside ``_handle_glucagon_rescue`` and then re-entered
        ``_handle_hypo_correction`` on the same step, stacking another CHO
        injection on top of the 50 g rescue while the CGM still read in
        the danger zone. The fix sets a ``_glucagon_fired_time`` timestamp and
        skips hypo correction for 20+ minutes (7 steps × 3 min) after rescue,
        so the counter remains at 0 right after rescue.
        """
        ctrl = _build_controller(adherent_params, seed=42)
        # First, trigger some hypo corrections
        _step(ctrl, cgm=65.0)
        assert ctrl._hypo_correction_count == 1

        # Now trigger glucagon at BG=15
        # Need to advance past recheck window first (15 min = 5 steps)
        for _ in range(5):
            _step(ctrl, cgm=65.0)
        _step(ctrl, cgm=15.0)
        # BUG #7 fix: glucagon resets the count to 0 AND the immediate hypo
        # correction is suppressed for the firing step, so count stays at 0.
        assert ctrl._hypo_correction_count == 0
        assert len(ctrl.glucagon_events) == 1
        assert ctrl.glucagon_events[0].bg_at_rescue == 15.0
        # H6#2: _glucagon_just_fired replaced by _glucagon_fired_time datetime
        assert ctrl._glucagon_fired_time is not None

    def test_glucagon_no_fire_above_threshold(
        self, adherent_params: ArchetypeParams
    ) -> None:
        """Glucagon should not fire when CGM >= 30 (BUG #1 fix).

        Updated: the old threshold was 20 mg/dL, which corresponds to deep
        coma -- far too late to administer a rescue. The new threshold is
        30 mg/dL, still severe neuroglycopenia but treatable.
        """
        ctrl = _build_controller(adherent_params, seed=42)
        _step(ctrl, cgm=35.0)
        assert len(ctrl.glucagon_events) == 0

    # -- A3: Trend-based hypo correction tests -----------------------------------

    def test_trend_rising_skips_correction(
        self, adherent_params: ArchetypeParams
    ) -> None:
        """If BG is rising after correction, don't correct again."""
        ctrl = _build_controller(adherent_params, seed=42)
        # First correction at BG=55
        _step(ctrl, cgm=55.0)
        assert ctrl._hypo_correction_count == 1
        assert ctrl._bg_at_last_correction == 55.0

        # Wait 15 min (5 steps at 3 min) then BG=62 (rising) — should NOT correct
        for _ in range(4):
            _step(ctrl, cgm=55.0)
        _step(ctrl, cgm=62.0)
        assert ctrl._hypo_correction_count == 1, (
            "Should not correct when BG is rising (62 > 55)"
        )

    def test_trend_falling_triggers_correction(
        self, adherent_params: ArchetypeParams
    ) -> None:
        """If BG is falling/flat after recheck, correct again."""
        ctrl = _build_controller(adherent_params, seed=42)
        # First correction at BG=60
        _step(ctrl, cgm=60.0)
        assert ctrl._hypo_correction_count == 1

        # Wait 15 min then BG=55 (falling) — SHOULD correct
        for _ in range(4):
            _step(ctrl, cgm=60.0)
        _step(ctrl, cgm=55.0)
        assert ctrl._hypo_correction_count == 2, (
            "Should correct again when BG is falling (55 <= 60)"
        )

    def test_trend_flat_triggers_correction(
        self, adherent_params: ArchetypeParams
    ) -> None:
        """If BG is flat (same as last correction), correct again."""
        ctrl = _build_controller(adherent_params, seed=42)
        # First correction at BG=60
        _step(ctrl, cgm=60.0)
        assert ctrl._hypo_correction_count == 1

        # Wait 15 min then BG=60 (flat) — SHOULD correct
        for _ in range(4):
            _step(ctrl, cgm=60.0)
        _step(ctrl, cgm=60.0)
        assert ctrl._hypo_correction_count == 2, (
            "Should correct again when BG is flat (60 == 60)"
        )

    def test_new_episode_after_recovery(
        self, adherent_params: ArchetypeParams
    ) -> None:
        """After BG >= 70, a new drop below 70 starts a fresh episode."""
        ctrl = _build_controller(adherent_params, seed=42)
        # First episode
        _step(ctrl, cgm=65.0)
        assert ctrl._hypo_correction_count == 1

        # Recover
        _step(ctrl, cgm=75.0)
        assert ctrl._hypo_correction_count == 0
        assert ctrl._bg_at_last_correction is None

        # New episode
        _step(ctrl, cgm=65.0)
        assert ctrl._hypo_correction_count == 1

    # -- WP2: Pump safety guard tests -------------------------------------------

    def test_max_bolus_capped(
        self, adherent_params: ArchetypeParams
    ) -> None:
        """Bolus dose should never exceed max_single_bolus_u (pump hardware limit).

        We set max_single_bolus_u=10 and create a large meal that would
        require > 10U. The delivered bolus must be capped.
        """
        # Override the max single bolus to 10U
        params = adherent_params.model_copy(update={"max_single_bolus_u": 10.0})
        meal_time = _START + timedelta(hours=6)
        # 150g carbs with CR=10 → ~15U needed (exceeds 10U limit)
        meal = _make_meal(meal_time, true_carbs=150.0, estimated_carbs=150.0)
        schedule = _make_schedule(meals=[meal])
        ctrl = _build_controller(params, schedules=[schedule], seed=42)

        # Advance to meal time (120 steps = 360 min = 6 hours)
        for _ in range(120):
            _step(ctrl, cgm=120.0, cho=0.0)

        # Deliver the meal
        action = _step(ctrl, cgm=120.0, cho=150.0)
        # bolus is delivered as U/min over sample_time=3min
        delivered_u = action.bolus * 3  # convert back to units
        assert delivered_u <= 10.0, (
            f"Delivered bolus {delivered_u:.2f}U exceeds max_single_bolus_u=10U"
        )

    def test_hypo_cho_capped_per_episode(
        self, nonadherent_params: ArchetypeParams
    ) -> None:
        """Total CHO in one hypo episode must not exceed hypo_max_episode_cho_g.

        Nonadherent config has hypo_max_episode_cho_g=120.0.
        """
        params = nonadherent_params.model_copy(update={
            "hypo_max_episode_cho_g": 60.0,
            "hypo_max_corrections": 20,  # high limit so CHO cap binds first
        })
        ctrl = _build_controller(params, seed=42)

        # Trigger many hypo corrections in a single episode
        # by keeping BG < 70 and stepping past recheck intervals
        for _ in range(100):
            _step(ctrl, cgm=55.0)

        total_cho = sum(ctrl._scenario._injected_cho.values())
        assert total_cho <= 60.0 + 0.01, (
            f"Total episode CHO {total_cho:.1f}g exceeds hypo_max_episode_cho_g=60g"
        )

    def test_escalation_capped_at_3_steps(
        self, adherent_params: ArchetypeParams
    ) -> None:
        """Overcorrection escalation exponent must stop growing after 3 corrections.

        Using adherent archetype with single treatment (glucose_tablet) to
        eliminate randomness from treatment selection. With BG < 60, base=15.2g
        (4 tablets), factor=1.25:
        Step 1: 15.2g (no escalation, exponent=0)
        Step 2: 15.2 * 1.25^1 = 19.0g
        Step 3: 15.2 * 1.25^2 = 23.75g
        Step 4: 15.2 * 1.25^3 = 29.69g (capped at exponent=3)
        Step 5: 15.2 * 1.25^3 = 29.69g (still capped)
        """
        params = adherent_params.model_copy(update={
            "hypo_max_corrections": 10,
            "hypo_max_episode_cho_g": 500.0,  # high so we don't hit CHO cap
            "hypo_overcorrection_factor": 1.25,
            "hypo_treatment_preference": ["tablete_glicose"],  # single item, no randomness
        })
        ctrl = _build_controller(params, seed=42)

        # Collect individual CHO injections across corrections
        corrections_cho: list[float] = []
        prev_total = 0.0
        for step in range(200):
            _step(ctrl, cgm=55.0)
            current_total = sum(ctrl._scenario._injected_cho.values())
            if current_total > prev_total:
                corrections_cho.append(current_total - prev_total)
                prev_total = current_total

        # Need at least 5 corrections to verify both monotone increase and the cap.
        assert len(corrections_cho) >= 5, (
            f"Expected at least 5 corrections, got {len(corrections_cho)}"
        )

        # --- Monotone increase: corrections 1-4 must be strictly increasing -------
        # If escalation is entirely absent, all corrections would be equal and the
        # cap assertion below would trivially pass (flat == flat). We check the
        # escalation is actually happening before verifying it is bounded.
        assert corrections_cho[0] < corrections_cho[1], (
            f"Correction 1 ({corrections_cho[0]:.2f}g) must be less than "
            f"correction 2 ({corrections_cho[1]:.2f}g); escalation not firing"
        )
        assert corrections_cho[1] < corrections_cho[2], (
            f"Correction 2 ({corrections_cho[1]:.2f}g) must be less than "
            f"correction 3 ({corrections_cho[2]:.2f}g); escalation not firing"
        )
        assert corrections_cho[2] < corrections_cho[3], (
            f"Correction 3 ({corrections_cho[2]:.2f}g) must be less than "
            f"correction 4 ({corrections_cho[3]:.2f}g); escalation not firing"
        )

        # --- Cap: correction 4 and 5 must be the same (exponent capped at 3) -----
        assert corrections_cho[3] == pytest.approx(corrections_cho[4], rel=0.01), (
            f"Correction 4 ({corrections_cho[3]:.2f}g) and 5 ({corrections_cho[4]:.2f}g) "
            f"should be equal (escalation capped at 3 steps)"
        )

    def test_iob_hard_limit_suppresses_correction(
        self, adherent_params: ArchetypeParams
    ) -> None:
        """When IOB >= iob_hard_limit_u, correction bolus should be suppressed.

        We set iob_hard_limit_u very low (0.1U) so any recorded bolus
        causes subsequent corrections to be blocked.
        """
        params = adherent_params.model_copy(update={"iob_hard_limit_u": 0.1})
        meal_time = _START + timedelta(hours=6)
        meal = _make_meal(meal_time, true_carbs=60.0, estimated_carbs=60.0)
        schedule = _make_schedule(meals=[meal])
        ctrl = _build_controller(params, schedules=[schedule], seed=42)

        # Advance to meal time and deliver meal bolus (this loads IOB)
        for _ in range(120):
            _step(ctrl, cgm=120.0, cho=0.0)
        _step(ctrl, cgm=120.0, cho=60.0)  # meal bolus → IOB > 0.1

        # Now advance to a correction check interval with high BG
        # Corrections should be suppressed because IOB > 0.1U
        bolus_seen = False
        for _ in range(_CORRECTION_CHECK_INTERVAL * 2):
            action = _step(ctrl, cgm=280.0, cho=0.0)
            if action.bolus > 0:
                bolus_seen = True

        assert not bolus_seen, (
            "Correction bolus should be suppressed when IOB >= iob_hard_limit_u"
        )

    # --- BUG #5 regression -----------------------------------------------------

    def test_correction_threshold_configurable_per_archetype(
        self,
        adherent_params: ArchetypeParams,
        moderate_params: ArchetypeParams,
        nonadherent_params: ArchetypeParams,
    ) -> None:
        """BUG #5 regression: correction_threshold_mg_dl must come from the
        archetype YAML and differ across profiles (170 / 190 / 220 by
        default; the same field must be overridable via model_copy)."""
        # YAML defaults after the BUG #5 fix.
        assert adherent_params.correction_threshold_mg_dl == 170.0
        assert moderate_params.correction_threshold_mg_dl == 190.0
        assert nonadherent_params.correction_threshold_mg_dl == 220.0

        # The same field must remain configurable (no hardcoding).
        custom = adherent_params.model_copy(
            update={"correction_threshold_mg_dl": 155.0}
        )
        assert custom.correction_threshold_mg_dl == 155.0

    def test_correction_threshold_drives_actual_behavior(
        self, adherent_params: ArchetypeParams
    ) -> None:
        """BUG #5 regression: the controller must honour the configured
        threshold -- a correction should fire above it but not below."""
        params = adherent_params.model_copy(update={
            "correction_threshold_mg_dl": 200.0,
            "correction_probability": 1.0,
            "rage_bolus_probability": 0.0,
            "phantom_bolus_probability": 0.0,
        })
        ctrl = _build_controller(params, seed=42)

        # BG=190 < 200 threshold -- no correction at the interval.
        for _ in range(_CORRECTION_CHECK_INTERVAL):
            action = _step(ctrl, cgm=190.0, cho=0.0)
        # No bolus should have fired (correction suppressed under threshold).
        # We can also assert that the controller's BG was treated as below
        # the correction-trigger via the archetype's logic.
        assert action.bolus == 0.0

        # Fresh controller: BG=250 > 200 -- correction must fire on the next
        # interval boundary.
        ctrl_high = _build_controller(params, seed=42)
        saw_bolus = False
        for _ in range(_CORRECTION_CHECK_INTERVAL + 1):
            action = _step(ctrl_high, cgm=250.0, cho=0.0)
            if action.bolus > 0:
                saw_bolus = True
        assert saw_bolus, "Correction should fire when BG > threshold"

    # --- BUG #6 regression -----------------------------------------------------

    def test_phantom_bolus_30min_cooldown(
        self, nonadherent_params: ArchetypeParams
    ) -> None:
        """BUG #6 regression: phantom boluses cannot stack within 30 minutes.

        Even if the random roll succeeds at every correction interval, the
        controller-side cooldown forbids any phantom firing inside 30 min
        of the previous phantom. We force phantom_bolus_probability=1.0 so
        every successful correction would otherwise be a PHANTOM event.
        """
        from simada.core.types import BolusType

        params = nonadherent_params.model_copy(update={
            "correction_threshold_mg_dl": 150.0,
            "correction_probability": 1.0,
            "phantom_bolus_probability": 1.0,
            "rage_bolus_probability": 0.0,
            "iob_hard_limit_u": 0.0,
            "cf_error_factor_mean": 1.0,
            "cf_error_factor_std": 0.0,
            "cr_error_factor_mean": 1.0,
            "cr_error_factor_std": 0.0,
        })
        ctrl = _build_controller(params, seed=7)

        # Drive 90 minutes of high BG (3 correction intervals).
        phantom_times: list[datetime] = []
        original_record = ctrl._insulin_model.process_correction_bolus

        def tracking_record(*args, **kwargs):
            event = original_record(*args, **kwargs)
            if event is not None and event.bolus_type == BolusType.PHANTOM:
                phantom_times.append(event.time)
            return event

        ctrl._insulin_model.process_correction_bolus = tracking_record  # type: ignore[method-assign]

        for _ in range(31):  # 31 steps = ~90 min, covers 3 intervals
            _step(ctrl, cgm=280.0, cho=0.0)

        # If any phantom fired, the next one must be at least 30 min later.
        for prev, nxt in zip(phantom_times, phantom_times[1:]):
            delta_min = (nxt - prev).total_seconds() / 60.0
            assert delta_min >= 30.0 - 1e-6, (
                f"Phantom fired only {delta_min:.1f} min after previous "
                f"(cooldown should be >= 30 min)"
            )

    # --- BUG #7 regression -----------------------------------------------------

    def test_hypo_correction_suppressed_step_after_glucagon(
        self, adherent_params: ArchetypeParams
    ) -> None:
        """BUG #7 regression: the hypo correction must be skipped on the
        glucagon-firing step so we do not stack CHO on top of the 50 g
        rescue while the CGM still reads in the danger zone.
        """
        ctrl = _build_controller(adherent_params, seed=42)
        # Fire glucagon at BG=15 -- the same step's hypo correction must
        # be suppressed.
        _step(ctrl, cgm=15.0)
        # Only the 50 g glucagon injection should be present; the 7.6 or
        # 15.2 g tablet correction must NOT have fired.
        injected_total = sum(ctrl._scenario._injected_cho.values())
        assert injected_total == pytest.approx(50.0), (
            f"Expected only glucagon (50g), got {injected_total:.1f}g -- "
            "hypo correction was not suppressed on the glucagon step"
        )
        assert ctrl._hypo_correction_count == 0
        # H6#2: _glucagon_just_fired replaced by _glucagon_fired_time datetime
        assert ctrl._glucagon_fired_time is not None

        # The very next step (still BG < 30, still within 20-min window) is
        # also gated by the suppression window.
        _step(ctrl, cgm=18.0)
        # Glucagon cannot re-fire (24h cooldown) AND hypo is suppressed.
        injected_after_two = sum(ctrl._scenario._injected_cho.values())
        assert injected_after_two == pytest.approx(50.0)
        # Suppression window is still active (only 3 min elapsed, need >=21 min).
        assert ctrl._glucagon_fired_time is not None


# ---------------------------------------------------------------------------
# Severe hyperglycemia self-rescue (high-side mirror of the carb rescue)
# ---------------------------------------------------------------------------


class TestSevereHyperRescue:
    """Patient self-rescue when BG stays at/above 350: fires after the
    archetype's onset delay, re-doses every 60 min, resets below threshold."""

    @staticmethod
    def _archetypes_dir():
        from pathlib import Path
        return Path(__file__).resolve().parent.parent.parent / "configs" / "archetypes"

    def test_rescue_fires_after_onset_and_redoses(self) -> None:
        from simada.core.config import load_archetype_params

        base = load_archetype_params(self._archetypes_dir() / "adherent.yaml")
        params = base.model_copy(update={"severe_hyper_onset_minutes": 15.0})
        ctrl = _build_controller(params)

        # Below threshold: nothing happens.
        ctrl._current_time = _START
        ctrl._handle_severe_hyper_rescue(300.0)
        assert ctrl._severe_hyper_doses == 0
        assert ctrl._severe_hyper_since is None

        # Cross 350: clock starts, but onset (15 min) not reached -> no dose.
        ctrl._handle_severe_hyper_rescue(400.0)
        assert ctrl._severe_hyper_since == _START
        assert ctrl._severe_hyper_doses == 0

        ctrl._current_time = _START + timedelta(minutes=14)
        ctrl._handle_severe_hyper_rescue(400.0)
        assert ctrl._severe_hyper_doses == 0

        # At onset (15 min sustained): first rescue dose.
        ctrl._current_time = _START + timedelta(minutes=15)
        ctrl._handle_severe_hyper_rescue(400.0)
        assert ctrl._severe_hyper_doses == 1
        assert ctrl._pending_bolus > 0

        # Inside the 60-min re-dose window: no new dose.
        ctrl._current_time = _START + timedelta(minutes=60)
        ctrl._handle_severe_hyper_rescue(400.0)
        assert ctrl._severe_hyper_doses == 1

        # 60 min after the last dose (t=15 -> t=76): re-doses.
        ctrl._current_time = _START + timedelta(minutes=76)
        ctrl._handle_severe_hyper_rescue(400.0)
        assert ctrl._severe_hyper_doses == 2

        # Drop below threshold: episode resets.
        ctrl._handle_severe_hyper_rescue(200.0)
        assert ctrl._severe_hyper_since is None
        assert ctrl._last_severe_hyper_dose is None

    def test_onset_increases_with_lower_adherence(self) -> None:
        """Blunted perception: adherent reacts soonest, nonadherent latest."""
        from simada.core.config import load_archetype_params

        base = self._archetypes_dir()
        adh = load_archetype_params(base / "adherent.yaml").severe_hyper_onset_minutes
        mod = load_archetype_params(base / "moderate.yaml").severe_hyper_onset_minutes
        non = load_archetype_params(base / "nonadherent.yaml").severe_hyper_onset_minutes
        assert adh < mod < non
        assert (adh, mod, non) == (15.0, 30.0, 45.0)

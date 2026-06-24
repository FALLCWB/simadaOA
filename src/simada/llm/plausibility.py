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
"""Plausibility / aberration checks for LLM-generated week scenarios.

Catches structural and temporal impossibilities that a naive LLM may emit
(eating during sleep, a 30-hour day, an exercise day with no food, a
starvation day) and flags soft out-of-literature-range values. Structural
checks are country-agnostic (operate on times/meals/exercise); carb-range
checks are optional and require a carb-per-meal function (e.g. TACO).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from simada.llm.scenario_schema import EXPECTED_DAYS, MAIN_MEAL_KEYS

if TYPE_CHECKING:
    from collections.abc import Callable


# Literature-grounded plausible ranges. Population dietary figures (meals/day,
# CHO/day) follow the Brazilian household budget/diet survey and TACO table
# (ibge2020pof, taco2011 in docs/references.bib); sleep/portion bounds are
# physiological hard limits.
@dataclass(frozen=True)
class PlausibleRanges:
    # awake hours = 24 - sleep_hours, so the awake-window bound is exactly the
    # sleep-window bound: only sleep_hours is enforced (adjust it to change the
    # effective awake bound). hard bounds (physiological).
    sleep_hours: tuple[float, float] = (3.0, 14.0)
    wake_minute: tuple[int, int] = (240, 660)           # 04:00-11:00 (soft)
    sleep_minute: tuple[int, int] = (1140, 1620)        # 19:00-03:00(+24) (soft)
    meals_per_day: tuple[int, int] = (3, 20)            # soft; eating occasions (ibge2020pof)
    cho_per_day: tuple[float, float] = (50.0, 450.0)    # g/day soft (ibge2020pof, taco2011)
    portion_g: tuple[float, float] = (1.0, 1500.0)      # hard (single item)
    exercise_duration_min: tuple[float, float] = (1.0, 300.0)  # min (soft upper)
    exercise_min_cho_g: float = 20.0                    # min CHO on an exercise day (hard)
    expected_days: int = EXPECTED_DAYS                  # full week (hard)


SEV_HARD = "hard"   # physical impossibility / aberration
SEV_SOFT = "soft"   # out of literature range (implausible, not impossible)


@dataclass
class Violation:
    day_index: int
    kind: str
    severity: str
    detail: str


@dataclass
class PlausibilityReport:
    violations: list[Violation] = field(default_factory=list)
    n_days: int = 0

    @property
    def hard(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == SEV_HARD]

    @property
    def soft(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == SEV_SOFT]

    @property
    def is_aberration(self) -> bool:
        return len(self.hard) > 0


def _to_min(hhmm: object) -> int | None:
    """Parse an ``HH:MM`` string to minutes-since-midnight, or ``None`` if the
    value is missing/malformed. Rejects out-of-range clock fields (hour must be
    0-23, minute 0-59) so that values like ``25:00``, ``12:99`` or ``-1:30`` are
    treated as unparseable rather than silently producing bogus minute counts."""
    try:
        parts = str(hhmm).strip().split(":")
    except (AttributeError, TypeError):
        return None
    if len(parts) != 2:
        return None
    try:
        h, m = int(parts[0]), int(parts[1])
    except (ValueError, TypeError):
        return None
    if not (0 <= h < 24 and 0 <= m < 60):
        return None
    return h * 60 + m


def _is_main_meal(meal_type: object) -> bool:
    """True if meal_type names a main meal (breakfast/lunch/dinner), where a
    near-zero mappable-carb result signals a food-mapping failure rather than a
    legitimately light snack. Keys come from the shared scenario schema."""
    s = str(meal_type or "").lower()
    return any(k in s for k in MAIN_MEAL_KEYS)


def _normalize_exercises(raw: object) -> list[dict[str, Any]]:
    """Coerce the day's ``exercise`` field into a list of exercise dicts.
    Accepts a single dict, a list, or nothing; drops non-dict entries."""
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        return [e for e in raw if isinstance(e, dict)]
    return []


def validate_week(week: dict[str, Any], ranges: PlausibleRanges | None = None,
                  carbs_per_meal: Callable[[dict[str, Any]], float] | None = None
                  ) -> PlausibilityReport:
    """Validate one week scenario. ``carbs_per_meal`` optional: callable that
    takes a meal dict and returns its total carbs (g) for carb-range checks.

    The validator is hardened against malformed LLM output: it never raises on
    wrong types (string ``day_index``, non-list ``meals``, etc.); it instead
    records a hard violation and keeps going. Unparseable times never silently
    skip a check."""
    r = ranges or PlausibleRanges()
    rep = PlausibilityReport()
    days = week.get("days", []) or []
    if not isinstance(days, list):
        rep.violations.append(Violation(-1, "bad_week_format", SEV_HARD,
                                        f"days nao e lista: {type(days).__name__}"))
        days = []
    rep.n_days = len(days)

    # --- week-level structure: a full scenario week is exactly 7 days ---
    if rep.n_days != r.expected_days:
        rep.violations.append(Violation(-1, "not_seven_days", SEV_HARD,
                                        f"{rep.n_days} dias (esperado {r.expected_days})"))

    for day in days:
        if not isinstance(day, dict):
            rep.violations.append(Violation(-1, "bad_day_format", SEV_HARD,
                                            f"dia nao e dict: {type(day).__name__}"))
            continue
        try:
            idx = int(day.get("day_index", -1))
        except (ValueError, TypeError):
            idx = -1
        wake = _to_min(day.get("wake_time", ""))
        sleep = _to_min(day.get("sleep_time", ""))
        meals = day.get("meals", []) or []
        if not isinstance(meals, list):
            rep.violations.append(Violation(idx, "bad_meals_format", SEV_HARD,
                                            f"meals nao e lista: {type(meals).__name__}"))
            meals = []

        # --- temporal sanity of wake/sleep ---
        if wake is None or sleep is None:
            rep.violations.append(Violation(idx, "bad_time_format", SEV_HARD,
                                            f"wake={day.get('wake_time')} sleep={day.get('sleep_time')}"))
            continue
        # sleep after midnight is encoded as <= wake; normalize to a 24h+ value
        sleep_norm = sleep if sleep > wake else sleep + 1440
        awake_h = (sleep_norm - wake) / 60.0
        sleep_h = 24.0 - awake_h
        # awake_h is the complement of sleep_h, so the awake-window bound is
        # exactly the sleep-window bound; a single sleep_hours check covers both
        # (a too-short/too-long awake day is a too-long/too-short sleep).
        if not (r.sleep_hours[0] <= sleep_h <= r.sleep_hours[1]):
            rep.violations.append(Violation(idx, "sleep_hours_out", SEV_HARD,
                                            f"sleep={sleep_h:.1f}h (awake={awake_h:.1f}h)"))
        if not (r.wake_minute[0] <= wake <= r.wake_minute[1]):
            rep.violations.append(Violation(idx, "wake_time_unusual", SEV_SOFT,
                                            f"wake={wake//60:02d}:{wake%60:02d}"))
        if not (r.sleep_minute[0] <= sleep_norm <= r.sleep_minute[1]):
            rep.violations.append(Violation(idx, "sleep_time_unusual", SEV_SOFT,
                                            f"sleep={sleep//60:02d}:{sleep%60:02d}"))

        # --- starvation / energy-balance ---
        n_meals = len(meals)
        if n_meals == 0:
            rep.violations.append(Violation(idx, "no_meals", SEV_HARD, "0 refeicoes no dia"))
        exercises = _normalize_exercises(day.get("exercise"))
        has_exercise = bool(exercises)

        day_carbs = 0.0
        carbs_ok = True
        meal_times: list[int] = []
        for meal in meals:
            if not isinstance(meal, dict):
                rep.violations.append(Violation(idx, "bad_meal_format", SEV_HARD,
                                                f"refeicao nao e dict: {type(meal).__name__}"))
                continue
            mt = _to_min(meal.get("time", ""))
            # an unparseable meal time must NOT slip past validation silently
            if mt is None:
                rep.violations.append(Violation(idx, "bad_meal_time", SEV_HARD,
                                                f"hora de refeicao invalida: {meal.get('time')!r}"))
            else:
                mt_norm = mt if mt >= wake else mt + 1440
                meal_times.append(mt_norm)
                # eating during sleep window = aberration
                if not (wake <= mt_norm <= sleep_norm):
                    rep.violations.append(Violation(idx, "meal_during_sleep", SEV_HARD,
                                                    f"refeicao {meal.get('time')} fora da janela acordada "
                                                    f"[{wake//60:02d}:{wake%60:02d}-{sleep//60:02d}:{sleep%60:02d}]"))
            items = meal.get("items", []) or []
            if not isinstance(items, list):
                rep.violations.append(Violation(idx, "bad_items_format", SEV_HARD,
                                                f"items nao e lista: {type(items).__name__}"))
                items = []
            for item in items:
                if not isinstance(item, dict):
                    rep.violations.append(Violation(idx, "bad_item_format", SEV_HARD,
                                                    f"item nao e dict: {type(item).__name__}"))
                    continue
                p = item.get("portion_g")
                if p is None:
                    continue
                try:
                    pf = float(p)
                except (ValueError, TypeError):
                    rep.violations.append(Violation(idx, "bad_portion", SEV_HARD, f"portion={p}"))
                    continue
                if not (r.portion_g[0] <= pf <= r.portion_g[1]):
                    rep.violations.append(Violation(idx, "portion_out", SEV_HARD,
                                                    f"{item.get('food')}={pf}g"))
            if carbs_per_meal is not None:
                try:
                    meal_carbs = float(carbs_per_meal(meal))
                except Exception:
                    # carb function blew up: do NOT treat as 0 carbs (that would
                    # fabricate a false exercise_no_food / cho_per_day_out). Mark
                    # the day so carb-based checks are skipped instead.
                    carbs_ok = False
                else:
                    day_carbs += meal_carbs
                    # A MAIN meal (breakfast/lunch/dinner) with items but ~0
                    # mappable carbs means its foods did not map to the food
                    # table (e.g. unmatched dish names). The adapter would
                    # silently DROP it, understating the scenario's carb load.
                    # Flag it instead of letting it vanish (H5 bug2).
                    if items and meal_carbs < 5.0 and _is_main_meal(meal.get("meal_type")):
                        rep.violations.append(Violation(
                            idx, "meal_carbs_collapsed", SEV_HARD,
                            f"refeicao principal {meal.get('meal_type')!r} com "
                            f"{meal_carbs:.0f}g de carbo mapeavel (pratos nao mapearam)"))

        # --- chronological order / duplicate meal times (soft) ---
        if len(meal_times) >= 2:
            if any(b < a for a, b in zip(meal_times, meal_times[1:], strict=False)):
                rep.violations.append(Violation(idx, "meals_out_of_order", SEV_SOFT,
                                                "refeicoes fora de ordem cronologica"))
            if len(set(meal_times)) != len(meal_times):
                rep.violations.append(Violation(idx, "duplicate_meal_time", SEV_SOFT,
                                                "duas refeicoes no mesmo horario"))

        # --- exercise sanity: must occur while awake, plausible duration ---
        for ex in exercises:
            st = _to_min(ex.get("start_time", ""))
            st_norm = None
            start_in_window = False
            if st is None:
                rep.violations.append(Violation(idx, "exercise_bad_time", SEV_HARD,
                                                f"hora invalida: {ex.get('start_time')!r}"))
            else:
                st_norm = st if st >= wake else st + 1440
                start_in_window = wake <= st_norm <= sleep_norm
                if not start_in_window:
                    rep.violations.append(Violation(idx, "exercise_during_sleep", SEV_HARD,
                                                    f"exercicio {ex.get('start_time')} fora da "
                                                    f"janela acordada"))
            dur = ex.get("duration_min")
            durf = None
            if dur is not None:
                try:
                    durf = float(dur)
                except (ValueError, TypeError):
                    durf = None
                    rep.violations.append(Violation(idx, "exercise_duration_out", SEV_SOFT,
                                                    f"duracao invalida: {dur!r}"))
                else:
                    if not (r.exercise_duration_min[0] <= durf <= r.exercise_duration_min[1]):
                        rep.violations.append(Violation(idx, "exercise_duration_out", SEV_SOFT,
                                                        f"exercicio de {durf:.0f} min"))
            # Starts while awake but runs past bedtime -> ends mid-sleep, which
            # is physiologically impossible and would inject activity into the
            # sleep window. Checking the start alone misses it (22:30 + 120 min
            # ends at 00:30).
            if (start_in_window and st_norm is not None and durf is not None
                    and st_norm + durf > sleep_norm):
                rep.violations.append(Violation(idx, "exercise_ends_during_sleep", SEV_HARD,
                                                f"exercicio {ex.get('start_time')} +{durf:.0f}min "
                                                f"termina depois de dormir"))

        # carb function failed somewhere today -> flag and skip carb checks
        if carbs_per_meal is not None and not carbs_ok:
            rep.violations.append(Violation(idx, "carbs_uncomputable", SEV_SOFT,
                                            "funcao de carbo falhou; checagens de carbo puladas"))

        # exercise but ~no food = aberration (energy balance)
        if (has_exercise and carbs_per_meal is not None and carbs_ok
                and day_carbs < r.exercise_min_cho_g):
            rep.violations.append(Violation(idx, "exercise_no_food", SEV_HARD,
                                            f"exercicio com {day_carbs:.0f}g carbo no dia"))

        # --- soft distributional bounds ---
        if not (r.meals_per_day[0] <= n_meals <= r.meals_per_day[1]):
            rep.violations.append(Violation(idx, "meals_per_day_out", SEV_SOFT, f"{n_meals} refeicoes"))
        if (carbs_per_meal is not None and carbs_ok and n_meals > 0
                and not (r.cho_per_day[0] <= day_carbs <= r.cho_per_day[1])):
            rep.violations.append(Violation(idx, "cho_per_day_out", SEV_SOFT, f"{day_carbs:.0f}g/dia"))

    return rep

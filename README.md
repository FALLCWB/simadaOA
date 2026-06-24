# simada — Simulation of AID Adherence

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

A behavioral and dietary simulation framework for type 1 diabetes (T1D), built
on top of the [simglucose](https://github.com/jxx123/simglucose) implementation
of the UVA/Padova metabolic model. simada adds the patient-behavior and dietary
realism that automated-insulin-delivery (AID) controller evaluation requires,
while leaving the validated physiological core unchanged.

## What it adds

- **Adherence archetypes** (adherent, moderate, nonadherent) that modulate
  pre-bolus timing, carbohydrate-estimation error, insulin-on-board accounting,
  bolus omission, dietary fidelity, and exercise, plus a literature-grounded
  self-rescue response to sustained severe hyperglycemia.
- **Multi-country dietary patterns** for Brazil, the United States, and Japan,
  each grounded in a national food-composition table, with country-specific
  meal structure and named foods.
- **Language-model dietary scenario generation** that drafts week-long dietary
  libraries from each country's food vocabulary and validates them for
  vocabulary mapping, nutritional plausibility, and food diversity.
- **Calibration** of carbohydrate-ratio / correction-factor to each patient's
  physiology, and an archetype-aware basal-bolus baseline controller.

## Install

Requires Python 3.12+ and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

## Run

```bash
uv run simada run configs/scenarios/7day_cohort.yaml   # run a cohort simulation
uv run simada analyze results/                          # compute clinical metrics
uv run pytest                                           # run the test suite
```

The language-model scenario generation expects a local
[Ollama](https://ollama.com) runtime; pull the model with `ollama pull phi4`.

## Layout

```
src/simada/      framework (patient archetypes, meals, behavior, insulin,
                 scenario engine, controller, analysis, LLM pipeline)
configs/         YAML configuration (archetypes, meals, scenarios, insulin)
data/            reference data (food-composition tables, clinical targets)
tests/           unit, integration, property-based, and validation tests
paper_results/   per-physiology metrics and reproducer scripts for the
                 reported cohort results
```

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).

# Fully-crossed cohort — per-physiology glycemic metrics

Per-physiology glycemic metrics from the fully-crossed cohort, preserved so the
reported numbers are reproducible without re-running the cohort.

## Provenance

- Design: every one of the 30 standard simglucose physiologies (10 adults,
  10 adolescents, 10 children) simulated under each of the three adherence
  archetypes, for each of the three countries (Brazil, United States, Japan),
  90 days, pump therapy, stock `dopri5` integrator, master seed 42.
- 30 physiologies x 3 archetypes x 3 countries = 270 cells per arm
  (minus integration failures: rule 265/270, carbohydrate-matched LLM 269/270,
  raw LLM 270/270).

## Files

`x3rule_<archetype>.csv` rule-based meals; `x3llmcm_<archetype>.csv` LLM (phi4)
meals carbohydrate-matched to the rule arm; `x4llmraw_<archetype>.csv` LLM meals
as generated. Each row is one (country, archetype, physiology) cell on both the
true plasma-glucose (`basis=BG`) and sensor (`basis=CGM`) traces, with TIR, TBR,
TAR, CV, mean BG, GMI, LBGI, HBGI, MAGE, and severe-hypoglycemia counts.

## Reproduce

From this directory:

```
uv run --project .. python reproduce_equivalence.py
```

Prints completions, the pooled and physiology-clustered TOST equivalence of the
carbohydrate-matched LLM arm versus the rule arm, the individual
paired-difference distribution, the per-archetype clustered TOST, and the
LLM-vs-rule metric deltas.

## Integrator check

The cohort runs use simglucose's stock `dopri5` solver. `integrator_check.py`
(with `integrator_check_worker.py`) compares `dopri5` against the stiffness-aware
LSODA alternative on all thirty physiologies (controller in the loop, 30 days)
and reports the difference in the clinical metrics; `integrator_metric_diffs.csv`
holds the per-physiology result (Time in Range within 2.3 pp per physiology,
mean glucose within 3 mg/dL, CV ~1.3 pp at the cohort mean). Requires the
LSODA-reset behavior in `src/simada/patient/resilient.py`; production uses stock
`dopri5`.

```
uv run --project .. python integrator_check.py
```

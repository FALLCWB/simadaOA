"""Reproduce the equivalence statistics from the fully-crossed three-country cohort.

Run from this directory:  uv run python reproduce_equivalence.py

Data files (per-patient glycemic metrics, BG and CGM bases) in this directory:
  x3rule_<archetype>.csv    rule-based meals
  x3llmcm_<archetype>.csv   LLM (phi4) meals, carbohydrate-matched to the rule arm
  x4llmraw_<archetype>.csv  LLM (phi4) meals, as generated (raw)
Each *_<archetype>.csv holds the three countries (brazil, japan, usa) x 30
physiologies for that archetype; concatenating the three archetypes gives the
270-cell fully-crossed cohort (minus integration failures)."""
import os
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ARCH = ("adherent", "moderate", "nonadherent")
KEY = ["country", "archetype", "simglucose_name"]


def load(prefix):
    d = pd.concat([pd.read_csv(os.path.join(HERE, f"{prefix}_{a}.csv")) for a in ARCH])
    return d[d.basis == "BG"]


def tost(d, margin):
    n = len(d); m = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
    p = max(1 - stats.t.cdf((m + margin) / se, n - 1), stats.t.cdf((m - margin) / se, n - 1))
    return m, stats.t.interval(0.90, n - 1, loc=m, scale=se), p


rule, cm, raw = load("x3rule"), load("x3llmcm"), load("x4llmraw")
print(f"completions: rule {len(rule)}/270, carb-matched LLM {len(cm)}/270, raw LLM {len(raw)}/270")

m = rule.merge(cm, on=KEY, suffixes=("_r", "_l"))
m["dtir"] = m["tir_l"] - m["tir_r"]
d = m["dtir"].to_numpy()
mm, ci, p = tost(d, 3.0)
print(f"\nPooled TOST (n={len(d)}):       mean={mm:+.2f}pp 90%CI=({ci[0]:+.2f},{ci[1]:+.2f}) p={p:.1e}")
pc = m.groupby("simglucose_name")["dtir"].mean().to_numpy()
mm, ci, p = tost(pc, 3.0)
print(f"Physiology-clustered (n={len(pc)}): mean={mm:+.2f}pp 90%CI=({ci[0]:+.2f},{ci[1]:+.2f}) p={p:.1e}")
print(f"Individual diffs within +/-5pp: {100*np.mean(np.abs(d)<=5):.0f}%  (median {np.median(d):+.1f}, "
      f"5-95pct {np.percentile(d,5):+.1f}/{np.percentile(d,95):+.1f})")
for a in ARCH:
    s = m[m.archetype == a]; pa = s.groupby("simglucose_name")["dtir"].mean().to_numpy()
    mc, cc, pcl = tost(pa, 3.0)
    print(f"  {a:12s} clustered n={len(pa)} mean={mc:+.2f} 90%CI=({cc[0]:+.2f},{cc[1]:+.2f}) p={pcl:.1e}")

print("\nPer-cell equivalence (Table: country x archetype, carb-matched vs rule):")
for c in ("brazil", "usa", "japan"):
    for a in ARCH:
        s = m[(m.country == c) & (m.archetype == a)]
        mu, ci, p = tost(s["dtir"].to_numpy(), 3.0)
        flag = "" if p < 0.05 else " (CI exceeds 3pp)"
        print(f"  {c:7s} {a:12s} n={len(s):2d} dTIR={mu:+.1f} 90%CI=({ci[0]:+.1f},{ci[1]:+.1f}){flag}")

print("\nLLM-vs-rule deltas (LLM minus rule, mean over three countries):")
for label, arm in (("raw", raw), ("carb-matched", cm)):
    ma = rule.merge(arm, on=KEY, suffixes=("_r", "_l"))
    g = ma.assign(dtir=ma.tir_l - ma.tir_r, dcv=ma.cv_l - ma.cv_r).groupby("archetype")
    for a in ARCH:
        print(f"  {label:12s} {a:12s} dTIR={g.get_group(a).dtir.mean():+.1f}  dCV={g.get_group(a).dcv.mean():+.1f}")

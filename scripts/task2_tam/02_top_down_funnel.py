#!/usr/bin/env python
"""Task 2 — Top-down epidemiological funnel for US Camzyos TAM.

Reproduces every number in the Task 2 section of `docs/CASE_STUDY.md`.
Emits three CSVs, one PNG, and 17 hard-asserted sanity checks against the
rounded-k values in the case study.

Outputs (all in outputs/task2_tam/):
    02_top_down_tam_2026.csv    — headline MC TAM distribution (p10, p50, p90, mean)
    02_top_down_scenarios.csv   — Bear / Base / Bull deterministic point estimates
    02_top_down_over_time.csv   — TAM at 2028/2030 under 2%/4.7%/7.4% growth from MC median
    02_top_down_tam_2026.png    — MC distribution histogram with median + mode-product markers

Run: .venv/bin/python scripts/task2_tam/02_top_down_funnel.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.task2_tam.top_down import (
    FunnelParams,
    deterministic_scenarios,
    eligibility_triangle,
    over_time_from_anchor,
    plot_tam_distribution_2026,
    prevalence_triangle_2026,
    run_mc,
    variance_decomposition_log,
)

OUT = Path("outputs/task2_tam")
OUT.mkdir(parents=True, exist_ok=True)


def _pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q))


def _k(x: float) -> int:
    """Round to nearest thousand — the reporting unit used in docs/CASE_STUDY.md."""
    return int(round(x / 1000))


# ── 1. Parameters ────────────────────────────────────────────────────────────
print("=" * 78)
print("Task 2 — Top-down epidemiological funnel for US Camzyos TAM")
print("=" * 78)

p = FunnelParams()

print("\nParameters:")
print(f"  US adults ({p.anchor_year}):        {p.us_adults:>15,.0f}")
print(
    f"  Prevalence triangle:     ({p.prev_min:.0f}, {p.prev_mode:.0f}, {p.prev_max:.0f})  per 100k"
)
print(
    f"  Growth scenarios:        "
    f"floor {p.growth_floor:.1%} / base {p.growth_base:.1%} / ceiling {p.growth_ceiling:.1%}"
)
print(f"  Obstructive triangle:    ({p.obstr_min:.2f}, {p.obstr_mode:.2f}, {p.obstr_max:.2f})")
print(f"  Symptomatic triangle:    ({p.sympt_min:.2f}, {p.sympt_mode:.2f}, {p.sympt_max:.2f})")


# ── 2. Prevalence triangle (2026) ────────────────────────────────────────────
prev_lo, prev_mode, prev_hi = prevalence_triangle_2026(p)
print(f"\n1. Prevalence triangle at {p.anchor_year} (per 100k, source-anchored):")
print(f"   min  = {prev_lo:.0f}  (Husser 2018, Germany 2015)")
print(f"   mode = {prev_mode:.0f}  (Butzner 2021, US 2019)")
print(f"   max  = {prev_hi:.0f}  (Massera 2023 imaging-phenotype ceiling)")


# ── 3. Eligibility triangle ──────────────────────────────────────────────────
elo, emode, ehi = eligibility_triangle(p)
print("\n2. Eligibility triangle (obstructive × symptomatic):")
print(f"   min  = {p.obstr_min:.2f} × {p.sympt_min:.2f} = {elo:.3f}  → {elo * 100:.0f}%")
print(f"   mode = {p.obstr_mode:.2f} × {p.sympt_mode:.2f} = {emode:.3f}  → {emode * 100:.0f}%")
print(f"   max  = {p.obstr_max:.2f} × {p.sympt_max:.2f} = {ehi:.3f}  → {ehi * 100:.0f}%")


# ── 4. Monte Carlo — headline 2026 TAM ───────────────────────────────────────
mc = run_mc(p, seed=42, n_draws=10_000)
tam, prev_draws, elig_draws = mc["tam"], mc["prev"], mc["elig"]

med, p10, p90 = _pct(tam, 50), _pct(tam, 10), _pct(tam, 90)
print(f"\n3. TAM {p.anchor_year} — Monte Carlo (10,000 draws, seed 42):")
print(f"   Median:  {_k(med):>4}k    (exact: {med:>10,.0f})")
print(f"   80% CI:  [{_k(p10):>3}k, {_k(p90):>3}k]   (exact: [{p10:,.0f}, {p90:,.0f}])")
print(f"   Mean:    {_k(np.mean(tam)):>4}k    (exact: {np.mean(tam):>10,.0f})")


# ── 5. Deterministic Bear / Base / Bull ──────────────────────────────────────
print("\n4. Bear / Base / Bull (deterministic — product of chosen values):")
scenarios = deterministic_scenarios(p)
for row in scenarios:
    print(
        f"   {row['scenario']:<5}: prev={row['prev_per_100k']:>6.1f}/100k, "
        f"elig={row['elig_fraction']:.3f} → TAM = {_k(row['tam']):>4}k  "
        f"(exact: {row['tam']:>7,})"
    )

base_tam_k = _k(scenarios[1]["tam"])
print(
    f"\n   Skew note: MC median ({_k(med)}k) > deterministic base ({base_tam_k}k) "
    f"because the prevalence triangle is right-skewed (mode {prev_mode:.0f}, max {prev_hi:.0f})."
)


# ── 6. Variance decomposition ────────────────────────────────────────────────
vd = variance_decomposition_log(prev_draws, elig_draws)
print("\n5. Variance decomposition of log(TAM):")
print(f"   Prevalence:  {vd['prevalence'] * 100:>5.1f}%")
print(f"   Eligibility: {vd['eligibility'] * 100:>5.1f}%")


# ── 7. Over-time table ───────────────────────────────────────────────────────
rows = over_time_from_anchor(med, p)
print(
    f"\n6. Over-time (point estimates, mode eligibility = {emode:.2f}, "
    f"growth from {p.anchor_year} mode prevalence):"
)
years = p.target_years
print(f"   {'scenario':<18} {'rate':>7}   " + "".join(f"{y:>8}" for y in years))
for scenario_key in ["floor_2pct", "base_4_7pct", "ceiling_7_4pct"]:
    scenario_rows = [r for r in rows if r["scenario"] == scenario_key]
    g = scenario_rows[0]["growth_rate"]
    k_vals = [f"{_k(r['tam'])}k" for r in scenario_rows]
    print(f"   {scenario_key:<18} {g:>7.1%}   " + "".join(f"{v:>8}" for v in k_vals))


# ── 8. Save outputs ──────────────────────────────────────────────────────────
mc_summary = pd.DataFrame(
    [
        {
            "quantity": "tam_2026_us_patients",
            "p10": int(p10),
            "p50": int(med),
            "p90": int(p90),
            "mean": int(np.mean(tam)),
            "n_draws": len(tam),
            "seed": 42,
        }
    ]
)
mc_summary.to_csv(OUT / "02_top_down_tam_2026.csv", index=False)

pd.DataFrame(scenarios).to_csv(OUT / "02_top_down_scenarios.csv", index=False)
pd.DataFrame(rows).to_csv(OUT / "02_top_down_over_time.csv", index=False)

plot_path = plot_tam_distribution_2026(tam, p, OUT / "02_top_down_tam_2026.png")

print("\n7. Outputs:")
for path in [
    OUT / "02_top_down_tam_2026.csv",
    OUT / "02_top_down_scenarios.csv",
    OUT / "02_top_down_over_time.csv",
    plot_path,
]:
    print(f"   → {path}")


# ── 9. Sanity checks against docs/CASE_STUDY.md ───────────────────────────
# Every doc number is rounded to the nearest 1,000 and reported in "k".
# These checks demand EXACT match on rounded-k values (no tolerance).
print("\n8. Sanity checks against docs/CASE_STUDY.md (exact rounded-k match):")


def _check_k(name: str, actual: float, expected_k: int) -> None:
    """Round `actual` to nearest 1k and demand exact equality with `expected_k`."""
    actual_k = _k(actual)
    ok = actual_k == expected_k
    marker = "PASS" if ok else "FAIL"
    print(f"   [{marker}] {name}: {actual_k}k  vs doc {expected_k}k  (exact: {actual:,.0f})")
    if not ok:
        raise AssertionError(f"{name} mismatch: {actual_k}k vs doc {expected_k}k")


def _check_pct(name: str, actual_fraction: float, expected_pct: int) -> None:
    """Round `actual_fraction` to nearest whole percent and demand exact equality."""
    actual_pct = int(round(actual_fraction * 100))
    ok = actual_pct == expected_pct
    marker = "PASS" if ok else "FAIL"
    print(f"   [{marker}] {name}: {actual_pct}%  vs doc {expected_pct}%")
    if not ok:
        raise AssertionError(f"{name} mismatch: {actual_pct}% vs doc {expected_pct}%")


# Headline TAM (2026)
_check_k("TAM 2026 median", med, 127)
_check_k("TAM 2026 p10", p10, 84)
_check_k("TAM 2026 p90", p90, 195)

# Bear / Base / Bull
_check_k("Bear TAM", scenarios[0]["tam"], 41)
_check_k("Base TAM", scenarios[1]["tam"], 94)
_check_k("Bull TAM", scenarios[2]["tam"], 340)

# Over-time — every cell of the 3-scenario × 3-year table.
# All scenarios share the 2026 mode-anchor (94k); they diverge from 2026 forward.
expected_over_time = {
    ("floor_2pct", 2026): 127,
    ("floor_2pct", 2028): 132,
    ("floor_2pct", 2030): 137,
    ("base_4_7pct", 2026): 127,
    ("base_4_7pct", 2028): 139,
    ("base_4_7pct", 2030): 152,
    ("ceiling_7_4pct", 2026): 127,
    ("ceiling_7_4pct", 2028): 146,
    ("ceiling_7_4pct", 2030): 169,
}
for r in rows:
    key = (r["scenario"], r["year"])
    _check_k(f"Over-time {r['scenario']} {r['year']}", r["tam"], expected_over_time[key])

# Variance shares
_check_pct("Prevalence variance share", vd["prevalence"], 58)
_check_pct("Eligibility variance share", vd["eligibility"], 42)

print("\nAll checks passed. Doc and script are byte-identical on rounded-k values.")

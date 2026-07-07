#!/usr/bin/env python
"""Task 2 — Total addressable market (TAM) Monte Carlo pipeline.

Grounded estimate of the US Camzyos addressable pool under two definitions:

    Pool A — theoretical ceiling (symptomatic oHCM in the US, diagnosed or not)
    Pool B — diagnosed & treatable today (anchored on Butzner 2021 US claims)

Plus a simple penetration forecast of on-drug patients and US revenue through
2030, anchored to observed BMS quarterly figures.

Every prior is documented in `outputs/task2_tam/09_tam_sources.csv` with
citation and source type ('peer_reviewed', 'company_disclosure',
'external_default', 'user_data_needed'). External defaults and
user-data-needed flags are the first things the IC should challenge.

Outputs (all under `outputs/task2_tam/`):
    09_tam_sources.csv          — every prior + source
    09_tam_forecast_summary.csv — median + 80% CI per forecast year
    09_tam_pools.png            — Pool A vs Pool B distributions
    09_tam_fanchart_patients.png
    09_tam_fanchart_revenue.png
    09_tam_tornado.png          — one-at-a-time sensitivity on each pool

Run from repo root:
    .venv/bin/python scripts/task2_tam/09_tam_monte_carlo.py
"""

import sys
from pathlib import Path

# Repo root is two levels above this file (scripts/task2_tam/foo.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.task2_tam.diffusion import bms_implied_on_drug_series
from src.task2_tam.priors import ExternalAssumptions, sources_dataframe
from src.task2_tam.sensitivity import tornado_for_pool_a, tornado_for_pool_b
from src.task2_tam.simulation import run_simulation

# ── Palette (repo convention) ─────────────────────────────────────────────────
C_MEDIAN = "#2a78d6"
C_FAN_INNER = "#7cb0e6"
C_FAN_OUTER = "#c3d8ef"
C_ALT = "#1baf7a"
C_HIST = "#0b0b0b"
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
INK_PRI = "#0b0b0b"
INK_SEC = "#52514e"
INK_MUT = "#898781"

OUTPUT_DIR = Path("outputs/task2_tam")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _rule(title: str):
    print(f"\n{'─' * 78}\n{title}\n{'─' * 78}")


# ── 1. Source audit ───────────────────────────────────────────────────────────
_rule("1. Source audit trail")
sources = sources_dataframe()
assert sources["source"].astype(str).str.len().gt(0).all()
sources_path = OUTPUT_DIR / "09_tam_sources.csv"
sources.to_csv(sources_path, index=False)
print(f"  {len(sources)} rows → {sources_path}")

flagged = sources[sources["source_type"].isin(["external_default", "user_data_needed"])]
print(f"\n  {len(flagged)} priors flagged for challenge (external_default / user_data_needed):")
for _, r in flagged.iterrows():
    print(f"    [{r['source_type']:>18}] {r['name']}")

# ── 2. Run Monte Carlo ────────────────────────────────────────────────────────
_rule("2. Monte Carlo simulation")
assumptions = ExternalAssumptions()
print(
    f"  n_draws={assumptions.n_monte_carlo_draws:,}, "
    f"horizon={assumptions.launch_month}→{assumptions.forecast_end_month}"
)
sim = run_simulation(assumptions)

# ── 3. Pool A vs Pool B distributions ─────────────────────────────────────────
_rule("3. Two definitions of 'addressable'")


def _summ(series: pd.Series) -> tuple[float, float, float]:
    return (
        float(np.percentile(series, 10)),
        float(np.percentile(series, 50)),
        float(np.percentile(series, 90)),
    )


a10, a50, a90 = _summ(sim.funnel["pool_a_theoretical"])
b10, b50, b90 = _summ(sim.funnel["pool_b_diagnosed_today"])
gap10, gap50, gap90 = _summ(sim.funnel["undiagnosed_gap"])

print(f"  Pool A (theoretical ceiling):  p50 {a50:>10,.0f}   80% CI [{a10:,.0f}, {a90:,.0f}]")
print(f"  Pool B (diagnosed today):      p50 {b50:>10,.0f}   80% CI [{b10:,.0f}, {b90:,.0f}]")
print(f"  Undiagnosed gap  (A−B):        p50 {gap50:>10,.0f}   80% CI [{gap10:,.0f}, {gap90:,.0f}]")

# ── 4. Backcast — does model 2024 exit-rate contain BMS-implied ~10k? ─────────
_rule("4. Backcast check — 2024 exit run-rate")
bms_implied = bms_implied_on_drug_series(net_price_per_year_usd=75_000)
print("  BMS-implied prevalent US patients on Camzyos (from quarterly US revenue):")
for pd_period, val in bms_implied.items():
    print(f"    {pd_period}:  {val:>7,.0f}")

target_month = pd.Period("2024-12", freq="M")
mask = sim.month_grid.to_period("M") == target_month
if not mask.any():
    raise RuntimeError(f"{target_month} not in grid")
i = int(np.where(mask)[0][0])
mp10, mp50, mp90 = np.percentile(sim.patients_by_draw[:, i], [10, 50, 90])
bms_2024 = float(bms_implied.loc[target_month])
print(f"\n  Model 2024-12: p10 {mp10:,.0f} / p50 {mp50:,.0f} / p90 {mp90:,.0f}")
print(f"  BMS-implied 2024-12: {bms_2024:,.0f}")
if mp10 <= bms_2024 <= mp90:
    print("  ✓ BMS-implied 2024 exit-rate within model 80% CI")
else:
    print("  ⚠ BMS-implied outside model 80% CI — flag in writeup limitations")

# ── 5. Forecast summary ───────────────────────────────────────────────────────
_rule("5. Forecast summary")
summary = sim.summary_table(years=(2025, 2028, 2030))
summary_path = OUTPUT_DIR / "09_tam_forecast_summary.csv"
summary.to_csv(summary_path)
print(summary.to_string())
print(f"\n  → {summary_path}")

# ── 6. Tornado sensitivity ────────────────────────────────────────────────────
_rule("6. Tornado sensitivity (one-at-a-time)")
tor_a = tornado_for_pool_a()
tor_b = tornado_for_pool_b()
print("Pool A:")
print(tor_a.round(0).to_string())
print("\nPool B:")
print(tor_b.round(0).to_string())

# ── 7. Figures ────────────────────────────────────────────────────────────────
_rule("7. Building figures")


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK_MUT, labelsize=9)


# 7a. Pool A vs B histogram
fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=SURFACE)
bins = np.linspace(0, max(a90, sim.funnel["pool_a_theoretical"].quantile(0.99)), 60)
ax.hist(
    sim.funnel["pool_a_theoretical"],
    bins=bins,
    color=C_MEDIAN,
    alpha=0.6,
    label=f"Pool A (theoretical ceiling)\nmedian {a50:,.0f}, 80% CI [{a10:,.0f}, {a90:,.0f}]",
)
ax.hist(
    sim.funnel["pool_b_diagnosed_today"],
    bins=bins,
    color=C_ALT,
    alpha=0.6,
    label=f"Pool B (diagnosed today)\nmedian {b50:,.0f}, 80% CI [{b10:,.0f}, {b90:,.0f}]",
)
ax.axvline(a50, color=C_MEDIAN, linewidth=1.2, linestyle="--")
ax.axvline(b50, color=C_ALT, linewidth=1.2, linestyle="--")
ax.set_xlabel("US Camzyos-addressable patients", fontsize=10, color=INK_SEC)
ax.set_ylabel("Monte Carlo density (n=10k draws)", fontsize=10, color=INK_SEC)
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x / 1000)}k"))
ax.legend(fontsize=9, frameon=False, loc="upper right")
_style_axes(ax)
ax.set_title(
    "Two definitions of 'addressable' — Pool A vs Pool B",
    fontsize=12,
    fontweight="bold",
    color=INK_PRI,
    loc="left",
    pad=10,
)
ax.text(
    0.98,
    0.55,
    "The gap between A and B is the undiagnosed pool.\n"
    "Camzyos' 5–10 year growth is gated by how fast\n"
    "this gap closes (rising diagnosis rate).",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=8.5,
    color=INK_SEC,
    bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=GRID, lw=0.8),
)
plt.tight_layout()
fp = OUTPUT_DIR / "09_tam_pools.png"
plt.savefig(fp, dpi=150, bbox_inches="tight", facecolor=SURFACE)
plt.close()
print(f"  {fp}")

# 7b. Fan chart — prevalent patients
fig, ax = plt.subplots(figsize=(11, 6), facecolor=SURFACE)
pct = sim.prevalent_percentiles((10, 25, 50, 75, 90))
ax.fill_between(sim.month_grid, pct["p10"], pct["p90"], color=C_FAN_OUTER, label="80% CI")
ax.fill_between(sim.month_grid, pct["p25"], pct["p75"], color=C_FAN_INNER, label="50% CI")
ax.plot(sim.month_grid, pct["p50"], color=C_MEDIAN, linewidth=2, label="Model median")

for period, val in bms_implied.items():
    ax.scatter([period.to_timestamp()], [val], color=C_HIST, s=45, zorder=6, marker="D")

handles = [
    Line2D([], [], color=C_MEDIAN, linewidth=2, label="Model median"),
    Patch(color=C_FAN_INNER, label="Model 50% CI"),
    Patch(color=C_FAN_OUTER, label="Model 80% CI"),
    Line2D([], [], color=C_HIST, marker="D", linestyle="", label="BMS-implied (from US revenue)"),
]
ax.legend(handles=handles, fontsize=9, frameon=False, loc="upper left")

pdufa_ts = pd.Timestamp(assumptions.aficamten_pdufa_month)
ax.axvline(pdufa_ts, color=INK_MUT, linestyle=":", linewidth=1)
ax.text(
    pdufa_ts,
    ax.get_ylim()[1] * 0.03,
    " aficamten PDUFA (assumed)",
    rotation=90,
    va="bottom",
    ha="left",
    fontsize=8,
    color=INK_MUT,
)

_style_axes(ax)
ax.grid(color=GRID, linewidth=0.7)
ax.set_xlabel("Month", fontsize=10, color=INK_SEC)
ax.set_ylabel("Prevalent US patients on Camzyos", fontsize=10, color=INK_SEC)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.set_title(
    "US Camzyos prevalent-patient forecast — Monte Carlo (n=10,000)",
    fontsize=12,
    fontweight="bold",
    color=INK_PRI,
    loc="left",
    pad=10,
)
plt.tight_layout()
fp = OUTPUT_DIR / "09_tam_fanchart_patients.png"
plt.savefig(fp, dpi=150, bbox_inches="tight", facecolor=SURFACE)
plt.close()
print(f"  {fp}")

# 7c. Fan chart — annual revenue
fig, ax = plt.subplots(figsize=(11, 6), facecolor=SURFACE)
r_pct = sim.revenue_percentiles((10, 25, 50, 75, 90))
years = r_pct.index.values
ax.fill_between(years, r_pct["p10"], r_pct["p90"], color=C_FAN_OUTER, label="80% CI")
ax.fill_between(years, r_pct["p25"], r_pct["p75"], color=C_FAN_INNER, label="50% CI")
ax.plot(years, r_pct["p50"], color=C_MEDIAN, linewidth=2, marker="o", label="Model median")

bms_actual = {2023: 226.0, 2024: 543.0}
for y, v in bms_actual.items():
    ax.scatter(
        [y],
        [v],
        color=C_HIST,
        s=55,
        zorder=6,
        marker="D",
        label="BMS reported (US)" if y == 2023 else None,
    )

ax.legend(fontsize=9, frameon=False, loc="upper left")
_style_axes(ax)
ax.grid(color=GRID, linewidth=0.7)
ax.set_xlabel("Calendar year", fontsize=10, color=INK_SEC)
ax.set_ylabel("US Camzyos net revenue (USD millions)", fontsize=10, color=INK_SEC)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${int(x):,}M"))
ax.set_title(
    "US Camzyos annual net revenue forecast — Monte Carlo",
    fontsize=12,
    fontweight="bold",
    color=INK_PRI,
    loc="left",
    pad=10,
)
plt.tight_layout()
fp = OUTPUT_DIR / "09_tam_fanchart_revenue.png"
plt.savefig(fp, dpi=150, bbox_inches="tight", facecolor=SURFACE)
plt.close()
print(f"  {fp}")

# 7d. Tornado — both pools side by side
fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=SURFACE)
for ax_, tor, title, colour in zip(
    axes,
    [tor_a, tor_b],
    ["Pool A (theoretical ceiling)", "Pool B (diagnosed today)"],
    [C_MEDIAN, C_ALT],
):
    y = np.arange(len(tor))
    baseline = tor["baseline"].iloc[0]
    ax_.barh(y, tor["high"] - baseline, left=baseline, color=colour, alpha=0.85, label="High (p95)")
    ax_.barh(y, tor["low"] - baseline, left=baseline, color=colour, alpha=0.4, label="Low (p05)")
    ax_.axvline(baseline, color=INK_PRI, linewidth=1.2)
    ax_.set_yticks(y)
    ax_.set_yticklabels(tor.index, fontsize=9)
    ax_.invert_yaxis()
    ax_.set_xlabel("US patients", fontsize=9, color=INK_SEC)
    ax_.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x / 1000)}k"))
    ax_.set_title(title, fontsize=11, fontweight="bold", color=INK_PRI, loc="left")
    ax_.legend(fontsize=8, frameon=False, loc="lower right")
    _style_axes(ax_)
    ax_.grid(axis="x", color=GRID, linewidth=0.7)

fig.suptitle(
    "Tornado sensitivity — which input moves the pool most?",
    fontsize=12,
    fontweight="bold",
    color=INK_PRI,
    x=0.02,
    ha="left",
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fp = OUTPUT_DIR / "09_tam_tornado.png"
plt.savefig(fp, dpi=150, bbox_inches="tight", facecolor=SURFACE)
plt.close()
print(f"  {fp}")

# ── 8. IC talking points ──────────────────────────────────────────────────────
_rule("8. IC talking points")
print(f"  Pool A (theoretical): median {a50:,.0f}, 80% CI [{a10:,.0f}, {a90:,.0f}]")
print(f"  Pool B (diagnosed today): median {b50:,.0f}, 80% CI [{b10:,.0f}, {b90:,.0f}]")
r_pct_final = sim.revenue_percentiles()
peak_year = int(r_pct_final.idxmax()["p50"])
peak_rev = float(r_pct_final.loc[peak_year, "p50"])
print(f"  Peak US revenue (median): {peak_year}, ${peak_rev:,.0f}M")
top_lever_a = tor_a["swing"].idxmax()
top_lever_b = tor_b["swing"].idxmax()
print(f"  Top sensitivity — Pool A: {top_lever_a}")
print(f"  Top sensitivity — Pool B: {top_lever_b}")

print("\nAll assertions passed. Pipeline complete.")

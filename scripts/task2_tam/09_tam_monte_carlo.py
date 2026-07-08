#!/usr/bin/env python
"""Task 2 — Total addressable market (TAM) pipeline.

Two Monte Carlo runs, side by side, so the IC can see what the claims data
actually changes:

    1. Prior-predictive: samples the literature-anchored priors only.
       This is a *forward simulation* / uncertainty propagation — no data
       enters the numbers. Historically we called this "the Monte Carlo
       TAM"; per the task brief we now label it precisely as
       prior-predictive.

    2. Posterior-predictive: same funnel, but the Camzyos-eligible fraction
       is replaced with the Beta-Binomial posterior updated on our 30k-patient
       claims cohort. This is the Bayesian update the brief actually asks for
       ("your claims data is one of the sources — combine them").

The prior on the eligible fraction (Beta, ESS ≈ 47) is much lighter than the
data (n ≈ 19,000 HCM-coded patients), so the posterior is data-dominated.
Three claims-based definitions of "treatable oHCM" are shown so the IC can
see how proxy choice moves the posterior. All three are strict LOWER BOUNDS
on true clinical eligibility because claims routinely under-code symptoms
(see FINDINGS F23 / METHODS.md).

Outputs (all under `outputs/task2_tam/`):
    09_tam_sources.csv              — every prior + source
    09_tam_forecast_summary.csv     — prior + posterior median + 80% CI per year
    09_tam_pools.png                — Pool A/B distributions (both runs)
    09_tam_fanchart_patients.png    — on-drug fan chart (both runs)
    09_tam_fanchart_revenue.png     — revenue fan chart (both runs)
    09_tam_tornado.png              — one-at-a-time sensitivity
    09_tam_bayes_update.png         — prior → data → posterior explainer
    09_tam_bayes_summary.csv        — posterior summary (3 definitions)

Run from repo root:
    .venv/bin/python scripts/task2_tam/09_tam_monte_carlo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

from src.task1_adoption.data_loading import load_data
from src.task2_tam.bayesian_update import update_eligible_fraction
from src.task2_tam.claims_evidence import collect_evidence, evidence_dataframe
from src.task2_tam.diffusion import bms_implied_on_drug_series
from src.task2_tam.priors import ExternalAssumptions, sources_dataframe
from src.task2_tam.sensitivity import tornado_for_pool_a, tornado_for_pool_b
from src.task2_tam.simulation import run_simulation

# ── Palette (repo convention) ─────────────────────────────────────────────────
C_PRIOR = "#c3d8ef"  # light blue — prior-predictive
C_MEDIAN = "#2a78d6"  # blue      — posterior median line
C_FAN_INNER = "#7cb0e6"
C_FAN_OUTER = "#c3d8ef"
C_POST = "#104281"  # deep blue — posterior fill
C_ALT = "#1baf7a"  # green     — Pool B / posterior histogram
C_DATA = "#e88a1a"  # amber     — data / likelihood
C_HIST = "#0b0b0b"
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
INK_PRI = "#0b0b0b"
INK_SEC = "#52514e"
INK_MUT = "#898781"

OUTPUT_DIR = Path("outputs/task2_tam")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _rule(title: str) -> None:
    print(f"\n{'─' * 78}\n{title}\n{'─' * 78}")


def _summ(series) -> tuple[float, float, float]:
    return (
        float(np.percentile(series, 10)),
        float(np.percentile(series, 50)),
        float(np.percentile(series, 90)),
    )


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK_MUT, labelsize=9)


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

# ── 2. Bayesian update — the eligible-fraction posterior ──────────────────────
_rule("2. Bayesian update on the Camzyos-eligible fraction")
print("  Combining literature prior with our 30k-patient claims cohort.")
print("  Prior:     Beta(α=14.3, β=32.3), mean 0.307  (specialty registry, k_eff ≈ 47)")
print("  Data:      Binomial(n, k) from HCM-coded patients meeting a")
print("             claims-based 'treatable oHCM' proxy")
print("  Posterior: Beta(α+k, β+n-k)   —  conjugate, closed-form")
print()

data = load_data()
evidence = collect_evidence(data)
ev_df = evidence_dataframe(evidence)
print(ev_df.drop(columns=["description"]).to_string(index=False))

# Compute posterior for each definition; use the strict definition (D1) as primary.
posteriors = [update_eligible_fraction(e.k, e.n, e.label) for e in evidence]
primary_posterior = posteriors[0]

print("\n  Posterior summaries (all three definitions):")
print(f"    {'definition':<40} {'post mean':>10} {'post 90% CI':>22} {'prior→post shift':>18}")
for post in posteriors:
    shift = post.posterior_mean / post.prior_mean
    print(
        f"    {post.definition_label:<40} "
        f"{post.posterior_mean:>10.4f} "
        f"[{post.posterior_p05:.4f}, {post.posterior_p95:.4f}] "
        f"{shift:>18.2%}"
    )

# Persist the Bayesian summary for reference.
bayes_rows = []
for post in posteriors:
    bayes_rows.append(
        {
            "definition": post.definition_label,
            "k": post.k,
            "n": post.n,
            "k_over_n": round(post.k / post.n, 4),
            "prior_alpha": round(post.prior_alpha, 3),
            "prior_beta": round(post.prior_beta, 3),
            "prior_mean": round(post.prior_mean, 4),
            "prior_ess": round(post.prior_ess, 2),
            "posterior_alpha": round(post.posterior_alpha, 3),
            "posterior_beta": round(post.posterior_beta, 3),
            "posterior_mean": round(post.posterior_mean, 4),
            "posterior_p05": round(post.posterior_p05, 4),
            "posterior_p95": round(post.posterior_p95, 4),
        }
    )
bayes_path = OUTPUT_DIR / "09_tam_bayes_summary.csv"
pd.DataFrame(bayes_rows).to_csv(bayes_path, index=False)
print(f"\n  → {bayes_path}")

print("\n  Important caveat: our proxy is a strict LOWER BOUND on true clinical")
print("  eligibility. Claims routinely miss symptom severity, so the posterior")
print("  answers 'what fraction shows up as treatable in this claims database',")
print("  not 'what fraction is truly clinically eligible'. The literature prior")
print("  (~30%) captures the latter in a well-coded referral setting. Report both.")

# ── 3. Two Monte Carlo runs: prior-predictive + posterior-predictive ──────────
_rule("3. Prior-predictive vs posterior-predictive TAM")
assumptions = ExternalAssumptions()
print(
    f"  n_draws={assumptions.n_monte_carlo_draws:,}, "
    f"horizon={assumptions.launch_month}→{assumptions.forecast_end_month}"
)

sim_prior = run_simulation(assumptions)
sim_post = run_simulation(assumptions, eligible_fraction_dist=primary_posterior.posterior)


def _print_pools(label: str, sim) -> tuple[float, float, float, float, float, float]:
    a10, a50, a90 = _summ(sim.funnel["pool_a_theoretical"])
    b10, b50, b90 = _summ(sim.funnel["pool_b_diagnosed_today"])
    print(f"  [{label}]")
    print(f"    Pool A (theoretical ceiling):  p50 {a50:>10,.0f}   80% CI [{a10:,.0f}, {a90:,.0f}]")
    print(f"    Pool B (diagnosed today):      p50 {b50:>10,.0f}   80% CI [{b10:,.0f}, {b90:,.0f}]")
    return a10, a50, a90, b10, b50, b90


pa10, pa50, pa90, pb10, pb50, pb90 = _print_pools("prior-predictive", sim_prior)
qa10, qa50, qa90, qb10, qb50, qb90 = _print_pools("posterior-predictive (D1 strict)", sim_post)

# ── 4. Backcast on the prior-predictive run ───────────────────────────────────
_rule("4. Backcast check — 2024 exit run-rate (prior-predictive)")
bms_implied = bms_implied_on_drug_series(net_price_per_year_usd=75_000)
print("  BMS-implied prevalent US patients on Camzyos (from quarterly US revenue):")
for pd_period, val in bms_implied.items():
    print(f"    {pd_period}:  {val:>7,.0f}")

target_month = pd.Period("2024-12", freq="M")
mask = sim_prior.month_grid.to_period("M") == target_month
if not mask.any():
    raise RuntimeError(f"{target_month} not in grid")
i = int(np.where(mask)[0][0])
mp10, mp50, mp90 = np.percentile(sim_prior.patients_by_draw[:, i], [10, 50, 90])
bms_2024 = float(bms_implied.loc[target_month])
print(f"\n  Model 2024-12: p10 {mp10:,.0f} / p50 {mp50:,.0f} / p90 {mp90:,.0f}")
print(f"  BMS-implied 2024-12: {bms_2024:,.0f}")
if mp10 <= bms_2024 <= mp90:
    print("  ✓ BMS-implied 2024 exit-rate within prior-predictive 80% CI")
else:
    print("  ⚠ BMS-implied outside prior-predictive 80% CI — flag in writeup limitations")

# ── 5. Forecast summary (both runs) ───────────────────────────────────────────
_rule("5. Forecast summary — prior-predictive vs posterior-predictive")


def _merge_summaries(sim_prior, sim_post, years=(2025, 2028, 2030)) -> pd.DataFrame:
    prior = sim_prior.summary_table(years).rename(
        columns={c: f"prior_{c}" for c in sim_prior.summary_table(years).columns}
    )
    post = sim_post.summary_table(years).rename(
        columns={c: f"post_{c}" for c in sim_post.summary_table(years).columns}
    )
    return prior.join(post)


summary = _merge_summaries(sim_prior, sim_post)
summary_path = OUTPUT_DIR / "09_tam_forecast_summary.csv"
summary.to_csv(summary_path)
print(summary.to_string())
print(f"\n  → {summary_path}")

# ── 6. Tornado sensitivity ────────────────────────────────────────────────────
_rule("6. Tornado sensitivity (one-at-a-time, prior-predictive)")
tor_a = tornado_for_pool_a()
tor_b = tornado_for_pool_b()
print("Pool A:")
print(tor_a.round(0).to_string())
print("\nPool B:")
print(tor_b.round(0).to_string())

# ── 7. Figures ────────────────────────────────────────────────────────────────
_rule("7. Building figures")

# 7a. Bayesian update explainer — prior / data likelihood / posterior
fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=SURFACE)
xs = np.linspace(0, 0.6, 500)

# Prior distribution
prior_pdf = stats.beta(primary_posterior.prior_alpha, primary_posterior.prior_beta).pdf(xs)
ax.plot(xs, prior_pdf, color=C_PRIOR, linewidth=2.5, label="Prior (literature, α=14.3, β=32.3)")
ax.fill_between(xs, 0, prior_pdf, color=C_PRIOR, alpha=0.35)

# Posterior distribution (primary = D1 strict)
post_pdf = primary_posterior.posterior.pdf(xs)
ax.plot(
    xs,
    post_pdf,
    color=C_POST,
    linewidth=2.5,
    label=(
        f"Posterior (D1 strict, n={primary_posterior.n:,}, k={primary_posterior.k}, "
        f"α={primary_posterior.posterior_alpha:.0f}, β={primary_posterior.posterior_beta:.0f})"
    ),
)
ax.fill_between(xs, 0, post_pdf, color=C_POST, alpha=0.35)

# Data likelihood mark: vertical dashed for each definition
for post, alpha_line in zip(posteriors, (1.0, 0.6, 0.3)):
    rate = post.k / post.n
    ax.axvline(rate, color=C_DATA, linestyle=":", linewidth=1.5, alpha=alpha_line)
    ax.text(
        rate + 0.003,
        ax.get_ylim()[1] * 0.02 if ax.get_ylim()[1] > 0 else 5,
        f"{post.definition_label}\nk/n = {rate:.3f}",
        fontsize=7.5,
        color=C_DATA,
        alpha=alpha_line,
        rotation=90,
        va="bottom",
        ha="left",
    )

ax.axvline(primary_posterior.prior_mean, color=INK_MUT, linestyle="--", linewidth=1)
ax.text(
    primary_posterior.prior_mean + 0.005,
    max(prior_pdf) * 0.9,
    f"Prior mean {primary_posterior.prior_mean:.3f}",
    color=INK_SEC,
    fontsize=8.5,
    va="top",
)

ax.set_xlabel("Camzyos-eligible fraction of HCM patients", fontsize=10, color=INK_SEC)
ax.set_ylabel("Density", fontsize=10, color=INK_SEC)
ax.set_title(
    "Bayesian update — prior meets data, out comes the posterior",
    fontsize=12,
    fontweight="bold",
    color=INK_PRI,
    loc="left",
    pad=10,
)
ax.legend(fontsize=9, frameon=False, loc="upper right")
_style_axes(ax)

ax.text(
    0.98,
    0.60,
    (
        "Prior effective sample size ≈ 47 (literature).\n"
        f"Data sample size = {primary_posterior.n:,} HCM-coded patients.\n"
        "Ratio ≈ 400× — the data dominates the prior.\n\n"
        "Caveat: k/n is a claims-observable LOWER BOUND on true\n"
        "clinical eligibility; claims routinely under-code symptoms."
    ),
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=8.5,
    color=INK_SEC,
    bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=GRID, lw=0.8),
)

plt.tight_layout()
fp = OUTPUT_DIR / "09_tam_bayes_update.png"
plt.savefig(fp, dpi=150, bbox_inches="tight", facecolor=SURFACE)
plt.close()
print(f"  {fp}")

# 7b. Pool A/B — prior-predictive vs posterior-predictive
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=SURFACE)
for ax_, pool_col, title, colour in zip(
    axes,
    ["pool_a_theoretical", "pool_b_diagnosed_today"],
    ["Pool A (theoretical ceiling)", "Pool B (diagnosed today)"],
    [C_MEDIAN, C_ALT],
):
    prior_vals = sim_prior.funnel[pool_col]
    post_vals = sim_post.funnel[pool_col]
    upper = max(prior_vals.quantile(0.99), post_vals.quantile(0.99))
    bins = np.linspace(0, upper, 60)
    p10, p50, p90 = _summ(prior_vals)
    q10, q50, q90 = _summ(post_vals)
    ax_.hist(
        prior_vals,
        bins=bins,
        color=C_PRIOR,
        alpha=0.75,
        label=f"Prior-predictive  p50 {p50:,.0f}\n80% CI [{p10:,.0f}, {p90:,.0f}]",
    )
    ax_.hist(
        post_vals,
        bins=bins,
        color=colour,
        alpha=0.65,
        label=f"Posterior-predictive  p50 {q50:,.0f}\n80% CI [{q10:,.0f}, {q90:,.0f}]",
    )
    ax_.axvline(p50, color=C_PRIOR, linewidth=1.2, linestyle="--")
    ax_.axvline(q50, color=colour, linewidth=1.4, linestyle="--")
    ax_.set_xlabel("US Camzyos-addressable patients", fontsize=10, color=INK_SEC)
    ax_.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x / 1000)}k"))
    ax_.legend(fontsize=8.5, frameon=False, loc="upper right")
    _style_axes(ax_)
    ax_.set_title(title, fontsize=11, fontweight="bold", color=INK_PRI, loc="left")

axes[0].set_ylabel("Monte Carlo density (n=10k draws)", fontsize=10, color=INK_SEC)
fig.suptitle(
    "What the claims data changes — Pool A vs Pool B, prior vs posterior",
    fontsize=12.5,
    fontweight="bold",
    color=INK_PRI,
    x=0.02,
    ha="left",
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fp = OUTPUT_DIR / "09_tam_pools.png"
plt.savefig(fp, dpi=150, bbox_inches="tight", facecolor=SURFACE)
plt.close()
print(f"  {fp}")

# 7c. Fan chart — prevalent patients (both runs overlaid)
fig, ax = plt.subplots(figsize=(11, 6), facecolor=SURFACE)
pct_prior = sim_prior.prevalent_percentiles((10, 25, 50, 75, 90))
pct_post = sim_post.prevalent_percentiles((10, 25, 50, 75, 90))

ax.fill_between(
    sim_prior.month_grid,
    pct_prior["p10"],
    pct_prior["p90"],
    color=C_PRIOR,
    alpha=0.5,
    label="Prior-predictive 80% CI",
)
ax.plot(
    sim_prior.month_grid,
    pct_prior["p50"],
    color=C_PRIOR,
    linewidth=1.5,
    linestyle="--",
    label="Prior-predictive median",
)

ax.fill_between(
    sim_post.month_grid,
    pct_post["p10"],
    pct_post["p90"],
    color=C_POST,
    alpha=0.35,
    label="Posterior-predictive 80% CI",
)
ax.plot(
    sim_post.month_grid,
    pct_post["p50"],
    color=C_POST,
    linewidth=2,
    label="Posterior-predictive median",
)

for period, val in bms_implied.items():
    ax.scatter([period.to_timestamp()], [val], color=C_HIST, s=45, zorder=6, marker="D")

handles = [
    Line2D([], [], color=C_PRIOR, linewidth=1.5, linestyle="--", label="Prior-predictive median"),
    Patch(color=C_PRIOR, alpha=0.5, label="Prior-predictive 80% CI"),
    Line2D([], [], color=C_POST, linewidth=2, label="Posterior-predictive median"),
    Patch(color=C_POST, alpha=0.35, label="Posterior-predictive 80% CI"),
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
    "US Camzyos prevalent-patient forecast — prior vs posterior",
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

# 7d. Fan chart — annual revenue (both runs)
fig, ax = plt.subplots(figsize=(11, 6), facecolor=SURFACE)
r_prior = sim_prior.revenue_percentiles((10, 25, 50, 75, 90))
r_post = sim_post.revenue_percentiles((10, 25, 50, 75, 90))
years = r_prior.index.values

ax.fill_between(
    years, r_prior["p10"], r_prior["p90"], color=C_PRIOR, alpha=0.5, label="Prior 80% CI"
)
ax.plot(
    years,
    r_prior["p50"],
    color=C_PRIOR,
    linewidth=1.5,
    linestyle="--",
    marker="o",
    label="Prior median",
)
ax.fill_between(
    years, r_post["p10"], r_post["p90"], color=C_POST, alpha=0.35, label="Posterior 80% CI"
)
ax.plot(years, r_post["p50"], color=C_POST, linewidth=2, marker="o", label="Posterior median")

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
    "US Camzyos annual net revenue — prior vs posterior",
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

# 7e. Tornado (unchanged — still on the prior-predictive baseline)
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
print("  Prior-predictive (literature only):")
print(f"    Pool A p50 {pa50:,.0f}   Pool B p50 {pb50:,.0f}")
r_prior_final = sim_prior.revenue_percentiles()
peak_year_p = int(r_prior_final.idxmax()["p50"])
peak_rev_p = float(r_prior_final.loc[peak_year_p, "p50"])
print(f"    Peak US revenue: {peak_year_p}, ${peak_rev_p:,.0f}M (median)")

print("  Posterior-predictive (literature + 30k-patient claims, D1 strict):")
print(f"    Pool A p50 {qa50:,.0f}   Pool B p50 {qb50:,.0f}")
r_post_final = sim_post.revenue_percentiles()
peak_year_q = int(r_post_final.idxmax()["p50"])
peak_rev_q = float(r_post_final.loc[peak_year_q, "p50"])
print(f"    Peak US revenue: {peak_year_q}, ${peak_rev_q:,.0f}M (median)")

print(
    f"\n  Prior → posterior shift on eligible fraction: "
    f"{primary_posterior.prior_mean:.3f} → {primary_posterior.posterior_mean:.4f} "
    f"({primary_posterior.posterior_mean / primary_posterior.prior_mean:.2%} of prior)"
)
print("  The gap between the two runs *is* the value of the claims subscription:")
print("  full-population data lets us collapse the eligible-fraction uncertainty,")
print("  and — with a coding-adjusted proxy — anchor the number honestly.")

print("\nAll assertions passed. Pipeline complete.")

#!/usr/bin/env python
"""Task 2 — US addressable market for Camzyos.

One PyMC model. TAM = diagnosed HCM (N) × true clinical eligibility (p).
Claims evidence enters as k successes out of n HCM patients with k ~ Binomial(n, p·s),
where s is the claims capture rate. The posterior on p drives the TAM;
the posterior on s is the calibration insight.

Outputs (all in outputs/task2_tam/):
    09_tam_summary.csv              — posterior summary of p, s, N, TAM
    09_tam_posterior.png            — 2-panel figure (TAM distribution + joint p,s)
    09_tam_prior_sensitivity.csv    — how TAM moves under alternative priors

Run: .venv/bin/python3.13 scripts/task2_tam/09_tam_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import arviz as az
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from src.task1_adoption.data_loading import load_data
from src.task2_tam.claims_evidence import collect_evidence
from src.task2_tam.tam_model import TamPriors, build_model, sample

# ── Palette (repo convention) ─────────────────────────────────────────────────
C_POST = "#104281"
C_DATA = "#e88a1a"
C_MED = "#2a78d6"
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
INK_PRI = "#0b0b0b"
INK_SEC = "#52514e"
INK_MUT = "#898781"

OUT = Path("outputs/task2_tam")
OUT.mkdir(parents=True, exist_ok=True)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK_MUT, labelsize=9)


def _pct(x, q):
    return float(np.percentile(x, q))


# ── 1. Data ───────────────────────────────────────────────────────────────────
print("─" * 78)
print("Task 2 — US addressable market for Camzyos")
print("─" * 78)

ev = collect_evidence(load_data())
print("\n  Claims evidence (I421 ∩ Disopyramide):")
print(f"    k = {ev.k:,}   coded-treatable HCM patients")
print(f"    n = {ev.n:,}   HCM-coded patients")
print(f"    k/n = {ev.rate:.3%}  observed rate")

# ── 2. Primary model ──────────────────────────────────────────────────────────
priors = TamPriors()
print("\n  Priors:")
print(
    f"    p ~ Beta({priors.p_alpha}, {priors.p_beta})  → mean {priors.p_alpha / (priors.p_alpha + priors.p_beta):.1%}  (Desai 2022)"
)
print(
    f"    s ~ Beta({priors.s_alpha}, {priors.s_beta})       → mean {priors.s_alpha / (priors.s_alpha + priors.s_beta):.1%}  (coding lit)"
)
print(
    f"    N ~ LogNormal(median={priors.n_hcm_median:,.0f}, p95={priors.n_hcm_p95:,.0f})  (Butzner 2021)"
)

print("\n  Sampling PyMC model...")
model = build_model(k=ev.k, n=ev.n, priors=priors)
idata = sample(model, draws=2000, tune=1000, seed=42)

post = idata.posterior
p_draws = post["p_true_eligibility"].values.flatten()
s_draws = post["s_capture_rate"].values.flatten()
n_draws = post["n_hcm_us"].values.flatten()
tam_draws = post["tam"].values.flatten()

# ── 3. Summary ────────────────────────────────────────────────────────────────
summary_rows = []
for name, x, fmt in [
    ("p_true_eligibility", p_draws, "{:.3f}"),
    ("s_capture_rate", s_draws, "{:.3f}"),
    ("n_hcm_us", n_draws, "{:,.0f}"),
    ("tam", tam_draws, "{:,.0f}"),
]:
    mean, lo, med, hi = float(np.mean(x)), _pct(x, 10), _pct(x, 50), _pct(x, 90)
    summary_rows.append(
        {
            "parameter": name,
            "posterior_mean": mean,
            "p10": lo,
            "p50": med,
            "p90": hi,
        }
    )
    print(
        f"    {name:<22} mean {fmt.format(mean):>12}   80% CI [{fmt.format(lo)}, {fmt.format(hi)}]"
    )

summary_df = pd.DataFrame(summary_rows)
summary_path = OUT / "09_tam_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"\n  → {summary_path}")

# ── 4. Prior sensitivity — one-at-a-time on the two structural priors ────────
# The IC wants to see how the TAM CI moves under alternative reasonable priors.
# Vary p (eligibility) and s (capture rate) endpoints; keep N_hcm fixed.
print("\n  Prior sensitivity (TAM 80% CI under alternative priors):")
sens_scenarios = [
    ("Base case", TamPriors()),
    ("Bull: p mean 40%", TamPriors(p_alpha=20.0, p_beta=30.0)),
    ("Bear: p mean 20%", TamPriors(p_alpha=8.0, p_beta=32.0)),
    ("s more diffuse", TamPriors(s_alpha=1.0, s_beta=6.0)),
    ("s tighter (chart-review-like)", TamPriors(s_alpha=20.0, s_beta=120.0)),
    ("N low: median 300k", TamPriors(n_hcm_median=300_000, n_hcm_p95=500_000)),
    ("N high: median 550k", TamPriors(n_hcm_median=550_000, n_hcm_p95=900_000)),
]
sens_rows = []
for label, sp in sens_scenarios:
    m = build_model(k=ev.k, n=ev.n, priors=sp)
    id_ = sample(m, draws=1000, tune=500, seed=42)
    t = id_.posterior["tam"].values.flatten()
    p = id_.posterior["p_true_eligibility"].values.flatten()
    s = id_.posterior["s_capture_rate"].values.flatten()
    row = {
        "scenario": label,
        "tam_p10": int(_pct(t, 10)),
        "tam_p50": int(_pct(t, 50)),
        "tam_p90": int(_pct(t, 90)),
        "p_mean": round(float(np.mean(p)), 3),
        "s_mean": round(float(np.mean(s)), 3),
    }
    sens_rows.append(row)
    print(
        f"    {label:<32}  TAM p50 {row['tam_p50']:>7,}   [{row['tam_p10']:>6,}, {row['tam_p90']:>6,}]"
    )

sens_df = pd.DataFrame(sens_rows)
sens_path = OUT / "09_tam_prior_sensitivity.csv"
sens_df.to_csv(sens_path, index=False)
print(f"\n  → {sens_path}")

# ── 5. TAM over time ─────────────────────────────────────────────────────────
# Diagnosed HCM in the US is growing (~9%/yr, Butzner 2013→2019 HIRD). The TAM
# grows in step: TAM(year) = N × (1+g)^(year - reference) × p, with N and p
# drawn jointly from the posterior. This is a *pool* projection — not a
# Camzyos-on-drug forecast (which is a diffusion question this model deliberately
# does not attempt: too few years of launch data to anchor peak penetration).
REF_YEAR = 2024
YEARS_OUT = (2024, 2027, 2030)
GROWTH_SCENARIOS = {"g_low_5pct": 0.05, "g_base_9pct": 0.09, "g_high_12pct": 0.12}
print("\n  TAM over time (posterior draws of N × (1+g)^t × p):")
tam_time_rows = []
for label, g in GROWTH_SCENARIOS.items():
    for year in YEARS_OUT:
        tam_t = n_draws * (1 + g) ** (year - REF_YEAR) * p_draws
        tam_time_rows.append(
            {
                "scenario": label,
                "growth_rate": g,
                "year": year,
                "tam_p10": int(_pct(tam_t, 10)),
                "tam_p50": int(_pct(tam_t, 50)),
                "tam_p90": int(_pct(tam_t, 90)),
            }
        )
    print(f"    {label} ({g:.0%}/yr):")
    for year in YEARS_OUT:
        r = tam_time_rows[-len(YEARS_OUT) + YEARS_OUT.index(year)]
        print(
            f"      {year}   p50 {r['tam_p50']:>7,}   80% CI [{r['tam_p10']:,}, {r['tam_p90']:,}]"
        )

tam_time_df = pd.DataFrame(tam_time_rows)
tam_time_path = OUT / "09_tam_over_time.csv"
tam_time_df.to_csv(tam_time_path, index=False)
print(f"\n  → {tam_time_path}")


# ── 6. Figure ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 5.5), facecolor=SURFACE)
gs = fig.add_gridspec(1, 2, width_ratios=(1.15, 1.0), wspace=0.28)

# 5a. Left panel — TAM posterior histogram
ax_tam = fig.add_subplot(gs[0, 0])
t_lo, t_med, t_hi = _pct(tam_draws, 10), _pct(tam_draws, 50), _pct(tam_draws, 90)
upper = _pct(tam_draws, 99.5)
bins = np.linspace(0, upper, 60)
ax_tam.hist(tam_draws, bins=bins, color=C_POST, alpha=0.75)
ax_tam.axvline(
    t_med, color=C_MED, linewidth=1.8, linestyle="--", label=f"Median  {t_med / 1000:,.0f}k"
)
ax_tam.axvspan(
    t_lo, t_hi, color=C_POST, alpha=0.10, label=f"80% CI  {t_lo / 1000:,.0f}k – {t_hi / 1000:,.0f}k"
)
ax_tam.set_xlabel("US Camzyos-addressable patients", fontsize=10, color=INK_SEC)
ax_tam.set_ylabel("Posterior draws", fontsize=10, color=INK_SEC)
ax_tam.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x / 1000):,}k"))
ax_tam.legend(fontsize=9, frameon=False, loc="upper right")
_style(ax_tam)
ax_tam.set_title(
    "US addressable market — posterior",
    fontsize=11.5,
    fontweight="bold",
    color=INK_PRI,
    loc="left",
    pad=8,
)

# 5b. Right panel — joint (p, s) posterior scatter with priors overlaid
ax_ps = fig.add_subplot(gs[0, 1])
# Subsample for readability
idx = np.random.default_rng(0).choice(len(p_draws), size=min(2000, len(p_draws)), replace=False)
ax_ps.scatter(s_draws[idx], p_draws[idx], s=6, color=C_POST, alpha=0.35, label="Posterior draws")

# Prior means as light crosshairs
p_prior_mean = priors.p_alpha / (priors.p_alpha + priors.p_beta)
s_prior_mean = priors.s_alpha / (priors.s_alpha + priors.s_beta)
ax_ps.axhline(p_prior_mean, color=INK_MUT, linestyle=":", linewidth=0.8)
ax_ps.axvline(s_prior_mean, color=INK_MUT, linestyle=":", linewidth=0.8)

# The data ridge p*s = k/n
s_grid = np.linspace(0.02, 0.5, 200)
p_ridge = ev.rate / s_grid
ok = (p_ridge > 0) & (p_ridge < 0.6)
ax_ps.plot(
    s_grid[ok],
    p_ridge[ok],
    color=C_DATA,
    linewidth=1.6,
    linestyle="--",
    label=f"Data ridge  p × s = {ev.rate:.1%}",
)

ax_ps.set_xlim(0, 0.4)
ax_ps.set_ylim(0.05, 0.55)
ax_ps.set_xlabel("s — claims capture rate", fontsize=10, color=INK_SEC)
ax_ps.set_ylabel("p — true clinical eligibility", fontsize=10, color=INK_SEC)
ax_ps.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
ax_ps.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
ax_ps.legend(fontsize=8.5, frameon=False, loc="upper right")
_style(ax_ps)
ax_ps.set_title(
    "Joint (p, s) posterior — why the priors matter",
    fontsize=11.5,
    fontweight="bold",
    color=INK_PRI,
    loc="left",
    pad=8,
)

fig.suptitle(
    f"US Camzyos TAM — median ~{t_med / 1000:.0f}k patients "
    f"(80% CI {t_lo / 1000:.0f}k–{t_hi / 1000:.0f}k)   |   "
    f"claims capture rate s ≈ {np.median(s_draws):.1%}",
    fontsize=12,
    fontweight="bold",
    color=INK_PRI,
    x=0.02,
    ha="left",
    y=1.02,
)
plt.tight_layout(rect=[0, 0, 1, 0.98])
fig_path = OUT / "09_tam_posterior.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
plt.close()
print(f"\n  → {fig_path}")

# ── 7. Convergence sanity ────────────────────────────────────────────────────
diag = az.summary(idata, var_names=["p_true_eligibility", "s_capture_rate", "n_hcm_us", "tam"])
rhat_max = float(pd.to_numeric(diag["r_hat"], errors="coerce").max())
ess_min = float(pd.to_numeric(diag["ess_bulk"], errors="coerce").min())
print(f"\n  Convergence: max r_hat = {rhat_max:.3f}, min ESS = {ess_min:.0f}")
if rhat_max > 1.01 or ess_min < 400:
    print("  ⚠ convergence concern — inspect trace before trusting results")
else:
    print("  ✓ chains converged")

print("\nDone.")

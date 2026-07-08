"""Registry of every prior used in the TAM pipeline.

Every scalar traces to a `Prior` here — no magic numbers elsewhere in
`src/task2_tam/`. This makes `outputs/task2_tam/09_tam_sources.csv` a complete
audit trail.

Source-type taxonomy
--------------------
- ``peer_reviewed``     : published epidemiology or clinical trial data
- ``company_disclosure``: BMS / Cytokinetics filings, earnings releases, labels
- ``government``        : Census, FDA, CMS
- ``external_default``  : defensible industry rule-of-thumb where no better
                          source exists. The IC should challenge these first.
- ``user_data_needed``  : the analyst has better proprietary data (e.g. IQVIA
                          scripts, house view on aficamten) that should
                          override this default. Flagged for follow-up.

Two eligible-pool definitions
-----------------------------
We carry two distinct "addressable" quantities through the pipeline because
they differ by nearly an order of magnitude and are routinely conflated:

- **Pool A (theoretical ceiling)**: symptomatic oHCM patients with LVEF ≥ 50%
  in the US population, whether diagnosed or not. This is the maximum
  population that could benefit from Camzyos if diagnosis were universal.

- **Pool B (diagnosed & treatable today)**: patients already in the healthcare
  system with an oHCM diagnosis code and NYHA II-III. This is the commercially
  relevant near-term number. Anchored on Butzner 2021 (US claims).

The gap between A and B is the undiagnosed pool — the largest driver of
Camzyos' long-run growth as diagnosis rates rise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

# scipy doesn't expose a public type alias for a "frozen" distribution instance;
# `stats.lognorm(...)` returns `_distn_infrastructure.FrozenDist`. We type as
# `Any` to avoid coupling to a private path.
FrozenDist = Any

SourceType = Literal[
    "peer_reviewed",
    "company_disclosure",
    "government",
    "external_default",
    "user_data_needed",
]


@dataclass(frozen=True)
class Prior:
    """A named, cited prior distribution."""

    name: str
    distribution: FrozenDist
    source: str
    source_type: SourceType
    notes: str = ""

    @property
    def median(self) -> float:
        """Median of the distribution."""
        return float(self.distribution.median())

    @property
    def p05(self) -> float:
        """5th percentile."""
        return float(self.distribution.ppf(0.05))

    @property
    def p95(self) -> float:
        """95th percentile."""
        return float(self.distribution.ppf(0.95))

    def sample(self, size: int, random_state: np.random.Generator) -> np.ndarray:
        """Draw `size` iid samples from the distribution."""
        return self.distribution.rvs(size=size, random_state=random_state)


# ── Distribution builders ─────────────────────────────────────────────────────


def _lognormal_from_ci(median: float, p95: float) -> FrozenDist:
    if median <= 0 or p95 <= median:
        raise ValueError(f"Require 0 < median < p95, got median={median}, p95={p95}")
    sigma = float(np.log(p95 / median) / 1.645)
    return stats.lognorm(s=sigma, scale=median)


def _beta_from_ci(mean: float, p05: float, p95: float) -> FrozenDist:
    """Beta on (0,1) approximating a target mean and 90% CI."""
    if not (0 < mean < 1) or not (0 <= p05 < mean < p95 <= 1):
        raise ValueError(f"Bad Beta params: mean={mean}, p05={p05}, p95={p95}")
    sd_target = (p95 - p05) / 3.29
    variance = min(sd_target**2, 0.9 * mean * (1 - mean))
    a = mean * ((mean * (1 - mean) / variance) - 1)
    b = (1 - mean) * ((mean * (1 - mean) / variance) - 1)
    return stats.beta(a, b)


# ── Fixed constants (no uncertainty) ──────────────────────────────────────────

US_ADULTS_20_PLUS = 261_000_000  # US Census ACS 2024
US_ADULTS_SOURCE = "US Census Bureau, 2024 ACS age-sex estimates"

LAUNCH_MONTH = "2022-04"
FORECAST_END_MONTH = "2030-12"


# ── Prior registry ────────────────────────────────────────────────────────────

PRIORS: dict[str, Prior] = {}


def _register(p: Prior) -> None:
    PRIORS[p.name] = p


# ── POOL A drivers — theoretical ceiling (phenotype-based) ────────────────────
#
# True HCM prevalence (imaging + genetic + claims-based cumulative). Anchored
# on the convergence between the classic imaging figure (1:500 = 0.2%) and the
# recent cumulative claims-based estimate (Butzner 2026 JACC:Advances,
# ~1:327 = 0.306% cumulative 2016-2023). Down-weighted genetic 1:200 estimates
# because sarcomere-variant penetrance in UK Biobank was only ~16% (Massera
# 2023 UK Biobank). Median at 0.23% blends imaging and claims-cumulative.

_register(
    Prior(
        name="hcm_prevalence",
        distribution=_lognormal_from_ci(median=0.0023, p95=0.0031),
        source=(
            "Maron BJ et al. Circulation 1995;92:785-9 (CARDIA, 0.17% young adults, verified); "
            "Massera D et al. Int J Cardiol 2023 (10.1016/j.ijcard.2023.04.005, UK Biobank); "
            "Butzner M et al. JACC Advances 2026 (10.1016/j.jacadv.2025.102552, "
            "US claims cumulative 2016-2023; VERIFY DOI — could not resolve via paywall)"
        ),
        source_type="peer_reviewed",
        notes=(
            "Anchored on convergence of imaging (0.2%) and claims-cumulative (0.31%). "
            "Median 0.23% gives implied US HCM ~600k, 90% CI ~450-800k. Sarcomere-"
            "variant genetic prevalence (1:200) explicitly excluded due to low penetrance."
        ),
    )
)

# Camzyos-eligible fraction of HCM — single joint prior replacing separate
# obstructive × symptomatic nodes. Uses the specialty-registry finding directly
# (30.7% of HCM adults are obstructive AND symptomatic enough for a myosin
# inhibitor) rather than multiplying two uncertain independent fractions.
# This avoids the independence assumption and is more directly grounded.
# Community claims imply ~20%; referral + provocation implies ~40%+.
_register(
    Prior(
        name="camzyos_eligible_fraction",
        distribution=_beta_from_ci(mean=0.307, p05=0.20, p95=0.42),
        source=(
            "US HCM specialty registry: 30.7% of HCM adults are obstructive AND "
            "symptomatic enough for a myosin inhibitor (direct observation). "
            "Concordant with Desai 2022 (MarketScan ~50% symptomatic) × Maron 2006 / "
            "Butzner 2026 (obstructive range 37-66%). Community claims imply ~20%; "
            "referral + provocation implies ~40%+"
        ),
        source_type="peer_reviewed",
        notes=(
            "Replaces separate obstructive_fraction × symptomatic_and_ef_preserved_fraction. "
            "Avoids the independence assumption between two uncertain fractions. "
            "Directly observed joint probability from a US specialty registry."
        ),
    )
)


# ── POOL B anchor — diagnosed & treatable today (claims-based) ────────────────
#
# Butzner 2021 (VERIFIED via PubMed) reports point-prevalent diagnosed HCM of
# 262,591 in 2019 in the HIRD database (grew from 164,403 in 2013 — 1.5x in
# 6 years). Ongoing tripling (per Butzner 2026, unverified) suggests
# ~500-800k diagnosed by 2024. Take the LOWER anchor to be conservative for
# "actively diagnosed and clinically recognised in a given year."

_register(
    Prior(
        name="diagnosed_hcm_us_current",
        distribution=_lognormal_from_ci(median=400_000, p95=650_000),
        source=(
            "Butzner M et al. Am J Cardiol 2021 (10.1016/j.amjcard.2021.08.024, "
            "HIRD, 262,591 point-prevalent HCM in 2019 — VERIFIED via PubMed); "
            "extrapolated forward at ~10%/year clinical-recognition growth to 2024; "
            "cross-check: Butzner M et al. JACC Advances 2026 (10.1016/j.jacadv.2025.102552, "
            "cumulative 833k 2016-2023 — DOI not verifiable via paywall, cite with caveat)"
        ),
        source_type="peer_reviewed",
        notes=(
            "Point-prevalent count of patients with active HCM diagnosis codes in US "
            "claims at ~2024. Median 400k = Butzner 2019 (263k) grown at 8.7%/yr "
            "over 5 years. Wide upper bound accepts the Butzner 2026 acceleration."
        ),
    )
)


# ── Camzyos commercial anchors ────────────────────────────────────────────────

# Camzyos net price per patient-year. WAC list ~$89k; specialty cardiology
# gross-to-net typically 15-20%. Static across horizon in base case — real-
# world price erosion from aficamten captured on volume, not price.
_register(
    Prior(
        name="net_price_per_year_usd",
        distribution=_lognormal_from_ci(median=75_000, p95=90_000),
        source="BMS 10-K FY2024 (list price ~$89k WAC); specialty cardiology gross-to-net 15-20%",
        source_type="company_disclosure",
        notes="Verify list price vs most recent BMS 10-K if forecast horizon extends past 2027.",
    )
)

# Peak penetration of the diagnosed treatable pool (Pool B) that Camzyos ever
# achieves. Bounded above by aficamten share loss and by prescriber access
# (REMS certification), bounded below by minimum specialty-drug adoption.
_register(
    Prior(
        name="peak_penetration_of_pool_b",
        distribution=_beta_from_ci(mean=0.30, p05=0.15, p95=0.50),
        source=(
            "External default — specialty cardiology analogue; capped by REMS + aficamten. "
            "In-sample validation: our synthetic dataset shows 146/775 = 18.8% Camzyos "
            "penetration among Disopyramide-experienced oHCM patients over 21 months "
            "(Task 1 pipeline); adjusting for the synthetic 98.8% Diso co-occurrence "
            "artefact (real-world ~60-75%), true steady-state ≈ 14-17%, within our CI."
        ),
        source_type="external_default",
        notes=(
            "Fraction of the diagnosed treatable pool that eventually receives Camzyos at "
            "peak. IC should challenge — this is the second-biggest lever after "
            "obstructive_fraction. BB Biotech house view welcome. In-sample 18.8% (Task 1) "
            "is a partial validation; median 30% assumes long-run penetration will exceed "
            "the 21-month snapshot because the launch trajectory is still ramping."
        ),
    )
)

# Diffusion rate — years to reach 80% of peak penetration from launch. Simple
# proxy replacing full Bass model. Median 6 years is consistent with observed
# BMS trajectory (10k patients by end 2024, ramping toward peak).
_register(
    Prior(
        name="years_to_80pct_peak",
        distribution=_lognormal_from_ci(median=6.0, p95=9.0),
        source="External default — observed BMS Camzyos trajectory 2022-2024",
        source_type="external_default",
        notes="Simple penetration ramp shape. See diffusion.py for exact functional form.",
    )
)

# Aficamten share of NEW starts once approved — external default; overridable
# by BB Biotech house view.
_register(
    Prior(
        name="aficamten_terminal_share_of_new_starts",
        distribution=_beta_from_ci(mean=0.50, p05=0.30, p95=0.65),
        source=(
            "Cytokinetics SEQUOIA-HCM positive readout Dec 2023 (verify current status); "
            "second-in-class analogue (DPP-4, SGLT2 second-entrants typically 40-60%)"
        ),
        source_type="external_default",
        notes=(
            "OVERRIDABLE via BB Biotech house view. Applied to NEW starts, not "
            "prevalent stock (limited switching from established Camzyos patients)."
        ),
    )
)


# ── Historic BMS Camzyos revenue anchors (US, USD millions) ───────────────────
#
# Sourced from BMS earnings releases (news.bms.com). Used for backcast checks
# and to calibrate the penetration curve shape. Not stochastic.

# BMS-reported US Camzyos net revenue by quarter-end month (USD millions).
# `verified=True` rows come directly from earnings releases (news.bms.com).
# `verified=False` rows are interpolated between reported points using the
# reported FY2024 total and the observed quarterly ramp.
BMS_QUARTERLY_REVENUE_US_M: dict[str, tuple[float, bool]] = {
    "2023-12": (84.0, True),  # Q4 2023 earnings release, Feb 2024
    "2024-03": (92.0, False),
    "2024-06": (118.0, False),
    "2024-09": (132.0, False),
    "2024-12": (201.0, True),  # Q4 2024 earnings release, Feb 2025
    "2025-03": (126.0, True),  # Q1 2025 earnings release, Apr 2025
}
BMS_VERIFIED_SOURCE = (
    "BMS quarterly earnings releases (news.bms.com): Q4 2023 (Feb 2024), "
    "Q4 2024 (Feb 2025), Q1 2025 (Apr 2025). Directly reported."
)
BMS_INTERPOLATED_SOURCE = (
    "Interpolated between adjacent BMS-reported quarters using the reported "
    "FY2024 total and observed ramp shape. Not directly disclosed."
)


# ── Non-stochastic external assumptions ───────────────────────────────────────


@dataclass(frozen=True)
class ExternalAssumptions:
    """Non-stochastic assumptions used by the pipeline; surfaced in the audit CSV."""

    aficamten_pdufa_month: str = "2025-09"
    aficamten_ramp_months: int = 18
    forecast_end_month: str = FORECAST_END_MONTH
    launch_month: str = LAUNCH_MONTH
    n_monte_carlo_draws: int = 10_000
    random_seed: int = 42

    provenance: dict[str, tuple[str, SourceType]] = field(
        default_factory=lambda: {
            "aficamten_pdufa_month": (
                "Cytokinetics NDA acceptance ~Q4 2024; PDUFA target Sept 2025. "
                "VERIFY current status — knowledge cutoff Jan 2026.",
                "user_data_needed",
            ),
            "aficamten_ramp_months": (
                "External default — typical second-in-class specialty ramp",
                "external_default",
            ),
            "forecast_end_month": ("Analyst choice", "external_default"),
            "launch_month": ("FDA approval April 28, 2022", "government"),
            "n_monte_carlo_draws": ("Sufficient for stable 80% CI", "external_default"),
            "random_seed": ("Reproducibility", "external_default"),
        }
    )


def bms_revenue_values() -> dict[str, float]:
    """Return just the revenue values (drops the verified/interpolated flag)."""
    return {m: v for m, (v, _) in BMS_QUARTERLY_REVENUE_US_M.items()}


def sources_dataframe() -> pd.DataFrame:
    """Return a DataFrame of every prior + external assumption for the audit CSV."""
    rows = []
    for name, p in PRIORS.items():
        rows.append(
            {
                "name": name,
                "kind": "prior",
                "value_or_median": round(p.median, 6),
                "p05": round(p.p05, 6),
                "p95": round(p.p95, 6),
                "source_type": p.source_type,
                "source": p.source,
                "notes": p.notes,
            }
        )

    ext = ExternalAssumptions()
    for name, (src, src_type) in ext.provenance.items():
        rows.append(
            {
                "name": name,
                "kind": "external_assumption",
                "value_or_median": getattr(ext, name),
                "p05": "",
                "p95": "",
                "source_type": src_type,
                "source": src,
                "notes": "",
            }
        )
    # US_ADULTS_20_PLUS is a module constant, not an ExternalAssumptions field,
    # but still worth surfacing in the audit CSV.
    rows.append(
        {
            "name": "us_adults_20_plus",
            "kind": "external_assumption",
            "value_or_median": US_ADULTS_20_PLUS,
            "p05": "",
            "p95": "",
            "source_type": "government",
            "source": US_ADULTS_SOURCE,
            "notes": "",
        }
    )

    for qmonth, (rev, verified) in BMS_QUARTERLY_REVENUE_US_M.items():
        rows.append(
            {
                "name": f"bms_us_camzyos_revenue_{qmonth}_usdm",
                "kind": "historic_anchor",
                "value_or_median": rev,
                "p05": "",
                "p95": "",
                "source_type": "company_disclosure",
                "source": BMS_VERIFIED_SOURCE if verified else BMS_INTERPOLATED_SOURCE,
                "notes": "US Camzyos net revenue, USD millions",
            }
        )

    return pd.DataFrame(rows)

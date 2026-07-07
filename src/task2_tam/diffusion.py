"""Simple penetration curve — Camzyos on-drug prevalent-patient trajectory.

We deliberately avoid a full Bass diffusion model here. Given only ~4 years of
launch data and one clean anchor point (the BMS-implied ~10k patients at end
of 2024), fitting p and q separately would be over-parameterised and
indefensible to challenge.

Instead:
    on_drug(t) = pool_B(t) × peak_penetration × logistic_ramp(t)

where logistic_ramp is a saturating S-curve pinned to reach ~80% of the peak
in `years_to_80pct_peak` years from launch. Aficamten haircut is applied to
NEW starts post-PDUFA (existing Camzyos patients don't switch en masse).

pool_B (diagnosed treatable) is treated as growing over time from its current
value toward pool_A (theoretical ceiling), reflecting the rising diagnosis
rate documented in Butzner 2021. Diagnosis-growth prior below.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.task2_tam.priors import bms_revenue_values


def build_month_grid(start_month: str, end_month: str) -> pd.DatetimeIndex:
    """Monthly DatetimeIndex from start_month to end_month, inclusive."""
    return pd.date_range(start=start_month, end=end_month, freq="MS")


def _years_since(month_grid: pd.DatetimeIndex, ref_month: str) -> np.ndarray:
    ref = pd.Timestamp(ref_month)
    return np.array([max(0, (m.year - ref.year) + (m.month - ref.month) / 12) for m in month_grid])


def _logistic_ramp(t_years: np.ndarray, t_80pct: float) -> np.ndarray:
    """S-curve rising from 0 to 1, hitting ~0.8 at t = t_80pct.

    Uses a standard logistic parameterised so f(0) ≈ 0 and f(t_80pct) = 0.8.
    """
    # logistic(x) = 1/(1+exp(-x)). Want logistic(k*(t - t_mid)) with
    # f(t_80pct)=0.8 and f(0) ≈ 0. Set t_mid = t_80pct/2, solve k.
    # 0.8 = 1/(1+exp(-k * t_80pct/2)) → k = 2/t_80pct * log(4)
    k = 2.0 / t_80pct * np.log(4)
    t_mid = t_80pct / 2
    return 1.0 / (1.0 + np.exp(-k * (t_years - t_mid)))


def _aficamten_share(
    month_grid: pd.DatetimeIndex,
    pdufa_month: str,
    ramp_months: int,
    terminal_share: float,
) -> np.ndarray:
    """Linear ramp of aficamten share of new starts post-PDUFA."""
    pdufa_ts = pd.Timestamp(pdufa_month)
    share = np.zeros(len(month_grid))
    for i, m in enumerate(month_grid):
        months_post = (m.year - pdufa_ts.year) * 12 + (m.month - pdufa_ts.month)
        share[i] = terminal_share * max(0.0, min(months_post / ramp_months, 1.0))
    return share


POOL_B_REFERENCE_MONTH = "2024-06"
"""Calendar month at which `pool_b_today` is calibrated.

Matches the `diagnosed_hcm_us_current` prior (Butzner 2019 grown ~5 years to
2024). Trajectory is back-scaled to earlier months and forward-grown from here.
"""


def _pool_b_over_time(
    month_grid: pd.DatetimeIndex,
    pool_b_today: float,
    pool_a: float,
    annual_growth_rate: float = 0.087,
    reference_month: str = POOL_B_REFERENCE_MONTH,
) -> np.ndarray:
    """Pool B trajectory: forward-growing from a calibration reference month.

    The `diagnosed_hcm_us_current` prior is a 2024 point estimate (Butzner 2019
    grown 5 years). We plant it at `reference_month` and back-scale to earlier
    months (constant compounding, no gap logic) and forward-grow from it
    (compounding scaled by remaining gap to pool_a, so pool_b never exceeds
    pool_a).

    Butzner 2021 documented 1.5× growth 2013→2019 in HIRD (~8.7%/yr); Butzner
    2026 (unverified) suggests further acceleration. Default 0.087 matches the
    extrapolation encoded in the prior.

    Args:
        month_grid: Monthly timestamps.
        pool_b_today: Pool B at `reference_month`.
        pool_a: Ceiling (theoretical addressable).
        annual_growth_rate: Base year-over-year growth rate.
        reference_month: YYYY-MM string identifying when `pool_b_today` applies.
    """
    n = len(month_grid)
    monthly_rate = (1 + annual_growth_rate) ** (1 / 12) - 1
    ref_idx = int(month_grid.to_period("M").get_loc(pd.Period(reference_month, freq="M")))

    b = np.zeros(n)
    b[ref_idx] = pool_b_today
    # Back-scale (constant compounding, no gap logic — pre-reference is historical).
    for i in range(ref_idx - 1, -1, -1):
        b[i] = b[i + 1] / (1 + monthly_rate)
    # Forward-grow with gap logic toward pool_a.
    for i in range(ref_idx + 1, n):
        gap_frac = max(0.0, 1.0 - b[i - 1] / pool_a) if pool_a > 0 else 0.0
        b[i] = b[i - 1] * (1 + monthly_rate * gap_frac)
    return b


@dataclass
class PenetrationInputs:
    """Per-draw scalars for a single trajectory (used by `prevalent_patients_trajectory`)."""

    pool_a: float
    pool_b_today: float
    peak_penetration: float
    years_to_80pct: float
    aficamten_terminal_share: float


def prevalent_patients_trajectory(
    inp: PenetrationInputs,
    month_grid: pd.DatetimeIndex,
    launch_month: str,
    aficamten_pdufa_month: str,
    aficamten_ramp_months: int,
    diagnosis_growth_rate: float = 0.087,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute prevalent-patient trajectory (flow-based, aficamten diverts new starts only).

    Two-step:
      1. Intact class trajectory: intact(t) = pool_b(t) × peak_penetration × ramp(t).
         This is the joint on-drug count for Camzyos + aficamten combined.
      2. Camzyos share = cumulative sum of NEW starts, with each month's new
         starts diluted by aficamten share at that month. This preserves the
         stock property that existing Camzyos patients don't switch en masse.

    A small first-year persistence loss (5%/year) is applied to Camzyos patients
    only — captures real-world discontinuation. Deliberately simple: not a full
    cohort-tracking persistence model.

    Args:
        inp: Per-draw scalars.
        month_grid: Monthly timestamps.
        launch_month: YYYY-MM string.
        aficamten_pdufa_month: YYYY-MM string.
        aficamten_ramp_months: Ramp length in months.
        diagnosis_growth_rate: Annual growth rate of pool B toward pool A.

    Returns:
        Tuple of (on_drug, pool_b_trajectory), both length n_months.
    """
    pool_b_traj = _pool_b_over_time(
        month_grid,
        inp.pool_b_today,
        inp.pool_a,
        diagnosis_growth_rate,
    )
    t_years = _years_since(month_grid, launch_month)
    ramp = _logistic_ramp(t_years, inp.years_to_80pct)
    intact = pool_b_traj * inp.peak_penetration * ramp

    afi_share = _aficamten_share(
        month_grid,
        aficamten_pdufa_month,
        aficamten_ramp_months,
        inp.aficamten_terminal_share,
    )

    # Monthly new starts to the intact class; diluted by aficamten at that month.
    intact_diffs = np.diff(intact, prepend=0.0)
    intact_diffs = np.maximum(intact_diffs, 0.0)
    camzyos_new = intact_diffs * (1.0 - afi_share)

    # Simple monthly retention on existing Camzyos patients (5%/yr = 0.427%/mo loss).
    monthly_retention = (1 - 0.05) ** (1 / 12)
    on_drug = np.zeros(len(month_grid))
    on_drug[0] = camzyos_new[0]
    for i in range(1, len(month_grid)):
        on_drug[i] = on_drug[i - 1] * monthly_retention + camzyos_new[i]

    return on_drug, pool_b_traj


def implied_on_drug_from_revenue(
    revenue_by_month: dict[str, float],
    net_price_per_year_usd: float,
) -> pd.Series:
    """Back-out avg on-drug patients from quarterly US revenue.

    Quarterly revenue $R M annualised: R * 4 * 1e6 = annual revenue at that
    run-rate. Divided by net price per patient-year gives implied prevalent
    patients during that quarter.

    Args:
        revenue_by_month: {"YYYY-MM": revenue_usd_millions}
        net_price_per_year_usd: Net price per patient-year.

    Returns:
        Series indexed by month period, values = implied prevalent count.
    """
    idx = pd.PeriodIndex(list(revenue_by_month.keys()), freq="M")
    vals = np.array(list(revenue_by_month.values())) * 1e6 * 4 / net_price_per_year_usd
    return pd.Series(vals, index=idx, name="implied_on_drug_patients")


def bms_implied_on_drug_series(net_price_per_year_usd: float = 75_000) -> pd.Series:
    """Historic BMS-anchored implied on-drug series — thin wrapper over the general primitive."""
    return implied_on_drug_from_revenue(bms_revenue_values(), net_price_per_year_usd)

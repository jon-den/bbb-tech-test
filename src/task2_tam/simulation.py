"""Monte Carlo orchestrator for the TAM pipeline.

Runs `n_draws` iterations:
    funnel → (pool_A, pool_B) per draw
      → sample penetration + diffusion + aficamten + price
      → prevalent-patient trajectory + annual revenue trajectory

Returns per-draw arrays for fan chart and summary percentiles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.task2_tam.diffusion import (
    PenetrationInputs,
    build_month_grid,
    prevalent_patients_trajectory,
)
from src.task2_tam.funnel import draw_funnel
from src.task2_tam.priors import PRIORS, ExternalAssumptions
from src.task2_tam.revenue import annual_revenue_from_monthly_patients


@dataclass
class SimulationResult:
    """Per-draw arrays and percentile summaries from a Monte Carlo run."""

    funnel: pd.DataFrame
    month_grid: pd.DatetimeIndex
    patients_by_draw: np.ndarray  # (n_draws, n_months)
    revenue_by_draw: pd.DataFrame  # (n_draws, n_years)
    draw_scalars: pd.DataFrame
    assumptions: ExternalAssumptions

    def prevalent_percentiles(self, percentiles=(10, 25, 50, 75, 90)) -> pd.DataFrame:
        """Percentile-summarise the monthly prevalent-patient trajectory."""
        p = np.percentile(self.patients_by_draw, percentiles, axis=0)
        return pd.DataFrame(p.T, index=self.month_grid, columns=[f"p{q}" for q in percentiles])

    def revenue_percentiles(self, percentiles=(10, 25, 50, 75, 90)) -> pd.DataFrame:
        """Percentile-summarise the annual US revenue trajectory."""
        p = np.percentile(self.revenue_by_draw.values, percentiles, axis=0)
        return pd.DataFrame(
            p.T, index=self.revenue_by_draw.columns, columns=[f"p{q}" for q in percentiles]
        )

    def summary_table(self, years=(2025, 2028, 2030)) -> pd.DataFrame:
        """Median + 80% CI on prevalent patients (year-end) and annual revenue."""
        rows = []
        for y in years:
            idxs = np.where(self.month_grid.year == y)[0]
            if len(idxs) == 0:
                continue
            i = idxs[-1]  # year-end month
            p = self.patients_by_draw[:, i]
            r = (
                self.revenue_by_draw[y].values
                if y in self.revenue_by_draw.columns
                else np.array([np.nan])
            )
            rows.append(
                {
                    "year": y,
                    "on_drug_p10": int(np.percentile(p, 10)),
                    "on_drug_p50": int(np.percentile(p, 50)),
                    "on_drug_p90": int(np.percentile(p, 90)),
                    "revenue_usdm_p10": round(np.percentile(r, 10), 0),
                    "revenue_usdm_p50": round(np.percentile(r, 50), 0),
                    "revenue_usdm_p90": round(np.percentile(r, 90), 0),
                }
            )
        return pd.DataFrame(rows).set_index("year")


def run_simulation(assumptions: ExternalAssumptions | None = None) -> SimulationResult:
    """Run the full Monte Carlo pipeline and return a `SimulationResult`.

    Args:
        assumptions: External assumptions; defaults to `ExternalAssumptions()`.
    """
    a = assumptions or ExternalAssumptions()
    rng = np.random.default_rng(a.random_seed)

    funnel = draw_funnel(a.n_monte_carlo_draws, rng)

    peak_pen = PRIORS["peak_penetration_of_pool_b"].sample(a.n_monte_carlo_draws, rng)
    yrs_to_80 = PRIORS["years_to_80pct_peak"].sample(a.n_monte_carlo_draws, rng)
    afi_share = PRIORS["aficamten_terminal_share_of_new_starts"].sample(a.n_monte_carlo_draws, rng)
    price = PRIORS["net_price_per_year_usd"].sample(a.n_monte_carlo_draws, rng)

    draw_scalars = pd.DataFrame(
        {
            "peak_penetration_of_pool_b": peak_pen,
            "years_to_80pct_peak": yrs_to_80,
            "aficamten_terminal_share": afi_share,
            "net_price_per_year_usd": price,
        }
    )

    month_grid = build_month_grid(a.launch_month, a.forecast_end_month)
    n_months = len(month_grid)
    n_draws = a.n_monte_carlo_draws

    patients = np.zeros((n_draws, n_months))
    revenue_records = {}

    for i in range(n_draws):
        inp = PenetrationInputs(
            pool_a=float(funnel["pool_a_theoretical"].iat[i]),
            pool_b_today=float(funnel["pool_b_diagnosed_today"].iat[i]),
            peak_penetration=peak_pen[i],
            years_to_80pct=yrs_to_80[i],
            aficamten_terminal_share=afi_share[i],
        )
        on_drug, _ = prevalent_patients_trajectory(
            inp,
            month_grid,
            a.launch_month,
            a.aficamten_pdufa_month,
            a.aficamten_ramp_months,
        )
        patients[i] = on_drug
        revenue_records[i] = annual_revenue_from_monthly_patients(on_drug, month_grid, price[i])

    revenue_df = pd.DataFrame(revenue_records).T
    revenue_df.columns.name = "year"

    return SimulationResult(
        funnel=funnel,
        month_grid=month_grid,
        patients_by_draw=patients,
        revenue_by_draw=revenue_df,
        draw_scalars=draw_scalars,
        assumptions=a,
    )

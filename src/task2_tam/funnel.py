"""Eligibility funnel: compute two definitions of "addressable" per draw.

Pool A — theoretical ceiling (phenotype-based)
    US adults × HCM prevalence × Camzyos-eligible fraction
    = symptomatic oHCM patients (NYHA II-III, LVEF ≥ 50%) in the US,
      whether diagnosed or not.

Pool B — diagnosed & treatable today (claims-based)
    Diagnosed HCM (Butzner) × Camzyos-eligible fraction
    = patients currently in the healthcare system with a clinical oHCM code
      and eligible symptoms — the near-term commercial addressable pool.

The Camzyos-eligible fraction can come from either:
    - the literature prior (default; forward Monte Carlo / "prior-predictive"), or
    - the Beta-Binomial posterior updated on our claims cohort
      (opt-in via `eligible_fraction_dist`; see `bayesian_update.py`).
Both paths use a Beta-distributed rate, so downstream code is unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.task2_tam.priors import PRIORS, US_ADULTS_20_PLUS

FrozenDist = Any


def draw_funnel(
    n_draws: int,
    rng: np.random.Generator,
    eligible_fraction_dist: FrozenDist | None = None,
) -> pd.DataFrame:
    """Draw one row per Monte Carlo iteration through the two-pool funnel.

    Args:
        n_draws: Number of Monte Carlo draws.
        rng: NumPy Generator for reproducibility.
        eligible_fraction_dist: Optional frozen scipy distribution to use for
            the Camzyos-eligible fraction. When None (default), samples the
            literature prior — this is the prior-predictive Monte Carlo.
            Pass the posterior from `bayesian_update.update_eligible_fraction`
            to run the posterior-predictive TAM.

    Returns:
        DataFrame with n_draws rows. Columns include per-node values and the
        two headline pools (pool_a_theoretical, pool_b_diagnosed_today).
    """
    hcm_prev = PRIORS["hcm_prevalence"].sample(n_draws, rng)
    diagnosed_hcm = PRIORS["diagnosed_hcm_us_current"].sample(n_draws, rng)

    if eligible_fraction_dist is None:
        eligible = PRIORS["camzyos_eligible_fraction"].sample(n_draws, rng)
    else:
        eligible = eligible_fraction_dist.rvs(size=n_draws, random_state=rng)

    hcm_phenotype = US_ADULTS_20_PLUS * hcm_prev
    pool_a = hcm_phenotype * eligible

    pool_b = diagnosed_hcm * eligible

    return pd.DataFrame(
        {
            "hcm_prevalence": hcm_prev,
            "hcm_phenotype_us": hcm_phenotype,
            "camzyos_eligible_fraction": eligible,
            "pool_a_theoretical": pool_a,
            "diagnosed_hcm_us_current": diagnosed_hcm,
            "pool_b_diagnosed_today": pool_b,
            "undiagnosed_gap": pool_a - pool_b,
        }
    )

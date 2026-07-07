"""Eligibility funnel: compute two definitions of "addressable" per draw.

Pool A — theoretical ceiling (phenotype-based)
    US adults × HCM prevalence × oHCM fraction × (symptomatic II-III with EF ≥ 50%)
    = symptomatic oHCM patients in the US population, whether diagnosed or not.

Pool B — diagnosed & treatable today (claims-based)
    Diagnosed HCM (Butzner) × oHCM fraction × (symptomatic II-III with EF ≥ 50%)
    = patients currently in the healthcare system with a clinical oHCM code
      and NYHA II-III symptoms — the near-term commercial addressable pool.

The gap A − B is the undiagnosed pool. Camzyos revenue over 5-10 years is
gated by how fast that gap closes (see diffusion.py). Both quantities are
computed per Monte Carlo draw sharing the same oHCM and symptomatic-fraction
draws, so their correlation is preserved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.task2_tam.priors import PRIORS, US_ADULTS_20_PLUS


def draw_funnel(n_draws: int, rng: np.random.Generator) -> pd.DataFrame:
    """Draw one row per Monte Carlo iteration through the two-pool funnel.

    Args:
        n_draws: Number of Monte Carlo draws.
        rng: NumPy Generator for reproducibility.

    Returns:
        DataFrame with n_draws rows. Columns include per-node values and the
        two headline pools (pool_a_theoretical, pool_b_diagnosed_today).
    """
    hcm_prev = PRIORS["hcm_prevalence"].sample(n_draws, rng)
    obstructive = PRIORS["obstructive_fraction"].sample(n_draws, rng)
    sym_ef = PRIORS["symptomatic_and_ef_preserved_fraction"].sample(n_draws, rng)
    diagnosed_hcm = PRIORS["diagnosed_hcm_us_current"].sample(n_draws, rng)

    hcm_phenotype = US_ADULTS_20_PLUS * hcm_prev
    ohcm_phenotype = hcm_phenotype * obstructive
    pool_a = ohcm_phenotype * sym_ef  # theoretical ceiling

    diagnosed_ohcm = diagnosed_hcm * obstructive
    pool_b = diagnosed_ohcm * sym_ef  # diagnosed today

    return pd.DataFrame(
        {
            "hcm_prevalence": hcm_prev,
            "hcm_phenotype_us": hcm_phenotype,
            "obstructive_fraction": obstructive,
            "ohcm_phenotype_us": ohcm_phenotype,
            "symptomatic_and_ef_preserved_fraction": sym_ef,
            "pool_a_theoretical": pool_a,
            "diagnosed_hcm_us_current": diagnosed_hcm,
            "diagnosed_ohcm_us": diagnosed_ohcm,
            "pool_b_diagnosed_today": pool_b,
            "undiagnosed_gap": pool_a - pool_b,
        }
    )

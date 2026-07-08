"""One PyMC model, one TAM number.

The question: how many US patients could realistically be prescribed Camzyos?

The setup: TAM = (diagnosed HCM in the US) × (Camzyos-eligible fraction).
Both inputs are uncertain, and the eligible fraction has two disagreeing
sources — literature (Desai 2022, ~30%) and our own claims cohort (~4%).

The reframe: our claims data does not measure true eligibility. It measures
who is *coded* as eligible. Two patients with the same clinical picture can
look different in claims. So:

    observed_in_claims = true_eligibility (p) × claims_capture_rate (s)

The model updates both jointly. TAM uses posterior draws of *true* eligibility
`p`. The posterior on `s` is the calibration insight — how much of the
literature-vs-claims gap is under-coding vs. genuine over-estimation.

The whole model is 8 lines of PyMC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pymc as pm


@dataclass(frozen=True)
class TamPriors:
    """Every prior with a citation. Change one number here — nothing else."""

    # True clinical eligibility p ~ Beta(a, b). Mean a/(a+b).
    # Desai 2022 US HCM specialty registry: ~30% of HCM adults are obstructive
    # AND symptomatic enough for a myosin inhibitor. Beta(14.3, 32.3) → mean
    # 30.7%, 90% CI ~20-42% (matches Desai + Butzner community-vs-referral range).
    p_alpha: float = 14.3
    p_beta: float = 32.3

    # Claims capture rate s ~ Beta(a, b). Weakly informative from coding-
    # validation literature: sensitivity of I421+antiarrhythmic proxies for
    # true symptomatic oHCM runs ~10-25%. Beta(2, 12) → mean 14.3%, ESS 14.
    s_alpha: float = 2.0
    s_beta: float = 12.0

    # Diagnosed HCM in the US. Butzner 2021 (verified): 263k in 2019 in HIRD;
    # grown at the observed ~9%/yr rate over 5 years to a 2024 point estimate
    # of ~400k. LogNormal with median 400k and 95th percentile 650k captures
    # the extrapolation uncertainty and the Butzner 2026 acceleration signal.
    n_hcm_median: float = 400_000.0
    n_hcm_p95: float = 650_000.0


def build_model(k: int, n: int, priors: TamPriors = TamPriors()) -> pm.Model:
    """Joint model for (p, s, N_hcm) and derived TAM.

    Args:
        k: observed "coded-treatable" HCM patients in the claims cohort
           (I421 ∩ Disopyramide fill).
        n: total HCM-coded patients in the claims cohort.
        priors: prior hyperparameters. Change one number to run a sensitivity.
    """
    sigma_lognorm = float(np.log(priors.n_hcm_p95 / priors.n_hcm_median) / 1.645)

    with pm.Model() as model:
        p = pm.Beta("p_true_eligibility", alpha=priors.p_alpha, beta=priors.p_beta)
        s = pm.Beta("s_capture_rate", alpha=priors.s_alpha, beta=priors.s_beta)
        n_hcm = pm.LogNormal("n_hcm_us", mu=np.log(priors.n_hcm_median), sigma=sigma_lognorm)

        pm.Binomial("k_observed", n=n, p=p * s, observed=k)

        pm.Deterministic("tam", n_hcm * p)

    return model


def sample(model: pm.Model, draws: int = 2000, tune: int = 1000, seed: int = 42):
    """Run NUTS. Returns an ArviZ InferenceData."""
    with model:
        return pm.sample(
            draws=draws,
            tune=tune,
            chains=4,
            random_seed=seed,
            progressbar=False,
            target_accept=0.9,
        )

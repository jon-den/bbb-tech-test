"""Beta-Binomial conjugate update for the Camzyos-eligible fraction.

The `camzyos_eligible_fraction` prior is a Beta distribution over the fraction
of HCM patients who are Camzyos-eligible (obstructive + symptomatic + treatable).
Beta is conjugate to Binomial: given `k` claims-observable "treatable oHCM"
patients out of `n` HCM-coded patients in our cohort, the posterior is

    Beta(alpha + k, beta + n - k)

That is it. One line of maths, no MCMC needed. The plumbing here just:

  1. reads the Beta(alpha, beta) parameters off the existing prior;
  2. combines with (k, n) from `claims_evidence.py`;
  3. returns a frozen scipy Beta the funnel can sample from exactly like the prior.

Two things worth being honest about, both surfaced in the summary text and the
IC-facing figure:

- **Prior effective sample size (alpha + beta ~ 47) vs data (n ~ 19,000)**:
  the posterior is data-dominated. The Bayesian machinery is doing real work —
  it tells us the prior is essentially irrelevant given our sample size — not
  hiding a subjective bet.
- **Proxy under-coverage**: what our claims data can *see* (I421 code plus a
  symptom/escalation marker like Disopyramide) is a strict lower bound on true
  clinical eligibility, because claims routinely miss symptoms. So the posterior
  answers "what fraction of HCM patients would show up as treatable in this
  claims database", which is not identical to the literature quantity. We report
  both and let the IC pick the framing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from src.task2_tam.priors import PRIORS

FrozenDist = Any


@dataclass(frozen=True)
class PosteriorSummary:
    """Beta prior + Binomial data → Beta posterior, plus a plain-language summary.

    Attributes:
        prior_alpha: Beta prior alpha (successes-like pseudo-count).
        prior_beta: Beta prior beta (failures-like pseudo-count).
        prior_mean: Prior mean = alpha / (alpha + beta).
        prior_ess: Prior effective sample size = alpha + beta.
        k: Observed successes in the claims cohort (numerator).
        n: Observed trials in the claims cohort (denominator).
        posterior_alpha: alpha + k.
        posterior_beta: beta + (n - k).
        posterior_mean: Posterior mean.
        posterior_p05: 5th percentile of posterior.
        posterior_p95: 95th percentile of posterior.
        posterior: Frozen scipy Beta distribution the funnel can sample from.
        definition_label: Short label for which numerator definition was used.
    """

    prior_alpha: float
    prior_beta: float
    prior_mean: float
    prior_ess: float
    k: int
    n: int
    posterior_alpha: float
    posterior_beta: float
    posterior_mean: float
    posterior_p05: float
    posterior_p95: float
    posterior: FrozenDist
    definition_label: str


def _beta_params(prior_name: str) -> tuple[float, float]:
    """Read the two Beta shape params off a registered Beta prior."""
    dist = PRIORS[prior_name].distribution
    if getattr(dist, "dist", None) is None or dist.dist.name != "beta":
        raise ValueError(
            f"Prior '{prior_name}' is not a scipy Beta — got {type(dist.dist).__name__}."
        )
    alpha, beta = float(dist.args[0]), float(dist.args[1])
    return alpha, beta


def update_eligible_fraction(
    k: int,
    n: int,
    definition_label: str = "I421 ∩ Disopyramide",
    prior_name: str = "camzyos_eligible_fraction",
) -> PosteriorSummary:
    """Update the Camzyos-eligible fraction with a Beta-Binomial conjugate step.

    Args:
        k: Number of HCM-coded patients meeting the claims-based "treatable
            eligible" definition (numerator).
        n: Number of HCM-coded patients in the cohort (denominator).
        definition_label: Human-readable label for the numerator definition,
            surfaced in figures and the sources CSV.
        prior_name: Name of a registered Beta prior in PRIORS. Defaults to
            'camzyos_eligible_fraction'.

    Returns:
        PosteriorSummary with prior params, data (k, n), and the posterior
        distribution the funnel can sample from.

    Raises:
        ValueError: if k < 0, n <= 0, k > n, or the named prior is not a Beta.
    """
    if n <= 0 or k < 0 or k > n:
        raise ValueError(f"Invalid data: need 0 <= k <= n and n > 0, got k={k}, n={n}.")

    alpha, beta = _beta_params(prior_name)
    post_alpha = alpha + k
    post_beta = beta + (n - k)
    posterior = stats.beta(post_alpha, post_beta)

    return PosteriorSummary(
        prior_alpha=alpha,
        prior_beta=beta,
        prior_mean=alpha / (alpha + beta),
        prior_ess=alpha + beta,
        k=int(k),
        n=int(n),
        posterior_alpha=post_alpha,
        posterior_beta=post_beta,
        posterior_mean=float(posterior.mean()),
        posterior_p05=float(posterior.ppf(0.05)),
        posterior_p95=float(posterior.ppf(0.95)),
        posterior=posterior,
        definition_label=definition_label,
    )


def sample_posterior(
    posterior: PosteriorSummary, size: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw `size` iid samples from the posterior Beta.

    Args:
        posterior: A PosteriorSummary returned by `update_eligible_fraction`.
        size: Number of draws.
        rng: NumPy Generator for reproducibility.
    """
    return posterior.posterior.rvs(size=size, random_state=rng)

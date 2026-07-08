"""Unit tests for the Beta-Binomial conjugate update on the eligible fraction."""

import numpy as np
import pytest
from scipy import stats

from src.task2_tam.bayesian_update import (
    sample_posterior,
    update_eligible_fraction,
)
from src.task2_tam.priors import PRIORS


class TestUpdateEligibleFraction:
    def test_posterior_matches_analytic_beta(self):
        """Posterior params must equal Beta(alpha + k, beta + n - k) exactly."""
        prior_dist = PRIORS["camzyos_eligible_fraction"].distribution
        alpha, beta = float(prior_dist.args[0]), float(prior_dist.args[1])
        k, n = 800, 20_000
        post = update_eligible_fraction(k, n)
        assert post.posterior_alpha == pytest.approx(alpha + k)
        assert post.posterior_beta == pytest.approx(beta + n - k)

    def test_posterior_mean_between_prior_mean_and_data_rate(self):
        """A basic sanity check: posterior mean must sit between prior mean and k/n."""
        k, n = 800, 20_000
        post = update_eligible_fraction(k, n)
        rate = k / n
        lo, hi = sorted([post.prior_mean, rate])
        assert lo <= post.posterior_mean <= hi

    def test_data_dominates_when_n_much_larger_than_ess(self):
        """When n >> alpha + beta, the posterior mean should be close to k/n."""
        k, n = 800, 20_000  # ESS ≈ 47, so n / ESS ≈ 400
        post = update_eligible_fraction(k, n)
        assert post.posterior_mean == pytest.approx(k / n, rel=0.05)

    def test_prior_dominates_when_n_much_smaller_than_ess(self):
        """When n << alpha + beta, the posterior mean should stay near the prior."""
        k, n = 1, 3
        post = update_eligible_fraction(k, n)
        # Prior ESS ≈ 47; n = 3 barely moves it. Difference from prior mean
        # should be tiny relative to the prior CI width.
        prior_ci_half_width = (post.posterior.ppf(0.95) - post.posterior.ppf(0.05)) / 2
        assert abs(post.posterior_mean - post.prior_mean) < prior_ci_half_width

    def test_ci_narrows_with_more_data(self):
        """CI width must be non-increasing in n at fixed k/n."""
        widths = []
        for n in (100, 1_000, 10_000, 100_000):
            k = int(n * 0.04)
            post = update_eligible_fraction(k, n)
            widths.append(post.posterior_p95 - post.posterior_p05)
        # Strict monotone decrease within numerical tolerance.
        for a, b in zip(widths, widths[1:]):
            assert b < a

    @pytest.mark.parametrize("k,n", [(-1, 100), (10, -5), (110, 100), (0, 0)])
    def test_invalid_inputs_raise(self, k, n):
        with pytest.raises(ValueError):
            update_eligible_fraction(k, n)

    def test_unknown_prior_name_raises(self):
        with pytest.raises(KeyError):
            update_eligible_fraction(10, 100, prior_name="not_a_prior")


class TestSamplePosterior:
    def test_returns_correct_shape(self):
        post = update_eligible_fraction(800, 20_000)
        rng = np.random.default_rng(0)
        samples = sample_posterior(post, size=500, rng=rng)
        assert samples.shape == (500,)

    def test_samples_within_unit_interval(self):
        post = update_eligible_fraction(800, 20_000)
        rng = np.random.default_rng(0)
        samples = sample_posterior(post, size=1_000, rng=rng)
        assert samples.min() > 0
        assert samples.max() < 1

    def test_sample_mean_matches_posterior_mean(self):
        post = update_eligible_fraction(800, 20_000)
        rng = np.random.default_rng(0)
        samples = sample_posterior(post, size=50_000, rng=rng)
        # 3 sigma for a mean of 50k Beta(~800+alpha, ~19200+beta) draws is tiny.
        assert samples.mean() == pytest.approx(post.posterior_mean, abs=0.001)

    def test_reproducible_with_seed(self):
        post = update_eligible_fraction(800, 20_000)
        s1 = sample_posterior(post, size=1_000, rng=np.random.default_rng(42))
        s2 = sample_posterior(post, size=1_000, rng=np.random.default_rng(42))
        assert np.array_equal(s1, s2)


class TestPriorRegistryContract:
    """The update assumes the eligible-fraction prior is a scipy Beta.

    Failing this test means someone changed the prior spec — the update is
    still valid but the ESS interpretation in figures/docs will drift.
    """

    def test_eligible_prior_is_beta(self):
        dist = PRIORS["camzyos_eligible_fraction"].distribution
        assert dist.dist.name == "beta"

    def test_prior_mean_matches_specialty_registry(self):
        dist = PRIORS["camzyos_eligible_fraction"].distribution
        # Registry value is 0.307; allow small drift from CI-based construction.
        assert float(stats.beta(dist.args[0], dist.args[1]).mean()) == pytest.approx(
            0.307, abs=0.01
        )

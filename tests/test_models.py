"""Smoke tests for DiscreteHazardGLM, GBMHazardBenchmark, and MarginalRateModel.

Covers the API surface actually used in the pipeline. Does not use
check_estimator() because DiscreteHazardGLM wraps statsmodels and
cannot satisfy the full sklearn contract.
"""

import numpy as np
import pandas as pd
import pytest

from src.models import DiscreteHazardGLM, GBMHazardBenchmark, MarginalRateModel


@pytest.fixture
def binary_data():
    """Minimal binary dataset with ~5% event rate (typical for monthly hazard panels)."""
    rng = np.random.RandomState(0)
    n = 200
    X = pd.DataFrame({"x1": rng.randn(n), "x2": rng.randn(n)})
    y = pd.Series((rng.rand(n) < 0.05).astype(int))
    return X, y


class TestDiscreteHazardGLM:
    def test_fit_returns_self(self, binary_data):
        X, y = binary_data
        model = DiscreteHazardGLM()
        assert model.fit(X, y) is model

    def test_predict_proba_shape(self, binary_data):
        X, y = binary_data
        proba = DiscreteHazardGLM().fit(X, y).predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_predict_uses_rate_threshold(self, binary_data):
        """At ~5% event rate, predict() thresholding at rate_ must return some positives."""
        X, y = binary_data
        preds = DiscreteHazardGLM().fit(X, y).predict(X)
        assert preds.sum() > 0, "threshold bug: predict() at 0.5 returns all zeros for rare events"

    def test_coef_table_has_hr_columns(self, binary_data):
        X, y = binary_data
        tbl = DiscreteHazardGLM(link="cloglog").fit(X, y).coef_table
        assert {"HR", "HR_lower", "HR_upper"}.issubset(tbl.columns)

    def test_coef_table_or_for_logit(self, binary_data):
        X, y = binary_data
        tbl = DiscreteHazardGLM(link="logit").fit(X, y).coef_table
        assert "OR" in tbl.columns


class TestGBMHazardBenchmark:
    def test_fit_returns_self(self, binary_data):
        X, y = binary_data
        model = GBMHazardBenchmark()
        assert model.fit(X, y) is model

    def test_predict_uses_rate_threshold(self, binary_data):
        """GBM must share the same rate_ threshold contract as DiscreteHazardGLM."""
        X, y = binary_data
        preds = GBMHazardBenchmark().fit(X, y).predict(X)
        assert preds.sum() > 0, "rate_ threshold not applied in GBMHazardBenchmark.predict()"


class TestMarginalRateModel:
    def test_predict_proba_constant(self, binary_data):
        """Null model: every row gets the training marginal rate as its predicted probability."""
        X, y = binary_data
        model = MarginalRateModel().fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        np.testing.assert_allclose(proba[:, 1], y.mean(), atol=1e-10)

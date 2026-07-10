"""Smoke tests for StabilitySelector.

Covers the API surface actually used in the pipeline. StabilitySelector is
intentionally NOT check_estimator-compliant because:
- fit() accepts an extra patient_ids argument (breaks CV pipelines)
- both fit() and transform() require a DataFrame with named columns (not a plain array)
"""

import numpy as np
import pandas as pd
import pytest

from src.task1_adoption.selection import StabilitySelector


@pytest.fixture
def person_month_data():
    """Small person-month dataset with 3 features and ~5% event rate."""
    rng = np.random.RandomState(1)
    n_patients, n_months = 80, 5
    patient_ids = np.repeat(np.arange(n_patients), n_months)
    X = pd.DataFrame(
        {
            "feat_a": rng.randn(n_patients * n_months),
            "feat_b": rng.randn(n_patients * n_months),
            "feat_c": rng.randn(n_patients * n_months),
        }
    )
    y = pd.Series((rng.rand(n_patients * n_months) < 0.05).astype(int))
    return X, y, patient_ids


class TestStabilitySelector:
    def test_fit_returns_self(self, person_month_data):
        X, y, pids = person_month_data
        sel = StabilitySelector(n_bootstrap=10, random_state=0)
        assert sel.fit(X, y, patient_ids=pids) is sel

    def test_selected_features_subset_of_columns(self, person_month_data):
        X, y, pids = person_month_data
        sel = StabilitySelector(n_bootstrap=10, threshold=0.0, random_state=0).fit(
            X, y, patient_ids=pids
        )
        assert set(sel.selected_features_).issubset(set(X.columns))

    def test_transform_returns_subset(self, person_month_data):
        X, y, pids = person_month_data
        sel = StabilitySelector(n_bootstrap=10, threshold=0.0, random_state=0).fit(
            X, y, patient_ids=pids
        )
        X_t = sel.transform(X)
        assert isinstance(X_t, pd.DataFrame)
        assert set(X_t.columns).issubset(set(X.columns))

    def test_transform_requires_dataframe(self, person_month_data):
        """Documents the DataFrame-only API contract — numpy arrays are rejected."""
        X, y, pids = person_month_data
        sel = StabilitySelector(n_bootstrap=10, threshold=0.0, random_state=0).fit(
            X, y, patient_ids=pids
        )
        with pytest.raises(ValueError, match="DataFrame"):
            sel.transform(X.values)

    def test_c_used_stored(self, person_month_data):
        """C='auto' must tune via CV and store the result for reproducibility."""
        X, y, pids = person_month_data
        sel = StabilitySelector(n_bootstrap=10, C="auto", random_state=0).fit(
            X, y, patient_ids=pids
        )
        assert hasattr(sel, "C_used_") and sel.C_used_ > 0

    def test_without_patient_ids(self, person_month_data):
        """Fallback: subsample rows when patient_ids not provided."""
        X, y, _ = person_month_data
        sel = StabilitySelector(n_bootstrap=10, random_state=0).fit(X, y)
        assert hasattr(sel, "selected_features_")

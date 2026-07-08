"""Unit tests for the claims-based (k, n) extraction.

These use the real synthetic dataset (small and cached), so they double as an
integration test that the definitions still resolve on the shipped data.
"""

import pytest

from src.task1_adoption.data_loading import load_data
from src.task2_tam.claims_evidence import collect_evidence, evidence_dataframe


@pytest.fixture(scope="module")
def evidence():
    return collect_evidence(load_data())


def test_three_definitions(evidence):
    assert len(evidence) == 3
    labels = [e.label for e in evidence]
    assert labels[0].startswith("D1")
    assert labels[1].startswith("D2")
    assert labels[2].startswith("D3")


def test_denominator_shared_across_definitions(evidence):
    n_values = {e.n for e in evidence}
    assert len(n_values) == 1, "All three definitions must share the HCM denominator"


def test_numerators_monotone_non_decreasing(evidence):
    """D1 ⊆ D2 ⊆ D3, so k values must be non-decreasing."""
    ks = [e.k for e in evidence]
    assert ks[0] <= ks[1] <= ks[2]


def test_rates_positive_and_below_one(evidence):
    for e in evidence:
        assert 0 < e.rate < 1


def test_evidence_dataframe_shape(evidence):
    df = evidence_dataframe(evidence)
    assert set(df.columns) == {
        "definition",
        "numerator_k",
        "denominator_n",
        "rate",
        "description",
    }
    assert len(df) == 3


def test_expected_denominator_matches_shipped_data(evidence):
    # Regression: 18,953 HCM-coded patients in the shipped synthetic cohort.
    # If this drifts, the write-up citation in METHODS.md needs updating.
    assert evidence[0].n == 18_953

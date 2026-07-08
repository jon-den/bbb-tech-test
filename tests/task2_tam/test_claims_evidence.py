"""Regression tests for the claims-based (k, n) extraction on the shipped data."""

from src.task1_adoption.data_loading import load_data
from src.task2_tam.claims_evidence import collect_evidence


def test_evidence_on_shipped_cohort():
    ev = collect_evidence(load_data())
    assert ev.n == 18_953, "HCM-coded denominator drifted — update case study"
    assert ev.k == 769, "I421 ∩ Diso numerator drifted — update case study"
    assert 0 < ev.rate < 1

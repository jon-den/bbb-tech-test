"""Extract (k, n) evidence from the claims cohort for the Bayesian update.

The prior on `camzyos_eligible_fraction` means:
    P(patient is obstructive AND symptomatic AND treatable | HCM patient)

To update this prior we need a within-HCM ratio the claims data can measure.

    Denominator (n): unique patients with any HCM-family diagnosis code
                     (I421, I422, I429). This is the population the prior
                     is defined over.

    Numerator (k):   patients meeting a claims-based proxy for "treatable oHCM".
                     We report three definitions so the IC can see how the
                     posterior moves under different operational choices —
                     the range brackets sensitivity to proxy selection.

The three definitions borrow directly from Task 1's escalation-ladder markers.
Task 1 showed that Disopyramide is a near-definitional precondition for
Camzyos initiation (98.8% of initiators, F6/F21) — so Disopyramide fill is
the single strongest single-signal marker of symptomatic obstructive disease.
SRT and multi-drug HCM regimens are complementary escalation signals.

    D1 — Strict     : I421 ∩ Disopyramide
                      Matches the prior's meaning most tightly (obstructive
                      AND symptomatic enough to warrant a label-indicated
                      antiarrhythmic).
    D2 — Middle     : I421 ∩ (Disopyramide OR SRT)
                      Adds septal reduction therapy as an equivalent
                      escalation signal.
    D3 — Broad      : I421 ∩ (Disopyramide OR SRT OR ≥3 HCM drug classes)
                      Adds a multi-drug HCM regimen as a looser escalation
                      proxy.

All three are strict LOWER BOUNDS on true clinical eligibility because
claims routinely fail to capture symptom severity (see FINDINGS.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.task1_adoption.config import (
    DISOPYRAMIDE_CODE,
    HCM_ELIGIBILITY_CODES,
    HCM_MEDS,
    OHCM_CODE,
    SRT_CODE,
)
from src.task1_adoption.data_loading import RawData


@dataclass(frozen=True)
class TreatableEvidence:
    """One (k, n) observation for the Beta-Binomial update.

    Attributes:
        label: Short human-readable label used in figures / CSVs.
        description: Full plain-language definition for the write-up.
        k: Observed numerator (patients meeting the proxy).
        n: Observed denominator (HCM-coded patients).
        rate: k / n, the empirical fraction.
    """

    label: str
    description: str
    k: int
    n: int

    @property
    def rate(self) -> float:
        """Observed rate k/n (nan if n == 0)."""
        return self.k / self.n if self.n > 0 else float("nan")


def _hcm_denominator(data: RawData) -> set[int]:
    """Patients with any HCM-family diagnosis code (I421, I422, I429)."""
    return set(
        data.diagnoses.loc[
            data.diagnoses["dx_code"].isin(HCM_ELIGIBILITY_CODES), "patient_id"
        ].unique()
    )


def _obstructive_coded(data: RawData) -> set[int]:
    """Patients with the strict obstructive HCM code (I421)."""
    return set(data.diagnoses.loc[data.diagnoses["dx_code"] == OHCM_CODE, "patient_id"].unique())


def _disopyramide_ever(data: RawData) -> set[int]:
    """Patients with ≥1 Disopyramide fill (label-indicated antiarrhythmic for oHCM)."""
    return set(
        data.prescriptions.loc[
            data.prescriptions["rx_code"] == DISOPYRAMIDE_CODE, "patient_id"
        ].unique()
    )


def _srt_ever(data: RawData) -> set[int]:
    """Patients with ≥1 septal reduction therapy procedure."""
    return set(data.procedures.loc[data.procedures["px_code"] == SRT_CODE, "patient_id"].unique())


def _multi_hcm_meds(data: RawData, min_distinct: int = 3) -> set[int]:
    """Patients with ≥ min_distinct distinct HCM-guideline medications ever."""
    hcm_rx = data.prescriptions[data.prescriptions["rx_code"].isin(HCM_MEDS)]
    counts = hcm_rx.groupby("patient_id")["rx_code"].nunique()
    return set(counts[counts >= min_distinct].index)


def collect_evidence(data: RawData) -> list[TreatableEvidence]:
    """Return three (k, n) definitions of "claims-observable treatable oHCM".

    Args:
        data: RawData loaded by src.task1_adoption.data_loading.load_data.

    Returns:
        List of three TreatableEvidence rows (strict, middle, broad) sharing
        the same denominator (HCM-coded patients). Ordered by k ascending.
    """
    hcm = _hcm_denominator(data)
    obs = _obstructive_coded(data)
    diso = _disopyramide_ever(data)
    srt = _srt_ever(data)
    multi = _multi_hcm_meds(data)

    n = len(hcm)
    d1 = obs & diso
    d2 = obs & (diso | srt)
    d3 = obs & (diso | srt | multi)

    return [
        TreatableEvidence(
            label="D1: I421 ∩ Diso",
            description=(
                "Obstructive HCM code (I421) AND Disopyramide fill. "
                "Disopyramide is label-indicated for symptomatic oHCM and was "
                "present in 98.8% of Camzyos initiators (Task 1 F6/F21) — the "
                "single strongest claims-based marker of symptomatic obstructive disease."
            ),
            k=len(d1),
            n=n,
        ),
        TreatableEvidence(
            label="D2: I421 ∩ (Diso ∪ SRT)",
            description=(
                "Obstructive HCM code AND (Disopyramide OR septal reduction "
                "therapy). Adds SRT as an equivalent escalation signal for "
                "patients who bypassed Disopyramide."
            ),
            k=len(d2),
            n=n,
        ),
        TreatableEvidence(
            label="D3: I421 ∩ (Diso ∪ SRT ∪ ≥3 HCM meds)",
            description=(
                "Obstructive HCM code AND any of (Disopyramide, SRT, ≥3 "
                "distinct HCM-guideline medications). Broadest escalation proxy."
            ),
            k=len(d3),
            n=n,
        ),
    ]


def evidence_dataframe(evidence: list[TreatableEvidence]) -> pd.DataFrame:
    """Tidy the evidence list for the sources CSV / figure captions."""
    return pd.DataFrame(
        {
            "definition": [e.label for e in evidence],
            "numerator_k": [e.k for e in evidence],
            "denominator_n": [e.n for e in evidence],
            "rate": [round(e.rate, 4) for e in evidence],
            "description": [e.description for e in evidence],
        }
    )

"""Extract (k, n) evidence from the claims cohort for the Bayesian update.

The Task 2 model needs one ratio from our claims cohort — the fraction of
HCM patients who look "treatable" by the Task 1 escalation markers.

    n  =  patients with any HCM-family diagnosis code (I421 / I422 / I429).
    k  =  those who also have I421 (obstructive) AND a Disopyramide fill.

Disopyramide is label-indicated for symptomatic oHCM and was present in
98.8% of Camzyos initiators in Task 1 — the single strongest claims-based
signal of symptomatic obstructive disease. This is a strict lower bound on
true clinical eligibility because claims routinely miss symptom detail.
The Task 2 model handles that gap explicitly via the capture-rate `s`.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.task1_adoption.config import (
    DISOPYRAMIDE_CODE,
    HCM_ELIGIBILITY_CODES,
    OHCM_CODE,
)
from src.task1_adoption.data_loading import RawData


@dataclass(frozen=True)
class TreatableEvidence:
    """One (k, n) observation for the Task 2 Binomial likelihood."""

    k: int
    n: int

    @property
    def rate(self) -> float:
        """Observed rate k/n."""
        return self.k / self.n if self.n > 0 else float("nan")


def collect_evidence(data: RawData) -> TreatableEvidence:
    """Return (k, n) for the I421 ∩ Disopyramide claims-treatable definition."""
    hcm = set(
        data.diagnoses.loc[
            data.diagnoses["dx_code"].isin(HCM_ELIGIBILITY_CODES), "patient_id"
        ].unique()
    )
    obs = set(data.diagnoses.loc[data.diagnoses["dx_code"] == OHCM_CODE, "patient_id"].unique())
    diso = set(
        data.prescriptions.loc[
            data.prescriptions["rx_code"] == DISOPYRAMIDE_CODE, "patient_id"
        ].unique()
    )
    return TreatableEvidence(k=len(obs & diso), n=len(hcm))

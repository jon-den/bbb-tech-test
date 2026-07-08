"""Task 2 — US total addressable market (TAM) for Camzyos.

Two modules, one question:

    claims_evidence : extract (k, n) from the claims cohort — Task 1's
                      I421 ∩ Disopyramide escalation markers.
    tam_model       : one PyMC model — joint update on true eligibility p,
                      claims capture rate s, and diagnosed HCM count N;
                      TAM = N × p.
"""

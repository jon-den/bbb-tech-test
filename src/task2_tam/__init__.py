"""Task 2 — US total addressable market (TAM) for Camzyos.

Top-down epidemiological funnel: `TAM = US adults × diagnosed HCM prevalence
× Camzyos-eligible fraction`. Each factor is a triangular distribution over
directly cited published bounds; Monte Carlo propagates the joint uncertainty
to a TAM distribution. Every parameter is anchored on a specific source in
`FunnelParams`.
"""

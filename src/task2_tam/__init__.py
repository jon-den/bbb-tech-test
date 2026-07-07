"""Total addressable market (TAM) Monte Carlo pipeline for Camzyos.

Modules
-------
priors      : Registry of every distribution used in the pipeline, with citations.
funnel      : Eligibility funnel — Pool A (theoretical) and Pool B (diagnosed today).
diffusion   : Logistic penetration curve with aficamten share haircut.
revenue     : Patient count → US net revenue conversion.
simulation  : Monte Carlo orchestrator (n_draws × time-steps arrays).
sensitivity : One-at-a-time tornado on each pool.

Design principle
----------------
No magic numbers outside `priors.py`. Every scalar in the pipeline traces to a
named `Prior` with a source string, so `outputs/task2_tam/09_tam_sources.csv`
is a complete audit trail for the IC.
"""

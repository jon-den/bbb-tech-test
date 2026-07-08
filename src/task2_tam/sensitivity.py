"""One-at-a-time (OAT) tornado sensitivity on Pool A and Pool B.

For each input, move to its 5th and 95th percentile while holding others at
median. The change in the output pool is the tornado bar.

Deliberately not Sobol — OAT is easier to explain to a non-quant audience and
sufficient for a multiplicative funnel where main effects dominate.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from src.task2_tam.priors import PRIORS, US_ADULTS_20_PLUS


def _tornado(inputs: list[str], f: Callable[[dict[str, float]], float]) -> pd.DataFrame:
    baseline = f({n: PRIORS[n].median for n in inputs})
    rows = []
    for name in inputs:
        med = {n: PRIORS[n].median for n in inputs}
        med[name] = PRIORS[name].p05
        low = f(med)
        med[name] = PRIORS[name].p95
        high = f(med)
        rows.append(
            {"input": name, "baseline": baseline, "low": low, "high": high, "swing": high - low}
        )
    return pd.DataFrame(rows).set_index("input").sort_values("swing", ascending=False)


_ELG = "camzyos_eligible_fraction"


def tornado_for_pool_a() -> pd.DataFrame:
    """Tornado on Pool A (theoretical addressable): pop × prev × eligible."""
    return _tornado(
        ["hcm_prevalence", _ELG],
        lambda v: US_ADULTS_20_PLUS * v["hcm_prevalence"] * v[_ELG],
    )


def tornado_for_pool_b() -> pd.DataFrame:
    """Tornado on Pool B (diagnosed & treatable today): diagnosed × eligible."""
    return _tornado(
        ["diagnosed_hcm_us_current", _ELG],
        lambda v: v["diagnosed_hcm_us_current"] * v[_ELG],
    )

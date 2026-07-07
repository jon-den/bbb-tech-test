"""Patient count → US net revenue conversion.

Simple: patients × net_price_per_year = annualised revenue. In this module we
work in ANNUAL revenue units (USD millions), computed from monthly prevalent
patient counts by taking the annual average of each calendar year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def annual_revenue_from_monthly_patients(
    monthly_patients: np.ndarray,
    month_grid: pd.DatetimeIndex,
    net_price_per_year_usd: float,
) -> pd.Series:
    """Convert monthly prevalent-patient count to annual net revenue (USD millions).

    Args:
        monthly_patients: 1D array of prevalent patients per month.
        month_grid: Monthly timestamps corresponding to `monthly_patients`.
        net_price_per_year_usd: Net price per patient-year.

    Returns:
        Series indexed by calendar year, values in USD millions.
    """
    df = pd.DataFrame({"patients": monthly_patients}, index=month_grid)
    yearly_avg = df.groupby(df.index.year)["patients"].mean()
    return yearly_avg * net_price_per_year_usd / 1e6

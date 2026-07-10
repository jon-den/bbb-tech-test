"""Evaluation metrics for discrete-time hazard models."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import brier_score_loss, roc_auc_score


@dataclass
class DatasetCols:
    """Column name configuration for person-month dataset DataFrames.

    Centralises the magic strings used across calibration and discrimination
    functions so that renaming a column requires changing one place.
    """

    patient_id: str = "patient_id"
    month: str = "study_month"
    event: str = "event"
    pred: str = "pred"


_DEFAULT_COLS = DatasetCols()


# ── Row-level calibration ────────────────────────────────────────────────────


def calibration_table(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    n_bins: int = 5,
) -> pd.DataFrame:
    """Predicted vs observed event rates in quantile bins.

    Returns:
        DataFrame with columns: bin, n, events, mean_pred, observed_rate.
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_pred)})
    df["bin"] = pd.qcut(df["p"], n_bins, duplicates="drop")
    return (
        df.groupby("bin", observed=True)
        .agg(
            n=("y", "size"),
            events=("y", "sum"),
            mean_pred=("p", "mean"),
            observed_rate=("y", "mean"),
        )
        .reset_index()
    )


def calibration_plot(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    n_bins: int = 5,
    ax: plt.Axes | None = None,
    label: str | None = None,
) -> plt.Axes:
    """Calibration plot: predicted vs observed event rates in quantile bins."""
    tbl = calibration_table(y_true, y_pred, n_bins)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        tbl["mean_pred"], tbl["observed_rate"], s=tbl["n"] * 2, zorder=3, label=label, alpha=0.8
    )
    lims = [0, max(tbl["mean_pred"].max(), tbl["observed_rate"].max()) * 1.2]
    ax.plot(lims, lims, "k--", alpha=0.4, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed event rate")
    ax.set_title("Calibration plot")
    if label:
        ax.legend()
    return ax


def count_calibration(
    dataset: pd.DataFrame,
    pred_col: str = _DEFAULT_COLS.pred,
    month_col: str = _DEFAULT_COLS.month,
    event_col: str = _DEFAULT_COLS.event,
) -> pd.DataFrame:
    """Compare predicted vs observed monthly new starts.

    Sums predicted hazards across the risk set each month → expected new
    starts. Compare to observed count.
    """
    return (
        dataset.groupby(month_col)
        .agg(
            observed=(event_col, "sum"),
            predicted=(pred_col, "sum"),
            n_at_risk=(event_col, "size"),
        )
        .reset_index()
    )


def count_calibration_plot(
    monthly: pd.DataFrame,
    month_col: str = _DEFAULT_COLS.month,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Bar chart: predicted vs observed monthly new starts."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    x = monthly[month_col]
    bar_width = 0.4
    ax.bar(x, monthly["observed"], alpha=0.5, label="Observed", width=bar_width, align="edge")
    ax.bar(
        x + bar_width,
        monthly["predicted"],
        alpha=0.5,
        label="Predicted",
        width=bar_width,
        align="edge",
        color="C1",
    )
    ax.set_xlabel("Study month")
    ax.set_ylabel("New initiations")
    ax.set_title("Count-level calibration: predicted vs observed monthly new starts")
    ax.legend()
    return ax


def brier_decomposition(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Brier score decomposed against the null (marginal rate) model."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    bs = brier_score_loss(y_true, y_pred)
    bs_null = brier_score_loss(y_true, np.full_like(y_pred, y_true.mean()))
    return {"brier_score": bs, "brier_null": bs_null, "brier_skill_score": 1 - bs / bs_null}


# ── Patient-level discrimination ─────────────────────────────────────────────


def time_dependent_auc(
    dataset: pd.DataFrame,
    pred_col: str = _DEFAULT_COLS.pred,
    cols: DatasetCols = _DEFAULT_COLS,
) -> dict[str, float | int]:
    """Monthly time-dependent AUROC, weighted by event count.

    For each study month t: AUROC of the predicted hazard h_t against the
    binary event indicator (cases = initiate at t; controls = at-risk
    non-initiators at t). Weighted average across all months with ≥1 event.

    This is the correct discrimination metric for a discrete-time hazard model:
    it evaluates whether the model ranks patients correctly *within* each month,
    without accumulating predictions over time. Cumulative-incidence summaries
    are fundamentally length-confounded — F(T_i) grows monotonically with T_i
    regardless of the model's quality, so patients followed longer always appear
    higher-risk.

    Returns nan if no month has both events and non-events.
    """
    aucs: list[float] = []
    weights: list[int] = []
    for _, grp in dataset.groupby(cols.month):
        n_ev = int(grp[cols.event].sum())
        n_ctrl = int((grp[cols.event] == 0).sum())
        if n_ev == 0 or n_ctrl == 0:
            continue
        aucs.append(float(roc_auc_score(grp[cols.event], grp[pred_col])))
        weights.append(n_ev)

    if not aucs:
        return {"time_dependent_auc": float("nan"), "n_months_evaluated": 0}

    weights_arr = np.array(weights, dtype=float)
    aucs_arr = np.array(aucs)
    return {
        "time_dependent_auc": float((aucs_arr * weights_arr).sum() / weights_arr.sum()),
        "n_months_evaluated": len(aucs),
    }


# ── Composite evaluation ─────────────────────────────────────────────────────


def evaluate_model(
    model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    dataset_test: pd.DataFrame,
    model_name: str = "Model",
    cols: DatasetCols = _DEFAULT_COLS,
) -> tuple[dict, pd.DataFrame]:
    """Run full evaluation suite; return results dict and monthly calibration table.

    All metrics are computed on the test set only:
    - Brier score / BSS: row-level calibration
    - Time-dependent AUC: weighted monthly AUROC (no length-confounding)
    - Count calibration: predicted vs observed monthly initiations

    Discrimination is evaluated month-by-month (time_dependent_auc) rather
    than via a patient-level C-index. Cumulative-incidence C-indices are
    inappropriate here because F(T_i) grows with follow-up length, causing
    censored patients followed longer to always outrank event patients who
    exit earlier — regardless of model quality.
    """
    y_pred_test = model.predict_proba(X_test)[:, 1]

    dataset_test = dataset_test.copy()
    dataset_test[cols.pred] = y_pred_test

    brier = brier_decomposition(y_test.values, y_pred_test)
    monthly = count_calibration(
        dataset_test, pred_col=cols.pred, month_col=cols.month, event_col=cols.event
    )
    disc = time_dependent_auc(dataset_test, pred_col=cols.pred, cols=cols)

    results = {
        "name": model_name,
        **brier,
        **disc,
        "mean_pred": float(y_pred_test.mean()),
        "mean_observed": float(y_test.mean()),
        "n_rows": len(y_test),
        "n_events": int(y_test.sum()),
    }
    return results, monthly

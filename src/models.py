"""Discrete-time hazard models and benchmarks with sklearn-compatible API."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import CLogLog as CLogLogLink
from statsmodels.genmod.families.links import Logit as LogitLink


class DiscreteHazardGLM(BaseEstimator, ClassifierMixin):
    """Binary GLM for discrete-time hazard estimation.

    Parameters
    ----------
    link : {"cloglog", "logit"}
        Link function. cloglog gives grouped proportional hazards
        (coefficients are log-hazard-ratios). logit gives log-odds-ratios.
    add_intercept : bool
        Whether to add a constant column.

    Notes:
    -----
    predict() thresholds at the training marginal rate (stored as rate_),
    not at 0.5. Monthly initiation hazards are typically 1-3%, so a 0.5
    threshold would classify everything as non-event. predict_proba()
    is the correct output to use downstream; hard labels from predict()
    are rarely meaningful for hazard models.
    """

    def __init__(self, link="cloglog", add_intercept=True):
        self.link = link
        self.add_intercept = add_intercept

    def fit(self, X, y):
        """Fit binary GLM on panel data; store result_ and rate_; return self."""
        link_fn = CLogLogLink() if self.link == "cloglog" else LogitLink()
        X_fit = self._prepare_X(X)
        self.feature_names_ = list(X_fit.columns) if hasattr(X_fit, "columns") else None
        self.result_ = sm.GLM(np.asarray(y), np.asarray(X_fit), family=Binomial(link=link_fn)).fit()
        self.classes_ = np.array([0, 1])
        self.rate_ = float(np.mean(y))  # marginal rate used as predict() threshold
        return self

    def predict_proba(self, X):
        """Return (n_samples, 2) predicted probability array."""
        X_pred = self._prepare_X(X)
        p = self.result_.predict(np.asarray(X_pred))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        """Hard labels thresholded at the training marginal rate.

        Use predict_proba() for risk scoring — hard labels are not
        meaningful for hazard models with rates well below 0.5.
        """
        return (self.predict_proba(X)[:, 1] >= self.rate_).astype(int)

    def _prepare_X(self, X):
        if self.add_intercept:
            if isinstance(X, pd.DataFrame):
                if "const" not in X.columns:
                    X = sm.add_constant(X)
            else:
                X = sm.add_constant(X)
        return X

    @property
    def coef_table(self) -> pd.DataFrame:
        """Coefficient table with hazard ratios and 95% CI.

        Schema depends on link function: columns are HR/HR_lower/HR_upper
        when link='cloglog', or OR/OR_lower/OR_upper when link='logit'.
        """
        ci = self.result_.conf_int()
        tbl = pd.DataFrame(
            {
                "coef": self.result_.params,
                "se": self.result_.bse,
                "z": self.result_.tvalues,
                "p": self.result_.pvalues,
                "ci_lower": ci[:, 0],
                "ci_upper": ci[:, 1],
            }
        )
        if self.feature_names_:
            tbl.index = self.feature_names_
        label = "HR" if self.link == "cloglog" else "OR"
        tbl[label] = np.exp(tbl["coef"])
        tbl[f"{label}_lower"] = np.exp(tbl["ci_lower"])
        tbl[f"{label}_upper"] = np.exp(tbl["ci_upper"])
        return tbl


class MarginalRateModel(BaseEstimator, ClassifierMixin):
    """Null model: predict the marginal event rate for everyone."""

    def fit(self, X, y):
        """Store training event rate; return self."""
        self.rate_ = float(np.mean(y))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        """Return constant marginal rate for all rows."""
        n = X.shape[0]
        p = np.full(n, self.rate_)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        """Return all-zeros (marginal rate is always below any useful threshold)."""
        return np.zeros(X.shape[0], dtype=int)


class GBMHazardBenchmark(BaseEstimator, ClassifierMixin):
    """HistGradientBoostingClassifier configured for low-event-rate hazard estimation.

    Nonlinear benchmark against DiscreteHazardGLM. Not interpretable for IC
    use (no meaningful coefficient table) — compare via time-dependent AUC only.

    Parameters
    ----------
    max_iter : int
        Boosting iterations.
    max_leaf_nodes : int
        Tree depth cap — 15 (vs. sklearn default 31) guards against overfitting
        on the small event counts typical in hazard panels.
    min_samples_leaf : int
        Minimum leaf size; helps with sparse event rows.
    random_state : int
    """

    def __init__(self, max_iter=100, max_leaf_nodes=15, min_samples_leaf=20, random_state=42):
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X, y):
        """Fit HistGradientBoosting on panel data; store gbm_ and rate_; return self."""
        self.gbm_ = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        self.gbm_.fit(X, y)
        self.classes_ = np.array([0, 1])
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        """Delegate to the fitted gbm_."""
        return self.gbm_.predict_proba(X)

    def predict(self, X):
        """Hard labels thresholded at rate_ — same contract as DiscreteHazardGLM.predict()."""
        return (self.predict_proba(X)[:, 1] >= self.rate_).astype(int)


def cox_discretization_check(
    panel_train: pd.DataFrame,
    feature_cols: list,
    glm: DiscreteHazardGLM,
    id_col: str = "patient_id",
    event_col: str = "event",
    month_col: str = "study_month",
) -> pd.DataFrame:
    """Compare cloglog GLM hazard ratios against lifelines CoxTimeVaryingFitter.

    Validates that person-month discretization is consistent with continuous-time
    partial likelihood: if HRs agree (same sign, similar magnitude), the grouped-
    Cox approximation is sound.

    Parameters
    ----------
    panel_train : person-month DataFrame with id_col, event_col, month_col, and feature_cols.
    feature_cols : clinical feature column names (NOT study_month — Cox absorbs time via
        the baseline hazard; no time covariate is needed).
    glm : fitted DiscreteHazardGLM whose coef_table provides the cloglog HRs.
    id_col, event_col, month_col : column name overrides.

    Returns:
    -------
    DataFrame indexed by feature with columns:
        cox_hr, cloglog_hr, ratio (cox/cloglog), sign_match (bool).
    """
    from lifelines import CoxTimeVaryingFitter

    # Build start/stop panel for lifelines (counting-process format)
    keep_cols = [id_col, event_col, month_col] + [
        f for f in feature_cols if f in panel_train.columns
    ]
    df = panel_train[keep_cols].copy()
    df["_start"] = df[month_col] - 1
    df["_stop"] = df[month_col]
    df = df.drop(columns=[month_col])

    ctv = CoxTimeVaryingFitter()
    try:
        ctv.fit(
            df,
            id_col=id_col,
            event_col=event_col,
            start_col="_start",
            stop_col="_stop",
            show_progress=False,
        )
        cox_hr = np.exp(ctv.params_)
    except Exception as exc:
        raise RuntimeError(f"CoxTimeVaryingFitter failed: {exc}") from exc

    # Extract cloglog HRs — strip ColumnTransformer prefix (e.g. "features__bb_ever" → "bb_ever")
    glm_tbl = glm.coef_table
    hr_col = "HR" if glm.link == "cloglog" else "OR"
    feature_set = set(feature_cols)
    glm_hr = {}
    for idx, val in glm_tbl[hr_col].items():
        # Strip ColumnTransformer prefix (e.g. "features__bb_ever" → "bb_ever")
        stripped = str(idx).removeprefix("features__")
        if stripped in feature_set:
            glm_hr[stripped] = float(val)

    rows = []
    for feat in feature_cols:
        if feat not in panel_train.columns:
            continue
        c = float(cox_hr.get(feat, float("nan")))
        g = glm_hr.get(feat, float("nan"))
        sign_match = None
        ratio = None
        if not (np.isnan(c) or np.isnan(g)) and g != 0:
            sign_match = bool((c > 1) == (g > 1))
            ratio = c / g
        rows.append(
            {
                "feature": feat,
                "cox_hr": c,
                "cloglog_hr": g,
                "ratio": ratio,
                "sign_match": sign_match,
            }
        )

    return pd.DataFrame(rows).set_index("feature")

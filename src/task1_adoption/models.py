"""Discrete-time hazard models and benchmarks with sklearn-compatible API."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator, ClassifierMixin
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import CLogLog as CLogLogLink
from statsmodels.genmod.families.links import Logit as LogitLink


class DiscreteHazardGLM(BaseEstimator, ClassifierMixin):
    """Binary GLM for discrete-time hazard estimation.

    Args:
        link: Link function — "cloglog" gives grouped proportional hazards
            (coefficients are log-hazard-ratios); "logit" gives log-odds-ratios.
        add_intercept: Whether to prepend a constant column.

    predict() thresholds at the training marginal rate (rate_), not 0.5.
    Monthly initiation hazards are ~1–3%, so 0.5 would label everything as
    non-event. Use predict_proba() for risk scoring.
    """

    def __init__(self, link="cloglog", add_intercept=True):
        self.link = link
        self.add_intercept = add_intercept

    def fit(self, X, y):
        """Fit binary GLM on panel data; return self.

        Args:
            X: Feature DataFrame. Passing a DataFrame (not a numpy array)
                preserves column names in statsmodels so coef_table is
                labelled automatically.
            y: Binary event indicator, one entry per person-month.
        """
        link_fn = CLogLogLink() if self.link == "cloglog" else LogitLink()
        X_fit = self._prepare_X(X)
        self.result_ = sm.GLM(np.asarray(y), X_fit, family=Binomial(link=link_fn)).fit()
        self.classes_ = np.array([0, 1])
        self.rate_ = float(np.mean(y))  # marginal rate used as predict() threshold
        return self

    def predict_proba(self, X):
        """Return (n_samples, 2) predicted probability array."""
        X_pred = self._prepare_X(X)
        p = self.result_.predict(X_pred)
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
                    # has_constant='add' bypasses statsmodels' ptp-based check, which
                    # incorrectly skips the intercept for single-row prediction DataFrames.
                    X = sm.add_constant(X, has_constant="add")
            else:
                X = sm.add_constant(X, has_constant="add")
        return X

    @property
    def coef_table(self) -> pd.DataFrame:
        """Coefficient table with exponentiated effect sizes and 95% CI.

        Returns:
            DataFrame indexed by feature name with columns coef, se, z, p,
            ci_lower, ci_upper, and HR (or OR when link='logit') plus
            HR_lower / HR_upper (exponentiated CI bounds).
        """
        ci = np.asarray(self.result_.conf_int())
        tbl = pd.DataFrame(
            {
                "coef": self.result_.params,
                "se": self.result_.bse,
                "z": self.result_.tvalues,
                "p": self.result_.pvalues,
                "ci_lower": ci[:, 0],
                "ci_upper": ci[:, 1],
            },
            index=self.result_.params.index,
        )
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
    """Gradient boosting with Cox partial likelihood loss (scikit-survival).

    Uses patient-level survival data internally: converts the person-month
    panel to one row per patient with (event, duration) and baseline features.
    Predictions are mapped back to per-month hazard probabilities via the
    fitted survival function.

    Args:
        n_estimators: Number of boosting iterations.
        max_depth: Maximum tree depth (3 is conservative for small n).
        min_samples_leaf: Minimum leaf size; stabilises sparse event rows.
        random_state: Random seed.
    """

    def __init__(self, n_estimators=100, max_depth=3, min_samples_leaf=20, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X, y, panel=None):
        """Fit survival GBM on patient-level data derived from the panel.

        Args:
            X: Feature DataFrame from ColumnTransformer (person-month rows).
                Must include a time column (first column) and feature columns.
            y: Binary event indicator per person-month.
            panel: Full panel DataFrame with patient_id and study_month.
                Required to derive patient-level survival data. If None,
                falls back to the panel passed to the constructor.
        """
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis

        if panel is None:
            panel = self._panel
        self._panel = panel

        feature_cols = [c for c in X.columns if c.startswith("features__")]
        time_col = [c for c in X.columns if c.startswith("time__")][0]

        Xp = X.copy()
        Xp["_patient_id"] = panel.loc[X.index, "patient_id"].values
        Xp["_event"] = y.values
        Xp["_month"] = Xp[time_col]

        patient_data = Xp.groupby("_patient_id").agg(
            event=("_event", "max"),
            duration=("_month", "max"),
            **{f: (f, "first") for f in feature_cols},
        )
        patient_data["duration"] = patient_data["duration"].astype(float)

        y_surv = np.array(
            [(bool(e), d) for e, d in zip(patient_data["event"], patient_data["duration"])],
            dtype=[("event", bool), ("duration", float)],
        )
        X_surv = patient_data[feature_cols].values

        self.gbm_ = GradientBoostingSurvivalAnalysis(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        self.gbm_.fit(X_surv, y_surv)
        self.feature_cols_ = feature_cols
        self._risk_mean = float(np.mean(self.gbm_.predict(X_surv)))
        self.classes_ = np.array([0, 1])
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        """Per-month hazard pseudo-probabilities from Cox risk scores.

        The Cox PH model produces relative risk scores, not calibrated
        monthly probabilities. We map scores to the [0, 1] range via
        logistic calibration against the training event rate. This
        preserves the patient ranking (AUC) and approximately calibrates
        to the correct probability scale for BSS.
        """
        from scipy.special import expit

        X_feat = X[self.feature_cols_].values
        risk = self.gbm_.predict(X_feat)

        risk_centered = risk - self._risk_mean
        p = expit(risk_centered) * 2 * self.rate_
        p = np.clip(p, 1e-8, 1.0 - 1e-8)
        return np.column_stack([1 - p, p])

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

    Validates that person-month discretization is consistent with the continuous-
    time partial likelihood. Matching signs and ratios within ~2× confirm the
    grouped-Cox approximation is sound.

    Args:
        panel_train: Person-month DataFrame containing id_col, event_col,
            month_col, and all feature_cols.
        feature_cols: Clinical feature names. Exclude study_month — Cox absorbs
            time via the baseline hazard.
        glm: Fitted DiscreteHazardGLM providing cloglog HRs via coef_table.
        id_col: Patient identifier column.
        event_col: Binary event indicator column.
        month_col: Study month column.

    Returns:
        DataFrame indexed by feature with columns cox_hr, cloglog_hr,
        ratio (cox / cloglog), and sign_match (bool).
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
            sign_match = bool(np.sign(np.log(c)) == np.sign(np.log(g)))
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

"""Feature selection: LASSO regularisation path and stability selection."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


class StabilitySelector(BaseEstimator, TransformerMixin):
    """Stability selection via bootstrap + L1-penalised logistic regression.

    Runs LASSO on n_bootstrap subsamples, records which features have
    non-zero coefficients. Selection probability = fraction of resamples
    where the feature survived penalisation.

    Both fit() and transform() require a pandas DataFrame — numpy arrays
    are rejected so that feature names are always preserved in
    selected_features_ and selection_probabilities_.

    Args:
        n_bootstrap (int): Number of bootstrap resamples.
        sample_fraction (float): Fraction of patients (not rows) to draw
            per resample.
        threshold (float): Minimum selection probability to keep a feature.
        C (float | str): Inverse regularisation strength. "auto" tunes via
            LogisticRegressionCV on the full training set before bootstrapping.
        random_state (int): Random seed.
    """

    def __init__(
        self, n_bootstrap=200, sample_fraction=0.7, threshold=0.6, C="auto", random_state=42
    ):
        self.n_bootstrap = n_bootstrap
        self.sample_fraction = sample_fraction
        self.threshold = threshold
        self.C = C
        self.random_state = random_state

    def fit(self, X, y, patient_ids=None):
        """Fit stability selector on a person-month panel.

        Args:
            X (pd.DataFrame): Feature DataFrame with named columns.
            y (array-like): Binary event indicator.
            patient_ids (array-like | None): Patient identifiers parallel to X
                rows. When provided, bootstrap subsamples whole patients to avoid
                data leakage across person-months. When None, subsamples rows.

        Returns:
            self
        """
        if not hasattr(X, "columns"):
            raise ValueError("StabilitySelector requires a DataFrame input with named columns.")
        rng = np.random.RandomState(self.random_state)
        n_features = X.shape[1]
        counts = np.zeros(n_features)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        feature_names = X.columns

        # Tune C if "auto"
        C = self._resolve_C(X_scaled, np.asarray(y), rng)
        self.C_used_ = C

        if patient_ids is not None:
            unique_patients = np.unique(patient_ids)
            n_sample = int(len(unique_patients) * self.sample_fraction)
        else:
            n_sample = int(X.shape[0] * self.sample_fraction)

        n_valid = 0
        for i in range(self.n_bootstrap):
            if patient_ids is not None:
                sampled_patients = resample(
                    unique_patients,
                    n_samples=n_sample,
                    replace=False,
                    random_state=rng.randint(0, 2**31),  # RandomState seed range: [0, 2^31-1]
                )
                mask = np.isin(patient_ids, sampled_patients)
                X_sub, y_sub = X_scaled[mask], np.asarray(y)[mask]
            else:
                idx = resample(
                    np.arange(X.shape[0]),
                    n_samples=n_sample,
                    replace=False,
                    random_state=rng.randint(0, 2**31),
                )
                X_sub, y_sub = X_scaled[idx], np.asarray(y)[idx]

            if y_sub.sum() < 2:  # skip degenerate bootstrap samples (0 or 1 events)
                continue

            model = LogisticRegression(
                penalty="l1",
                C=C,
                solver="saga",
                max_iter=5000,
                random_state=rng.randint(0, 2**31),
            )
            model.fit(X_sub, y_sub)
            counts += (model.coef_[0] != 0).astype(int)
            n_valid += 1

        denom = n_valid if n_valid > 0 else self.n_bootstrap
        self.selection_probabilities_ = pd.Series(counts / denom, index=feature_names).sort_values(
            ascending=False
        )
        self.selected_features_ = list(
            self.selection_probabilities_[self.selection_probabilities_ >= self.threshold].index
        )
        return self

    def _resolve_C(self, X: np.ndarray, y: np.ndarray, rng: np.random.RandomState) -> float:
        """Return C, tuning via LogisticRegressionCV when C='auto'.

        Args:
            X (np.ndarray): Scaled feature matrix.
            y (np.ndarray): Binary event indicator.
            rng (np.random.RandomState): Seeded state for reproducibility.

        Returns:
            float: Regularisation strength C.
        """
        if self.C != "auto":
            return self.C
        cv_model = LogisticRegressionCV(
            penalty="l1",
            solver="saga",
            Cs=np.logspace(-3, 1, 20),
            cv=5,
            max_iter=5000,
            scoring="neg_log_loss",
            random_state=rng.randint(0, 2**31),
        )
        cv_model.fit(X, y)
        return float(cv_model.C_[0])

    def transform(self, X):
        """Return DataFrame with only the stability-selected features."""
        if not hasattr(X, "columns"):
            raise ValueError("StabilitySelector requires a DataFrame input with named columns.")
        return X[self.selected_features_]

    def plot(self, ax=None, figsize=(10, 6)):
        """Bar chart of selection probabilities."""
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        probs = self.selection_probabilities_.sort_values(ascending=True)
        colors = ["C0" if p >= self.threshold else "C7" for p in probs]
        ax.barh(range(len(probs)), probs.values, color=colors, edgecolor="white")
        ax.set_yticks(range(len(probs)))
        ax.set_yticklabels(probs.index)
        ax.axvline(
            self.threshold,
            color="red",
            linestyle="--",
            alpha=0.6,
            label=f"Threshold ({self.threshold})",
        )
        ax.set_xlabel("Selection probability")
        ax.set_title(f"Stability selection ({self.n_bootstrap} bootstrap resamples)")
        ax.legend()
        return ax


def lasso_path_plot(X, y, Cs=None, ax=None, figsize=(10, 6)):
    """Plot LASSO regularisation path: coefficients vs penalty strength.

    Args:
        X (pd.DataFrame | np.ndarray): Feature matrix.
        y (array-like): Binary event indicator.
        Cs (array-like | None): Sequence of C values to evaluate. Defaults to
            30 log-spaced values from 1e-3 to 1e2.
        ax (plt.Axes | None): Axes to draw on; created if None.
        figsize (tuple[int, int]): Figure size when ax is None.

    Returns:
        plt.Axes: Axes with one line per feature showing coefficient vs C.
    """
    if Cs is None:
        Cs = np.logspace(-3, 2, 30)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    feature_names = X.columns if hasattr(X, "columns") else [f"f{i}" for i in range(X.shape[1])]

    coefs = []
    for C in Cs:
        model = LogisticRegression(penalty="l1", C=C, solver="saga", max_iter=5000, random_state=42)
        model.fit(X_scaled, y)
        coefs.append(model.coef_[0])

    coefs = np.array(coefs)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    for i, name in enumerate(feature_names):
        ax.plot(Cs, coefs[:, i], label=name)
    ax.set_xscale("log")
    ax.set_xlabel("C (inverse regularisation strength →)")
    ax.set_ylabel("Coefficient")
    ax.set_title("LASSO regularisation path")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.axhline(0, color="grey", linewidth=0.5)
    return ax

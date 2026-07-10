#!/usr/bin/env python
"""Feature-selection experiments.

Question we are trying to resolve
---------------------------------
The case study documents a stability-selection rule ("bootstrap + L1,
≥60% selection probability across 200 patient-level resamples") and
then reports a REFINED model with 6 features. As currently written,
that rule does not cleanly yield those 6 features — one feature
(`n_hcm_meds`) sits well below the 0.60 threshold in the current run,
so it is retained "on clinical grounds" outside the rule.

This script sweeps the selection procedure across:
  1. L1 penalty strength `C`         (grid + auto)
  2. Stability threshold             (0.5 / 0.6 / 0.7 / 0.8)
  3. Bootstrap count / fraction      (200/0.7 baseline + variants)
  4. Candidate pool                  (EXPANDED, COMPACT_hand_pruned)
  5. CV split strategy for auto-C    (KFold random, GroupKFold by patient,
                                       TimeSeriesSplit)
  6. A plain LASSO baseline          (single LogisticRegressionCV under
                                       each CV strategy — no bootstrap)

For every configuration we record:
  - selected features and their selection probabilities
  - CV-tuned C (where applicable)
  - size of selected set
  - whether the current REFINED 6 are all present

Output: outputs/task1_adoption/experiments/feature_selection_matrix.csv
        outputs/task1_adoption/experiments/feature_selection_summary.md

Run:
    .venv/bin/python scripts/task1_adoption/experiments_feature_selection.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from src.task1_adoption.config import (
    EXPANDED_FEATURES,
    LAUNCH_MONTH,
    REFINED_FEATURES,
    DatasetConfig,
    SplitConfig,
)
from src.task1_adoption.data_loading import load_data
from src.task1_adoption.dataset import build_dataset

warnings.filterwarnings("ignore")

OUT = Path("outputs/task1_adoption/experiments")
OUT.mkdir(parents=True, exist_ok=True)

# ---------- Candidate pools ----------
# EXPANDED keeps all engineered features including rare composites and 12m
# rolling counts. COMPACT keeps only the mechanistically defensible subset —
# the treatment-journey + specialist-workup features the case study is built around.

EXPANDED = list(EXPANDED_FEATURES)

# Compact pool — drops rolling-window features and rare label-only composites
# that were never expected to matter at n=91 events.
COMPACT = [
    "age",
    "sex_F",
    "months_since_diso",
    "n_hcm_meds",
    "hf_flag",
    "bb_ever",
    "ccb_ever",
    "bb_current",
    "ccb_current",
    "months_since_last_med_change",
    "mri_ever",
    "af_flag",
    "mitral_flag",
    "diso_mpr_12m",
]

POOLS = {
    "EXPANDED": EXPANDED,
    "COMPACT_14": COMPACT,
}


# ---------- CV strategies for auto-C tuning ----------
def _cv_random(n_splits=5, random_state=42):
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _cv_group(n_splits=5):
    return GroupKFold(n_splits=n_splits)


def _cv_time(n_splits=5):
    return TimeSeriesSplit(n_splits=n_splits)


def _tune_C(X, y, groups, months, strategy, Cs=None):
    """Tune inverse regularisation strength via cross-validated logistic LASSO."""
    Cs = Cs if Cs is not None else np.logspace(-3, 1, 20)
    if strategy == "random_5":
        cv = _cv_random(5)
        cv_iter = cv.split(X, y)
    elif strategy == "random_10":
        cv = _cv_random(10)
        cv_iter = cv.split(X, y)
    elif strategy == "group_patient":
        cv = _cv_group(5)
        cv_iter = cv.split(X, y, groups=groups)
    elif strategy == "time_5":
        order = np.argsort(months)
        X = X[order]
        y = np.asarray(y)[order]
        cv = _cv_time(5)
        cv_iter = cv.split(X, y)
    else:
        raise ValueError(f"unknown CV strategy: {strategy}")

    cv_model = LogisticRegressionCV(
        penalty="l1",
        solver="saga",
        Cs=Cs,
        cv=list(cv_iter),
        max_iter=5000,
        scoring="neg_log_loss",
        random_state=42,
    )
    cv_model.fit(X, y)
    return float(cv_model.C_[0])


# ---------- Stability selection (patient-level bootstrap) ----------
def stability_run(
    X: pd.DataFrame,
    y: np.ndarray,
    patient_ids: np.ndarray,
    C: float,
    n_bootstrap: int = 200,
    sample_fraction: float = 0.7,
    random_state: int = 42,
) -> pd.Series:
    """Return selection probability per feature."""
    rng = np.random.RandomState(random_state)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    names = X.columns

    unique_patients = np.unique(patient_ids)
    n_sample = int(len(unique_patients) * sample_fraction)

    counts = np.zeros(X.shape[1])
    valid = 0
    for _ in range(n_bootstrap):
        sampled = resample(
            unique_patients,
            n_samples=n_sample,
            replace=False,
            random_state=rng.randint(0, 2**31),
        )
        mask = np.isin(patient_ids, sampled)
        Xb, yb = Xs[mask], y[mask]
        if yb.sum() < 2:
            continue
        model = LogisticRegression(
            penalty="l1",
            C=C,
            solver="saga",
            max_iter=5000,
            random_state=rng.randint(0, 2**31),
        )
        model.fit(Xb, yb)
        counts += (model.coef_[0] != 0).astype(int)
        valid += 1
    denom = valid if valid > 0 else n_bootstrap
    return pd.Series(counts / denom, index=names).sort_values(ascending=False)


# ---------- Plain LASSO (single fit) ----------
def lasso_run(
    X: pd.DataFrame,
    y: np.ndarray,
    patient_ids: np.ndarray,
    months: np.ndarray,
    cv_strategy: str,
) -> tuple[float, list[str]]:
    """Fit LogisticRegressionCV once, return (C, non-zero features)."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    C = _tune_C(Xs, y, patient_ids, months, cv_strategy)
    model = LogisticRegression(
        penalty="l1", C=C, solver="saga", max_iter=5000, random_state=42
    ).fit(Xs, y)
    kept = X.columns[model.coef_[0] != 0].tolist()
    return C, kept


# ---------- Main ----------
def main() -> None:
    print("Loading data + building dataset...")
    data = load_data()
    dataset = build_dataset(data, DatasetConfig())

    split_cfg = SplitConfig()
    split_month = (
        pd.Period(split_cfg.train_end_month, freq="M") - pd.Period(LAUNCH_MONTH, freq="M")
    ).n + 1
    train = dataset[dataset["study_month"] <= split_month].copy()
    y = train["event"].to_numpy()
    pids = train["patient_id"].to_numpy()
    months = train["study_month"].to_numpy()

    print(f"Train rows={len(train):,}, patients={len(np.unique(pids)):,}, events={int(y.sum())}\n")

    refined_set = set(REFINED_FEATURES)
    thresholds = [0.5, 0.6, 0.7, 0.8]
    manual_Cs = [0.01, 0.05, 0.1, 0.5, 1.0]
    cv_strategies = ["random_5", "random_10", "group_patient", "time_5"]

    prob_records: list[dict] = []  # long-form per (config, feature) selection probability
    summary_records: list[dict] = []

    # --- Stability selection sweep ---
    for pool_name, pool in POOLS.items():
        pool_avail = [f for f in pool if f in train.columns]
        X = train[pool_avail]
        print(f"[stability] pool={pool_name} ({len(pool_avail)} features)")

        # 1a. auto-C under each CV strategy at baseline 200 boot / 0.7 sample
        for cv in cv_strategies:
            C = _tune_C(StandardScaler().fit_transform(X), y, pids, months, cv)
            probs = stability_run(X, y, pids, C, n_bootstrap=200, sample_fraction=0.7)
            for thr in thresholds:
                selected = probs[probs >= thr].index.tolist()
                summary_records.append(
                    {
                        "method": "stability",
                        "pool": pool_name,
                        "cv_for_C": cv,
                        "C": round(C, 5),
                        "n_bootstrap": 200,
                        "sample_frac": 0.7,
                        "threshold": thr,
                        "n_selected": len(selected),
                        "selected": ", ".join(selected),
                        "refined_covered": sum(f in selected for f in REFINED_FEATURES),
                        "refined_missing": ", ".join(sorted(refined_set - set(selected))),
                    }
                )
            for feat, p in probs.items():
                prob_records.append(
                    {
                        "method": "stability",
                        "pool": pool_name,
                        "cv_for_C": cv,
                        "C_source": "auto",
                        "C": round(C, 5),
                        "feature": feat,
                        "selection_prob": round(float(p), 3),
                    }
                )

        # 1b. Manual C sweep at baseline CV, threshold 0.6
        for C in manual_Cs:
            probs = stability_run(X, y, pids, C, n_bootstrap=200, sample_fraction=0.7)
            for thr in thresholds:
                selected = probs[probs >= thr].index.tolist()
                summary_records.append(
                    {
                        "method": "stability",
                        "pool": pool_name,
                        "cv_for_C": "manual",
                        "C": C,
                        "n_bootstrap": 200,
                        "sample_frac": 0.7,
                        "threshold": thr,
                        "n_selected": len(selected),
                        "selected": ", ".join(selected),
                        "refined_covered": sum(f in selected for f in REFINED_FEATURES),
                        "refined_missing": ", ".join(sorted(refined_set - set(selected))),
                    }
                )
            for feat, p in probs.items():
                prob_records.append(
                    {
                        "method": "stability",
                        "pool": pool_name,
                        "cv_for_C": "manual",
                        "C_source": "manual",
                        "C": C,
                        "feature": feat,
                        "selection_prob": round(float(p), 3),
                    }
                )

        # 1c. Higher-resource variants at group_patient CV, threshold 0.6
        for n_boot, sfrac in [(400, 0.7), (200, 0.5), (200, 0.9)]:
            C = _tune_C(StandardScaler().fit_transform(X), y, pids, months, "group_patient")
            probs = stability_run(X, y, pids, C, n_bootstrap=n_boot, sample_fraction=sfrac)
            for thr in thresholds:
                selected = probs[probs >= thr].index.tolist()
                summary_records.append(
                    {
                        "method": "stability",
                        "pool": pool_name,
                        "cv_for_C": "group_patient",
                        "C": round(C, 5),
                        "n_bootstrap": n_boot,
                        "sample_frac": sfrac,
                        "threshold": thr,
                        "n_selected": len(selected),
                        "selected": ", ".join(selected),
                        "refined_covered": sum(f in selected for f in REFINED_FEATURES),
                        "refined_missing": ", ".join(sorted(refined_set - set(selected))),
                    }
                )

    # --- Plain LASSO (no bootstrap) ---
    for pool_name, pool in POOLS.items():
        pool_avail = [f for f in pool if f in train.columns]
        X = train[pool_avail]
        for cv in cv_strategies:
            C, kept = lasso_run(X, y, pids, months, cv)
            summary_records.append(
                {
                    "method": "lasso_1shot",
                    "pool": pool_name,
                    "cv_for_C": cv,
                    "C": round(C, 5),
                    "n_bootstrap": None,
                    "sample_frac": None,
                    "threshold": None,
                    "n_selected": len(kept),
                    "selected": ", ".join(kept),
                    "refined_covered": sum(f in kept for f in REFINED_FEATURES),
                    "refined_missing": ", ".join(sorted(refined_set - set(kept))),
                }
            )

    summary = pd.DataFrame(summary_records)
    probs_df = pd.DataFrame(prob_records)

    summary.to_csv(OUT / "feature_selection_matrix.csv", index=False)
    probs_df.to_csv(OUT / "feature_selection_probabilities.csv", index=False)
    print(f"\nWrote {len(summary)} configurations → {OUT / 'feature_selection_matrix.csv'}")
    print(f"Wrote {len(probs_df)} per-feature rows → {OUT / 'feature_selection_probabilities.csv'}")

    # --- Compact console summary (most informative slices) ---
    hdr = [
        "method",
        "pool",
        "cv_for_C",
        "C",
        "n_bootstrap",
        "sample_frac",
        "threshold",
        "n_selected",
        "refined_covered",
        "refined_missing",
    ]
    print("\n=== Stability, threshold=0.6, sample_frac=0.7, n_boot=200 ===")
    view = summary[
        (summary["method"] == "stability")
        & (summary["threshold"] == 0.6)
        & (summary["sample_frac"] == 0.7)
        & (summary["n_bootstrap"] == 200)
    ].sort_values(["pool", "cv_for_C"])
    print(view[hdr].to_string(index=False))

    print("\n=== Plain LASSO (one-shot) ===")
    view = summary[summary["method"] == "lasso_1shot"].sort_values(["pool", "cv_for_C"])
    print(view[hdr].to_string(index=False))

    print("\n=== Threshold sensitivity, COMPACT_14 pool, group_patient CV ===")
    view = summary[
        (summary["pool"] == "COMPACT_14")
        & (summary["cv_for_C"] == "group_patient")
        & (summary["n_bootstrap"] == 200)
        & (summary["sample_frac"] == 0.7)
    ].sort_values("threshold")
    print(view[hdr + ["selected"]].to_string(index=False))


if __name__ == "__main__":
    main()

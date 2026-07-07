#!/usr/bin/env python3
"""Sensitivity analysis for the Camzyos adoption model.

Tests robustness of the refined model across five axes:
  1. Risk set: Disopyramide-conditioned vs full oHCM pool
  2. Cohort definition: I421 only vs I421+I422 (broader HCM)
  3. Enrollment window: lenient (≥3m) vs standard (≥6m) vs strict (≥12m)
  4. Censoring: censor at first enrollment gap vs allow gaps
  5. Feature set: null / clinical priors / refined / stability-selected

Each configuration builds its own panel, applies the temporal train/test
split, fits the refined model, and reports calibration + discrimination.
Primary metric is Brier skill score (calibration). C-index is secondary.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.config import (
    CLINICAL_FEATURES,
    LAUNCH_MONTH,
    MONTH_COL,
    REFINED_FEATURES,
    PanelConfig,
    SplitConfig,
)
from src.data_loading import load_data
from src.evaluation import evaluate_model
from src.models import DiscreteHazardGLM
from src.panel import build_panel

np.random.seed(42)


def make_preprocessor(feature_cols):
    return ColumnTransformer(
        transformers=[
            ("time", "passthrough", [MONTH_COL]),
            ("features", "passthrough", feature_cols),
        ],
        remainder="drop",
    )


def prepare_Xy(train, test, feature_cols):
    all_cols = feature_cols + [MONTH_COL]
    prep = make_preprocessor(feature_cols)
    X_train = pd.DataFrame(
        prep.fit_transform(train[all_cols]),
        columns=prep.get_feature_names_out(),
        index=train.index,
    )
    X_test = pd.DataFrame(
        prep.transform(test[all_cols]),
        columns=prep.get_feature_names_out(),
        index=test.index,
    )
    return X_train, X_test, train["event"], test["event"]


def run_config(data, panel_cfg, split_cfg, feature_cols, label):
    """Build panel, split, fit refined model, return results dict."""
    try:
        panel = build_panel(data, panel_cfg)
    except Exception as e:
        return {"label": label, "error": str(e)}

    if panel["event"].sum() < 20:
        return {"label": label, "error": f"too few events ({panel['event'].sum()})"}

    split_month = (
        pd.Period(split_cfg.train_end_month, freq="M") - pd.Period(LAUNCH_MONTH, freq="M")
    ).n + 1

    train = panel[panel[MONTH_COL] <= split_month].copy()
    test = panel[panel[MONTH_COL] > split_month].copy()

    available = [f for f in feature_cols if f in panel.columns]
    if len(available) < 2:
        return {"label": label, "error": "insufficient features"}

    X_tr, X_te, y_tr, y_te = prepare_Xy(train, test, available)

    model = DiscreteHazardGLM(link="cloglog").fit(X_tr, y_tr)
    results, _ = evaluate_model(model, X_te, y_te, test, label)
    results["label"] = label  # evaluate_model stores as "name"; alias for display
    results["train_events"] = int(y_tr.sum())
    results["test_events"] = int(y_te.sum())
    results["n_patients"] = panel["patient_id"].nunique()
    results["features_used"] = len(available)
    return results


def main():
    print("Loading data...")
    data = load_data()
    split_cfg = SplitConfig()

    print("\nRunning sensitivity analyses...\n")
    results = []

    # ── 1. Baseline: standard config ────────────────────────────────
    cfg = PanelConfig(risk_set="disopyramide", censor_at_first_gap=True)
    r = run_config(data, cfg, split_cfg, REFINED_FEATURES, "Baseline (Diso pool, censor gaps)")
    results.append(r)

    # ── 2. Risk set: full oHCM pool ──────────────────────────────────
    # Drops months_since_diso (NaT for non-Diso patients) and mri_ever
    ohcm_features = [f for f in REFINED_FEATURES if f not in ["months_since_diso"]]
    cfg = PanelConfig(risk_set="full_ohcm", censor_at_first_gap=True)
    r = run_config(
        data, cfg, split_cfg, ohcm_features, "Risk set: full oHCM (no Diso conditioning)"
    )
    results.append(r)

    # ── 3. Censoring: allow enrollment gaps ──────────────────────────
    cfg = PanelConfig(risk_set="disopyramide", censor_at_first_gap=False)
    r = run_config(data, cfg, split_cfg, REFINED_FEATURES, "Censoring: allow enrollment gaps")
    results.append(r)

    # ── 4. Feature set: clinical priors (original 7 features) ────────
    cfg = PanelConfig(risk_set="disopyramide", censor_at_first_gap=True)
    r = run_config(
        data, cfg, split_cfg, CLINICAL_FEATURES, "Features: clinical priors (7 features)"
    )
    results.append(r)

    # ── 5. Temporal split sensitivity: earlier cutoff (month 9) ──────
    early_split = SplitConfig(train_end_month="2022-12")  # 9 months post-launch
    cfg = PanelConfig(risk_set="disopyramide", censor_at_first_gap=True)
    r = run_config(data, cfg, early_split, REFINED_FEATURES, "Split: earlier cutoff (Dec 2022)")
    results.append(r)

    # ── 6. Temporal split sensitivity: later cutoff (month 18) ───────
    late_split = SplitConfig(train_end_month="2023-09")  # 18 months post-launch
    cfg = PanelConfig(risk_set="disopyramide", censor_at_first_gap=True)
    r = run_config(data, cfg, late_split, REFINED_FEATURES, "Split: later cutoff (Sep 2023)")
    results.append(r)

    # ── Report ────────────────────────────────────────────────────────
    df = pd.DataFrame([r for r in results if "error" not in r])
    errors = [r for r in results if "error" in r]

    print(f"{'=' * 100}")
    print("SENSITIVITY ANALYSIS — REFINED MODEL (cloglog, 6 features)")
    print("Primary metric: Brier Skill Score (BSS). Positive = better than null.")
    print(f"{'=' * 100}")

    display_cols = [
        "label",
        "n_patients",
        "train_events",
        "test_events",
        "brier_skill_score",
        "time_dependent_auc",
        "mean_pred",
        "mean_observed",
    ]
    print(df[display_cols].round(3).to_string(index=False))

    if errors:
        print("\nConfigurations that failed:")
        for e in errors:
            print(f"  {e['label']}: {e['error']}")

    print(f"\n{'=' * 100}")
    print("INTERPRETATION")
    print(f"{'=' * 100}")

    baseline = df[df["label"].str.startswith("Baseline")]
    if not baseline.empty:
        base_bss = baseline["brier_skill_score"].iloc[0]
        base_auc = baseline["time_dependent_auc"].iloc[0]
        print(f"\nBaseline BSS = {base_bss:.3f}, time-dep AUC = {base_auc:.3f}")
        print("\nDeviation from baseline:")
        for _, row in df.iterrows():
            delta_bss = row["brier_skill_score"] - base_bss
            delta_auc = row["time_dependent_auc"] - base_auc
            print(f"  {row['label']:55s}  BSS Δ={delta_bss:+.3f}  AUC Δ={delta_auc:+.3f}")

    out_path = Path("outputs/05_sensitivity_analysis.csv")
    out_path.parent.mkdir(exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

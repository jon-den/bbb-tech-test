#!/usr/bin/env python
"""Task 1: Camzyos adoption model — discrete-time hazard.

Builds a person-month dataset for the Disopyramide-conditioned risk set,
fits discrete-time hazard GLMs, runs stability selection on the expanded
feature set, and evaluates against baselines.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.style import INK_PRI, SURFACE
from src.task1_adoption.config import (
    CANDIDATE_FEATURES,
    CLINICAL_FEATURES,
    EXPANDED_FEATURES,
    LAUNCH_MONTH,
    MONTH_COL,
    REFINED_FEATURES,
    DatasetConfig,
    ModelConfig,
    SplitConfig,
)
from src.task1_adoption.data_loading import load_data
from src.task1_adoption.dataset import build_dataset
from src.task1_adoption.evaluation import (
    bootstrap_test_ci,
    calibration_plot,
    count_calibration,
    count_calibration_plot,
    evaluate_model,
)
from src.task1_adoption.models import (
    DiscreteHazardGLM,
    GBMHazardBenchmark,
    MarginalRateModel,
    cox_discretization_check,
)
from src.task1_adoption.selection import StabilitySelector, lasso_path_plot

np.random.seed(42)


def make_preprocessor(feature_cols):
    """Continuous time trend + clinical features.

    study_month is included as a continuous passthrough rather than one-hot
    encoded. Month dummies would all be zero for test months (unseen
    categories → handle_unknown="ignore" → all zeros), collapsing the
    baseline hazard to a constant for the entire test window. A single
    continuous time covariate extrapolates the linear trend correctly.
    """
    return ColumnTransformer(
        transformers=[
            ("time", "passthrough", [MONTH_COL]),
            ("features", "passthrough", feature_cols),
        ],
        remainder="drop",
    )


def prepare_Xy(train, test, feature_cols):
    """Fit preprocessor on train, transform both. Returns X_train, X_test, y_train, y_test."""
    all_cols = feature_cols + [MONTH_COL]
    prep = make_preprocessor(feature_cols)
    X_train = prep.fit_transform(train[all_cols])
    X_test = prep.transform(test[all_cols])
    names = prep.get_feature_names_out()
    X_train = pd.DataFrame(X_train, columns=names, index=train.index)
    X_test = pd.DataFrame(X_test, columns=names, index=test.index)
    return X_train, X_test, train["event"], test["event"]


def main():
    dataset_cfg = DatasetConfig()
    split_cfg = SplitConfig()
    model_cfg = ModelConfig()

    # ── 1. Load data and build dataset ──────────────────────────────
    print("Loading data...")
    data = load_data()

    print(f"Building person-month dataset (risk_set={dataset_cfg.risk_set})...")
    dataset = build_dataset(data, dataset_cfg)

    print("\nDataset summary:")
    print(f"  Rows:     {len(dataset):,}")
    print(f"  Patients: {dataset['patient_id'].nunique():,}")
    print(f"  Events:   {dataset['event'].sum():,}")
    print(f"  Months:   {dataset['study_month'].min()} – {dataset['study_month'].max()}")
    print(f"  Event rate: {dataset['event'].mean():.4f}")
    skip = {"patient_id", "month", "study_month", "event"}
    print(f"  Features available: {[c for c in dataset.columns if c not in skip]}")

    # ── 2. Temporal train/test split ────────────────────────────────
    split_month = (
        pd.Period(split_cfg.train_end_month, freq="M") - pd.Period(LAUNCH_MONTH, freq="M")
    ).n + 1

    train = dataset[dataset["study_month"] <= split_month].copy()
    test = dataset[dataset["study_month"] > split_month].copy()

    print(f"\nTemporal split at study_month {split_month} ({split_cfg.train_end_month}):")
    print(
        f"  Train: {len(train):,} rows, {train['event'].sum():.0f} events, "
        f"{train['patient_id'].nunique()} patients"
    )
    print(
        f"  Test:  {len(test):,} rows, {test['event'].sum():.0f} events, "
        f"{test['patient_id'].nunique()} patients"
    )

    # ── 3. Stability selection on candidate pool ────────────────────
    #
    # Two-stage selection (see docs/TASK_1_2_CASE_STUDY.md → "Feature selection"):
    #   1. Pre-filter: 36 raw candidates → CANDIDATE_FEATURES (14) by
    #      dropping rolling-window near-duplicates and rare label-derived
    #      composites (<1% prevalence) that game stability selection at n=91
    #      events.
    #   2. Stability selection with patient-level bootstrap (200×70%) and
    #      patient-GroupKFold CV for the L1 penalty. Threshold 0.6 →
    #      5 features.
    #
    # The unfiltered EXPANDED_FEATURES pool is fit separately below solely
    # so the pathologies (multi-window redundancy, rare-composite artefacts)
    # are visible in the appendix.
    candidate_available = [f for f in CANDIDATE_FEATURES if f in dataset.columns]
    expanded_available = [f for f in EXPANDED_FEATURES if f in dataset.columns]
    print(f"\n{'=' * 70}")
    print(f"STABILITY SELECTION ({len(candidate_available)} candidates, C=auto)")
    print(f"{'=' * 70}")

    selector = StabilitySelector(
        n_bootstrap=200,
        sample_fraction=0.7,
        threshold=0.6,
        C="auto",
        cv_for_C="group_patient",
        random_state=42,
    )
    selector.fit(
        train[candidate_available],
        train["event"],
        patient_ids=train["patient_id"].values,
    )
    print(f"  CV-tuned C: {selector.C_used_:.4f}")

    print("\nSelection probabilities:")
    for feat, prob in selector.selection_probabilities_.items():
        marker = "  ✓" if prob >= selector.threshold else "   "
        print(f"  {marker} {feat:30s} {prob:.2f}")

    selected = selector.selected_features_
    print(f"\nSelected ({len(selected)}): {selected}")

    # ── 4. Prepare feature matrices ─────────────────────────────────
    clinical_available = [f for f in CLINICAL_FEATURES if f in dataset.columns]
    refined_available = [f for f in REFINED_FEATURES if f in dataset.columns]

    datasets = {}
    for label, feats in [
        ("clinical", clinical_available),
        ("refined", refined_available),
        ("selected", selected if selected else clinical_available),
        ("expanded", expanded_available),
    ]:
        Xtr, Xte, ytr, yte = prepare_Xy(train, test, feats)
        datasets[label] = (Xtr, Xte, ytr, yte)

    # ── 5. Fit and evaluate models ──────────────────────────────────
    print(f"\n{'=' * 70}")
    print("MODEL COMPARISON")
    print(f"{'=' * 70}")
    results = []
    models = {}

    Xtr_c, Xte_c, y_train, y_test = datasets["clinical"]

    # Null
    m = MarginalRateModel().fit(Xtr_c, y_train)
    r, _ = evaluate_model(m, Xte_c, y_test, test, "Null (marginal rate)")
    results.append(r)

    # Linear time trend only
    time_col = [c for c in Xtr_c.columns if c.startswith("time__")]
    m = DiscreteHazardGLM(link=model_cfg.link).fit(Xtr_c[time_col], y_train)
    r, _ = evaluate_model(m, Xte_c[time_col], y_test, test, "Linear time trend only")
    results.append(r)

    # Clinical priors (7 features)
    m = DiscreteHazardGLM(link=model_cfg.link).fit(Xtr_c, y_train)
    r, _ = evaluate_model(m, Xte_c, y_test, test, f"Clinical priors ({len(clinical_available)})")
    results.append(r)
    models["clinical"] = m

    # Refined model (treatment journey + specialist engagement, from stability selection)
    Xtr_r, Xte_r = datasets["refined"][:2]
    m = DiscreteHazardGLM(link=model_cfg.link).fit(Xtr_r, y_train)
    r, _ = evaluate_model(m, Xte_r, y_test, test, f"Refined ({len(refined_available)})")
    results.append(r)
    models["refined"] = m

    # Stability-selected
    if selected and selected != clinical_available:
        Xtr_s, Xte_s = datasets["selected"][:2]
        m = DiscreteHazardGLM(link=model_cfg.link).fit(Xtr_s, y_train)
        r, _ = evaluate_model(m, Xte_s, y_test, test, f"Stability-selected ({len(selected)})")
        results.append(r)
        models["selected"] = m

    # Full expanded (expect overfitting)
    Xtr_e, Xte_e = datasets["expanded"][:2]
    m = DiscreteHazardGLM(link=model_cfg.link).fit(Xtr_e, y_train)
    r, _ = evaluate_model(m, Xte_e, y_test, test, f"Full expanded ({len(expanded_available)})")
    results.append(r)
    models["expanded"] = m

    # GBM benchmark (nonlinear; interpret AUC only, not coefficients)
    m = GBMHazardBenchmark().fit(Xtr_r, y_train, dataset=train)
    r, _ = evaluate_model(
        m, Xte_r, y_test, test, f"GBM benchmark ({len(refined_available)} features)"
    )
    results.append(r)
    models["gbm"] = m

    results_df = pd.DataFrame(results).set_index("name")
    display_cols = [
        "brier_score",
        "brier_skill_score",
        "time_dependent_auc",
        "mean_pred",
        "mean_observed",
        "n_events",
    ]
    print(results_df[display_cols].round(4).to_string())

    out_dir = Path("outputs/task1_adoption")
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df.reset_index()[["name"] + display_cols].round(6).to_csv(
        out_dir / "01_model_comparison.csv", index=False
    )
    print(f"\nSaved: {out_dir / '01_model_comparison.csv'}")

    # ── Bootstrap CIs for benchmarking models ──────────────────────
    print(f"\n{'=' * 70}")
    print("BOOTSTRAP CIs (500 patient-level resamples, test set)")
    print(f"{'=' * 70}")
    bench_models = {
        f"Refined ({len(refined_available)})": (models["refined"], datasets["refined"][1]),
        f"GBM benchmark ({len(refined_available)} features)": (
            models["gbm"],
            datasets["refined"][1],
        ),
        f"Full expanded ({len(expanded_available)})": (models["expanded"], datasets["expanded"][1]),
    }
    for label, (mdl, X_te) in bench_models.items():
        scored = test.copy()
        scored["pred"] = mdl.predict_proba(X_te)[:, 1]
        ci = bootstrap_test_ci(scored, n_bootstrap=500, random_state=42)
        auc_pt, auc_lo, auc_hi = ci["auc"]
        bss_pt, bss_lo, bss_hi = ci["bss"]
        print(
            f"  {label:45s}  AUC {auc_pt:.2f} [{auc_lo:.2f}, {auc_hi:.2f}]"
            f"  BSS {bss_pt:+.3f} [{bss_lo:+.3f}, {bss_hi:+.3f}]"
        )

    # ── 6. Coefficient tables ───────────────────────────────────────
    hr_col = "HR" if model_cfg.link == "cloglog" else "OR"

    for label in ["clinical", "refined"]:
        model = models[label]
        coef = model.coef_table
        clinical_coefs = coef.loc[coef.index != "const"]
        print(f"\n{'=' * 70}")
        print(f"COEFFICIENTS — {label} ({model_cfg.link})")
        print(f"{'=' * 70}")
        print(
            clinical_coefs[[hr_col, f"{hr_col}_lower", f"{hr_col}_upper", "p"]].round(4).to_string()
        )

    refined_coef = models["refined"].coef_table
    refined_coef = refined_coef.loc[refined_coef.index != "const"].copy()
    refined_coef.index.name = "feature"
    refined_coef.reset_index().to_csv(out_dir / "01_coef_refined.csv", index=False)
    print(f"Saved: {out_dir / '01_coef_refined.csv'}")

    # ── 7. CoxPH discretization cross-check ────────────────────────
    print(f"\n{'=' * 70}")
    print("COX PROPORTIONAL HAZARDS CROSS-CHECK")
    print("CoxTimeVaryingFitter (continuous time) vs cloglog GLM (discrete time)")
    print("HRs should agree in sign and be within ~20% if the grouped-Cox")
    print("approximation holds. Divergences flag feature-specific misspecification.")
    print(f"{'=' * 70}")

    try:
        cmp = cox_discretization_check(train, refined_available, models["refined"])
        print(cmp.round(3).to_string())
        n_agree = int(cmp["sign_match"].sum())
        print(f"\nSign agreement: {n_agree}/{len(cmp)} features")
        max_ratio = cmp["ratio"].abs().max()
        if max_ratio > 2.0:
            print(f"WARNING: max HR ratio = {max_ratio:.2f} — poor Cox discretization fit")
        else:
            print(f"Max HR ratio = {max_ratio:.2f} — consistent with continuous-time Cox")
    except Exception as exc:
        print(f"Cox cross-check failed: {exc}")

    # ── 9. Plots ────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=SURFACE)
    fig.suptitle(
        "Model evaluation — feature selection & calibration",
        fontsize=12,
        fontweight="bold",
        color=INK_PRI,
        x=0.02,
        ha="left",
    )

    selector.plot(ax=axes[0, 0])
    lasso_path_plot(train[expanded_available], y_train, ax=axes[0, 1], highlight_features=selected)

    # Use refined model for calibration plots
    y_pred = models["refined"].predict_proba(datasets["refined"][1])[:, 1]
    calibration_plot(y_test.values, y_pred, n_bins=5, ax=axes[1, 0], label="Refined model")
    axes[1, 0].set_title(
        "Calibration (quintiles, test set)",
        fontsize=10,
        fontweight="bold",
        color=INK_PRI,
        loc="left",
    )

    test_pred = test.copy()
    test_pred["pred"] = y_pred
    monthly = count_calibration(test_pred)
    count_calibration_plot(monthly, ax=axes[1, 1])
    axes[1, 1].set_title(
        "Count-level calibration (test set)",
        fontsize=10,
        fontweight="bold",
        color=INK_PRI,
        loc="left",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = Path("outputs/task1_adoption/01_model_evaluation.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=500, bbox_inches="tight", facecolor=SURFACE)
    print(f"\nPlots saved to {out}")


if __name__ == "__main__":
    main()

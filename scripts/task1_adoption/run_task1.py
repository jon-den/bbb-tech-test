#!/usr/bin/env python
"""Task 1: Camzyos adoption modelling — full pipeline.

Loads data once, fits the discrete-time hazard model, and runs all
analysis steps:
  1. Model comparison & stability selection
  2. Adoption figure (4-panel, case study)
  3. Calibration assessment (3-panel)
  4. Hazard ratio forest plot (case study)

Output directory: outputs/task1_adoption/
Run:   .venv/bin/python scripts/task1_adoption/run_task1.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.style import (
    C_ARCH,
    C_OBS,
    C_PRED,
    C_REF,
    C_SPLIT,
    GRID,
    INK_MUT,
    INK_PRI,
    INK_SEC,
    SPINE,
    SURFACE,
    style_ax,
)
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
    hosmer_lemeshow,
    subgroup_cal,
    wilson_ci,
)
from src.task1_adoption.models import (
    DiscreteHazardGLM,
    GBMHazardBenchmark,
    MarginalRateModel,
    cox_discretization_check,
)
from src.task1_adoption.selection import StabilitySelector, lasso_path_plot

np.random.seed(42)

OUT_DIR = Path("outputs/task1_adoption")

N_BOOTSTRAP = 500

ARCHETYPES = [
    {
        "label": "Not escalated\n(on BB, no CCB history)",
        "ccb_ever": 0,
        "bb_current": 1,
        "ccb_current": 0,
    },
    {
        "label": "Currently managed\n(on BB + CCB, CCB history)",
        "ccb_ever": 1,
        "bb_current": 1,
        "ccb_current": 1,
    },
    {
        "label": "CCB-tried, still on BB\n(escalated, not off meds)",
        "ccb_ever": 1,
        "bb_current": 1,
        "ccb_current": 0,
    },
    {
        "label": "Escalated, off meds\n(CCB history, off both)",
        "ccb_ever": 1,
        "bb_current": 0,
        "ccb_current": 0,
    },
]

FEATURE_LABELS = {
    "features__ccb_ever": "Ever prescribed CCB\n(verapamil / diltiazem)",
    "features__bb_current": "Active beta-blocker\n(30-day coverage window)",
    "features__ccb_current": "Active CCB prescription\n(30-day coverage window)",
    "features__months_since_diso": "Months since first\nDisopyramide fill",
}


# ── Shared helpers ───────────────────────────────────────────────────────────


def make_preprocessor(feature_cols):
    """Continuous time trend + clinical features."""
    return ColumnTransformer(
        [("time", "passthrough", [MONTH_COL]), ("features", "passthrough", feature_cols)],
        remainder="drop",
    )


def prepare_Xy(train, test, feature_cols):
    """Fit preprocessor on train, transform both. Returns (X_train, X_test, y_train, y_test, ct)."""
    all_cols = feature_cols + [MONTH_COL]
    ct = make_preprocessor(feature_cols)
    X_train = pd.DataFrame(
        ct.fit_transform(train[all_cols]), columns=ct.get_feature_names_out(), index=train.index
    )
    X_test = pd.DataFrame(
        ct.transform(test[all_cols]), columns=ct.get_feature_names_out(), index=test.index
    )
    return X_train, X_test, train["event"], test["event"], ct


def load_and_split():
    """Load data, build dataset, temporal train/test split."""
    print("Loading data and building dataset…")
    data = load_data()
    dataset = build_dataset(data, DatasetConfig())

    split_month = (
        pd.Period(SplitConfig().train_end_month, freq="M") - pd.Period(LAUNCH_MONTH, freq="M")
    ).n + 1

    train = dataset[dataset[MONTH_COL] <= split_month].copy()
    test = dataset[dataset[MONTH_COL] > split_month].copy()

    print(
        f"  Panel: {len(dataset):,} person-months, {dataset['patient_id'].nunique()} patients, "
        f"{int(dataset['event'].sum())} events"
    )
    print(
        f"  Train: {len(train):,} rows, {int(train['event'].sum())} events (months 1-{split_month})"
    )
    print(
        f"  Test:  {len(test):,} rows, {int(test['event'].sum())} events (months {split_month + 1}+)"
    )

    return dataset, train, test, split_month


# ── 1. Model comparison & stability selection ────────────────────────────────


def run_model_comparison(train, test, dataset):
    """Stability selection, multi-model comparison, bootstrap CIs, Cox cross-check."""
    model_cfg = ModelConfig()
    candidate_available = [f for f in CANDIDATE_FEATURES if f in dataset.columns]
    expanded_available = [f for f in EXPANDED_FEATURES if f in dataset.columns]
    clinical_available = [f for f in CLINICAL_FEATURES if f in dataset.columns]
    refined_available = [f for f in REFINED_FEATURES if f in dataset.columns]

    # Stability selection
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
    selector.fit(train[candidate_available], train["event"], patient_ids=train["patient_id"].values)
    print(f"  CV-tuned C: {selector.C_used_:.4f}")

    print("\nSelection probabilities:")
    for feat, prob in selector.selection_probabilities_.items():
        marker = "  ✓" if prob >= selector.threshold else "   "
        print(f"  {marker} {feat:30s} {prob:.2f}")

    selected = selector.selected_features_
    print(f"\nSelected ({len(selected)}): {selected}")

    # Prepare all feature sets
    datasets = {}
    for label, feats in [
        ("clinical", clinical_available),
        ("refined", refined_available),
        ("selected", selected if selected else clinical_available),
        ("expanded", expanded_available),
    ]:
        Xtr, Xte, ytr, yte, _ = prepare_Xy(train, test, feats)
        datasets[label] = (Xtr, Xte, ytr, yte)

    # Model comparison
    print(f"\n{'=' * 70}")
    print("MODEL COMPARISON")
    print(f"{'=' * 70}")
    results = []
    models = {}

    Xtr_c, Xte_c, y_train, y_test = datasets["clinical"]

    m = MarginalRateModel().fit(Xtr_c, y_train)
    r, _ = evaluate_model(m, Xte_c, y_test, test, "Null (marginal rate)")
    results.append(r)

    time_col = [c for c in Xtr_c.columns if c.startswith("time__")]
    m = DiscreteHazardGLM(link=model_cfg.link).fit(Xtr_c[time_col], y_train)
    r, _ = evaluate_model(m, Xte_c[time_col], y_test, test, "Linear time trend only")
    results.append(r)

    m = DiscreteHazardGLM(link=model_cfg.link).fit(Xtr_c, y_train)
    r, _ = evaluate_model(m, Xte_c, y_test, test, f"Clinical priors ({len(clinical_available)})")
    results.append(r)
    models["clinical"] = m

    Xtr_r, Xte_r = datasets["refined"][:2]
    m = DiscreteHazardGLM(link=model_cfg.link).fit(Xtr_r, y_train)
    r, _ = evaluate_model(m, Xte_r, y_test, test, f"Refined ({len(refined_available)})")
    results.append(r)
    models["refined"] = m

    if selected and selected != clinical_available:
        Xtr_s, Xte_s = datasets["selected"][:2]
        m = DiscreteHazardGLM(link=model_cfg.link).fit(Xtr_s, y_train)
        r, _ = evaluate_model(m, Xte_s, y_test, test, f"Stability-selected ({len(selected)})")
        results.append(r)

    Xtr_e, Xte_e = datasets["expanded"][:2]
    m = DiscreteHazardGLM(link=model_cfg.link).fit(Xtr_e, y_train)
    r, _ = evaluate_model(m, Xte_e, y_test, test, f"Full expanded ({len(expanded_available)})")
    results.append(r)
    models["expanded"] = m

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

    # Bootstrap CIs
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

    # Coefficient tables
    hr_col = "HR" if model_cfg.link == "cloglog" else "OR"
    for label in ["clinical", "refined"]:
        coef = models[label].coef_table
        clinical_coefs = coef.loc[coef.index != "const"]
        print(f"\n{'=' * 70}")
        print(f"COEFFICIENTS — {label} ({model_cfg.link})")
        print(f"{'=' * 70}")
        print(
            clinical_coefs[[hr_col, f"{hr_col}_lower", f"{hr_col}_upper", "p"]].round(4).to_string()
        )

    # Cox PH cross-check
    print(f"\n{'=' * 70}")
    print("COX PROPORTIONAL HAZARDS CROSS-CHECK")
    print(f"{'=' * 70}")
    try:
        cmp = cox_discretization_check(train, refined_available, models["refined"])
        print(cmp.round(3).to_string())
        max_ratio = cmp["ratio"].abs().max()
        print(f"Max HR ratio = {max_ratio:.2f}")
    except Exception as exc:
        print(f"Cox cross-check failed: {exc}")

    # Plots
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

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(
        OUT_DIR / "01_model_evaluation.png", dpi=500, bbox_inches="tight", facecolor=SURFACE
    )
    print(f"\nSaved: {OUT_DIR}/01_model_evaluation.png")
    plt.close()


# ── 2. Adoption figure ───────────────────────────────────────────────────────


def median_time_to_initiation(h: float) -> float:
    """Median months until initiation under a geometric(h) distribution."""
    return np.log(0.5) / np.log(1 - h)


def plot_adoption_figure(model, ct, X_train, X_test, train, test, dataset, split_month):
    """4-panel adoption dynamics figure + structured outputs."""
    refined_available = [f for f in REFINED_FEATURES if f in dataset.columns]
    y_train = train["event"]
    max_month = int(dataset[MONTH_COL].max())
    y_pred_test = model.predict_proba(X_test)[:, 1]

    test_scored = test.copy()
    test_scored["pred"] = y_pred_test

    # Bootstrap CIs
    print(f"\nBootstrapping ({N_BOOTSTRAP} resamples) for adoption figure…")
    rng = np.random.RandomState(42)

    ci = bootstrap_test_ci(test_scored, n_bootstrap=N_BOOTSTRAP, random_state=42)
    auc_pt, auc_lo, auc_hi = ci["auc"]
    bss_pt, bss_lo, bss_hi = ci["bss"]
    mae_pt, mae_lo, mae_hi = ci["count_mae"]

    # Monthly count CI: train-patient bootstrap (model uncertainty)
    unique_train_pats = train["patient_id"].unique()
    boot_monthly = []
    for _ in range(N_BOOTSTRAP):
        sampled = rng.choice(unique_train_pats, size=len(unique_train_pats), replace=True)
        mask = train["patient_id"].isin(sampled)
        Xb, yb = X_train[mask], y_train[mask]
        if yb.sum() < 2:
            continue
        mb = DiscreteHazardGLM(link="cloglog").fit(Xb, yb)
        tp = test.copy()
        tp["pred"] = mb.predict_proba(X_test)[:, 1]
        boot_monthly.append(tp.groupby(MONTH_COL)["pred"].sum())

    boot_monthly_df = pd.DataFrame(boot_monthly).fillna(0)

    monthly_obs = test_scored.groupby(MONTH_COL)["event"].sum()
    monthly_pred = test_scored.groupby(MONTH_COL)["pred"].sum()
    monthly_ci_lo = boot_monthly_df.quantile(0.025)
    monthly_ci_hi = boot_monthly_df.quantile(0.975)
    obs_ci_lo = monthly_obs - 1.96 * np.sqrt(monthly_obs)
    obs_ci_hi = monthly_obs + 1.96 * np.sqrt(monthly_obs)

    # Patient archetypes
    ref_month = split_month
    ref_months_diso = float(
        dataset.loc[dataset[MONTH_COL] == ref_month, "months_since_diso"].median()
    )

    arch_hazards, arch_medians = [], []
    for arch in ARCHETYPES:
        row = pd.DataFrame(
            {
                MONTH_COL: [ref_month],
                "months_since_diso": [ref_months_diso],
                "ccb_ever": [arch["ccb_ever"]],
                "bb_current": [arch["bb_current"]],
                "ccb_current": [arch["ccb_current"]],
            }
        )
        X_arch = pd.DataFrame(
            ct.transform(row[refined_available + [MONTH_COL]]), columns=ct.get_feature_names_out()
        )
        h = float(model.predict_proba(X_arch)[0, 1])
        arch_hazards.append(h)
        arch_medians.append(median_time_to_initiation(h))

    # Score remaining at-risk patients
    initiated = set(dataset.loc[dataset["event"] == 1, "patient_id"])
    remaining = dataset[~dataset["patient_id"].isin(initiated)]
    last_obs = remaining.loc[remaining.groupby("patient_id")[MONTH_COL].idxmax()]
    X_rem = pd.DataFrame(
        ct.transform(last_obs[refined_available + [MONTH_COL]]),
        columns=ct.get_feature_names_out(),
        index=last_obs.index,
    )
    last_obs = last_obs.copy()
    last_obs["pred_hazard"] = model.predict_proba(X_rem)[:, 1]

    # Cumulative S-curve data
    cum_obs = dataset.groupby(MONTH_COL)["event"].sum().cumsum()
    pool_size = dataset["patient_id"].nunique()

    # Print summary
    SEP = "=" * 68
    THIN = "-" * 68
    print(f"\n{SEP}")
    print("TASK 1: CAMZYOS ADOPTION — FULL ANSWER")
    print(SEP)

    print("\n[A] WHICH PATIENTS INITIATE")
    print(THIN)
    for arch, h, med in zip(ARCHETYPES, arch_hazards, arch_medians):
        label = arch["label"].replace("\n", " ")
        print(f"  {label:<40} {h:>11.1%} {med:>9.0f} mo")

    print("\n[B] WHEN — UPTAKE TRAJECTORY")
    print(THIN)
    print(
        f"  Cumulative penetration: {int(cum_obs.iloc[-1])}/{pool_size} = {cum_obs.iloc[-1] / pool_size:.1%}"
    )

    print("\n[C] UNCERTAINTY")
    print(THIN)
    print(f"  AUC: {auc_pt:.3f} [{auc_lo:.3f}, {auc_hi:.3f}]")
    print(f"  BSS: {bss_pt:.4f} [{bss_lo:.4f}, {bss_hi:.4f}]")
    print(f"  MAE: {mae_pt:.1f}/mo [{mae_lo:.1f}, {mae_hi:.1f}]")

    # Save structured outputs
    pd.DataFrame(
        [
            {
                "label": arch["label"].replace("\n", " "),
                "ccb_ever": arch["ccb_ever"],
                "bb_current": arch["bb_current"],
                "ccb_current": arch["ccb_current"],
                "hazard_per_month": round(h, 6),
                "median_tti_months": round(med, 1),
            }
            for arch, h, med in zip(ARCHETYPES, arch_hazards, arch_medians)
        ]
    ).to_csv(OUT_DIR / "03_archetypes.csv", index=False)

    _all_obs = dataset.groupby(MONTH_COL)["event"].sum()
    _train_pred = (
        train.copy()
        .assign(pred=model.predict_proba(X_train)[:, 1])
        .groupby(MONTH_COL)["pred"]
        .sum()
    )
    pd.DataFrame(
        [
            {
                "study_month": int(mo),
                "period": "test" if mo > split_month else "train",
                "observed": int(_all_obs.get(mo, 0)),
                "predicted": round(float(monthly_pred.get(mo, _train_pred.get(mo, 0))), 3),
                "pred_ci_lo": round(float(monthly_ci_lo.get(mo, float("nan"))), 3)
                if mo > split_month
                else None,
                "pred_ci_hi": round(float(monthly_ci_hi.get(mo, float("nan"))), 3)
                if mo > split_month
                else None,
            }
            for mo in sorted(_all_obs.index)
        ]
    ).to_csv(OUT_DIR / "03_monthly_predictions.csv", index=False)

    _mean_rem_h = last_obs["pred_hazard"].mean()
    (OUT_DIR / "03_summary.json").write_text(
        json.dumps(
            {
                "auc_point": round(auc_pt, 4),
                "auc_ci_lo": round(auc_lo, 4),
                "auc_ci_hi": round(auc_hi, 4),
                "brier_skill_score": round(bss_pt, 4),
                "count_mae": round(mae_pt, 3),
                "pool_size": int(pool_size),
                "initiated": int(cum_obs.iloc[-1]),
                "cumulative_penetration_pct": round(cum_obs.iloc[-1] / pool_size * 100, 2),
                "mean_hazard_remaining_pct": round(_mean_rem_h * 100, 4),
                "split_month": int(split_month),
                "max_month": int(max_month),
            },
            indent=2,
        )
    )

    print(f"\nSaved structured outputs to {OUT_DIR}/03_*.csv|json")

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(22, 15), facecolor=SURFACE)
    fig.suptitle(
        "Camzyos Adoption Dynamics",
        fontsize=22,
        fontweight="bold",
        color=INK_PRI,
        x=0.02,
        ha="left",
    )

    for ax in axes.flat:
        style_ax(ax, hide_top_right=True, grid_axis="y")
        ax.tick_params(labelsize=15)

    # Panel 1: Monthly adoption
    ax1 = axes[0, 0]
    all_monthly_obs = dataset.groupby(MONTH_COL)["event"].sum()
    train_scored = train.copy().assign(pred=model.predict_proba(X_train)[:, 1])
    all_monthly_pred = pd.concat([train_scored.groupby(MONTH_COL)["pred"].sum(), monthly_pred])
    months = all_monthly_obs.index.tolist()
    test_months = monthly_obs.index.tolist()

    ax1.bar(
        months,
        all_monthly_obs.values,
        color=C_OBS,
        alpha=0.55,
        width=0.6,
        zorder=2,
        label="Observed",
    )
    ax1.errorbar(
        test_months,
        monthly_obs.values,
        yerr=[monthly_obs.values - obs_ci_lo.values.clip(0), obs_ci_hi.values - monthly_obs.values],
        fmt="none",
        color=C_OBS,
        capsize=3,
        linewidth=1.2,
        zorder=3,
    )
    ax1.plot(
        months,
        all_monthly_pred.reindex(months).values,
        color=C_PRED,
        linewidth=2,
        zorder=4,
        label="Model predicted",
    )
    ax1.fill_between(
        test_months,
        monthly_ci_lo.reindex(test_months).values,
        monthly_ci_hi.reindex(test_months).values,
        color=C_PRED,
        alpha=0.18,
        zorder=1,
        label="95% bootstrap CI",
    )
    ax1.axvline(split_month + 0.5, color=C_SPLIT, linewidth=1.2, linestyle="--", zorder=5)
    ax1.text(
        split_month + 0.6,
        ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 10,
        "test →",
        color=C_SPLIT,
        fontsize=15,
        va="top",
    )
    ax1.set_xlabel("Study month (1 = Apr 2022)", fontsize=17, color=INK_SEC)
    ax1.set_ylabel("New Camzyos initiations", fontsize=17, color=INK_SEC)
    ax1.set_title(
        "Monthly new patients — observed vs predicted",
        fontsize=15,
        fontweight="bold",
        color=INK_PRI,
        loc="left",
    )
    ax1.legend(fontsize=14, frameon=False)
    ax1.set_xlim(0.5, 21.5)

    # Panel 2: Cumulative S-curve
    ax2 = axes[0, 1]
    cum_all = all_monthly_obs.cumsum()
    cum_pred_full = all_monthly_pred.reindex(months).cumsum()

    ax2.step(
        months,
        (cum_all / pool_size * 100).values,
        color=C_OBS,
        linewidth=2.5,
        where="post",
        label="Observed cumulative",
        zorder=3,
    )
    ax2.plot(
        months,
        (cum_pred_full / pool_size * 100).values,
        color=C_PRED,
        linewidth=2,
        linestyle="--",
        label="Model cumulative",
        zorder=3,
    )
    ax2.axhline(100, color=GRID, linewidth=1, linestyle=":")
    ax2.axvline(split_month + 0.5, color=C_SPLIT, linewidth=1.2, linestyle="--", zorder=4)
    ax2.text(
        months[-1] + 0.3,
        cum_all.iloc[-1] / pool_size * 100,
        f"{int(cum_all.iloc[-1])}/{pool_size}\n({cum_all.iloc[-1] / pool_size:.0%})",
        fontsize=15,
        color=C_OBS,
        va="center",
    )
    ax2.set_xlabel("Study month", fontsize=17, color=INK_SEC)
    ax2.set_ylabel("Cumulative initiations (% of at-risk pool)", fontsize=17, color=INK_SEC)
    ax2.set_title(
        "S-curve: cumulative penetration of Disopyramide pool",
        fontsize=15,
        fontweight="bold",
        color=INK_PRI,
        loc="left",
    )
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.legend(fontsize=14, frameon=False)
    ax2.set_xlim(0.5, 21.5)
    ax2.set_ylim(0, 105)

    # Panel 3: Patient archetypes
    ax3 = axes[1, 0]
    labels = [a["label"] for a in ARCHETYPES]
    y_pos = np.arange(len(labels))
    bars = ax3.barh(y_pos, [h * 100 for h in arch_hazards], color=C_ARCH, height=0.55, zorder=2)
    for bar, h, med in zip(bars, arch_hazards, arch_medians):
        ax3.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{h:.1%}/mo  |  median TTI {med:.0f} mo",
            va="center",
            fontsize=15,
            color=INK_SEC,
        )
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(labels, fontsize=17)
    ax3.set_xlabel("Predicted monthly initiation hazard (%)", fontsize=17, color=INK_SEC)
    ax3.set_title(
        f"Which patients? Predicted hazard by archetype\n(study month {ref_month}, median Diso duration {ref_months_diso:.0f} mo)",
        fontsize=15,
        fontweight="bold",
        color=INK_PRI,
        loc="left",
    )
    ax3.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    ax3.grid(axis="y", visible=False)
    ax3.set_xlim(0, max(arch_hazards) * 100 * 2.4)
    ax3.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax3.invert_yaxis()

    # Panel 4: Risk distribution
    ax4 = axes[1, 1]
    hazards_pct = last_obs["pred_hazard"] * 100
    ax4.hist(
        hazards_pct, bins=30, color=C_OBS, alpha=0.75, edgecolor=SURFACE, linewidth=0.5, zorder=2
    )
    ax4.axvline(
        hazards_pct.mean(),
        color=C_PRED,
        linewidth=2,
        linestyle="--",
        label=f"Mean {hazards_pct.mean():.2f}%",
        zorder=3,
    )
    ax4.axvline(
        hazards_pct.quantile(0.75),
        color=INK_MUT,
        linewidth=1.5,
        linestyle=":",
        label=f"75th pct {hazards_pct.quantile(0.75):.2f}%",
        zorder=3,
    )
    ax4.set_xlabel("Predicted monthly hazard (%)", fontsize=17, color=INK_SEC)
    ax4.set_ylabel("Number of remaining patients", fontsize=17, color=INK_SEC)
    ax4.set_title(
        f"Next adopters: risk distribution of {len(last_obs)} remaining patients",
        fontsize=15,
        fontweight="bold",
        color=INK_PRI,
        loc="left",
    )
    ax4.legend(fontsize=16, frameon=False, loc="upper left")
    ax4.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax4.text(
        0.97,
        0.85,
        f"AUC = {auc_pt:.3f}\n95% CI [{auc_lo:.3f}, {auc_hi:.3f}]",
        transform=ax4.transAxes,
        fontsize=15.5,
        color=INK_SEC,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=SPINE, lw=0.8),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUT_DIR / "03_adoption_figure.png", dpi=500, bbox_inches="tight", facecolor=SURFACE)
    print(f"Saved: {OUT_DIR}/03_adoption_figure.png")
    plt.close()


# ── 3. Calibration assessment ────────────────────────────────────────────────


def run_calibration(model, test, X_test):
    """Reliability diagram, Hosmer-Lemeshow test, subgroup calibration, 3-panel figure."""
    test = test.copy()
    test["pred"] = model.predict_proba(X_test)[:, 1]

    n_test_events = int(test["event"].sum())
    print(f"\nCalibration: {len(test):,} person-months, {n_test_events} events")

    # Reliability diagram data
    N_BINS = 10
    test["bin"] = pd.qcut(test["pred"], N_BINS, labels=False, duplicates="drop")
    reliability = (
        test.groupby("bin")
        .agg(mean_pred=("pred", "mean"), n_events=("event", "sum"), n_rows=("event", "count"))
        .reset_index()
    )
    reliability["obs_rate"] = reliability["n_events"] / reliability["n_rows"]

    ci_lo, ci_hi = wilson_ci(reliability["n_events"].values, reliability["n_rows"].values)
    reliability["ci_lo"] = ci_lo
    reliability["ci_hi"] = ci_hi

    # Hosmer-Lemeshow
    hl = hosmer_lemeshow(
        reliability["n_events"].values,
        reliability["n_rows"].values * reliability["mean_pred"].values,
        reliability["n_rows"].values,
        n_bins=N_BINS,
    )
    print(f"  Hosmer-Lemeshow: χ²({hl['dof']}) = {hl['chi2']:.2f}, p = {hl['p_value']:.3f}")

    # Monthly calibration
    monthly = test.groupby(MONTH_COL).agg(obs=("event", "sum"), pred=("pred", "sum")).reset_index()

    # Subgroup calibration
    ccb_cal = subgroup_cal(test, "ccb_ever", "CCB naive (ccb_ever=0)", "CCB tried (ccb_ever=1)")
    bb_cal = subgroup_cal(test, "bb_current", "Not on BB (bb_current=0)", "On BB (bb_current=1)")

    print("\nSubgroup calibration:")
    for _, r in pd.concat([ccb_cal, bb_cal]).iterrows():
        print(f"  {r['label']}: obs {r['obs_rate']:.3%}  pred {r['pred_rate']:.3%}  (n={r['n']:,})")

    # Save outputs
    reliability.to_csv(OUT_DIR / "04_reliability.csv", index=False)
    mae = float((monthly["pred"] - monthly["obs"]).abs().mean())
    monthly.to_csv(OUT_DIR / "04_monthly_calibration.csv", index=False)
    pd.concat([ccb_cal, bb_cal], ignore_index=True).to_csv(
        OUT_DIR / "04_subgroup_calibration.csv", index=False
    )
    (OUT_DIR / "04_hl_test.json").write_text(
        json.dumps(
            {
                "hl_chi2": round(hl["chi2"], 3),
                "hl_dof": hl["dof"],
                "hl_pval": round(hl["p_value"], 4),
                "n_bins": N_BINS,
                "mae_patients_per_month": round(mae, 3),
                "n_test_events": n_test_events,
            },
            indent=2,
        )
    )
    print(f"Saved calibration outputs to {OUT_DIR}/04_*.csv|json")

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), facecolor=SURFACE)
    fig.suptitle(
        "Model Calibration — Discrete-Time Hazard (Cloglog)",
        fontsize=12,
        fontweight="bold",
        color=INK_PRI,
        x=0.02,
        ha="left",
    )
    for ax in axes:
        style_ax(ax, hide_top_right=True, grid_axis="none")

    # Panel 1: Reliability diagram
    ax1 = axes[0]
    max_val = max(reliability["mean_pred"].max(), reliability["obs_rate"].max()) * 1.15
    ax1.plot(
        [0, max_val * 100],
        [0, max_val * 100],
        color=C_REF,
        linewidth=1.2,
        linestyle="--",
        zorder=1,
        label="Perfect calibration",
    )
    ax1.errorbar(
        reliability["mean_pred"] * 100,
        reliability["obs_rate"] * 100,
        yerr=[
            (reliability["obs_rate"] - reliability["ci_lo"]) * 100,
            (reliability["ci_hi"] - reliability["obs_rate"]) * 100,
        ],
        fmt="o",
        color=C_OBS,
        markersize=7,
        capsize=4,
        linewidth=1.5,
        zorder=3,
        label="Observed (95% Wilson CI)",
    )
    for _, row in reliability.iterrows():
        ax1.annotate(
            f"n={int(row['n_rows'])}",
            (row["mean_pred"] * 100, row["obs_rate"] * 100),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            color=INK_MUT,
        )
    ax1.set_xlabel("Mean predicted hazard per bin (%)", fontsize=9, color=INK_SEC)
    ax1.set_ylabel("Observed event rate (%)", fontsize=9, color=INK_SEC)
    ax1.set_title(
        f"Reliability diagram\n(test set, {N_BINS} quantile bins)\nHL χ²({hl['dof']})={hl['chi2']:.1f}  p={hl['p_value']:.2f}",
        fontsize=9.5,
        fontweight="bold",
        color=INK_PRI,
        loc="left",
    )
    ax1.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.legend(fontsize=8, frameon=False)
    style_ax(ax1, hide_top_right=True, grid_axis="both")
    ax1.set_xlim(-0.02 * 100 * max_val, max_val * 100 * 1.05)
    ax1.set_ylim(-0.02 * 100 * max_val, max_val * 100 * 1.05)

    # Panel 2: Monthly calibration
    ax2 = axes[1]
    m_months = monthly[MONTH_COL].values
    x = np.arange(len(m_months))
    w = 0.35
    ax2.bar(
        x - w / 2,
        monthly["obs"].values,
        width=w,
        color=C_OBS,
        alpha=0.8,
        label="Observed",
        zorder=2,
    )
    ax2.bar(
        x + w / 2,
        monthly["pred"].values,
        width=w,
        color=C_PRED,
        alpha=0.8,
        label="Predicted",
        zorder=2,
    )
    obs_ci_lo_m = np.maximum(0, monthly["obs"] - 1.96 * np.sqrt(monthly["obs"]))
    obs_ci_hi_m = monthly["obs"] + 1.96 * np.sqrt(monthly["obs"])
    ax2.errorbar(
        x - w / 2,
        monthly["obs"].values,
        yerr=[
            monthly["obs"].values - obs_ci_lo_m.values,
            obs_ci_hi_m.values - monthly["obs"].values,
        ],
        fmt="none",
        color=C_OBS,
        capsize=3,
        linewidth=1.2,
        zorder=3,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"M{int(m)}" for m in m_months], fontsize=8, rotation=45)
    ax2.set_xlabel("Study month (test period)", fontsize=9, color=INK_SEC)
    ax2.set_ylabel("New initiations", fontsize=9, color=INK_SEC)
    ax2.set_title(
        "Monthly calibration\n(predicted vs observed counts, test period)\nError bars = Poisson 95% CI",
        fontsize=9.5,
        fontweight="bold",
        color=INK_PRI,
        loc="left",
    )
    ax2.legend(fontsize=8, frameon=False)
    style_ax(ax2, hide_top_right=True, grid_axis="y")
    ax2.text(
        0.97,
        0.97,
        f"MAE = {mae:.1f} patients/month",
        transform=ax2.transAxes,
        fontsize=8,
        color=INK_SEC,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec=SPINE, lw=0.8),
    )

    # Panel 3: Subgroup calibration
    ax3 = axes[2]
    all_sub = pd.concat([ccb_cal, bb_cal], ignore_index=True)
    y_pos = np.arange(len(all_sub))
    ax3.barh(
        y_pos,
        all_sub["obs_rate"] * 100,
        height=0.35,
        color=C_OBS,
        alpha=0.8,
        label="Observed",
        zorder=2,
    )
    ax3.barh(
        y_pos - 0.38,
        all_sub["pred_rate"] * 100,
        height=0.35,
        color=C_PRED,
        alpha=0.8,
        label="Predicted",
        zorder=2,
    )
    ax3.errorbar(
        all_sub["obs_rate"] * 100,
        y_pos,
        xerr=[
            (all_sub["obs_rate"] - all_sub["ci_lo"]) * 100,
            (all_sub["ci_hi"] - all_sub["obs_rate"]) * 100,
        ],
        fmt="none",
        color=C_OBS,
        capsize=3,
        linewidth=1.2,
        zorder=3,
    )
    ax3.set_yticks(y_pos - 0.19)
    ax3.set_yticklabels(all_sub["label"], fontsize=8.5)
    ax3.set_xlabel("Event rate (%/person-month)", fontsize=9, color=INK_SEC)
    ax3.set_title(
        "Subgroup calibration\n(key predictors: ccb_ever, bb_current)",
        fontsize=9.5,
        fontweight="bold",
        color=INK_PRI,
        loc="left",
    )
    ax3.legend(fontsize=8, frameon=False)
    style_ax(ax3, hide_top_right=True, grid_axis="x")
    ax3.grid(axis="y", visible=False)
    ax3.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax3.invert_yaxis()
    for i, row in all_sub.iterrows():
        ax3.text(
            ax3.get_xlim()[1] * 0.02,
            i - 0.19,
            f"  n={row['n']:,}  events={row['obs']}",
            va="center",
            fontsize=7.5,
            color=INK_MUT,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(OUT_DIR / "04_calibration.png", dpi=500, bbox_inches="tight", facecolor=SURFACE)
    print(f"Saved: {OUT_DIR}/04_calibration.png")
    plt.close()


# ── 4. Hazard ratio forest plot ──────────────────────────────────────────────


def plot_forest(model):
    """HR forest plot for the refined model features."""
    tbl = model.coef_table
    features = [f for f in tbl.index if f != "const" and not f.startswith("time__")]
    tbl_feat = tbl.loc[features]

    labels = [FEATURE_LABELS.get(f, f) for f in features]
    hrs = tbl_feat["HR"].values
    lo = tbl_feat["HR_lower"].values
    hi = tbl_feat["HR_upper"].values
    pvals = tbl_feat["p"].values

    order = np.argsort(hrs)[::-1]
    labels = [labels[i] for i in order]
    hrs, lo, hi, pvals = hrs[order], lo[order], hi[order], pvals[order]

    n_rows = len(labels)
    fig, ax = plt.subplots(figsize=(9, 1.1 + n_rows * 0.55), facecolor=SURFACE)
    style_ax(ax, hide_top_right=True, grid_axis="x")
    ax.grid(axis="y", visible=False)

    y_pos = np.arange(len(labels))
    ax.axvline(1.0, color=INK_MUT, linewidth=1.2, linestyle="--", zorder=1)
    ax.errorbar(
        hrs,
        y_pos,
        xerr=[hrs - lo, hi - hrs],
        fmt="o",
        color=C_OBS,
        ecolor=C_ARCH[0],
        elinewidth=2.5,
        capsize=5,
        capthick=1.5,
        markersize=8,
        zorder=3,
    )

    for i, (hr, p, ci_lo, ci_hi) in enumerate(zip(hrs, pvals, lo, hi)):
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        ax.text(
            max(ci_hi, hr) * 1.15,
            i,
            f"HR {hr:.2f} ({ci_lo:.2f}–{ci_hi:.2f}){sig}",
            va="center",
            fontsize=8.5,
            color=INK_SEC,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9, color=INK_PRI)
    ax.set_xscale("log")
    ax.set_xlabel("Hazard Ratio (log scale)", fontsize=10, color=INK_SEC)
    ax.set_title(
        f"Feature hazard ratios — {len(features)}-feature refined model\n"
        "95% confidence intervals  |  ** p<0.01  * p<0.05",
        fontsize=11,
        fontweight="bold",
        color=INK_PRI,
        loc="left",
    )
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:g}"))

    # Save HR table
    pd.DataFrame(
        {
            "feature": [features[i] for i in order],
            "label": labels,
            "HR": hrs,
            "HR_lower": lo,
            "HR_upper": hi,
            "p": pvals,
        }
    ).to_csv(OUT_DIR / "05_hr_table.csv", index=False)
    print(f"\nSaved: {OUT_DIR}/05_hr_table.csv")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_hr_forest_plot.png", dpi=500, bbox_inches="tight", facecolor=SURFACE)
    print(f"Saved: {OUT_DIR}/05_hr_forest_plot.png")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset, train, test, split_month = load_and_split()

    # Fit the refined model once
    refined_available = [f for f in REFINED_FEATURES if f in dataset.columns]
    X_train, X_test, y_train, y_test, ct = prepare_Xy(train, test, refined_available)
    model = DiscreteHazardGLM(link=ModelConfig().link).fit(X_train, y_train)

    print(f"\nRefined model fitted ({len(refined_available)} features)")

    # Run all analysis steps
    run_model_comparison(train, test, dataset)
    plot_adoption_figure(model, ct, X_train, X_test, train, test, dataset, split_month)
    run_calibration(model, test, X_test)
    plot_forest(model)

    print(f"\n{'=' * 70}")
    print("All Task 1 outputs written to outputs/task1_adoption/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

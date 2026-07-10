#!/usr/bin/env python
"""Task 1 adoption figure — four-panel summary for the case study.

Panels:
  1. Monthly new starts (observed + predicted) with bootstrap CI
  2. Cumulative S-curve (observed + predicted, % of at-risk pool)
  3. Patient archetypes — predicted hazard by treatment-trajectory profile
  4. Risk distribution of remaining at-risk patients

Bootstrap CIs: 500 iterations, patient-level resampling.

Output: outputs/task1_adoption/03_adoption_figure.png
Run:    .venv/bin/python scripts/task1_adoption/03_adoption_figure.py
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
    C_SPLIT,
    GRID,
    INK_MUT,
    INK_PRI,
    INK_SEC,
    SURFACE,
    style_ax,
)
from src.task1_adoption.config import (
    LAUNCH_MONTH,
    MONTH_COL,
    REFINED_FEATURES,
    DatasetConfig,
    SplitConfig,
)
from src.task1_adoption.data_loading import load_data
from src.task1_adoption.dataset import build_dataset
from src.task1_adoption.evaluation import (
    bootstrap_test_ci,
)
from src.task1_adoption.models import DiscreteHazardGLM

np.random.seed(42)

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


def make_preprocessor(feature_cols):
    return ColumnTransformer(
        [("time", "passthrough", [MONTH_COL]), ("features", "passthrough", feature_cols)],
        remainder="drop",
    )


def prepare(train, test, feats):
    all_cols = feats + [MONTH_COL]
    ct = make_preprocessor(feats)
    Xtr = pd.DataFrame(
        ct.fit_transform(train[all_cols]), columns=ct.get_feature_names_out(), index=train.index
    )
    Xte = pd.DataFrame(
        ct.transform(test[all_cols]), columns=ct.get_feature_names_out(), index=test.index
    )
    return Xtr, Xte, train["event"], test["event"], ct


def median_time_to_initiation(h: float) -> float:
    """Median months until initiation under a geometric(h) distribution."""
    return np.log(0.5) / np.log(1 - h)


# ── 1. Build data ─────────────────────────────────────────────────────────────

print("Loading and building dataset…")
data = load_data()
dataset = build_dataset(data, DatasetConfig())
split_month = (
    pd.Period(SplitConfig().train_end_month, freq="M") - pd.Period(LAUNCH_MONTH, freq="M")
).n + 1
max_month = int(dataset[MONTH_COL].max())

train = dataset[dataset[MONTH_COL] <= split_month].copy()
test = dataset[dataset[MONTH_COL] > split_month].copy()

refined_available = [f for f in REFINED_FEATURES if f in dataset.columns]
X_train, X_test, y_train, y_test, ct = prepare(train, test, refined_available)

model = DiscreteHazardGLM(link="cloglog").fit(X_train, y_train)
y_pred_test = model.predict_proba(X_test)[:, 1]

test_scored = test.copy()
test_scored["pred"] = y_pred_test

# ── 2. Bootstrap CIs ─────────────────────────────────────────────────────────
# Test-patient bootstrap: evaluation uncertainty for AUC, BSS, count MAE.
# Train-patient bootstrap: model uncertainty on predicted monthly counts (for plot).

print(f"Bootstrapping ({N_BOOTSTRAP} resamples)…")
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
    Xb = X_train[mask]
    yb = y_train[mask]
    if yb.sum() < 2:
        continue
    mb = DiscreteHazardGLM(link="cloglog").fit(Xb, yb)
    pred_b = mb.predict_proba(X_test)[:, 1]
    tp = test.copy()
    tp["pred"] = pred_b
    tp_month = tp.groupby(MONTH_COL)["pred"].sum()
    boot_monthly.append(tp_month)

boot_monthly_df = pd.DataFrame(boot_monthly).fillna(0)

monthly_obs = test_scored.groupby(MONTH_COL)["event"].sum()
monthly_pred = test_scored.groupby(MONTH_COL)["pred"].sum()
monthly_ci_lo = boot_monthly_df.quantile(0.025)
monthly_ci_hi = boot_monthly_df.quantile(0.975)

# Poisson 95% CI on observed counts
obs_ci_lo = monthly_obs - 1.96 * np.sqrt(monthly_obs)
obs_ci_hi = monthly_obs + 1.96 * np.sqrt(monthly_obs)

# ── 3. Patient archetypes ─────────────────────────────────────────────────────

ref_month = split_month  # end of training period
ref_months_diso = float(dataset.loc[dataset[MONTH_COL] == ref_month, "months_since_diso"].median())

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

# ── 4. Score remaining at-risk patients ───────────────────────────────────────

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

# ── 5. Cumulative S-curve ─────────────────────────────────────────────────────

cum_obs = dataset.groupby(MONTH_COL)["event"].sum().cumsum()
cum_pred_test = monthly_pred.cumsum() + train["event"].sum()  # add train events for continuity
cum_pred_train = (
    train.copy()
    .assign(pred=model.predict_proba(X_train)[:, 1])
    .groupby(MONTH_COL)["pred"]
    .sum()
    .cumsum()
)

pool_size = dataset["patient_id"].nunique()

# ── 6. Print summary ──────────────────────────────────────────────────────────

SEP = "=" * 68
THIN = "-" * 68

print(f"\n{SEP}")
print("TASK 1: CAMZYOS ADOPTION — FULL ANSWER")
print(SEP)

print("\n[A] WHICH PATIENTS INITIATE")
print(THIN)
print(f"{'Archetype':<42} {'Hazard/month':>12} {'Median TTI':>12}")
print(THIN)
for arch, h, med in zip(ARCHETYPES, arch_hazards, arch_medians):
    label = arch["label"].replace("\n", " ")
    print(f"  {label:<40} {h:>11.1%} {med:>9.0f} mo")

print(
    f"\n  Reference month: {ref_month} ({SplitConfig().train_end_month}),",
    f"months_since_diso={ref_months_diso:.0f}",
)
print(
    f"\n  Key driver: ccb_ever HR = {np.exp(model.result_.params['features__ccb_ever']):.2f}",
    f"(95% CI {np.exp(model.coef_table.loc['features__ccb_ever', 'ci_lower']):.2f}–",
    f"{np.exp(model.coef_table.loc['features__ccb_ever', 'ci_upper']):.2f})",
    f"p={model.coef_table.loc['features__ccb_ever', 'p']:.4f}",
)
print(
    f"  Current BB:      bb_current HR = {np.exp(model.result_.params['features__bb_current']):.2f}",
    f"(95% CI {np.exp(model.coef_table.loc['features__bb_current', 'ci_lower']):.2f}–",
    f"{np.exp(model.coef_table.loc['features__bb_current', 'ci_upper']):.2f})",
    f"p={model.coef_table.loc['features__bb_current', 'p']:.4f}",
)

print("\n[B] WHEN — UPTAKE TRAJECTORY")
print(THIN)
train_monthly_obs = train.groupby(MONTH_COL)["event"].sum()
print(
    f"  Training period (months 1–{split_month}): {int(train['event'].sum())} events,",
    f"mean {train_monthly_obs.mean():.1f}/month (range {int(train_monthly_obs.min())}–{int(train_monthly_obs.max())})",
)
print(
    f"  Test period     (months {split_month + 1}–{max_month}):  {int(y_test.sum())} events,",
    f"mean {monthly_obs.mean():.1f}/month (range {int(monthly_obs.min())}–{int(monthly_obs.max())})",
)
print(
    f"\n  Cumulative penetration of at-risk pool ({pool_size} patients): {int(cum_obs.iloc[-1])}/{pool_size} = {cum_obs.iloc[-1] / pool_size:.1%}"
)
print(f"  Remaining at-risk (not yet initiated):  {pool_size - int(cum_obs.iloc[-1])} patients")
mean_rem_h = last_obs["pred_hazard"].mean()
print(f"  Mean predicted hazard for remaining patients: {mean_rem_h:.2%}/month")
print(
    f"  At steady state: ~{mean_rem_h * (pool_size - int(cum_obs.iloc[-1])):.1f} new starts/month from remaining pool"
)

print("\n[C] UNCERTAINTY")
print(THIN)
print(f"  Time-dependent AUC (test): {auc_pt:.3f}  95% CI [{auc_lo:.3f}, {auc_hi:.3f}]")
print(f"  Brier Skill Score:         {bss_pt:.4f}  95% CI [{bss_lo:.4f}, {bss_hi:.4f}]")
print(f"  Count MAE:                 {mae_pt:.1f}/mo   95% CI [{mae_lo:.1f}, {mae_hi:.1f}]")
print(f"\n  Monthly prediction intervals (test months {split_month + 1}–{max_month}):")
print(f"  {'Month':>6}  {'Obs':>5}  {'Pred':>6}  {'95% CI':>14}")
for m in sorted(monthly_obs.index):
    ob = int(monthly_obs.get(m, 0))
    pr = monthly_pred.get(m, 0)
    lo = monthly_ci_lo.get(m, 0)
    hi = monthly_ci_hi.get(m, 0)
    print(f"  {m:>6}  {ob:>5}  {pr:>6.1f}  [{lo:>5.1f}, {hi:>5.1f}]")

print("\n[D] TOP NEXT-ADOPTER CANDIDATES")
print(THIN)
top10 = last_obs.nlargest(10, "pred_hazard")[
    ["patient_id", MONTH_COL, "pred_hazard", "ccb_ever", "bb_current", "ccb_current"]
]
print(top10.to_string(index=False))

# ── 6b. Save structured outputs ──────────────────────────────────────────────

out_dir = Path("outputs/task1_adoption")
out_dir.mkdir(parents=True, exist_ok=True)

pd.DataFrame(
    [
        {
            "label": arch["label"].replace("\n", " "),
            "ccb_ever": arch["ccb_ever"],
            "bb_current": arch["bb_current"],
            "ccb_current": arch["ccb_current"],
            "hazard_per_month": round(h, 6),
            "hazard_per_month_pct": round(h * 100, 4),
            "median_tti_months": round(med, 1),
        }
        for arch, h, med in zip(ARCHETYPES, arch_hazards, arch_medians)
    ]
).to_csv(out_dir / "03_archetypes.csv", index=False)

_all_obs = dataset.groupby(MONTH_COL)["event"].sum()
_train_pred = (
    train.copy().assign(pred=model.predict_proba(X_train)[:, 1]).groupby(MONTH_COL)["pred"].sum()
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
).to_csv(out_dir / "03_monthly_predictions.csv", index=False)

_mean_rem_h = last_obs["pred_hazard"].mean()
(out_dir / "03_summary.json").write_text(
    json.dumps(
        {
            "auc_point": round(auc_pt, 4),
            "auc_ci_lo": round(auc_lo, 4),
            "auc_ci_hi": round(auc_hi, 4),
            "brier_skill_score": round(bss_pt, 4),
            "bss_ci_lo": round(bss_lo, 4),
            "bss_ci_hi": round(bss_hi, 4),
            "count_mae": round(mae_pt, 3),
            "count_mae_ci_lo": round(mae_lo, 3),
            "count_mae_ci_hi": round(mae_hi, 3),
            "n_test_events": int(y_test.sum()),
            "pool_size": int(pool_size),
            "initiated": int(cum_obs.iloc[-1]),
            "remaining": int(pool_size - cum_obs.iloc[-1]),
            "cumulative_penetration_pct": round(cum_obs.iloc[-1] / pool_size * 100, 2),
            "mean_hazard_remaining_pct": round(_mean_rem_h * 100, 4),
            "steady_state_new_starts_per_month": round(
                _mean_rem_h * (pool_size - int(cum_obs.iloc[-1])), 1
            ),
            "split_month": int(split_month),
            "max_month": int(max_month),
        },
        indent=2,
    )
)

print(f"\nSaved: {out_dir}/03_archetypes.csv")
print(f"Saved: {out_dir}/03_monthly_predictions.csv")
print(f"Saved: {out_dir}/03_summary.json")

# ── 7. Figure ─────────────────────────────────────────────────────────────────

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

# ── Panel 1: Monthly adoption (training + test) ──────────────────────────────
ax1 = axes[0, 0]

all_monthly_obs = dataset.groupby(MONTH_COL)["event"].sum()
train_scored = train.copy().assign(pred=model.predict_proba(X_train)[:, 1])
all_monthly_pred_train = train_scored.groupby(MONTH_COL)["pred"].sum()
all_monthly_pred = pd.concat([all_monthly_pred_train, monthly_pred])

months = all_monthly_obs.index.tolist()
# Bars: observed (full period)
ax1.bar(
    months, all_monthly_obs.values, color=C_OBS, alpha=0.55, width=0.6, zorder=2, label="Observed"
)
# Poisson CI on observed (test only)
test_months = monthly_obs.index.tolist()
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
# Line: model predicted (full period)
ax1.plot(
    months,
    all_monthly_pred.reindex(months).values,
    color=C_PRED,
    linewidth=2,
    zorder=4,
    label="Model predicted",
)
# Bootstrap CI band (test only)
ax1.fill_between(
    test_months,
    monthly_ci_lo.reindex(test_months).values,
    monthly_ci_hi.reindex(test_months).values,
    color=C_PRED,
    alpha=0.18,
    zorder=1,
    label="95% bootstrap CI",
)
# Train/test split line
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

# ── Panel 2: Cumulative S-curve ───────────────────────────────────────────────
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

# ── Panel 3: Patient archetypes ───────────────────────────────────────────────
ax3 = axes[1, 0]

labels = [a["label"] for a in ARCHETYPES]
y_pos = np.arange(len(labels))
bars = ax3.barh(y_pos, [h * 100 for h in arch_hazards], color=C_ARCH, height=0.55, zorder=2)
# Direct labels
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
    "Which patients? Predicted hazard by archetype\n"
    f"(study month {ref_month}, median Diso duration {ref_months_diso:.0f} mo)",
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

# ── Panel 4: Risk distribution of remaining patients ─────────────────────────
ax4 = axes[1, 1]

hazards_pct = last_obs["pred_hazard"] * 100
ax4.hist(hazards_pct, bins=30, color=C_OBS, alpha=0.75, edgecolor=SURFACE, linewidth=0.5, zorder=2)
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

# Annotation box — positioned below the legend area to avoid overlap
ax4.text(
    0.97,
    0.85,
    f"AUC = {auc_pt:.3f}\n95% CI [{auc_lo:.3f}, {auc_hi:.3f}]",
    transform=ax4.transAxes,
    fontsize=15.5,
    color=INK_SEC,
    va="top",
    ha="right",
    bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=GRID, lw=0.8),
)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out_path = Path("outputs/task1_adoption/03_adoption_figure.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=500, bbox_inches="tight", facecolor=SURFACE)
print(f"\nFigure saved to {out_path}")
plt.close()

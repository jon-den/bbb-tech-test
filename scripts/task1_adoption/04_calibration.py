#!/usr/bin/env python
"""Calibration assessment for the discrete-time hazard model.

Three-panel figure:
  1. Reliability diagram — predicted deciles vs observed event rate (test set)
  2. Monthly calibration — predicted vs observed new starts per month (test)
  3. Subgroup calibration — ccb_ever (main driver) and bb_current (main guard)

Calibration matters for the investment case: predicted hazard directly feeds the
expected-new-starts projection used in market sizing. If the model says 2%/month
for a patient cohort, that should mean ~2% initiate per month.

Output: outputs/task1_adoption/04_calibration.png
Run:    .venv/bin/python scripts/task1_adoption/04_calibration.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer

from src.task1_adoption.config import (
    LAUNCH_MONTH,
    MONTH_COL,
    REFINED_FEATURES,
    DatasetConfig,
    SplitConfig,
)
from src.task1_adoption.data_loading import load_data
from src.task1_adoption.dataset import build_dataset
from src.task1_adoption.models import DiscreteHazardGLM

np.random.seed(42)

# ── Palette ───────────────────────────────────────────────────────────────────
C_OBS = "#2a78d6"
C_PRED = "#1baf7a"
C_REF = "#c3c2b7"  # 45° perfect-calibration diagonal
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
INK_PRI = "#0b0b0b"
INK_SEC = "#52514e"
INK_MUT = "#898781"


def make_preprocessor(feature_cols):
    return ColumnTransformer(
        [("time", "passthrough", [MONTH_COL]), ("features", "passthrough", feature_cols)],
        remainder="drop",
    )


# ── 1. Build and fit ──────────────────────────────────────────────────────────

print("Building dataset and fitting model…")
data = load_data()
dataset = build_dataset(data, DatasetConfig())
split_month = (
    pd.Period(SplitConfig().train_end_month, freq="M") - pd.Period(LAUNCH_MONTH, freq="M")
).n + 1

train = dataset[dataset[MONTH_COL] <= split_month].copy()
test = dataset[dataset[MONTH_COL] > split_month].copy()

refined_available = [f for f in REFINED_FEATURES if f in dataset.columns]
all_cols = refined_available + [MONTH_COL]

ct = make_preprocessor(refined_available)
X_train = pd.DataFrame(
    ct.fit_transform(train[all_cols]), columns=ct.get_feature_names_out(), index=train.index
)
X_test = pd.DataFrame(
    ct.transform(test[all_cols]), columns=ct.get_feature_names_out(), index=test.index
)

model = DiscreteHazardGLM(link="cloglog").fit(X_train, train["event"])
test = test.copy()
test["pred"] = model.predict_proba(X_test)[:, 1]

n_test_events = int(test["event"].sum())
print(
    f"  Test set: {len(test):,} person-months, {n_test_events} events, "
    f"event rate {n_test_events / len(test):.3%}"
)

# ── 2. Reliability diagram data ───────────────────────────────────────────────
# Quantile-based bins (equal-count rather than equal-width) so each bin has
# enough person-months for stable rate estimates despite the low overall rate.

N_BINS = 10
test["bin"] = pd.qcut(test["pred"], N_BINS, labels=False, duplicates="drop")
reliability = (
    test.groupby("bin")
    .agg(mean_pred=("pred", "mean"), n_events=("event", "sum"), n_rows=("event", "count"))
    .reset_index()
)
reliability["obs_rate"] = reliability["n_events"] / reliability["n_rows"]


# Wilson score 95% CI on observed rate
def wilson_ci(k, n, z=1.96):
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return np.maximum(0, centre - half), np.minimum(1, centre + half)


ci_lo, ci_hi = wilson_ci(reliability["n_events"].values, reliability["n_rows"].values)
reliability["ci_lo"] = ci_lo
reliability["ci_hi"] = ci_hi

# Hosmer-Lemeshow χ² statistic
hl_chi2 = (
    (reliability["n_events"] - reliability["n_rows"] * reliability["mean_pred"]) ** 2
    / (reliability["n_rows"] * reliability["mean_pred"] * (1 - reliability["mean_pred"]))
).sum()
hl_dof = N_BINS - 2
hl_pval = 1 - stats.chi2.cdf(hl_chi2, df=hl_dof)

print(f"  Hosmer-Lemeshow: χ²({hl_dof}) = {hl_chi2:.2f}, p = {hl_pval:.3f}")

# ── 3. Monthly calibration ────────────────────────────────────────────────────

monthly = test.groupby(MONTH_COL).agg(obs=("event", "sum"), pred=("pred", "sum")).reset_index()

# ── 4. Subgroup calibration ───────────────────────────────────────────────────


def subgroup_cal(df, feature_col, label_0, label_1):
    rows = []
    for val, label in [(0, label_0), (1, label_1)]:
        sub = df[df[feature_col] == val]
        n = len(sub)
        obs = int(sub["event"].sum())
        pred = float(sub["pred"].sum())
        obs_rate = obs / n
        pred_rate = pred / n
        lo, hi = wilson_ci(np.array([obs]), np.array([n]))
        rows.append(
            {
                "label": label,
                "obs_rate": obs_rate,
                "pred_rate": pred_rate,
                "obs": obs,
                "pred": pred,
                "n": n,
                "ci_lo": lo[0],
                "ci_hi": hi[0],
            }
        )
    return pd.DataFrame(rows)


ccb_cal = subgroup_cal(test, "ccb_ever", "CCB naive (ccb_ever=0)", "CCB tried (ccb_ever=1)")
bb_cal = subgroup_cal(test, "bb_current", "Not on BB (bb_current=0)", "On BB (bb_current=1)")

print("\nSubgroup calibration (test set):")
print("  ccb_ever:")
for _, r in ccb_cal.iterrows():
    print(
        f"    {r['label']}: obs {r['obs_rate']:.3%}  pred {r['pred_rate']:.3%}  "
        f"(n={r['n']:,}, events={r['obs']})"
    )
print("  bb_current:")
for _, r in bb_cal.iterrows():
    print(
        f"    {r['label']}: obs {r['obs_rate']:.3%}  pred {r['pred_rate']:.3%}  "
        f"(n={r['n']:,}, events={r['obs']})"
    )

# ── 5. Figure ─────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=SURFACE)
fig.suptitle(
    "Model Calibration — Discrete-Time Hazard (Cloglog)",
    fontsize=13,
    fontweight="bold",
    color=INK_PRI,
    x=0.02,
    ha="left",
)

for ax in axes:
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=INK_MUT, labelsize=9)

# ── Panel 1: Reliability diagram ──────────────────────────────────────────────
ax1 = axes[0]

# Perfect calibration reference
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

# Annotate each dot with bin sample size
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
    f"Reliability diagram\n(test set, {N_BINS} quantile bins)\n"
    f"HL χ²({hl_dof})={hl_chi2:.1f}  p={hl_pval:.2f}",
    fontsize=9.5,
    fontweight="bold",
    color=INK_PRI,
    loc="left",
)
ax1.xaxis.set_major_formatter(mtick.PercentFormatter())
ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
ax1.legend(fontsize=8, frameon=False)
ax1.grid(color=GRID, linewidth=0.7, zorder=0)
ax1.set_xlim(-0.02 * 100 * max_val, max_val * 100 * 1.05)
ax1.set_ylim(-0.02 * 100 * max_val, max_val * 100 * 1.05)

# ── Panel 2: Monthly calibration ──────────────────────────────────────────────
ax2 = axes[1]

months = monthly[MONTH_COL].values
x = np.arange(len(months))
w = 0.35

obs_bars = ax2.bar(
    x - w / 2, monthly["obs"].values, width=w, color=C_OBS, alpha=0.8, label="Observed", zorder=2
)
pred_bars = ax2.bar(
    x + w / 2, monthly["pred"].values, width=w, color=C_PRED, alpha=0.8, label="Predicted", zorder=2
)

# Poisson CI on observed
obs_ci_lo = np.maximum(0, monthly["obs"] - 1.96 * np.sqrt(monthly["obs"]))
obs_ci_hi = monthly["obs"] + 1.96 * np.sqrt(monthly["obs"])
ax2.errorbar(
    x - w / 2,
    monthly["obs"].values,
    yerr=[monthly["obs"].values - obs_ci_lo.values, obs_ci_hi.values - monthly["obs"].values],
    fmt="none",
    color=C_OBS,
    capsize=3,
    linewidth=1.2,
    zorder=3,
)

ax2.set_xticks(x)
ax2.set_xticklabels([f"M{int(m)}" for m in months], fontsize=8, rotation=45)
ax2.set_xlabel("Study month (test period)", fontsize=9, color=INK_SEC)
ax2.set_ylabel("New initiations", fontsize=9, color=INK_SEC)
ax2.set_title(
    "Monthly calibration\n(predicted vs observed counts, test period)\n"
    "Error bars = Poisson 95% CI on observed",
    fontsize=9.5,
    fontweight="bold",
    color=INK_PRI,
    loc="left",
)
ax2.legend(fontsize=8, frameon=False)
ax2.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)

# MAE annotation
mae = float((monthly["pred"] - monthly["obs"]).abs().mean())
ax2.text(
    0.97,
    0.97,
    f"MAE = {mae:.1f} patients/month",
    transform=ax2.transAxes,
    fontsize=8,
    color=INK_SEC,
    va="top",
    ha="right",
    bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec=GRID, lw=0.8),
)

# ── Panel 3: Subgroup calibration ─────────────────────────────────────────────
ax3 = axes[2]

all_sub = pd.concat([ccb_cal, bb_cal], ignore_index=True)
y_pos = np.arange(len(all_sub))

# Observed rate + CI
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

# CI on observed
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
    "Subgroup calibration\n(key predictors: ccb_ever, bb_current)\n"
    "Each bar pair: observed (blue) vs predicted (aqua)",
    fontsize=9.5,
    fontweight="bold",
    color=INK_PRI,
    loc="left",
)
ax3.legend(fontsize=8, frameon=False)
ax3.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
ax3.grid(axis="y", visible=False)
ax3.xaxis.set_major_formatter(mtick.PercentFormatter())
ax3.invert_yaxis()

# Annotate with n and events
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
out_path = Path("outputs/task1_adoption/04_calibration.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
print(f"\nFigure saved to {out_path}")
plt.close()

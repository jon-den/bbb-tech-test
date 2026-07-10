#!/usr/bin/env python
"""Hazard ratio forest plot for the 4-feature refined model.

Output: outputs/task1_adoption/05_hr_forest_plot.png
Run:    .venv/bin/python scripts/task1_adoption/05_hr_forest_plot.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.task1_adoption.config import (
    LAUNCH_MONTH,
    MONTH_COL,
    REFINED_FEATURES,
    DatasetConfig,
    ModelConfig,
    SplitConfig,
)
from src.task1_adoption.data_loading import load_data
from src.task1_adoption.dataset import build_dataset
from src.task1_adoption.models import DiscreteHazardGLM

np.random.seed(42)

SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
INK_PRI = "#0b0b0b"
INK_SEC = "#52514e"
INK_MUT = "#898781"
C_POINT = "#2a78d6"
C_CI = "#86b6ef"

FEATURE_LABELS = {
    "features__ccb_ever": "Ever prescribed CCB\n(verapamil / diltiazem)",
    "features__bb_current": "Active beta-blocker\n(within 90 days)",
    "features__ccb_current": "Active CCB prescription\n(within 90 days)",
    "features__mri_ever": "Ever had cardiac MRI\n(CPT 75561)",
    "features__months_since_diso": "Months since first\nDisopyramide fill",
    "features__n_hcm_meds": "Distinct HCM meds\never tried (0–7)",
}


def main():
    data = load_data()
    dataset = build_dataset(data, DatasetConfig())
    split_month = (
        pd.Period(SplitConfig().train_end_month, freq="M") - pd.Period(LAUNCH_MONTH, freq="M")
    ).n + 1
    train = dataset[dataset[MONTH_COL] <= split_month].copy()

    refined = [f for f in REFINED_FEATURES if f in dataset.columns]
    ct = ColumnTransformer(
        [("time", "passthrough", [MONTH_COL]), ("features", "passthrough", refined)],
        remainder="drop",
    )
    X_train = pd.DataFrame(
        ct.fit_transform(train), columns=ct.get_feature_names_out(), index=train.index
    )
    y_train = train["event"]
    model = DiscreteHazardGLM(link=ModelConfig().link).fit(X_train, y_train)
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
    hrs = hrs[order]
    lo = lo[order]
    hi = hi[order]
    pvals = pvals[order]

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color(GRID)

    y_pos = np.arange(len(labels))

    ax.axvline(1.0, color=INK_MUT, linewidth=1.2, linestyle="--", zorder=1)

    ax.errorbar(
        hrs,
        y_pos,
        xerr=[hrs - lo, hi - hrs],
        fmt="o",
        color=C_POINT,
        ecolor=C_CI,
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
    ax.tick_params(colors=INK_MUT, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()

    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:g}"))

    plt.tight_layout()
    out_path = Path("outputs/task1_adoption/05_hr_forest_plot.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()

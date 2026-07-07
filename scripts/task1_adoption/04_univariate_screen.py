#!/usr/bin/env python
"""Univariate screening of all dx/px/rx codes against Camzyos initiation.

For each code in the data, compute its association with the event
using multiple encodings (binary ever-in-window, count) at multiple
time windows (3m, 6m, 12m). Rank by strength of association.

This is EXPLORATORY — hypothesis-generating, not confirmatory.
Run on training set only; no test set leakage.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

from src.task1_adoption.config import (
    CAMZYOS_CODE,
    SplitConfig,
)
from src.task1_adoption.data_loading import load_data

np.random.seed(42)

WINDOWS_MONTHS = [3, 6, 12]
MIN_PREVALENCE = 10  # minimum patients with the code to test


def main():
    split_cfg = SplitConfig()
    print("Loading data...")
    data = load_data()

    # Build the at-risk population: Diso-experienced oHCM patients
    ohcm = set(data.diagnoses.query("dx_code == @OHCM_CODE")["patient_id"].unique())
    diso = data.prescriptions.query("rx_code == @DISOPYRAMIDE_CODE")
    diso_patients = set(diso["patient_id"].unique()) & ohcm

    # Camzyos initiators with initiation date
    cam = data.prescriptions.query("rx_code == @CAMZYOS_CODE")
    first_cam = cam.groupby("patient_id")["date"].min().rename("camzyos_date")

    # Train period: events before split boundary
    split_date = pd.Timestamp(split_cfg.train_end_month + "-01") + pd.offsets.MonthEnd(0)
    train_cam = first_cam[(first_cam.index.isin(diso_patients)) & (first_cam <= split_date)]
    train_initiators = set(train_cam.index)
    train_pool = diso_patients  # all Diso patients are in the pool

    print(f"Train pool: {len(train_pool)} patients, {len(train_initiators)} initiators")

    # Collect all events with dates
    events = []
    for source, code_col, df in [
        ("dx", "dx_code", data.diagnoses),
        ("px", "px_code", data.procedures),
        ("rx", "rx_code", data.prescriptions),
    ]:
        sub = df[df["patient_id"].isin(train_pool)].copy()
        sub = sub.rename(columns={code_col: "code"})
        sub["source"] = source
        events.append(sub[["patient_id", "date", "code", "source"]])

    all_events = pd.concat(events, ignore_index=True)
    all_events = all_events[all_events["code"] != CAMZYOS_CODE]  # exclude the outcome

    print(f"Total events: {len(all_events):,} across {all_events['code'].nunique()} codes")

    # For each code × window, compute association
    results = []
    codes = all_events["code"].value_counts()
    codes = codes[codes >= MIN_PREVALENCE].index

    print(f"Testing {len(codes)} codes with prevalence >= {MIN_PREVALENCE}...")

    for window in WINDOWS_MONTHS:
        window_days = window * 30

        for code in codes:
            code_events = all_events[all_events["code"] == code]
            source = code_events["source"].iloc[0]

            # For initiators: code present in [camzyos_date - window_days, camzyos_date)?
            # For non-initiators: code present in [split_date - window_days, split_date)?
            code_by_patient = {}
            for pid in train_pool:
                if pid in train_initiators:
                    ref_date = train_cam[pid]
                else:
                    ref_date = split_date
                pat_events = code_events[code_events["patient_id"] == pid]
                window_start = ref_date - pd.Timedelta(days=window_days)
                in_window = pat_events[
                    (pat_events["date"] >= window_start) & (pat_events["date"] < ref_date)
                ]
                code_by_patient[pid] = {
                    "has_code": int(len(in_window) > 0),
                    "count": len(in_window),
                    "initiator": int(pid in train_initiators),
                }

            df_code = pd.DataFrame.from_dict(code_by_patient, orient="index")

            # Skip if too few have the code
            n_with = df_code["has_code"].sum()
            if n_with < MIN_PREVALENCE:
                continue

            # Fisher's exact test on binary encoding
            a = ((df_code["has_code"] == 1) & (df_code["initiator"] == 1)).sum()
            b = ((df_code["has_code"] == 1) & (df_code["initiator"] == 0)).sum()
            c = ((df_code["has_code"] == 0) & (df_code["initiator"] == 1)).sum()
            d = ((df_code["has_code"] == 0) & (df_code["initiator"] == 0)).sum()

            if min(a + b, c + d, a + c, b + d) == 0:
                continue

            odds_ratio, fisher_p = fisher_exact([[a, b], [c, d]])

            # Mann-Whitney U on count encoding
            init_counts = df_code.loc[df_code["initiator"] == 1, "count"]
            non_counts = df_code.loc[df_code["initiator"] == 0, "count"]
            if init_counts.sum() + non_counts.sum() > 0:
                try:
                    _, mw_p = mannwhitneyu(init_counts, non_counts, alternative="two-sided")
                except ValueError:
                    _, mw_p = np.nan, np.nan
            else:
                _, mw_p = np.nan, np.nan

            # Prevalence
            prev_init = a / (a + c) if (a + c) > 0 else 0
            prev_non = b / (b + d) if (b + d) > 0 else 0

            results.append(
                {
                    "code": code,
                    "source": source,
                    "window_months": window,
                    "n_with_code": n_with,
                    "prev_initiators": prev_init,
                    "prev_non_initiators": prev_non,
                    "odds_ratio": odds_ratio,
                    "fisher_p": fisher_p,
                    "mw_p": mw_p,
                }
            )

    results_df = pd.DataFrame(results)
    results_df["fisher_p_adj"] = np.minimum(
        results_df["fisher_p"] * len(results_df), 1.0
    )  # Bonferroni

    # Merge with code descriptions
    code_dict = pd.read_csv("synthetic_data/code_dictionary.csv")
    results_df = results_df.merge(
        code_dict[["code", "description", "category"]],
        on="code",
        how="left",
    )

    # Sort by raw p-value and display top results
    results_df = results_df.sort_values("fisher_p")

    print(f"\n{'=' * 90}")
    print("TOP 30 CODES BY FISHER'S EXACT TEST (per window)")
    print(f"{'=' * 90}")

    for window in WINDOWS_MONTHS:
        sub = results_df[results_df["window_months"] == window].head(10)
        print(f"\n--- Window: {window} months ---")
        display_cols = [
            "code",
            "source",
            "description",
            "n_with_code",
            "prev_initiators",
            "prev_non_initiators",
            "odds_ratio",
            "fisher_p",
            "fisher_p_adj",
        ]
        print(sub[display_cols].to_string(index=False))

    # Codes significant after Bonferroni at any window
    sig = results_df[results_df["fisher_p_adj"] < 0.05].copy()
    print(f"\n{'=' * 90}")
    print(f"CODES SIGNIFICANT AFTER BONFERRONI (p_adj < 0.05): {len(sig)}")
    print(f"{'=' * 90}")
    if len(sig) > 0:
        sig_summary = sig.sort_values("fisher_p").drop_duplicates("code", keep="first")
        print(
            sig_summary[
                [
                    "code",
                    "source",
                    "description",
                    "window_months",
                    "prev_initiators",
                    "prev_non_initiators",
                    "odds_ratio",
                    "fisher_p_adj",
                ]
            ].to_string(index=False)
        )

    # Save full results
    out_path = Path("outputs/task1_adoption/04_univariate_screen.csv")
    out_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()

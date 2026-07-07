"""Person-month panel construction with as-of feature timing.

All feature computations obey the as-of rule: a feature visible at panel
month t is computed from events strictly before month t (< t, or shift(1)
for rolling counts). Violating this rule introduces look-ahead bias.
"""

import pandas as pd

from src.atc import classify_prescriptions
from src.config import (
    AF_CODES_PREFIX,
    BETA_BLOCKERS,
    CARDIAC_MRI_CODE,
    CCBS,
    CYP_INHIBITORS,
    CYP_WASHOUT_DAYS,
    DISOPYRAMIDE_CODE,
    END_MONTH,
    ER_CODES,
    HCM_MEDS,
    HF_CODES_PREFIX,
    INPATIENT_CODES,
    LAUNCH_MONTH,
    MITRAL_CODE,
    ROLLING_FEATURE_CODES,
    ROLLING_WINDOWS,
    STRAIN_CODE,
    PanelConfig,
)
from src.data_loading import RawData


def _to_period(date_or_str):
    return pd.Period(str(date_or_str)[:7], freq="M")


def _period_range(start, end):
    return pd.period_range(start, end, freq="M")


def build_panel(data: RawData, config: PanelConfig = PanelConfig()) -> pd.DataFrame:
    """Build person-month panel for the at-risk population.

    Returns DataFrame with one row per patient-month, binary event
    indicator, and all time-varying features computed as-of each month.
    """
    skeleton = _build_skeleton(data, config)
    panel = _add_features(skeleton, data, config)
    return panel


# ── Panel skeleton ──────────────────────────────────────────────────────────


def _build_skeleton(data: RawData, config: PanelConfig) -> pd.DataFrame:
    launch = _to_period(LAUNCH_MONTH)
    end = _to_period(END_MONTH)

    first_cam = (
        data.prescriptions.query("rx_code == @CAMZYOS_CODE")
        .groupby("patient_id")["date"]
        .min()
        .rename("camzyos_date")
        .reset_index()
    )
    first_cam["camzyos_month"] = first_cam["camzyos_date"].dt.to_period("M")

    first_srt = (
        data.procedures.query("px_code == @SRT_CODE")
        .groupby("patient_id")["date"]
        .min()
        .rename("srt_date")
        .reset_index()
    )
    first_srt["srt_month"] = first_srt["srt_date"].dt.to_period("M")

    if config.risk_set == "disopyramide":
        eligible = _disopyramide_eligible(data, launch)
    else:
        eligible = _ohcm_eligible(data, launch)

    eligible = eligible.merge(first_cam, on="patient_id", how="left")
    eligible = eligible.merge(first_srt[["patient_id", "srt_month"]], on="patient_id", how="left")

    enrolled_set = _enrolled_months_set(data.enrollment)

    if config.censor_at_first_gap:
        gap_months = _first_gap_month(data.enrollment, eligible, launch)
        eligible = eligible.merge(gap_months, on="patient_id", how="left")
    else:
        eligible["gap_month"] = pd.NaT

    eligible["exit_month"] = eligible.apply(lambda r: _exit_month(r, end), axis=1)
    eligible["event_in_exit"] = (
        eligible["camzyos_month"].notna() & (eligible["camzyos_month"] == eligible["exit_month"])
    ).astype(int)

    rows = []
    for _, pat in eligible.iterrows():
        if pat["entry_month"] > pat["exit_month"]:
            continue
        for m in _period_range(pat["entry_month"], pat["exit_month"]):
            if (pat["patient_id"], m) not in enrolled_set:
                break  # treat enrollment gap as censoring
            is_event = int(m == pat["exit_month"] and pat["event_in_exit"] == 1)
            rows.append(
                {
                    "patient_id": pat["patient_id"],
                    "month": m,
                    "study_month": int((m - launch).n)
                    + 1,  # .n: integer month count from Period offset
                    "event": is_event,
                    "entry_month": pat["entry_month"],
                    "first_diso_date": pat.get("first_diso_date", pd.NaT),
                }
            )

    return pd.DataFrame(rows)


def _disopyramide_eligible(data: RawData, launch: pd.Period) -> pd.DataFrame:
    ohcm = set(data.diagnoses.query("dx_code == @OHCM_CODE")["patient_id"].unique())
    diso = data.prescriptions.query("rx_code == @DISOPYRAMIDE_CODE")
    first_diso = diso.groupby("patient_id")["date"].min().rename("first_diso_date").reset_index()
    first_diso = first_diso[first_diso["patient_id"].isin(ohcm)]
    first_diso["diso_month"] = first_diso["first_diso_date"].dt.to_period("M")
    first_diso["entry_month"] = first_diso["diso_month"].clip(lower=launch)
    return first_diso[["patient_id", "entry_month", "first_diso_date"]]


def _ohcm_eligible(data: RawData, launch: pd.Period) -> pd.DataFrame:
    ohcm = data.diagnoses.query("dx_code == @OHCM_CODE")
    first_dx = ohcm.groupby("patient_id")["date"].min().rename("first_dx_date").reset_index()
    first_dx["dx_month"] = first_dx["first_dx_date"].dt.to_period("M")
    first_dx["entry_month"] = first_dx["dx_month"].clip(lower=launch)
    first_dx["first_diso_date"] = pd.NaT
    return first_dx[["patient_id", "entry_month", "first_diso_date"]]


def _enrolled_months_set(enrollment: pd.DataFrame) -> set:
    enr = enrollment.query("enrolled == 1")
    return set(zip(enr["patient_id"], enr["month"].map(_to_period)))


def _first_gap_month(enrollment, eligible, launch):
    not_enrolled = enrollment.query("enrolled == 0").copy()
    not_enrolled["month_p"] = not_enrolled["month"].map(_to_period)
    merged = not_enrolled.merge(eligible[["patient_id", "entry_month"]], on="patient_id")
    merged = merged[merged["month_p"] >= merged["entry_month"]]
    return merged.groupby("patient_id")["month_p"].min().rename("gap_month").reset_index()


def _exit_month(row, end_period):
    candidates = [end_period]
    if pd.notna(row.get("camzyos_month")):
        candidates.append(row["camzyos_month"])
    if pd.notna(row.get("srt_month")):
        candidates.append(row["srt_month"])
    if pd.notna(row.get("gap_month")):
        prev = row["gap_month"] - 1
        if prev >= row["entry_month"]:
            candidates.append(prev)
    return min(candidates)


# ── Feature engineering helpers ─────────────────────────────────────────────


def _add_ever_before_flag(panel, data_df, code_col, codes, col_name, prefix=False):
    """Binary flag: any matching code ever strictly before each panel month."""
    lookup = _first_event_month(data_df, code_col, codes, prefix=prefix)
    panel = panel.merge(
        lookup.rename(columns={"first_event_month": f"_tmp_{col_name}"}),
        on="patient_id",
        how="left",
    )
    panel[col_name] = (
        panel[f"_tmp_{col_name}"].notna() & (panel[f"_tmp_{col_name}"] < panel["month"])
    ).astype(int)
    panel.drop(columns=[f"_tmp_{col_name}"], inplace=True)
    return panel


def _ever_before_from_df(pm, filtered_df, date_col, out_col):
    """Binary flag from a pre-filtered DataFrame: any row with date < panel month.

    Reusable alternative to _add_ever_before_flag when filtering has already
    been applied (e.g. ATC-classified columns rather than raw code columns).
    """
    first = filtered_df.groupby("patient_id")[date_col].min().reset_index()
    first["_first_month"] = first[date_col].dt.to_period("M")
    merged = pm.merge(first[["patient_id", "_first_month"]], on="patient_id", how="left")
    merged[out_col] = (
        merged["_first_month"].notna() & (merged["_first_month"] < merged["month"])
    ).astype(int)
    return merged[["patient_id", "month", out_col]]


def _add_current_med_flag(panel, prescriptions, codes, col_name, lookback_days=90):
    """Binary flag: any fill of given codes within lookback_days before each month."""
    rx = prescriptions[prescriptions["rx_code"].isin(codes)].copy()
    if rx.empty:
        panel[col_name] = 0
        return panel

    rx["fill_month"] = rx["date"].dt.to_period("M")
    rx["coverage_end_month"] = (rx["date"] + pd.Timedelta(days=lookback_days)).dt.to_period("M")

    pm = panel[["patient_id", "month"]].drop_duplicates()
    merged = pm.merge(
        rx[["patient_id", "fill_month", "coverage_end_month"]], on="patient_id", how="inner"
    )
    covered = merged[
        (merged["month"] >= merged["fill_month"])
        & (merged["month"] <= merged["coverage_end_month"])
    ][["patient_id", "month"]].drop_duplicates()
    covered[col_name] = 1

    panel = panel.merge(covered, on=["patient_id", "month"], how="left")
    panel[col_name] = panel[col_name].fillna(0).astype(int)
    return panel


# ── Thematic feature sub-functions ──────────────────────────────────────────


def _add_demographics(panel: pd.DataFrame, data: RawData) -> pd.DataFrame:
    """Demographics (age, sex) and time since Disopyramide first fill."""
    panel = panel.merge(data.patients, on="patient_id", how="left")
    panel["age"] = panel["month"].apply(lambda m: m.year) - panel["birth_year"]
    panel["sex_F"] = (panel["sex"] == "F").astype(int)

    panel["months_since_diso"] = 0
    has_diso = panel["first_diso_date"].notna()
    if has_diso.any():
        diso_periods = panel.loc[has_diso, "first_diso_date"].apply(_to_period)
        panel.loc[has_diso, "months_since_diso"] = (
            panel.loc[has_diso, "month"] - diso_periods
        ).apply(lambda x: int(x.n))
    return panel


def _add_clinical_features(panel: pd.DataFrame, data: RawData, config: PanelConfig) -> pd.DataFrame:
    """Cumulative HCM med count, HF comorbidity, and rolling procedure/symptom counts."""
    med_counts = _cumulative_med_count(data.prescriptions, panel)
    panel = panel.merge(med_counts, on=["patient_id", "month"], how="left")

    panel = _add_ever_before_flag(
        panel, data.diagnoses, "dx_code", HF_CODES_PREFIX, "hf_flag", prefix=True
    )

    data_sources = {"diagnoses": data.diagnoses, "procedures": data.procedures}
    for feat_base, (source, code_col, codes) in ROLLING_FEATURE_CODES.items():
        df = data_sources[source]
        for win in ROLLING_WINDOWS:
            col_name = f"{feat_base}_{win}m"
            feat = _rolling_event_count(df, code_col, codes, col_name, win)
            panel = panel.merge(feat, on=["patient_id", "month"], how="left")
    return panel


def _add_treatment_features(panel: pd.DataFrame, data: RawData, w: int) -> pd.DataFrame:
    """Treatment history: ever/current beta-blocker and CCB use, medication switches."""
    panel = _add_ever_before_flag(panel, data.prescriptions, "rx_code", BETA_BLOCKERS, "bb_ever")
    panel = _add_ever_before_flag(panel, data.prescriptions, "rx_code", CCBS, "ccb_ever")
    panel = _add_current_med_flag(panel, data.prescriptions, BETA_BLOCKERS, "bb_current")
    panel = _add_current_med_flag(panel, data.prescriptions, CCBS, "ccb_current")

    med_change = _med_change_features(data.prescriptions, panel, w)
    panel = panel.merge(med_change, on=["patient_id", "month"], how="left")
    return panel


def _add_procedure_features(panel: pd.DataFrame, data: RawData, w: int) -> pd.DataFrame:
    """Specialist workup markers: cardiac MRI ever, strain imaging, ER/inpatient utilisation."""
    panel = _add_ever_before_flag(panel, data.procedures, "px_code", CARDIAC_MRI_CODE, "mri_ever")

    strain = _rolling_event_count(data.procedures, "px_code", STRAIN_CODE, "strain_count_12m", w)
    panel = panel.merge(strain, on=["patient_id", "month"], how="left")

    er_inp = _rolling_event_count(
        data.procedures, "px_code", ER_CODES + INPATIENT_CODES, "_er_inp_count", w
    )
    panel = panel.merge(er_inp, on=["patient_id", "month"], how="left")
    panel["er_or_inpatient_12m"] = (panel["_er_inp_count"].fillna(0) > 0).astype(int)
    panel.drop(columns=["_er_inp_count"], inplace=True)
    return panel


def _add_comorbidity_features(panel: pd.DataFrame, data: RawData) -> pd.DataFrame:
    """Comorbidity ever-flags: atrial fibrillation, mitral valve regurgitation."""
    panel = _add_ever_before_flag(
        panel, data.diagnoses, "dx_code", AF_CODES_PREFIX, "af_flag", prefix=True
    )
    panel = _add_ever_before_flag(panel, data.diagnoses, "dx_code", MITRAL_CODE, "mitral_flag")
    return panel


def _add_atc_features(panel: pd.DataFrame, data: RawData, w: int) -> pd.DataFrame:
    """ATC-hierarchy features from classified prescriptions.

    Covers: distinct cardiac ATC classes ever (n_cardiac_classes),
    antiarrhythmic/anticoagulant/SGLT2i ever-flags, and total
    cardiac drug-days in the rolling window (cardiac_drug_days_12m).
    """
    rx_classified = classify_prescriptions(data.prescriptions)
    rx = rx_classified.copy()
    rx["fill_month"] = rx["date"].dt.to_period("M")
    pm = panel[["patient_id", "month"]].drop_duplicates()

    # Distinct cardiac ATC 2nd-level classes ever before month t
    cardiac_classes = [c for c in rx.columns if c.startswith("class_")]
    rx_cardiac = rx[rx[cardiac_classes].any(axis=1)].copy()
    if not rx_cardiac.empty:
        first_per_class = rx_cardiac.groupby(["patient_id", "atc_2nd"])["date"].min().reset_index()
        first_per_class["class_month"] = first_per_class["date"].dt.to_period("M")
        merged = pm.merge(first_per_class, on="patient_id", how="left")
        merged = merged[merged["class_month"] < merged["month"]]
        n_classes = (
            merged.groupby(["patient_id", "month"])["atc_2nd"]
            .nunique()
            .rename("n_cardiac_classes")
            .reset_index()
        )
    else:
        n_classes = pd.DataFrame(columns=["patient_id", "month", "n_cardiac_classes"])

    # Ever-before flags for antiarrhythmic (C01BA-BD), anticoagulant (B01A), SGLT2i.
    # All three follow the same pattern via _ever_before_from_df.
    antiarr_codes = ["C01BA", "C01BB", "C01BC", "C01BD"]
    rx["is_antiarr"] = (
        rx["atc_4th"].fillna("").isin(antiarr_codes) if "atc_4th" in rx.columns else False
    )
    antiarr_feat = _ever_before_from_df(pm, rx[rx["is_antiarr"]], "date", "antiarrhythmic_ever")

    rx["is_anticoag"] = rx["atc_2nd"].fillna("") == "B01" if "atc_2nd" in rx.columns else False
    ac_feat = _ever_before_from_df(pm, rx[rx["is_anticoag"]], "date", "anticoagulant_ever")

    if "sglt2_inhibitor" in rx.columns:
        rx["is_sglt2"] = rx["sglt2_inhibitor"].astype(object).fillna(False).astype(bool)
    else:
        rx["is_sglt2"] = False
    sg_feat = _ever_before_from_df(pm, rx[rx["is_sglt2"]], "date", "sglt2i_ever")
    # _ever_before_from_df returns only patients in filtered_df; fill 0 for the rest
    sg_feat = pm.merge(sg_feat, on=["patient_id", "month"], how="left")
    sg_feat["sglt2i_ever"] = sg_feat["sglt2i_ever"].fillna(0).astype(int)

    # Total cardiac drug-days in rolling window
    if "days_supply" in rx_cardiac.columns:
        rx_cardiac["days_supply"] = pd.to_numeric(
            rx_cardiac["days_supply"], errors="coerce"
        ).fillna(30)
        monthly_days = (
            rx_cardiac.groupby(["patient_id", "fill_month"])["days_supply"]
            .sum()
            .rename("count")
            .reset_index()
            .rename(columns={"fill_month": "month"})
        )
        drug_days = _compute_rolling_from_monthly(monthly_days, "count", "cardiac_drug_days_12m", w)
    else:
        drug_days = pd.DataFrame(columns=["patient_id", "month", "cardiac_drug_days_12m"])

    result = n_classes
    for feat_df in [antiarr_feat, ac_feat, sg_feat, drug_days]:
        result = result.merge(feat_df, on=["patient_id", "month"], how="outer")
    panel = panel.merge(result, on=["patient_id", "month"], how="left")
    return panel


def _add_derived_features(panel: pd.DataFrame, data: RawData, w: int) -> pd.DataFrame:
    """Derived composite features: MPR, CYP contraindication, dual-drug flags, echo acceleration."""
    # Disopyramide adherence (medication possession ratio)
    diso_adherence = _medication_possession_ratio(
        data.prescriptions, DISOPYRAMIDE_CODE, panel, w, "diso_mpr_12m"
    )
    panel = panel.merge(diso_adherence, on=["patient_id", "month"], how="left")

    # CYP inhibitor active — contraindication barrier with washout
    cyp = _drug_active_with_washout(
        data.prescriptions, CYP_INHIBITORS, CYP_WASHOUT_DAYS, panel, "cyp_inhibitor_active"
    )
    panel = panel.merge(cyp, on=["patient_id", "month"], how="left")

    # Dual BB+CCB current — EXPLORER-HCM exclusion criterion
    panel["dual_bb_ccb_current"] = (
        (panel.get("bb_current", 0) == 1) & (panel.get("ccb_current", 0) == 1)
    ).astype(int)

    # Diso+CCB combo current — label warns against this combination with Camzyos
    diso_current = _add_current_med_flag(
        panel[["patient_id", "month"]].drop_duplicates(),
        data.prescriptions,
        [DISOPYRAMIDE_CODE],
        "_diso_current",
    )
    panel = panel.merge(diso_current, on=["patient_id", "month"], how="left")
    panel["diso_ccb_combo_current"] = (
        (panel.get("_diso_current", 0) == 1) & (panel.get("ccb_current", 0) == 1)
    ).astype(int)
    panel.drop(columns=["_diso_current"], inplace=True, errors="ignore")

    # Echo acceleration: recent 6m TTE count > prior 6m TTE count
    tte_6 = panel.get("tte_count_6m", 0)
    tte_12 = panel.get("tte_count_12m", 0)
    # tte_12 is cumulative 12-month count; subtract 6m window to isolate the prior-6m count
    panel["echo_acceleration"] = (tte_6 > (tte_12 - tte_6)).astype(int)
    return panel


def _finalize(panel: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN from left-joins with 0 and drop construction columns."""
    numeric_cols = panel.select_dtypes(include="number").columns
    panel[numeric_cols] = panel[numeric_cols].fillna(0)
    panel.drop(
        columns=["birth_year", "sex", "entry_month", "first_diso_date"],
        inplace=True,
        errors="ignore",
    )
    return panel


def _add_features(skeleton: pd.DataFrame, data: RawData, config: PanelConfig) -> pd.DataFrame:
    """Orchestrate all feature sub-functions."""
    w = config.rolling_window
    panel = _add_demographics(skeleton, data)
    panel = _add_clinical_features(panel, data, config)
    panel = _add_treatment_features(panel, data, w)
    panel = _add_procedure_features(panel, data, w)
    panel = _add_comorbidity_features(panel, data)
    panel = _add_atc_features(panel, data, w)
    panel = _add_derived_features(panel, data, w)
    panel = _finalize(panel)
    return panel


# ── Core computation helpers ─────────────────────────────────────────────────


def _cumulative_med_count(prescriptions, panel):
    hcm_rx = prescriptions[prescriptions["rx_code"].isin(HCM_MEDS)]
    first_per_med = hcm_rx.groupby(["patient_id", "rx_code"])["date"].min().reset_index()
    first_per_med["fill_month"] = first_per_med["date"].dt.to_period("M")

    pm = panel[["patient_id", "month"]].drop_duplicates()
    merged = pm.merge(first_per_med, on="patient_id", how="left")
    merged = merged[merged["fill_month"] < merged["month"]]
    return (
        merged.groupby(["patient_id", "month"])["rx_code"]
        .nunique()
        .rename("n_hcm_meds")
        .reset_index()
    )


def _rolling_event_count(df, code_col, codes, out_col, window):
    if isinstance(codes, str):
        mask = df[code_col] == codes
    else:
        mask = df[code_col].isin(codes)
    events = df[mask].copy()
    events["month"] = events["date"].dt.to_period("M")
    monthly = events.groupby(["patient_id", "month"]).size().rename("count").reset_index()

    if monthly.empty:
        return pd.DataFrame(columns=["patient_id", "month", out_col])

    all_months = _period_range(monthly["month"].min(), monthly["month"].max())
    idx = pd.MultiIndex.from_product(
        [monthly["patient_id"].unique(), all_months], names=["patient_id", "month"]
    )
    dense = monthly.set_index(["patient_id", "month"]).reindex(idx, fill_value=0).reset_index()
    dense[out_col] = dense.groupby("patient_id")["count"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).sum()
    )
    return dense[["patient_id", "month", out_col]]


def _first_event_month(df, code_col, code_or_prefix, prefix=False):
    if prefix:
        mask = df[code_col].str.startswith(code_or_prefix)
    elif isinstance(code_or_prefix, list):
        mask = df[code_col].isin(code_or_prefix)
    else:
        mask = df[code_col] == code_or_prefix
    first = df[mask].groupby("patient_id")["date"].min().reset_index()
    first["first_event_month"] = first["date"].dt.to_period("M")
    return first[["patient_id", "first_event_month"]]


def _med_change_features(prescriptions, panel, window):
    """Months since last HCM med change and number of switches in rolling window."""
    hcm_rx = prescriptions[prescriptions["rx_code"].isin(HCM_MEDS)].copy()
    hcm_rx = hcm_rx.sort_values(["patient_id", "date"])

    first_per_med = hcm_rx.groupby(["patient_id", "rx_code"])["date"].min().reset_index()
    first_per_med = first_per_med.rename(columns={"date": "change_date"})
    first_per_med["change_month"] = first_per_med["change_date"].dt.to_period("M")

    pm = panel[["patient_id", "month"]].drop_duplicates()

    merged = pm.merge(first_per_med, on="patient_id", how="left")
    merged = merged[merged["change_month"] < merged["month"]]
    last_change = merged.groupby(["patient_id", "month"])["change_month"].max().reset_index()
    last_change["months_since_last_med_change"] = (
        last_change["month"] - last_change["change_month"]
    ).apply(lambda x: x.n)
    last_change = last_change[["patient_id", "month", "months_since_last_med_change"]]

    merged2 = pm.merge(first_per_med, on="patient_id", how="left")
    merged2["months_before"] = (merged2["month"] - merged2["change_month"]).apply(
        lambda x: x.n if pd.notna(x) else None
    )
    merged2 = merged2[
        (merged2["months_before"].notna())
        & (merged2["months_before"] > 0)
        & (merged2["months_before"] <= window)
    ]
    switches = (
        merged2.groupby(["patient_id", "month"]).size().rename("med_switches_12m").reset_index()
    )

    result = last_change.merge(switches, on=["patient_id", "month"], how="outer")
    return result


def _compute_rolling_from_monthly(monthly_counts, count_col, out_col, window):
    """Rolling sum from pre-aggregated monthly counts."""
    if monthly_counts.empty:
        return pd.DataFrame(columns=["patient_id", "month", out_col])

    all_months = _period_range(monthly_counts["month"].min(), monthly_counts["month"].max())
    idx = pd.MultiIndex.from_product(
        [monthly_counts["patient_id"].unique(), all_months], names=["patient_id", "month"]
    )
    dense = (
        monthly_counts.set_index(["patient_id", "month"]).reindex(idx, fill_value=0).reset_index()
    )
    dense[out_col] = dense.groupby("patient_id")[count_col].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).sum()
    )
    return dense[["patient_id", "month", out_col]]


def _medication_possession_ratio(prescriptions, drug_code, panel, window, out_col):
    """MPR = covered days / observation days in rolling window."""
    rx = prescriptions[prescriptions["rx_code"] == drug_code].copy()
    if rx.empty or "days_supply" not in rx.columns:
        result = panel[["patient_id", "month"]].drop_duplicates()
        result[out_col] = 0.0
        return result

    rx["days_supply"] = pd.to_numeric(rx["days_supply"], errors="coerce").fillna(30)
    rx["fill_month"] = rx["date"].dt.to_period("M")

    monthly_supply = (
        rx.groupby(["patient_id", "fill_month"])["days_supply"]
        .sum()
        .rename("covered_days")
        .reset_index()
        .rename(columns={"fill_month": "month"})
    )

    rolling = _compute_rolling_from_monthly(
        monthly_supply, "covered_days", "_covered_days_rolling", window
    )
    observation_days = window * 30.44  # 365.25 / 12 = average calendar days per month
    rolling[out_col] = (rolling["_covered_days_rolling"] / observation_days).clip(0, 1)
    return rolling[["patient_id", "month", out_col]]


def _drug_active_with_washout(prescriptions, drug_codes, washout_days, panel, out_col):
    """Binary flag: drug coverage window (fill + days_supply + washout) overlaps panel month."""
    rx = prescriptions[prescriptions["rx_code"].isin(drug_codes)].copy()
    if rx.empty:
        result = panel[["patient_id", "month"]].drop_duplicates()
        result[out_col] = 0
        return result

    rx["days_supply"] = pd.to_numeric(rx["days_supply"], errors="coerce").fillna(30)
    rx["fill_month"] = rx["date"].dt.to_period("M")
    rx["coverage_end"] = rx["date"] + pd.to_timedelta(rx["days_supply"] + washout_days, unit="D")
    rx["coverage_end_month"] = rx["coverage_end"].dt.to_period("M")

    pm = panel[["patient_id", "month"]].drop_duplicates()
    merged = pm.merge(
        rx[["patient_id", "fill_month", "coverage_end_month"]],
        on="patient_id",
        how="inner",
    )
    covered = merged[
        (merged["month"] >= merged["fill_month"])
        & (merged["month"] <= merged["coverage_end_month"])
    ][["patient_id", "month"]].drop_duplicates()
    covered[out_col] = 1

    result = pm.merge(covered, on=["patient_id", "month"], how="left")
    result[out_col] = result[out_col].fillna(0).astype(int)
    return result

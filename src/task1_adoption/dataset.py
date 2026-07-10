"""Person-month dataset construction with as-of feature timing.

All feature computations obey the as-of rule: a feature visible at row
month t is computed from events strictly before month t (< t, or shift(1)
for rolling counts). Violating this rule introduces look-ahead bias.
"""

import pandas as pd

from src.task1_adoption.config import (
    AF_CODES_PREFIX,
    BETA_BLOCKERS,
    CAMZYOS_CODE,  # noqa: F401 — used via pandas .query("@CAMZYOS_CODE")
    CARDIAC_MRI_CODE,
    CCBS,
    CYP_INHIBITORS,
    CYP_WASHOUT_DAYS,
    DISOPYRAMIDE_CODE,
    END_MONTH,
    ER_CODES,
    HCM_ELIGIBILITY_CODES,
    HCM_MEDS,
    HF_CODES_PREFIX,
    INPATIENT_CODES,
    LAUNCH_MONTH,
    MITRAL_CODE,
    OHCM_CODE,  # noqa: F401 — used via pandas .query("@OHCM_CODE")
    ROLLING_FEATURE_CODES,
    ROLLING_WINDOWS,
    SRT_CODE,  # noqa: F401 — used via pandas .query("@SRT_CODE")
    STRAIN_CODE,
    DatasetConfig,
)
from src.task1_adoption.data_loading import RawData


def _to_period(date_or_str):
    return pd.Period(str(date_or_str)[:7], freq="M")


def _period_range(start, end):
    return pd.period_range(start, end, freq="M")


def build_dataset(data: RawData, config: DatasetConfig = DatasetConfig()) -> pd.DataFrame:
    """Build person-month dataset for the at-risk population.

    Returns DataFrame with one row per patient-month, binary event
    indicator, and all time-varying features computed as-of each month.
    """
    skeleton = _build_skeleton(data, config)
    dataset = _add_features(skeleton, data, config)
    return dataset


# ── Skeleton construction ───────────────────────────────────────────────────


def _build_skeleton(data: RawData, config: DatasetConfig) -> pd.DataFrame:
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
    # HCM_ELIGIBILITY_CODES expands beyond I421 to include I422/I429 (see config).
    ohcm = set(
        data.diagnoses[data.diagnoses["dx_code"].isin(HCM_ELIGIBILITY_CODES)]["patient_id"].unique()
    )
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


def _add_ever_before_flag(dataset, data_df, code_col, codes, col_name, prefix=False):
    """Binary flag: any matching code ever strictly before each row's month."""
    lookup = _first_event_month(data_df, code_col, codes, prefix=prefix)
    dataset = dataset.merge(
        lookup.rename(columns={"first_event_month": f"_tmp_{col_name}"}),
        on="patient_id",
        how="left",
    )
    dataset[col_name] = (
        dataset[f"_tmp_{col_name}"].notna() & (dataset[f"_tmp_{col_name}"] < dataset["month"])
    ).astype(int)
    dataset.drop(columns=[f"_tmp_{col_name}"], inplace=True)
    return dataset


def _add_current_med_flag(dataset, prescriptions, codes, col_name, lookback_days=30):
    """Binary flag: any fill of given codes within lookback_days before each month."""
    rx = prescriptions[prescriptions["rx_code"].isin(codes)].copy()
    if rx.empty:
        dataset[col_name] = 0
        return dataset

    rx["fill_month"] = rx["date"].dt.to_period("M")
    rx["coverage_end_month"] = (rx["date"] + pd.Timedelta(days=lookback_days)).dt.to_period("M")

    pm = dataset[["patient_id", "month"]].drop_duplicates()
    merged = pm.merge(
        rx[["patient_id", "fill_month", "coverage_end_month"]], on="patient_id", how="inner"
    )
    covered = merged[
        (merged["month"] > merged["fill_month"])  # strict: fill in same month is not yet "current"
        & (merged["month"] <= merged["coverage_end_month"])
    ][["patient_id", "month"]].drop_duplicates()
    covered[col_name] = 1

    dataset = dataset.merge(covered, on=["patient_id", "month"], how="left")
    dataset[col_name] = dataset[col_name].fillna(0).astype(int)
    return dataset


# ── Thematic feature sub-functions ──────────────────────────────────────────


def _add_demographics(dataset: pd.DataFrame, data: RawData) -> pd.DataFrame:
    """Demographics (age, sex) and time since Disopyramide first fill."""
    dataset = dataset.merge(data.patients, on="patient_id", how="left")
    dataset["age"] = dataset["month"].apply(lambda m: m.year) - dataset["birth_year"]
    dataset["sex_F"] = (dataset["sex"] == "F").astype(int)

    dataset["months_since_diso"] = 0
    has_diso = dataset["first_diso_date"].notna()
    if has_diso.any():
        diso_periods = dataset.loc[has_diso, "first_diso_date"].apply(_to_period)
        dataset.loc[has_diso, "months_since_diso"] = (
            dataset.loc[has_diso, "month"] - diso_periods
        ).apply(lambda x: int(x.n))
    return dataset


def _add_clinical_features(
    dataset: pd.DataFrame, data: RawData, config: DatasetConfig
) -> pd.DataFrame:
    """Cumulative HCM med count, HF comorbidity, and rolling procedure/symptom counts."""
    med_counts = _cumulative_med_count(data.prescriptions, dataset)
    dataset = dataset.merge(med_counts, on=["patient_id", "month"], how="left")

    dataset = _add_ever_before_flag(
        dataset, data.diagnoses, "dx_code", HF_CODES_PREFIX, "hf_flag", prefix=True
    )

    data_sources = {"diagnoses": data.diagnoses, "procedures": data.procedures}
    for feat_base, (source, code_col, codes) in ROLLING_FEATURE_CODES.items():
        df = data_sources[source]
        for win in ROLLING_WINDOWS:
            col_name = f"{feat_base}_{win}m"
            feat = _rolling_event_count(df, code_col, codes, col_name, win)
            dataset = dataset.merge(feat, on=["patient_id", "month"], how="left")
    return dataset


def _add_treatment_features(dataset: pd.DataFrame, data: RawData, w: int) -> pd.DataFrame:
    """Treatment history: ever/current beta-blocker and CCB use, medication switches."""
    dataset = _add_ever_before_flag(
        dataset, data.prescriptions, "rx_code", BETA_BLOCKERS, "bb_ever"
    )
    dataset = _add_ever_before_flag(dataset, data.prescriptions, "rx_code", CCBS, "ccb_ever")
    dataset = _add_current_med_flag(dataset, data.prescriptions, BETA_BLOCKERS, "bb_current")
    dataset = _add_current_med_flag(dataset, data.prescriptions, CCBS, "ccb_current")

    med_change = _med_change_features(data.prescriptions, dataset, w)
    dataset = dataset.merge(med_change, on=["patient_id", "month"], how="left")
    return dataset


def _add_procedure_features(dataset: pd.DataFrame, data: RawData, w: int) -> pd.DataFrame:
    """Specialist workup markers: cardiac MRI ever, strain imaging, ER/inpatient utilisation."""
    dataset = _add_ever_before_flag(
        dataset, data.procedures, "px_code", CARDIAC_MRI_CODE, "mri_ever"
    )

    strain = _rolling_event_count(data.procedures, "px_code", STRAIN_CODE, "strain_count_12m", w)
    dataset = dataset.merge(strain, on=["patient_id", "month"], how="left")

    er_inp = _rolling_event_count(
        data.procedures, "px_code", ER_CODES + INPATIENT_CODES, "_er_inp_count", w
    )
    dataset = dataset.merge(er_inp, on=["patient_id", "month"], how="left")
    dataset["er_or_inpatient_12m"] = (dataset["_er_inp_count"].fillna(0) > 0).astype(int)
    dataset.drop(columns=["_er_inp_count"], inplace=True)
    return dataset


def _add_comorbidity_features(dataset: pd.DataFrame, data: RawData) -> pd.DataFrame:
    """Comorbidity ever-flags: atrial fibrillation, mitral valve regurgitation."""
    dataset = _add_ever_before_flag(
        dataset, data.diagnoses, "dx_code", AF_CODES_PREFIX, "af_flag", prefix=True
    )
    dataset = _add_ever_before_flag(dataset, data.diagnoses, "dx_code", MITRAL_CODE, "mitral_flag")
    return dataset


def _add_derived_features(dataset: pd.DataFrame, data: RawData, w: int) -> pd.DataFrame:
    """Derived composite features: MPR, CYP contraindication, dual-drug flags."""
    # Disopyramide adherence (medication possession ratio)
    diso_adherence = _medication_possession_ratio(
        data.prescriptions, DISOPYRAMIDE_CODE, dataset, w, "diso_mpr_12m"
    )
    dataset = dataset.merge(diso_adherence, on=["patient_id", "month"], how="left")

    # CYP inhibitor active — contraindication barrier with washout
    cyp = _drug_active_with_washout(
        data.prescriptions, CYP_INHIBITORS, CYP_WASHOUT_DAYS, dataset, "cyp_inhibitor_active"
    )
    dataset = dataset.merge(cyp, on=["patient_id", "month"], how="left")

    # Dual BB+CCB current — EXPLORER-HCM exclusion criterion
    dataset["dual_bb_ccb_current"] = (
        (dataset.get("bb_current", 0) == 1) & (dataset.get("ccb_current", 0) == 1)
    ).astype(int)

    # Diso+CCB combo current — label warns against this combination with Camzyos
    diso_current = _add_current_med_flag(
        dataset[["patient_id", "month"]].drop_duplicates(),
        data.prescriptions,
        [DISOPYRAMIDE_CODE],
        "_diso_current",
    )
    dataset = dataset.merge(diso_current, on=["patient_id", "month"], how="left")
    dataset["diso_ccb_combo_current"] = (
        (dataset.get("_diso_current", 0) == 1) & (dataset.get("ccb_current", 0) == 1)
    ).astype(int)
    dataset.drop(columns=["_diso_current"], inplace=True, errors="ignore")
    return dataset


def _finalize(dataset: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN from left-joins with 0 and drop construction columns."""
    numeric_cols = dataset.select_dtypes(include="number").columns
    dataset[numeric_cols] = dataset[numeric_cols].fillna(0)
    dataset.drop(
        columns=["birth_year", "sex", "entry_month", "first_diso_date"],
        inplace=True,
        errors="ignore",
    )
    return dataset


def _add_features(skeleton: pd.DataFrame, data: RawData, config: DatasetConfig) -> pd.DataFrame:
    """Orchestrate all feature sub-functions."""
    w = config.rolling_window
    dataset = _add_demographics(skeleton, data)
    dataset = _add_clinical_features(dataset, data, config)
    dataset = _add_treatment_features(dataset, data, w)
    dataset = _add_procedure_features(dataset, data, w)
    dataset = _add_comorbidity_features(dataset, data)
    dataset = _add_derived_features(dataset, data, w)
    dataset = _finalize(dataset)
    return dataset


# ── Core computation helpers ─────────────────────────────────────────────────


def _cumulative_med_count(prescriptions, dataset):
    hcm_rx = prescriptions[prescriptions["rx_code"].isin(HCM_MEDS)]
    first_per_med = hcm_rx.groupby(["patient_id", "rx_code"])["date"].min().reset_index()
    first_per_med["fill_month"] = first_per_med["date"].dt.to_period("M")

    pm = dataset[["patient_id", "month"]].drop_duplicates()
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


def _med_change_features(prescriptions, dataset, window):
    """Months since last HCM med change and number of switches in rolling window."""
    hcm_rx = prescriptions[prescriptions["rx_code"].isin(HCM_MEDS)].copy()
    hcm_rx = hcm_rx.sort_values(["patient_id", "date"])

    first_per_med = hcm_rx.groupby(["patient_id", "rx_code"])["date"].min().reset_index()
    first_per_med = first_per_med.rename(columns={"date": "change_date"})
    first_per_med["change_month"] = first_per_med["change_date"].dt.to_period("M")

    pm = dataset[["patient_id", "month"]].drop_duplicates()

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


def _medication_possession_ratio(prescriptions, drug_code, dataset, window, out_col):
    """MPR = covered days / observation days in rolling window."""
    rx = prescriptions[prescriptions["rx_code"] == drug_code].copy()
    if rx.empty or "days_supply" not in rx.columns:
        result = dataset[["patient_id", "month"]].drop_duplicates()
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


def _drug_active_with_washout(prescriptions, drug_codes, washout_days, dataset, out_col):
    """Binary flag: coverage window (fill + days_supply + washout) overlaps each row's month."""
    rx = prescriptions[prescriptions["rx_code"].isin(drug_codes)].copy()
    if rx.empty:
        result = dataset[["patient_id", "month"]].drop_duplicates()
        result[out_col] = 0
        return result

    rx["days_supply"] = pd.to_numeric(rx["days_supply"], errors="coerce").fillna(30)
    rx["fill_month"] = rx["date"].dt.to_period("M")
    rx["coverage_end"] = rx["date"] + pd.to_timedelta(rx["days_supply"] + washout_days, unit="D")
    rx["coverage_end_month"] = rx["coverage_end"].dt.to_period("M")

    pm = dataset[["patient_id", "month"]].drop_duplicates()
    merged = pm.merge(
        rx[["patient_id", "fill_month", "coverage_end_month"]],
        on="patient_id",
        how="inner",
    )
    covered = merged[
        (merged["month"] > merged["fill_month"])  # strict: fill in same month is not yet "current"
        & (merged["month"] <= merged["coverage_end_month"])
    ][["patient_id", "month"]].drop_duplicates()
    covered[out_col] = 1

    result = pm.merge(covered, on=["patient_id", "month"], how="left")
    result[out_col] = result[out_col].fillna(0).astype(int)
    return result

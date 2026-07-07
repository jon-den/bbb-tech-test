"""ATC drug classification from OMOP vocabulary.

Maps generic drug names in prescriptions.csv to ATC therapeutic classes
using the OMOP CONCEPT and CONCEPT_ANCESTOR tables.
"""

from pathlib import Path

import pandas as pd

OMOP_DIR = Path(__file__).parent.parent / "data" / "omop"

# ATC 2nd-level therapeutic classes relevant to cardiac patients
CARDIAC_ATC_CLASSES = {
    "C01": "cardiac_therapy",
    "C02": "antihypertensives",
    "C03": "diuretics",
    "C07": "beta_blockers",
    "C08": "ccbs",
    "C09": "raas_inhibitors",
    "C10": "lipid_modifying",
    "B01": "antithrombotics",
}

# ATC 4th-level subclasses of specific interest
HCM_ATC_SUBCLASSES = {
    "C01BA": "antiarrhythmic_Ia",  # disopyramide
    "C01BD": "antiarrhythmic_III",  # amiodarone, dronedarone
    "C01BC": "antiarrhythmic_Ic",  # flecainide
    "C01EB": "other_cardiac",  # mavacamten, ivabradine, ranolazine
    "C07AA": "bb_nonselective",  # propranolol, nadolol
    "C07AB": "bb_selective",  # metoprolol, atenolol
    "C07AG": "bb_alpha_beta",  # carvedilol
    "C08DA": "ccb_phenylalkylamine",  # verapamil (direct cardiac)
    "C08DB": "ccb_benzothiazepine",  # diltiazem (direct cardiac)
    "C08CA": "ccb_dihydropyridine",  # amlodipine, nifedipine (vascular)
    "A10BK": "sglt2_inhibitor",  # dapagliflozin, empagliflozin
}


def build_atc_mapping(omop_dir=OMOP_DIR):
    """Build a mapping from ATC 5th-level concept names to ATC codes and classes.

    Returns DataFrame with columns: atc_code, atc_name, atc_2nd, atc_4th,
    plus boolean columns for each HCM_ATC_SUBCLASS.
    """
    concepts = pd.read_csv(omop_dir / "CONCEPT.csv", sep="\t", dtype=str)
    atc5 = concepts[concepts["concept_class_id"] == "ATC 5th"].copy()
    atc5 = atc5[atc5["invalid_reason"].isna() | (atc5["invalid_reason"] == "")]
    atc5 = atc5.rename(columns={"concept_code": "atc_code", "concept_name": "atc_name"})
    atc5 = atc5[["atc_code", "atc_name"]].copy()

    atc5["atc_2nd"] = atc5["atc_code"].str[:3]
    atc5["atc_4th"] = atc5["atc_code"].str[:5]

    for prefix, label in HCM_ATC_SUBCLASSES.items():
        atc5[label] = atc5["atc_4th"].str.startswith(prefix)

    for prefix, label in CARDIAC_ATC_CLASSES.items():
        atc5[f"class_{label}"] = atc5["atc_2nd"] == prefix

    return atc5


# Manual ATC mapping for drugs missing from the OMOP extract
_MANUAL_ATC = {
    "Metoprolol Succinate": "C07AB02",
    "Metoprolol Tartrate": "C07AB02",
    "Propranolol Hydrochloride": "C07AA05",
    "Verapamil Hydrochloride": "C08DA01",
    "Disopyramide Phosphate": "C01BA03",
    "Amiodarone Hydrochloride": "C01BD01",
    "Flecainide Acetate": "C01BC04",
    "Digoxin": "C01AA05",
    "Furosemide": "C03CA01",
    "Bumetanide": "C03CA02",
    "Torsemide": "C03CA04",
    "Nifedipine": "C08CA05",
    "Nitroglycerin": "C01DA02",
    "Isosorbide Mononitrate": "C01DA14",
    "Sacubitril/Valsartan": "C09DX04",
    "Sotalol Hydrochloride": "C07AA07",
}


def map_prescriptions_to_atc(prescriptions, atc_mapping):
    """Map rx_code generic drug names to ATC classes.

    Combines stem-matching against the OMOP vocabulary with a manual
    fallback for drugs missing from the extract.
    """
    rx_names = prescriptions["rx_code"].unique()
    atc = atc_mapping.copy()

    atc["stem"] = atc["atc_name"].str.split(";").str[0].str.strip().str.lower()

    rx_stems = pd.DataFrame({"rx_code": rx_names})
    rx_stems["stem"] = rx_stems["rx_code"].str.split().str[0].str.lower()

    matched = rx_stems.merge(atc, on="stem", how="inner")
    # Longest ATC code = most specific (5th-level) → prefer the most granular match per drug name
    matched = matched.sort_values(
        "atc_code", key=lambda s: s.str.len(), ascending=False
    ).drop_duplicates(subset=["rx_code"], keep="first")

    # Add manual mappings for drugs not in OMOP
    matched_names = set(matched["rx_code"])
    manual_rows = []
    for rx_name, atc_code in _MANUAL_ATC.items():
        if rx_name not in matched_names and rx_name in rx_names:
            row = {"rx_code": rx_name, "atc_code": atc_code, "stem": rx_name.split()[0].lower()}
            row["atc_2nd"] = atc_code[:3]
            row["atc_4th"] = atc_code[:5]
            for prefix, label in HCM_ATC_SUBCLASSES.items():
                row[label] = atc_code[: len(prefix)] == prefix
            for prefix, label in CARDIAC_ATC_CLASSES.items():
                row[f"class_{label}"] = atc_code[:3] == prefix
            manual_rows.append(row)

    if manual_rows:
        manual_df = pd.DataFrame(manual_rows)
        # Align columns
        for col in matched.columns:
            if col not in manual_df.columns:
                manual_df[col] = False if matched[col].dtype == bool else None
        matched = pd.concat([matched, manual_df[matched.columns]], ignore_index=True)

    return matched


def classify_prescriptions(prescriptions, omop_dir=OMOP_DIR):
    """Add ATC classification columns to prescription data.

    Returns prescriptions DataFrame with added columns for each
    ATC subclass (boolean) and atc_code, atc_4th, atc_2nd.
    """
    atc_mapping = build_atc_mapping(omop_dir)
    rx_atc = map_prescriptions_to_atc(prescriptions, atc_mapping)

    # Select useful columns for merge
    merge_cols = ["rx_code", "atc_code", "atc_2nd", "atc_4th"]
    merge_cols += [
        c
        for c in rx_atc.columns
        if c
        in list(HCM_ATC_SUBCLASSES.values()) + [f"class_{v}" for v in CARDIAC_ATC_CLASSES.values()]
    ]

    rx_lookup = rx_atc[merge_cols].drop_duplicates(subset=["rx_code"])
    return prescriptions.merge(rx_lookup, on="rx_code", how="left")

"""Configuration constants for the Camzyos adoption analysis.

Sections
--------
1. Paths
2. Clinical codes  (ICD-10 diagnosis, CPT procedure, drug brand/generic names)
3. Feature sets    (CLINICAL / REFINED / EXPANDED)
4. Rolling-window config
5. Dataclass configs  (PanelConfig, SplitConfig, ModelConfig)
"""

from dataclasses import dataclass, field
from pathlib import Path

# ── 1. Paths ────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "synthetic_data"


# ── 2. Clinical codes ────────────────────────────────────────────────────────
#
# ICD-10 diagnosis codes use the format I[chapter][code] (e.g. I421).
# CPT/HCPCS procedure codes are 5-digit numeric strings.
# Drug codes are brand/generic name strings as they appear in the rx_code field.

# ICD-10: obstructive hypertrophic cardiomyopathy
OHCM_CODE = "I421"

# ICD-10 codes included as events in the Disopyramide-conditioned risk set.
# I422 (Other HCM) and I429 (Unspecified CM) are included alongside I421
# because 15 Camzyos initiators carry these codes (likely miscoding of oHCM —
# Disopyramide is specifically indicated for oHCM, so Diso + any HCM code is a
# strong signal of obstructive pathophysiology). Only I421 is used for the
# broader control pool (_ohcm_eligible) to preserve a consistent denominator.
HCM_ELIGIBILITY_CODES = ["I421", "I422", "I429"]

# Drug name as it appears in prescriptions.rx_code
CAMZYOS_CODE = "Mavacamten"
DISOPYRAMIDE_CODE = "Disopyramide Phosphate"

# CPT: alcohol septal ablation (septal reduction therapy)
SRT_CODE = "93583"

# HCM guideline-recommended medications (drug name strings)
HCM_MEDS = [
    "Metoprolol Succinate",
    "Metoprolol Tartrate",
    "Propranolol Hydrochloride",
    "Carvedilol",
    "Verapamil Hydrochloride",
    "Diltiazem Hydrochloride",
    "Disopyramide Phosphate",
]

# Subsets of HCM_MEDS by drug class
BETA_BLOCKERS = [
    "Metoprolol Succinate",
    "Metoprolol Tartrate",
    "Propranolol Hydrochloride",
    "Carvedilol",
]
CCBS = ["Verapamil Hydrochloride", "Diltiazem Hydrochloride"]

# CPT: transthoracic echocardiogram (TTE)
TTE_CODE = "93306"
# CPT: cardiac MRI without and with contrast
CARDIAC_MRI_CODE = "75561"
# CPT: strain/speckle-tracking echocardiography
STRAIN_CODE = "93356"
# CPT: stress testing codes (exercise + pharmacological)
STRESS_CODES = ["93015", "93017", "93018"]
# CPT: high-complexity E&M office visits (99214=moderate, 99215=high complexity)
HIGH_EM_CODES = ["99214", "99215"]
# CPT: emergency department visits
ER_CODES = ["99281", "99282", "99283", "99284", "99285"]
# CPT: initial hospital inpatient care
INPATIENT_CODES = ["99221", "99222", "99223"]
# CPT: BNP / NT-proBNP test (heart failure biomarker)
BNP_CODE = "83880"

# CYP2C19/CYP3A4 inhibitors — Camzyos label (Section 5.2) lists contraindications.
# These drug names appear in our prescription data as moderate-to-strong inhibitors.
CYP_INHIBITORS = [
    "Omeprazole",  # moderate CYP2C19 inhibitor
    "Esomeprazole Magnesium",  # moderate CYP2C19 inhibitor
    "Fluoxetine Hydrochloride",  # moderate CYP2C19 inhibitor
    "Fluconazole",  # strong CYP2C19, moderate CYP3A4 inhibitor
    "Ketoconazole",  # strong CYP3A4 inhibitor
]
# Days of washout after last CYP inhibitor fill + days_supply before Camzyos eligibility
CYP_WASHOUT_DAYS = 30

# ICD-10 prefixes for comorbidity flags (matched by str.startswith)
HF_CODES_PREFIX = "I50"  # heart failure
AF_CODES_PREFIX = "I48"  # atrial fibrillation / flutter

# ICD-10: mitral valve regurgitation NOS
MITRAL_CODE = "I340"

# ICD-10: symptom burden proxies (dyspnoea, syncope, chest pain)
SYMPTOM_CODES = ["R0600", "R0602", "R0609", "R55", "R079", "R0789"]

# Study period: Camzyos FDA approval through end of available claims data
LAUNCH_MONTH = "2022-04"
END_MONTH = "2023-12"


# ── 3. Feature sets ──────────────────────────────────────────────────────────
#
# CLINICAL  — 7 features derived from clinical domain knowledge before any
#             data-driven selection. These are the primary interpretable features
#             reported to the IC and used as the reference model.
#
# REFINED   — 6 features after stability selection and coefficient review.
#             Drops age, sex, tte_count_12m, hf_flag, symptom_burden_12m
#             (no signal); adds ccb_ever, bb_current, ccb_current, mri_ever.
#             This is the production model for scoring.
#
# EXPANDED  — 41 features including all multi-window rolling variants.
#             Used only for stability selection (feature discovery). Not
#             reported directly due to overfitting risk at n=83 events.

CLINICAL_FEATURES = [
    "age",
    "sex_F",
    "months_since_diso",
    "n_hcm_meds",
    "tte_count_12m",
    "hf_flag",
    "symptom_burden_12m",
]

REFINED_FEATURES = [
    "months_since_diso",
    "n_hcm_meds",
    "ccb_ever",
    "bb_current",
    "ccb_current",
    "mri_ever",
]

EXPANDED_FEATURES = [
    "age",
    "sex_F",
    "months_since_diso",
    "n_hcm_meds",
    "hf_flag",
    "bb_ever",
    "ccb_ever",
    "bb_current",
    "ccb_current",
    "months_since_last_med_change",
    "med_switches_12m",
    "mri_ever",
    "strain_count_12m",
    "er_or_inpatient_12m",
    "af_flag",
    "mitral_flag",
    "n_cardiac_classes",
    "antiarrhythmic_ever",
    "anticoagulant_ever",
    "sglt2i_ever",
    "cardiac_drug_days_12m",
    "diso_mpr_12m",
    "cyp_inhibitor_active",
    "dual_bb_ccb_current",
    "echo_acceleration",
    "diso_ccb_combo_current",
]


# ── 4. Rolling-window config ─────────────────────────────────────────────────
#
# Two distinct "window" concepts:
#   ROLLING_WINDOWS     — list of window sizes (months) used during the
#                         stability-selection sweep to discover the best window
#                         per feature. Produces columns like bnp_test_3m,
#                         bnp_test_6m, bnp_test_12m for each feature.
#   PanelConfig.rolling_window — single default window (months) used for
#                         features that are not swept (e.g. strain_count,
#                         er_or_inpatient). Set to 12 months by default.
#
# ROLLING_FEATURE_CODES: maps feature_base_name → (source, code_column, code(s))
#   source   : "diagnoses" | "procedures"  — which RawData table to query
#   code_col : column name to filter on
#   codes    : string or list of strings matching the code values

ROLLING_WINDOWS = [3, 6, 12]

ROLLING_FEATURE_CODES = {
    "tte_count": ("procedures", "px_code", TTE_CODE),
    "symptom_burden": ("diagnoses", "dx_code", SYMPTOM_CODES),
    "stress_count": ("procedures", "px_code", STRESS_CODES),
    "high_em_count": ("procedures", "px_code", HIGH_EM_CODES),
    "bnp_test": ("procedures", "px_code", BNP_CODE),
}

# Auto-generates multi-window column names, e.g. tte_count_3m, tte_count_6m, tte_count_12m.
# These are appended to EXPANDED_FEATURES so stability selection sees all window variants.
_ROLLING_MULTI = [f"{feat}_{win}m" for feat in ROLLING_FEATURE_CODES for win in ROLLING_WINDOWS]
EXPANDED_FEATURES += _ROLLING_MULTI

MONTH_COL = "study_month"


# ── 5. Dataclass configs ─────────────────────────────────────────────────────


@dataclass
class PanelConfig:
    """Configuration for person-month panel construction."""

    risk_set: str = "disopyramide"  # "disopyramide" (conditioning on prior Diso) | "full_ohcm"
    censor_at_first_gap: bool = True  # True = censor at first unenrolled month; False = allow gaps
    min_pre_launch_months: int = 6  # minimum months of pre-launch enrollment required
    rolling_window: int = 12  # default window (months) for non-swept rolling features


@dataclass
class SplitConfig:
    """Temporal train/test split configuration."""

    train_end_month: str = "2023-03"  # last calendar month included in training set (YYYY-MM)


@dataclass
class ModelConfig:
    """Discrete-time hazard model configuration."""

    link: str = "cloglog"  # "cloglog" (log-hazard-ratio) | "logit" (log-odds-ratio)
    baseline: str = "month_dummies"  # baseline hazard specification
    features: list = field(default_factory=lambda: CLINICAL_FEATURES.copy())
    random_seed: int = 42

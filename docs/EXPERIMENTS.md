# Experiments Log

Chronological record of modelling experiments for Task 1 (Camzyos adoption). Each entry documents what was tried, why, and what happened — including negative results.

---

## E1: Clinical prior model (7 features, cloglog)

**What:** Discrete-time hazard GLM with cloglog link on the Disopyramide-conditioned risk set (710 patients, 132 events). 7 features selected a priori on clinical grounds: age, sex, months_since_diso, n_hcm_meds, tte_count_12m, hf_flag, symptom_burden_12m. Month dummies for baseline hazard. Temporal split: train months 1–15 (102 events), test months 16–21 (30 events).

**Result:**
- BSS: +0.002, C-index: 0.63
- Two significant features: `months_since_diso` (HR=0.96, p=0.001) and `n_hcm_meds` (HR=1.68, p=0.002)
- Age, sex, TTE, HF flag, symptom burden all non-significant

**Outcome:** Establishes the baseline. Treatment trajectory features (time since Diso, medication count) carry signal. Demographics and symptom proxies do not. → F2, F10

---

## E2: Expanded features + stability selection (20 features, C=1.0)

**What:** Added 13 features to the clinical set: bb_ever, ccb_ever, bb_current, ccb_current, months_since_last_med_change, med_switches_12m, mri_ever, strain_count_12m, stress_count_12m, high_em_count_12m, er_or_inpatient_12m, af_flag, mitral_flag. Ran stability selection with 200 bootstrap resamples.

**Result:** FAILURE — all 20 features passed the 0.6 threshold because C=1.0 was too weak (insufficient regularisation). The full 20-feature model overfitted: BSS=-0.005, worse than the clinical prior model.

**Outcome:** Stability selection with a fixed C is unreliable. Need CV-tuned regularisation. → Led to E3.

---

## E3: Stability selection with CV-tuned C (20 features)

**What:** Same 20 features, but stability selection now auto-tunes C via 5-fold CV (LogisticRegressionCV). CV selected C=0.127 (much stronger regularisation than C=1.0).

**Result:**
- 7/20 features selected (clear separation): ccb_current (1.00), ccb_ever (1.00), bb_current (1.00), mri_ever (0.97), months_since_diso (0.93), strain_count_12m (0.72), er_or_inpatient_12m (0.64)
- 13 features below 0.28 — clean gap, no ambiguity
- `n_hcm_meds` dropped to π=0.04 — its signal fully captured by ccb_ever/bb_current

**Key finding:** The "ever tried" vs "currently on" distinction captures treatment trajectory precisely (F8). `ccb_ever` (HR=5.80) = escalation depth. `bb_current` (HR=0.07) = actively managed, not switching. `mri_ever` (HR=1.75) = specialist engagement proxy (F9).

**Outcome:** Identified the refined feature set. → F8, F9, F10

---

## E4: Refined model (6 features)

**What:** Hand-picked 6 features based on E3 findings: months_since_diso, n_hcm_meds, ccb_ever, bb_current, ccb_current, mri_ever. Motivated by clinical interpretability + stability selection evidence.

**Result:**
- BSS: +0.001, **C-index: 0.66** (best discrimination achieved)
- Slight overprediction: mean_pred=0.018 vs observed=0.012

**Outcome:** Best C-index of any model. Confirms treatment state + specialist engagement is the signal. → F13

---

## E5: ATC-derived features (26 features)

**What:** Used OMOP ATC vocabulary to classify all prescriptions by therapeutic class. Added 6 features: n_cardiac_classes (distinct ATC 2nd-level cardiac drug classes), antiarrhythmic_ever, anticoagulant_ever, sglt2i_ever, cardiac_drug_days_12m, diso_mpr_12m (Disopyramide Medication Possession Ratio from days_supply).

**Result:**
- Stability selection: `n_cardiac_classes` selected (π=0.94). All other ATC features below threshold.
- `diso_mpr_12m` (π=0.03) — Disopyramide adherence carries zero signal
- `cardiac_drug_days_12m` (π=0.33) — total drug-days adds nothing beyond binary flags
- Full 26-feature model: BSS=-0.013 (worst yet), C-index: 0.66

**Outcome:** ATC classification validates our manual drug groupings but adds no new predictive signal. Days-supply-based features are uninformative. → F11, F12, F13

---

## E6: Heart failure subtypes and intensity

**What:** Compared HF prevalence, subtypes (systolic I5022, diastolic I5032, acute I5023), and claim intensity between Camzyos initiators vs non-initiators in the Disopyramide pool.

**Result:** Near-identical distributions:
- Any HF: 75.8% vs 74.7%
- Systolic: 36.2% vs 36.5%
- Diastolic: 60.4% vs 59.2%
- Acute: 10.1% vs 9.4%
- Claim intensity: median 2, mean 1.9 for both groups

**Outcome:** HF in all its claims-data forms carries no signal. The clinical decision variables (LVOT gradient, NYHA class) are not captured in billing codes. → Confirms F2

---

## E7: Label-derived features (in progress)

**What:** Derived features from the Camzyos FDA label (Section 1–7, 14):
- `bnp_test_12m` — BNP/NT-proBNP test count (83880). Severity assessment marker per label Section 5.1.
- `cyp_inhibitor_active` — Currently on a contraindicated CYP2C19/CYP3A4 inhibitor (omeprazole, esomeprazole, fluconazole, ketoconazole, fluoxetine) with 30-day washout buffer. Hard contraindication per label Section 4.
- `dual_bb_ccb_current` — Concurrent beta-blocker + CCB. EXPLORER-HCM excluded these patients (Section 14).
- `diso_ccb_combo_current` — Concurrent disopyramide + verapamil/diltiazem. Label warns against this combination (Section 5.1, 7.3).
- `echo_acceleration` — Increased TTE frequency in recent 6m vs prior 6m. Label requires echo before initiation and at weeks 4/8/12 (Section 2.1).

**Rationale:** These features capture hard prescribing constraints (contraindications, trial exclusion criteria) rather than soft clinical signals. CYP inhibitor in particular is a binary barrier — if you're on omeprazole, you can't start Camzyos.

**Result:** Three label-derived features enter stability selection (31 candidates, C=0.207):
- `diso_ccb_combo_current` (π=0.98) — Diso+CCB combo is a hard barrier per label §5.1/7.3
- `echo_acceleration` (π=0.86) — increased echo frequency signals pre-initiation workup per label §2.1
- `cyp_inhibitor_active` (π=0.86) — CYP inhibitor with 30-day washout is a contraindication per label §4
- `bnp_test_12m` (π=0.77) — BNP testing as severity assessment per label §5.1/12.2
- `dual_bb_ccb_current` (π=0.00) — near-zero variance in this population, correctly ignored

However, the 14-feature stability-selected model (BSS=-0.022) performs worse than the 6-feature refined model (BSS=+0.001). Too many features for 102 events.

**Outcome:** Label-derived features validate the clinical logic but don't improve prediction. Confirms F13 — the 6-feature model is the ceiling. → F14

---

## E8: Multi-window rolling features (3m, 6m, 12m)

**What:** Replaced fixed 12-month rolling windows with 3m/6m/12m variants for 5 features (TTE count, symptom burden, stress test, high E&M, BNP test). 41 total candidate features. Let stability selection pick the preferred window per feature.

**Result:** Clear window preferences emerge:
- `bnp_test_3m` (π=0.91) >> `bnp_test_12m` (π=0.27) — 3m strongly preferred
- `stress_count_3m` (π=0.77) >> `stress_count_12m` (π=0.21) — 3m strongly preferred
- `symptom_burden_6m` (π=0.74) >> `symptom_burden_12m` (π=0.10) — 6m preferred
- `tte_count_3m` (π=0.57) > `tte_count_12m` (π=0.20) — 3m marginal (below threshold but best variant)

The 12-month window is actively worse for every monitoring feature. However, the 15-feature stability-selected model (BSS=-0.021) still underperforms the 6-feature refined model (BSS=+0.001).

**Outcome:** The 12-month window was diluting recency signal for monitoring/severity features. Shorter windows (3m for workup intensity, 6m for symptoms) are preferred. However, sample size remains the binding constraint — the insight improves methodology for future work with larger data. → F15

---

## E9: Univariate code screening across all codes × windows

**What:** Fisher's exact test (binary ever-in-window) and Mann-Whitney U (count) for every code with ≥10 patient prevalence across 3m/6m/12m windows. 621 codes × 3 windows = 1,863 tests on train set only. Bonferroni correction applied: p_adj = p × n_tests.

**Result:** Only 2 codes survive Bonferroni at p_adj < 0.05:
- Disopyramide Phosphate (3m, OR=9.17, p_adj=2.1e-11) — positive, already defines the risk set
- Metoprolol Succinate (3m, OR=0.062, p_adj=0.022) — negative (being on BB = not switching), captured by `bb_current`

**Interpretation failures:** Several procedure codes (urinalysis, magnesium, PTT) have p < 0.01 raw but p_adj >> 1. These associate with Camzyos via healthcare utilisation confounding — Camzyos patients attend more cardiology visits overall, so they appear for unrelated procedures. The raw test cannot distinguish "on the path to Camzyos" from "high healthcare utiliser."

**Outcome:** Definitive negative. The entire claims vocabulary adds no new predictive signal beyond our hand-engineered features. Confirms F13 and F16. The binding constraint is sample size (102 events), not feature space.

---

## Summary: What works and what doesn't

### Features that carry signal (stability selection π ≥ 0.60, 31 candidates):
| Feature | π | HR | Source | Interpretation |
|---|---|---|---|---|
| ccb_ever | 1.00 | 5.80 | E3 | Deeper escalation → closer to Camzyos |
| bb_current | 1.00 | 0.07 | E3 | Currently managed on BB → not switching now |
| ccb_current | 1.00 | 0.43 | E3 | Currently managed on CCB → not switching now |
| mri_ever | 0.99 | 1.75 | E3 | Specialist workup → at a REMS-certified centre |
| diso_ccb_combo_current | 0.98 | — | E7/label | Diso+CCB = must stop CCB first (label §5.1) |
| n_cardiac_classes | 0.98 | — | E5 | Broader cardiovascular complexity |
| strain_count_12m | 0.94 | — | E3 | Advanced monitoring |
| months_since_diso | 0.88 | 0.96 | E1 | Longer on Diso → stable, less likely to switch |
| echo_acceleration | 0.86 | — | E7/label | Increased echo = pre-initiation workup (label §2.1) |
| cyp_inhibitor_active | 0.86 | — | E7/label | CYP inhibitor contraindication (label §4) |
| er_or_inpatient_12m | 0.83 | — | E3 | Acute events |
| bb_ever | 0.78 | — | E3 | Treatment history depth |
| bnp_test_12m | 0.77 | — | E7/label | Severity assessment (label §5.1) |

### Features that don't carry signal:
| Feature | π | Why it doesn't work |
|---|---|---|
| age | 0.34 | Adoption is prescriber-driven, not age-driven |
| sex_F | 0.34 | Same |
| hf_flag | 0.58 | Claims HF codes ≠ LVOT gradient / NYHA class |
| symptom_burden_12m | 0.41 | Same — billing proxies, not clinical severity |
| n_hcm_meds | 0.08 | Signal fully captured by ccb_ever/bb_current |
| diso_mpr_12m | 0.27 | Adherence doesn't predict switching |
| dual_bb_ccb_current | 0.00 | Near-zero variance in Diso pool |

### Key insight for the IC:
Camzyos adoption within the Disopyramide pool is driven by **treatment trajectory** (escalation depth, current medication state) and **specialist engagement** (cardiac MRI), not by patient demographics, symptom severity, or disease burden as captured in claims. This is a prescriber-driven adoption pattern proxied through treatment history.
